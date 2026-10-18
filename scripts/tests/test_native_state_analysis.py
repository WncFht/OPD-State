"""Check prefix-state counting against explicit state sets and raw trace files."""

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "eval/analyze_native_states.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("native_state_analysis", SOURCE)
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def trajectory(tokens, *, prompt="question", visit=0, sample=0, step=1):
    return {
        "request": {
            "prompt_id": prompt,
            "prompt_ids": [10, 11],
            "visit": visit,
            "sample": sample,
        },
        "token_ids": tokens,
        "step": step,
        "termination": "eos" if tokens[-1] == 2 else "length",
        "segments": [
            {
                "identities": [["teacher", "/q", "sha"]],
                "start": 0,
                "end": len(tokens),
                "teacher_weight": None,
            }
        ],
    }


class StateAnalysisTests(unittest.TestCase):
    def test_exact_overlap_matches_explicit_states_including_eos_and_short_prefix(self):
        cases = [([3, 2], [3, 2]), ([3, 2], [3, 4, 2]), ([3], [4]), ([3], [3, 4, 2])]
        rng = random.Random(42)
        cases += [
            (
                [rng.randrange(8) for _ in range(rng.randint(1, 20))],
                [rng.randrange(8) for _ in range(rng.randint(1, 20))],
            )
            for _ in range(200)
        ]
        for left, right in cases:
            states_left = {tuple(left[:i]) for i in range(len(left))}
            states_right = {tuple(right[:i]) for i in range(len(right))}
            _, shared, jaccard = analysis.shared_prefix_states(left, right)
            self.assertEqual(shared, len(states_left & states_right))
            self.assertAlmostEqual(
                jaccard,
                len(states_left & states_right) / len(states_left | states_right),
            )
        self.assertEqual(
            analysis.shared_prefix_states([3, 2], [3, 4, 2]), (1, 2, 2 / 3)
        )
        with self.assertRaises(ValueError):
            analysis.shared_prefix_states([], [3])

    def test_pairs_use_question_visit_sample_and_restore_candidate_order(self):
        left = [trajectory([3, 2], visit=0), trajectory([4, 2], visit=1)]
        right = [trajectory([4, 2], visit=1), trajectory([3, 5, 2], visit=0)]
        pairs = analysis.compare(left, right)
        self.assertEqual([row["visit"] for row in pairs], [1, 0])
        self.assertEqual([row["shared_sampled_states"] for row in pairs], [2, 2])
        with self.assertRaisesRegex(ValueError, "exactly the same"):
            analysis.compare(left, right[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.compare(left, right + right[:1])
        right[0]["request"]["prompt_ids"] = [12]
        with self.assertRaisesRegex(ValueError, "question tokens"):
            analysis.compare(left, right)

    def test_switch_statistics_keep_action_policy_and_inherited_state_depth_distinct(
        self,
    ):
        row = trajectory([3, 4, 5, 6, 2])
        row["segments"] = [
            {
                "identities": [["teacher", "/q", "sha"]],
                "start": 0,
                "end": 3,
                "teacher_weight": None,
            },
            {
                "identities": [["current_student", "/p", 0]],
                "start": 3,
                "end": 5,
                "teacher_weight": None,
            },
        ]
        stats = analysis.summarize([row], 5)
        self.assertEqual(stats["sampled_training_positions"], 5)
        self.assertEqual(stats["termination"], {"eos": 1})
        self.assertEqual(stats["response_survival_at_depth"], {"0": 1, "1": 1})
        self.assertEqual(
            stats["action_policy_segments"]["teacher"]["mean_response_prefix_depth"], 1
        )
        self.assertEqual(
            stats["action_policy_segments"]["current_student"][
                "mean_response_prefix_depth"
            ],
            3.5,
        )

    def test_real_trace_reader_rejects_partial_steps_and_analysis_records_input_hashes(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [root / "reference", root / "candidate"]
            for run, tokens in zip(runs, [[3, 2], [3, 4, 2]], strict=True):
                trace = run / "checkpoints/native-trajectories"
                trace.mkdir(parents=True)
                (trace / "manifest.json").write_text(
                    json.dumps(
                        {
                            "train_files": {"train": "data-sha"},
                            "response_cap": 4,
                            "prompt_cap": 4,
                            "eos_token_id": 2,
                        }
                    )
                )
                (run / "command.json").write_text(
                    json.dumps({"environment": {"MINI_BATCH_SIZE": "1"}})
                )
                (trace / "step-1.jsonl").write_text(
                    json.dumps(trajectory(tokens)) + "\n"
                )
            result = analysis.analyze(*runs, root / "result", 1)
            self.assertEqual(result["paired"]["shared_sampled_states"], 2)
            self.assertEqual(result["by_step"]["1"]["identical_responses"], 0)
            self.assertEqual(result["by_step"]["1"]["first_action_agreement"], 1)
            self.assertAlmostEqual(
                result["paired"]["pooled_state_jaccard_on_matched_trajectories"], 2 / 3
            )
            self.assertEqual(
                result["paired_csv_sha256"],
                analysis.sha256(root / "result/paired-prefixes.csv"),
            )
            trace_path = runs[1] / "checkpoints/native-trajectories/step-1.jsonl"
            self.assertEqual(
                result["candidate"]["files_sha256"]["step-1.jsonl"],
                analysis.sha256(trace_path),
            )
            trace_path.write_text("")
            with self.assertRaisesRegex(ValueError, "incomplete global batch"):
                analysis.analyze(*runs, root / "incomplete", 1)
            trace_path.write_text(json.dumps(trajectory([2, 3])) + "\n")
            with self.assertRaisesRegex(ValueError, "after EOS"):
                analysis.analyze(*runs, root / "corrupt", 1)


if __name__ == "__main__":
    unittest.main()
