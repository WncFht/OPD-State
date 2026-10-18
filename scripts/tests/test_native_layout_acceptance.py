"""Reject physical source misrouting and inconsistent diagnostic reductions."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
spec = importlib.util.spec_from_file_location(
    "native_layout_acceptance", ROOT / "eval/verify_native_layout.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def fixture():
    counts = Counter({"current_student": 7, "teacher": 1})
    batch = {
        "step": 1,
        "validate": False,
        "generation": {
            "groups": [
                {"slot": slot, "phase": 0, "requests": n} for slot, n in counts.items()
            ]
        },
        "generation_placement": [
            {
                "layout": "split",
                "requests": counts,
                "replicas": {"current_student": [0, 1, 2], "teacher": [3]},
            }
        ],
        "generation_execution": [
            {
                "slot": source,
                "phase": 0,
                "replica_rank": rank,
                "device_uuid": f"GPU-card-{rank}",
                "pid": 100 + rank,
                "requests": count,
                "request_ids": [f"{source}-{rank}-{i}" for i in range(count)],
            }
            for source, rank, count in [
                ("current_student", 0, 3),
                ("current_student", 1, 2),
                ("current_student", 2, 2),
                ("teacher", 3, 1),
            ]
        ],
    }
    return batch, {0: counts}


class NativeLayoutAcceptanceTests(unittest.TestCase):
    def test_cuda_aliases_canonicalize_to_physical_nvml_before_rank_uniqueness_checks(
        self,
    ):
        batch, expected = fixture()
        aliases = {f"card-{i}": f"card-{i}" for i in range(4)}
        aliases.update({f"cuda-{i}": f"card-{i}" for i in range(4)})
        for row in batch["generation_execution"]:
            row["device_uuid"] = f"GPU-cuda-{row['replica_rank']}"
        ranks = {}
        verify.check_batch(batch, expected, 4, ranks, aliases)
        self.assertEqual(ranks, {i: f"card-{i}" for i in range(4)})
        # A different alias of the same hardware must not make two ranks valid.
        batch["generation_execution"][1]["device_uuid"] = "GPU-card-0"
        with self.assertRaisesRegex(ValueError, "share one physical GPU"):
            verify.check_batch(batch, expected, 4, {}, aliases)

    def test_plan_only_preserves_short_acceptance_and_unhashed_abba_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = path / "manifest.json"
            manifest.write_text("{}")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "eval/run_native_followups.py"),
                    "--plan-only",
                    "--source-commit",
                    "HEAD",
                    "--data-manifest",
                    str(manifest),
                    "--output",
                    str(path / "plan"),
                ],
                check=True,
                cwd=ROOT,
            )
            state = json.loads((path / "plan/state.json").read_text())
            self.assertEqual(state["state"], "planned")
            self.assertEqual(state["validation_questions"], 6)
            self.assertEqual(state["validation_responses"], 96)
            self.assertEqual(len(state["cases"]), 10)
            correctness, observations = state["cases"][:6], state["cases"][6:]
            self.assertTrue(
                all(c["verify_frozen"] and c["steps"] == 2 for c in correctness)
            )
            self.assertTrue(
                all(not c["verify_frozen"] and c["steps"] == 5 for c in observations)
            )
            self.assertEqual(
                [c["layout"] for c in observations],
                ["overlap", "split", "split", "overlap"],
            )
            self.assertTrue(
                all("--full-eval" not in c["command"] for c in state["cases"])
            )
            for case in state["cases"]:
                command = case["command"]
                values = [
                    command[i + 1]
                    for i, value in enumerate(command)
                    if value == "--validation"
                ]
                self.assertEqual(len(values), 3)
                self.assertTrue(all("validation-subset" in value for value in values))

    def test_actual_worker_shards_match_split_source_quotas_and_physical_devices(self):
        batch, expected = fixture()
        ranks = {}
        rows = verify.check_batch(
            batch, expected, 4, ranks, {f"card-{i}" for i in range(4)}
        )
        self.assertEqual(sum(row["requests"] for row in rows), 8)
        self.assertEqual(ranks, {i: f"card-{i}" for i in range(4)})
        self.assertEqual(rows[-1]["slot"], "teacher")

    def test_wrong_device_duplicate_request_short_shard_and_missing_receipt_are_rejected(
        self,
    ):
        batch, expected = fixture()
        for mutation in ["device", "rank", "request", "count", "missing"]:
            changed = copy.deepcopy(batch)
            records = changed["generation_execution"]
            if mutation == "device":
                records[0]["device_uuid"] = "GPU-foreign-card"
            elif mutation == "rank":
                records[0]["replica_rank"] = 7
            elif mutation == "request":
                records[-1]["request_ids"] = records[0]["request_ids"][:1]
            elif mutation == "count":
                records[0]["requests"] = 2
            else:
                records.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.check_batch(
                    changed, expected, 4, {}, {f"card-{i}" for i in range(4)}
                )

    def test_replica_device_changes_across_batches_are_rejected(self):
        batch, expected = fixture()
        with self.assertRaisesRegex(ValueError, "changed its physical device"):
            verify.check_batch(
                batch, expected, 4, {0: "card-3"}, {f"card-{i}" for i in range(4)}
            )

    def test_action_weighted_statistics_reconcile_with_opd_reward(self):
        actions = {"teacher": 2, "current_student": 1}
        metrics = {
            "native_state/all/tokens": 3,
            "opd/token_reward_mean": -1 / 3,
            "native_state/all/sampled_log_ratio_mean": 1 / 3,
            "native_state/all/log_ratio_rms": 1,
            "native_state/all/teacher_selected_action_probability_mean": 1.25 / 3,
        }
        for name, count, delta, probability in [
            ("teacher", 2, 1, 0.5),
            ("current_student", 1, -1, 0.25),
        ]:
            prefix = f"native_state/action_{name}/"
            metrics.update(
                {
                    prefix + "tokens": count,
                    prefix + "sampled_log_ratio_mean": delta,
                    prefix + "log_ratio_rms": 1,
                    prefix + "teacher_selected_action_probability_mean": probability,
                }
            )
        verify.check_metrics(metrics, actions)
        for key, value in [
            ("native_state/action_teacher/tokens", 3),
            ("native_state/all/log_ratio_rms", 2),
            (
                "native_state/action_teacher/teacher_selected_action_probability_mean",
                1.5,
            ),
            ("opd/token_reward_mean", 0),
        ]:
            changed = {**metrics, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.check_metrics(changed, actions)


if __name__ == "__main__":
    unittest.main()
