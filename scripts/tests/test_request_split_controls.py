"""Keep explicit request boundaries and matched APC protocols auditable."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import run_request_split_controls as CONTROL


def fixture(root):
    directory = root / "checkpoints/run/native-trajectories"
    directory.mkdir(parents=True)
    policy = {
        "kind": "sequence_switch",
        "first": "current_student",
        "second": "current_student",
        "boundary": {"kind": "response_tokens", "value": 4},
        "keep_request_boundary": True,
    }
    manifest = {"policy": policy}
    identity = [["current_student", "/student", 0]]
    row = {
        "token_ids": [3] * 8,
        "segments": [
            {"start": 0, "end": 4, "identities": identity},
            {"start": 4, "end": 8, "identities": identity},
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "step-1.jsonl").write_text(json.dumps(row) + "\n")
    return directory, row, manifest


class RequestSplitControlTests(unittest.TestCase):
    def test_protocol_has_six_engineering_checks_and_four_matched_full_kl_conditions(
        self,
    ):
        conditions = CONTROL.cases()
        short = [case for case in conditions if not case["formal"]]
        formal = [case for case in conditions if case["formal"]]
        self.assertEqual(len(short), 6)
        self.assertEqual(len(formal), 4)
        self.assertEqual(
            {case["policy"] for case in short},
            {f"{role}_request_split" for role in ("current", "initial", "teacher")},
        )
        for apc in (False, True):
            pair = [case for case in formal if case["apc"] is apc]
            self.assertEqual(
                {case["policy"] for case in pair},
                {"student_fresh", "current_request_split"},
            )
            for case in pair:
                cmd = CONTROL.command(
                    case,
                    Path("out"),
                    "source",
                    [Path(name) for name in ("math", "amc", "aime")],
                    Path("bank.json"),
                )
                self.assertIn("--full-eval", cmd)
                self.assertIn("trainer.validation_kl.enabled=True", cmd)
                self.assertIn(
                    f"actor_rollout_ref.rollout.enable_prefix_caching={apc}", cmd
                )
                self.assertEqual(cmd[cmd.index("--steps") + 1], "50")
                self.assertEqual(cmd[cmd.index("--batch") + 1], "64")
                self.assertEqual(cmd.count("--validation"), 3)
        for case in short:
            cmd = CONTROL.command(
                case, Path("out"), "source", [Path("subset")], Path("bank.json")
            )
            self.assertNotIn("--full-eval", cmd)
            self.assertIn("trainer.validation_kl.enabled=False", cmd)

    def test_actual_two_segments_keep_one_model_identity_and_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            result = CONTROL.audit_boundary(root, 1, require_split=True)
            self.assertEqual(result["responses"], 1)
            self.assertEqual(result["two_request_responses"], 1)
            self.assertEqual(result["model_switches"], 0)

    def test_wrong_model_version_removed_boundary_and_nonexercised_control_fail(self):
        for problem in ("version", "boundary", "optimized", "stopped"):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                directory, row, manifest = fixture(root)
                if problem == "version":
                    row["segments"][1]["identities"] = [
                        ["current_student", "/student", 1]
                    ]
                elif problem == "boundary":
                    row["segments"][0]["end"] = 3
                elif problem == "optimized":
                    manifest["policy"]["keep_request_boundary"] = False
                else:
                    row["token_ids"] = [3, 2]
                    row["segments"] = [copy.deepcopy(row["segments"][0])]
                    row["segments"][0]["end"] = 2
                (directory / "manifest.json").write_text(json.dumps(manifest))
                (directory / "step-1.jsonl").write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    CONTROL.audit_boundary(root, 1, require_split=True)

    def test_single_request_reference_cannot_silently_use_two_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, row, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "Single-request"):
                CONTROL.audit_boundary(root, 1, require_split=False)
            row["segments"] = [
                {"start": 0, "end": 8, "identities": row["segments"][0]["identities"]}
            ]
            (directory / "step-1.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(
                CONTROL.audit_boundary(root, 1, require_split=False)[
                    "two_request_responses"
                ],
                0,
            )


if __name__ == "__main__":
    unittest.main()
