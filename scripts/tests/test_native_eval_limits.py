"""Reject declared limits that disagree with physical native worker receipts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from run_native_eval_limits import cases, command
from verify_native_engine_limits import verify


def fixture(root):
    directory = root / "checkpoints/native-trajectories"
    directory.mkdir(parents=True)
    engines = {
        "current_student": {
            "max_num_seqs": 8,
            "max_num_batched_tokens": 1088,
            "context_limit": 1088,
        },
        "eval:current_student": {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1152,
            "context_limit": 1152,
        },
    }
    manifest = {"engines": engines}
    batches = []
    for step, evaluation in ((1, False), (2, False), (2, True)):
        slot = "eval:current_student" if evaluation else "current_student"
        receipts = [
            {
                "slot": slot,
                "replica_rank": rank,
                "device_uuid": f"GPU-card-{rank}",
                "max_num_seqs": engines[slot]["max_num_seqs"],
                "max_num_batched_tokens": engines[slot]["max_num_batched_tokens"],
                "max_model_len": engines[slot]["context_limit"],
            }
            for rank in range(4)
        ]
        batches.append(
            {"step": step, "validate": evaluation, "generation_execution": receipts}
        )
    return directory, manifest, batches


def save(directory, manifest, batches):
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "batches.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in batches)
    )


class NativeEvalLimitTests(unittest.TestCase):
    def test_control_protocol_holds_training_batch_fixed_and_exercises_default_input(
        self,
    ):
        configs = cases()
        self.assertEqual(len(configs), 3)
        self.assertEqual(
            [case["evaluation_sequences"] for case in configs], [None, 64, 64]
        )
        self.assertEqual(
            {case["policy"] for case in configs},
            {"student_fresh", "initial_teacher_fusion"},
        )
        for case in configs:
            cmd = command(case, Path("output"), "frozen", [Path("validation")])
            self.assertEqual(cmd[cmd.index("--batch") + 1], "8")
            self.assertEqual(cmd[cmd.index("--steps") + 1], "2")
            self.assertEqual(cmd[cmd.index("--val-response-cap") + 1], "128")
            self.assertEqual(
                any("generation_eval_max_num_seqs=" in value for value in cmd),
                case["evaluation_sequences"] is not None,
            )

    def test_train_and_eval_limits_are_checked_from_all_actual_worker_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, manifest, batches = fixture(root)
            save(folder, manifest, batches)
            result = verify(
                root,
                {i: f"GPU-card-{i}" for i in range(4)},
                training_sequences=8,
                evaluation_sequences=64,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["observed_ranks"]["eval:current_student"], [0, 1, 2, 3]
            )
            self.assertEqual(result["scheduler_receipt_counts"]["current_student"], 8)

    def test_declared_values_cannot_hide_wrong_limits_devices_or_missing_validation(
        self,
    ):
        for problem in (
            "eval_limit",
            "train_limit",
            "context",
            "old_receipt",
            "gpu",
            "missing_rank",
            "no_eval",
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                folder, manifest, batches = fixture(root)
                if problem == "eval_limit":
                    batches[-1]["generation_execution"][0]["max_num_seqs"] = 8
                elif problem == "train_limit":
                    batches[0]["generation_execution"][0]["max_num_seqs"] = 64
                elif problem == "context":
                    batches[-1]["generation_execution"][0]["max_model_len"] = 1088
                elif problem == "old_receipt":
                    batches[-1]["generation_execution"][0].pop("max_num_seqs")
                elif problem == "gpu":
                    batches[-1]["generation_execution"][0]["device_uuid"] = (
                        "GPU-foreign"
                    )
                elif problem == "missing_rank":
                    batches[-1]["generation_execution"].pop()
                else:
                    batches.pop()
                save(folder, manifest, batches)
                with self.assertRaises(ValueError):
                    verify(
                        root,
                        {i: f"GPU-card-{i}" for i in range(4)},
                        training_sequences=8,
                        evaluation_sequences=64,
                    )


if __name__ == "__main__":
    unittest.main()
