"""Bind periodic route/version evidence; task grading is tested separately."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import collect_native_acceptance as COLLECT


class PeriodicNativeCollectionTests(unittest.TestCase):
    def fixture(self, directory):
        run, output = directory / "run", directory / "result"
        output.mkdir()
        trajectory = run / "checkpoints/native-trajectories"
        trajectory.mkdir(parents=True)
        source = run / "source/verl/verl/workers/rollout/policy.py"
        source.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "verl/verl/workers/rollout/policy.py", source)
        policy = COLLECT.load_policy(source)
        manifest = {
            "policy": {"kind": "single", "model": "current_student"},
            "generation_models": {},
            "response_cap": 2,
            "seed": 42,
            "trainer_gpu_count": 4,
            "verify_frozen_parameters": False,
        }
        (trajectory / "manifest.json").write_text(json.dumps(manifest))
        for step in range(1, 5):
            plan = policy.compile_policy(
                manifest["policy"], cap=2, batch_size=1, seed=42, batch_id=step
            )
            row = {
                "request": {"prompt_id": "prompt", "visit": step - 1, "sample": 0},
                "route": plan.requests[0].route,
                "token_ids": [5, 2],
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "teacher_weight": None,
                        "identities": [["current_student", "/student", step - 1]],
                    }
                ],
            }
            (trajectory / f"step-{step}.jsonl").write_text(json.dumps(row) + "\n")
        batches = [
            {
                "step": step,
                "validate": True,
                "identities": [["current_student", "/student", step]],
                "metrics": {"source/response_count": 16},
            }
            for step in (2, 4)
        ]
        (trajectory / "batches.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in batches)
        )
        (run / "status.json").write_text(
            json.dumps({"state": "finished", "exit_code": 0})
        )
        (run / "command.json").write_text(
            json.dumps(
                {
                    "command": [],
                    "environment": {"MINI_BATCH_SIZE": "1", "MAX_RESP_LENGTH": "2"},
                }
            )
        )
        (run / "gpu.csv").write_text(
            "elapsed_s,gpu,used_mib,utilization_pct\n"
            + "".join(f"1,{gpu},1000,0\n" for gpu in range(4))
        )
        return run, output, trajectory, batches

    def test_periodic_validation_preserves_final_identity_and_binds_every_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, output, _, _ = self.fixture(Path(temporary))
            with patch.object(
                COLLECT, "collect", return_value={"validation_steps": [2, 4]}
            ) as grading:
                result = COLLECT.audit(run, output, 4, validation_interval=2)
            self.assertEqual(grading.call_args.kwargs["interval"], 2)
            self.assertEqual(result["evaluation_responses"], 16)
            self.assertEqual(result["final_student_version"], 4)
            self.assertEqual(
                result["evaluation_model_identities"],
                [("current_student", "/student", 4)],
            )
            self.assertEqual(set(result["validations_by_step"]), {"2", "4"})
            with self.assertRaisesRegex(ValueError, "final-only"):
                COLLECT.audit(run, output, 4)

    def test_wrong_intermediate_version_and_missing_validation_point_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, output, trajectory, batches = self.fixture(Path(temporary))
            batches[0]["identities"][0][2] = 0
            (trajectory / "batches.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in batches)
            )
            with self.assertRaisesRegex(ValueError, "matching current"):
                COLLECT.audit(run, output, 4, validation_interval=2)
            (trajectory / "batches.jsonl").write_text(json.dumps(batches[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "interval"):
                COLLECT.audit(run, output, 4, validation_interval=2)


if __name__ == "__main__":
    unittest.main()
