"""Verify historical checkpoints independently of a run's mutable summary."""

import csv
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

VERIFY = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "eval/verify_wandb_math.py")
)["verify"]


def fixture(directory, *, snapshot=True):
    metrics = {"eval/macro/mean@16": 0.5, "eval/AIME25/pass@16": 0.6}
    summary = {
        "wandb_url": "https://wandb.ai/test/OPD-State/runs/abc",
        "training_steps": 2,
        "validation_steps": [2],
        "source_commit": "test-revision",
    }
    if snapshot:
        summary["collection_scope"] = "through_step_snapshot"
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "validation-step2.json").write_text(json.dumps({"metrics": metrics}))
    with (directory / "training-1-2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["step", "actor/loss", "actor/grad_norm"]
        )
        writer.writeheader()
        for i in (1, 2):
            writer.writerow({"step": i, "actor/loss": i / 10, "actor/grad_norm": i})
    rows = [
        {
            "training/global_step": i,
            "actor/loss": i / 10,
            "actor/grad_norm": i,
        }
        for i in range(1, 5)
    ]
    rows[1].update(metrics)
    rows[3].update({key: 0.9 for key in metrics})
    run = SimpleNamespace(
        id="abc",
        url=summary["wandb_url"],
        state="running",
        summary={key: 0.9 for key in metrics},
        config={
            "trainer": {
                "source_commit": "test-revision",
                "save_freq": -1,
                "math_eval_samples": 16,
                "math_eval_require_full": True,
            },
            "actor_rollout_ref": {
                "model": {"path": "student"},
                "rollout": {
                    "val_kwargs": {"n": 16},
                    "prompt_length": 1024,
                    "response_length": 4096,
                },
            },
            "distillation": {"fsdp_teacher": {"model_path": "teacher"}},
            "data": {
                "train_files": ["train"],
                "val_files": ["validation"],
                "seed": 42,
                "train_batch_size": 64,
            },
        },
        scan_history=lambda *, keys, page_size: (
            {key: row.get(key) for key in keys} for row in rows
        ),
    )
    return run, SimpleNamespace(run=lambda path: run), rows, metrics


class WandbSnapshotTests(unittest.TestCase):
    def test_snapshot_uses_exact_history_not_future_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run, api, _, metrics = fixture(directory)
            report = VERIFY(directory, api, snapshot=True)
            self.assertEqual(report["metrics"], metrics)
            self.assertEqual(report["observed_remote_validation_steps"], [2, 4])
            self.assertEqual(report["history_training_rows"], 2)
            self.assertEqual(report["observed_remote_training_rows"], 4)
            self.assertEqual(report["state"], run.state)
            self.assertEqual(
                report["verification_scope"], "completed_validation_prefix"
            )
            with self.assertRaises(ValueError):
                VERIFY(directory, api)

    def test_default_verification_requires_finished_complete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run, api, rows, metrics = fixture(directory, snapshot=False)
            with self.assertRaises(ValueError):
                VERIFY(directory, api)
            run.state = "finished"
            with self.assertRaises(ValueError):
                VERIFY(directory, api)
            del rows[2:]
            run.summary = metrics
            self.assertEqual(VERIFY(directory, api)["metrics"], metrics)
            with self.assertRaises(ValueError):
                VERIFY(directory, api, snapshot=True)

    def test_nonmacro_metric_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, api, rows, _ = fixture(directory)
            rows[1]["eval/AIME25/pass@16"] = 0.7
            with self.assertRaisesRegex(ValueError, "snapshot metric"):
                VERIFY(directory, api, snapshot=True)

    def test_missing_training_and_changed_gradient_are_rejected(self):
        for key, value in (("actor/grad_norm", 99), ("actor/loss", float("nan"))):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _, api, rows, _ = fixture(directory)
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    VERIFY(directory, api, snapshot=True)
                del rows[0]
                with self.assertRaises(ValueError):
                    VERIFY(directory, api, snapshot=True)


if __name__ == "__main__":
    unittest.main()
