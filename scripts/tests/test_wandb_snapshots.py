"""Verify historical checkpoints independently of a run's mutable summary."""

import csv
import hashlib
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
    def test_partial_native_verification_is_explicit_and_checks_exact_kl_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run, api, _ = self.native_fixture(directory)
            run.config["trainer"]["math_eval_require_full"] = False
            run.config["trainer"]["validation_kl"] = {
                "enabled": True,
                "position_chunk_size": 32,
                "vocab_chunk_size": 8192,
            }
            summary_path = directory / "summary.json"
            local = json.loads(summary_path.read_text())
            local["full_benchmarks"] = False
            summary_path.write_text(json.dumps(local))
            with self.assertRaisesRegex(ValueError, "engineering"):
                VERIFY(directory, api, snapshot=True)
            original_scan = run.scan_history
            kl_metrics = {
                "validation_kl/on_validation_states/micro/responses": 96,
                "validation_kl/on_validation_states/micro/student_teacher_kl/question_mean": 0.1,
            }

            def scan(*, keys, page_size):
                values = list(original_scan(keys=keys, page_size=page_size))
                for row in values:
                    if row["training/global_step"] == 2:
                        row.update({k: v for k, v in kl_metrics.items() if k in keys})
                return iter(values)

            run.scan_history = scan
            kl = directory / "kl.json"
            data = {
                "student_version": 2,
                "position_chunk_size": 32,
                "vocab_chunk_size": 8192,
                "metrics": kl_metrics,
            }
            kl.write_text(json.dumps(data))
            report = VERIFY(
                directory, api, snapshot=True, engineering=True, kl_report=kl
            )
            self.assertEqual(report["quality_scope"], "engineering_only")
            self.assertEqual(report["kl_verification"]["metrics"], kl_metrics)
            bad = json.loads(kl.read_text())
            bad["metrics"][next(iter(kl_metrics))] = 95
            kl.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "KL metrics"):
                VERIFY(directory, api, snapshot=True, engineering=True, kl_report=kl)

    def native_fixture(self, directory):
        run, api, rows, metrics = fixture(directory)
        policy = {"kind": "single", "model": "initial_student"}
        run.config["actor_rollout_ref"]["rollout"].update(
            policy=policy, generation_backend="vllm_native"
        )
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["run_directory"] = str(directory / "run")
        summary_path.write_text(json.dumps(summary))
        manifest = directory / "run/checkpoints/actor/native-trajectories/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"policy": policy}))
        (directory / "native-acceptance.json").write_text(
            json.dumps(
                {
                    "native_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest()
                }
            )
        )
        return run, api, manifest

    def test_native_policy_is_verified_and_not_mislabeled_as_legacy_student_online(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run, api, manifest = self.native_fixture(directory)
            report = VERIFY(directory, api, snapshot=True)
            self.assertEqual(report["settings"]["rollout_source"], "native_policy")
            self.assertEqual(
                report["settings"]["rollout_policy"]["model"], "initial_student"
            )
            self.assertEqual(
                report["native_policy_verification"]["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )

    def test_remote_native_policy_or_local_generation_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run, api, manifest = self.native_fixture(directory)
            run.config["actor_rollout_ref"]["rollout"]["policy"] = {
                "kind": "single",
                "model": "teacher",
            }
            with self.assertRaisesRegex(ValueError, "policy"):
                VERIFY(directory, api, snapshot=True)
            run.config["actor_rollout_ref"]["rollout"]["policy"] = {
                "kind": "single",
                "model": "initial_student",
            }
            manifest.write_text(
                json.dumps(
                    {
                        "policy": run.config["actor_rollout_ref"]["rollout"]["policy"],
                        "changed": True,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "policy"):
                VERIFY(directory, api, snapshot=True)

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
