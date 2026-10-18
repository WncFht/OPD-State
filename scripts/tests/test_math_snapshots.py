"""Distinguish complete validation snapshots from a completed training run."""

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

COLLECTOR = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "eval/collect_math_experiment.py")
)


def fixture(base):
    run = base / "run"
    for name in ("logs", "source", "validation"):
        (run / name).mkdir(parents=True)
    (run / "source/COMMIT").write_text("test-revision\n")
    raw = [
        {
            "step": 2,
            "data_source": source,
            "validation_uid": source,
            "score": int(i < 8),
            "input": f"question-{source}",
            "gts": "0",
        }
        for source in ("MATH-500", "AMC23", "AIME25")
        for i in range(16)
    ]
    metrics, _ = COLLECTOR["MATH"].exact_math_metrics(
        [r["data_source"] for r in raw],
        [r["validation_uid"] for r in raw],
        [r["score"] for r in raw],
        samples_per_question=16,
    )
    (run / "validation/2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in raw)
    )
    log = run / "logs/train.log"
    log.write_text(
        "step:1 - actor/loss:0.1\nstep:2 - actor/loss:0.05 - "
        + " - ".join(f"{k}:{v}" for k, v in metrics.items())
        + "\n"
    )
    return run, log


class SnapshotTests(unittest.TestCase):
    def test_runtime_snapshot_is_hashed_instead_of_claiming_only_its_base_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, log = fixture(base)
            with log.open("a") as stream:
                stream.write("Final validation metrics:\n")
            source = run / "source/runtime.py"
            source.write_text("runtime = 1\n")
            manifest = {
                "base_commit": "base-revision",
                "files": {
                    "runtime.py": hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
            (run / "source-manifest.json").write_text(json.dumps(manifest))
            result = COLLECTOR["collect"](
                run, base / "final", steps=2, interval=2, require_full=False
            )
            self.assertEqual(result["source_identity"], "runtime_snapshot")
            self.assertEqual(result["source_commit"], "base-revision")
            source.write_text("runtime = 2\n")
            with self.assertRaisesRegex(ValueError, "snapshot does not match"):
                COLLECTOR["collect"](
                    run, base / "changed", steps=2, interval=2, require_full=False
                )

    def test_runtime_snapshot_rejects_empty_or_escaping_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, log = fixture(base)
            with log.open("a") as stream:
                stream.write("Final validation metrics:\n")
            for files in [{}, {"../outside.py": "ignored"}]:
                (run / "source-manifest.json").write_text(
                    json.dumps({"base_commit": "base", "files": files})
                )
                with self.assertRaises(ValueError):
                    COLLECTOR["collect"](
                        run, base / "refused", steps=2, interval=2, require_full=False
                    )

    def test_live_snapshot_ignores_later_rows_and_partial_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, log = fixture(base)
            with log.open("a") as stream:
                stream.write("step:3 - actor/loss:0.02\nstep:4 - ")
            (run / "validation/4.jsonl").write_text("{unfinished")
            result = COLLECTOR["collect"](
                run,
                base / "snapshot",
                steps=2,
                interval=2,
                require_full=False,
                snapshot=True,
            )
            self.assertEqual(result["collection_scope"], "through_step_snapshot")
            self.assertFalse(result["final_validation_marker_verified"])
            self.assertEqual(result["validation_steps"], [2])
            prefix = log.read_bytes()[: result["training_log_observed_bytes"]]
            self.assertEqual(
                hashlib.sha256(prefix).hexdigest(), result["training_log_sha256"]
            )

    def test_default_mode_still_requires_final_validation_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _ = fixture(base)
            with self.assertRaisesRegex(ValueError, "final validation success"):
                COLLECTOR["collect"](
                    run, base / "final", steps=2, interval=2, require_full=False
                )

    def test_snapshot_requires_logged_validation_metrics_and_all_replicas(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, log = fixture(base)
            original = log.read_text()
            log.write_text("step:1 - actor/loss:0.1\n")
            with self.assertRaisesRegex(ValueError, "exactly steps"):
                COLLECTOR["collect"](
                    run,
                    base / "missing",
                    steps=2,
                    interval=2,
                    require_full=False,
                    snapshot=True,
                )
            log.write_text(original)
            path = run / "validation/2.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
            with self.assertRaises(ValueError):
                COLLECTOR["collect"](
                    run,
                    base / "incomplete",
                    steps=2,
                    interval=2,
                    require_full=False,
                    snapshot=True,
                )

    def test_snapshot_does_not_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _ = fixture(base)
            output = base / "snapshot"
            output.mkdir()
            (output / "existing.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                COLLECTOR["collect"](
                    run,
                    output,
                    steps=2,
                    interval=2,
                    require_full=False,
                    snapshot=True,
                )


if __name__ == "__main__":
    unittest.main()
