"""Check frozen workload scope and tamper rejection without GPU access."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scoring_benchmark", ROOT / "eval/benchmark_validation_scoring.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class ScoringBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = []
        for source in ("MATH-500", "AMC23", "AIME25"):
            for sample, length in enumerate((2, 5, 9, 21)):
                rows.append(
                    {
                        "state_source": "on_validation_states",
                        "source": source,
                        "uid": source,
                        "sample": sample,
                        "prompt_ids": [1, 3],
                        "response_ids": [6] * (length - 1) + [2],
                    }
                )
        for state in ("on_initial_states", "on_teacher_states"):
            rows.append(
                {
                    "state_source": state,
                    "source": "MATH-500",
                    "uid": "bank",
                    "sample": 0,
                    "prompt_ids": [1, 3],
                    "response_ids": [6, 6, 2],
                    "positions": [0, 2],
                    "weights": [1, 2],
                }
            )
        self.records = self.root / "input.jsonl"
        self.records.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.args = SimpleNamespace(
            output=self.root / "prepared",
            records=self.records,
            per_benchmark=2,
            baseline_commit="b528c6d29425e368ff7add630bfc845f060cb8b8",
        )

    def test_real_git_baseline_is_frozen_with_all_positions_and_unchanged_bank_weights(
        self,
    ):
        manifest = BENCH.prepare(self.args)
        self.assertEqual(manifest["responses"], 8)
        self.assertEqual(manifest["effective_main_positions"], 3 * (2 + 21))
        _, rows = BENCH.frozen_workload(self.args.output)
        self.assertEqual([len(row["response_ids"]) for row in rows[:6]], [2, 21] * 3)
        self.assertEqual(rows[-1]["weights"], [1, 2])
        self.assertEqual(
            set(manifest["baseline_sources_sha256"]), {"core.py", "worker.py"}
        )

    def test_changed_workload_and_changed_baseline_fail_before_model_import(self):
        BENCH.prepare(self.args)
        path = self.args.output / "records.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "workload changed"):
            BENCH.frozen_workload(self.args.output)
        path.write_bytes(original)
        path = self.args.output / "baseline/core.py"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Baseline"):
            BENCH.frozen_workload(self.args.output)

    def test_gpu_admission_rejects_an_eight_gpu_or_unreserved_launch(self):
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "WORLD_SIZE": "8"}
        ):
            with self.assertRaisesRegex(ValueError, "0–3 only"):
                BENCH.run(self.args)


if __name__ == "__main__":
    unittest.main()
