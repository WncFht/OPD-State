"""Check workload identity, response routing and absent GPU measurements."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
SPEC = importlib.util.spec_from_file_location(
    "generation_benchmark", ROOT / "eval/benchmark_validation_generation.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class GenerationBenchmarkTests(unittest.TestCase):
    def prepared(self):
        return {
            "questions": [
                {
                    "source": source,
                    "uid": str(i),
                    "selection_key": str(i),
                    "prompt_ids": [1, 3],
                }
                for source in ("MATH-500", "AMC23", "AIME25")
                for i in range(30)
            ]
        }

    def test_32_prompts_times_16_samples_route_without_changing_the_question_budget(
        self,
    ):
        rows = BENCH.requests(self.prepared())
        self.assertEqual(len(rows), 512)
        self.assertEqual(len({row["key"] for row in rows}), 512)
        self.assertEqual(len({row["seed"] for row in rows}), 512)
        for gpus in (4, 8):
            ranks = [rows[rank::gpus] for rank in range(gpus)]
            self.assertEqual({len(rank) for rank in ranks}, {512 // gpus})
            self.assertEqual(
                {row["key"] for rank in ranks for row in rank},
                {row["key"] for row in rows},
            )
        self.assertEqual(rows, BENCH.requests(self.prepared()))

    def test_small_question_pool_and_undefined_metrics_do_not_produce_false_evidence(
        self,
    ):
        with self.assertRaises(ValueError):
            BENCH.requests({"questions": []})
        data = BENCH.clean_metrics(
            {"hit_rate": float("nan"), "other": [float("inf"), 0.0]}
        )
        self.assertIsNone(data["hit_rate"])
        self.assertIsNone(data["other"][0])
        self.assertEqual(data["other"][1], 0.0)
        self.assertTrue(math.isfinite(data["other"][1]))


if __name__ == "__main__":
    unittest.main()
