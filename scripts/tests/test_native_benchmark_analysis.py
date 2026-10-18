"""Reject incomplete or inconsistent native benchmark work before reporting speed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "native_benchmark_analysis", ROOT / "eval/summarize_native_benchmarks.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def fixture(directory):
    summary = {
        "arguments": {"layout": "overlap", "replicas": 4, "graph": True, "repeats": 2},
        "engines": {},
        "cold_seconds": {},
        "observations": 3,
    }
    rows = [
        {
            "policy": {"kind": "single", "model": "initial_student"},
            "batch": 4,
            "cap": 8,
            "repeat": repeat,
            "natural_eos": False,
            "tokens": 32,
            "seconds": seconds,
            "tokens_per_second": 32 / seconds,
            "responses_per_second": 4 / seconds,
            "groups": [{"input_tokens": 40, "output_tokens": 32}],
        }
        for repeat, seconds in [(-1, 4), (0, 2), (1, 1)]
    ]
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "observations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return rows


class NativeBenchmarkAnalysisTests(unittest.TestCase):
    def test_warmup_is_excluded_and_work_is_reconciled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            fixture(path)
            timing, _, receipt = MODULE.collect(path)
            self.assertEqual(timing[0]["seconds_mean"], 1.5)
            self.assertEqual(timing[0]["tokens_per_second_mean"], 24)
            self.assertEqual(receipt["observations"], 3)

    def test_short_work_missing_repeat_and_false_throughput_are_rejected(self):
        for change in ["tokens", "repeat", "rate"]:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                rows = fixture(path)
                if change == "tokens":
                    rows[1]["tokens"] = 16
                elif change == "repeat":
                    rows[2]["repeat"] = 0
                else:
                    rows[1]["tokens_per_second"] = 1000
                (path / "observations.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    MODULE.collect(path)


if __name__ == "__main__":
    unittest.main()
