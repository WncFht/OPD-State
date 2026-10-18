"""Reject incomplete or inconsistent native benchmark work before reporting speed."""

import hashlib
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
    def test_scheduler_metadata_and_physical_receipts_are_preserved_and_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = fixture(path)
            summary = json.loads((path / "summary.json").read_text())
            summary["arguments"]["budget"] = 0.4
            summary["scheduler"] = {
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048,
                "chunked_prefill": True,
            }
            summary["physical_receipts_required"] = True
            summary["physical_gpu_uuids"] = {"2": "GPU-card-2"}
            summary["engines"] = {
                "initial_student": {
                    "gpus": [2],
                    "budget_per_engine": 0.4,
                    "layouts": [{"parameter_bytes": 4, "kv_regions": []}],
                }
            }
            telemetry = path / "gpu.csv"
            telemetry.write_text("unix_s,gpu,used_mib,utilization_pct\n0,2,1024,50\n")
            summary["gpu_telemetry"] = {
                "sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
                "sampled_peak_mib": {"2": 1024},
                "errors": [],
            }
            for row in rows:
                row["groups"][0].update(phase=0, slot="initial_student", requests=4)
                row["physical_execution"] = [
                    {
                        "slot": "initial_student",
                        "phase": 0,
                        "replica_rank": 0,
                        "device_uuid": "card-2",
                        "requests": 4,
                        "request_ids": [str(i) for i in range(4)],
                    }
                ]
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            timings, _, receipt = MODULE.collect(path)
            self.assertEqual(timings[0]["max_num_seqs"], 32)
            self.assertEqual(timings[0]["max_num_batched_tokens"], 2048)
            self.assertTrue(timings[0]["chunked_prefill"])
            self.assertEqual(receipt["whole_card_sampled_peak_mib"], {"2": 1024})
            telemetry.write_text("changed")
            with self.assertRaisesRegex(ValueError, "GPU telemetry"):
                MODULE.collect(path)

    def test_missing_warmup_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = fixture(path)[1:]
            summary = json.loads((path / "summary.json").read_text())
            summary["observations"] = 2
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "warmups"):
                MODULE.collect(path)

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
