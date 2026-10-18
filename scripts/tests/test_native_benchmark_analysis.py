"""Reject incomplete or inconsistent native benchmark work before reporting speed."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
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
    def test_cuda_and_nvml_uuid_aliases_require_the_recorded_physical_index_binding(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = fixture(path)
            summary = json.loads((path / "summary.json").read_text())
            summary.update(
                physical_receipts_required=True,
                physical_gpu_uuids={"0": "GPU-nvml-card"},
                physical_gpu_identities={
                    "0": {
                        "physical_index": 0,
                        "nvml_uuid": "GPU-nvml-card",
                        "cuda_uuid": "GPU-cuda-card",
                        "cuda_uuid_v2": "GPU-cuda-card",
                        "pci_bus_id": "0000:63:00.0",
                    }
                },
            )
            summary["engines"] = {
                "initial_student": {
                    "gpus": [0],
                    "budget_per_engine": 0.4,
                    "layouts": [{"parameter_bytes": 1, "kv_regions": []}],
                }
            }
            for row in rows:
                row["groups"][0].update(phase=0, slot="initial_student", requests=4)
                row["physical_execution"] = [
                    {
                        "slot": "initial_student",
                        "phase": 0,
                        "replica_rank": 0,
                        "device_uuid": "GPU-cuda-card",
                        "requests": 4,
                        "request_ids": list(map(str, range(4))),
                    }
                ]
            self.save_fixture(path, summary, rows)
            MODULE.collect(path)
            summary["physical_gpu_identities"]["0"]["physical_index"] = 1
            self.save_fixture(path, summary, rows)
            with self.assertRaisesRegex(ValueError, "index/UUID"):
                MODULE.collect(path)

    def staged_fixture(self, path):
        import numpy as np

        rows = fixture(path)
        summary = json.loads((path / "summary.json").read_text())
        summary["arguments"]["replicas"] = 1
        source = ["initial_student", "/p0", "fp-p0"]
        summary.update(
            weight_mode="staged",
            effective_vocab_size=3,
            model_refs={
                "initial_student": source,
                "teacher": ["teacher", "/q", "fp-q"],
            },
            physical_receipts_required=True,
            physical_gpu_uuids={"0": "GPU-card-0"},
            engines={
                "current_student": {
                    "gpus": [0],
                    "budget_per_engine": 0.4,
                    "layouts": [{"parameter_bytes": 4, "kv_regions": []}],
                }
            },
            numerical_checks=[
                {
                    "phase": phase,
                    "source": role,
                    "reference_created": False,
                    "max_logit_error": 0,
                    "max_tv": 0,
                    "logit_tolerance": 0.5,
                    "tv_tolerance": 0.05,
                }
                for phase in ("before_measurement", "after_measurement")
                for role in ("initial_student", "teacher")
            ],
        )
        reference = path / "reference"
        reference.mkdir()
        arrays = {}
        for role in summary["model_refs"]:
            file = reference / f"{role}.npy"
            np.save(file, np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32))
            arrays[role] = hashlib.sha256(file.read_bytes()).hexdigest()
        metadata = reference / "reference.json"
        metadata.write_text(
            json.dumps(
                {
                    "models": summary["model_refs"],
                    "arrays": arrays,
                    "producer": {"weight_mode": "resident_serial"},
                    "prompts": [[1, 2], [3, 4]],
                    "vocabulary": 3,
                }
            )
        )
        for check in summary["numerical_checks"]:
            observed = path / f"{check['phase']}-{check['source']}.npy"
            observed.write_bytes((reference / f"{check['source']}.npy").read_bytes())
            check.update(
                prefix_count=2,
                vocab_size=3,
                observed_logits=str(observed),
                observed_sha256=hashlib.sha256(observed.read_bytes()).hexdigest(),
            )
        summary["numerical_reference"] = {
            "directory": str(reference),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
        for version, row in enumerate(rows):
            runtime = ["current_student", "/p0", version]
            row.update(
                source_bindings=[
                    {
                        "source_identity": source,
                        "runtime_identity": runtime,
                        "committed": True,
                        "copied": True,
                        "seconds": 0.25,
                        "transfers": [
                            {"fingerprint": "fp-p0", "parameters": 339, "seconds": 0.1}
                        ],
                        "server_status": [
                            {
                                "valid": True,
                                "awake": True,
                                "active": 0,
                                "identity": runtime,
                            }
                        ],
                    }
                ],
                weight_activation_seconds=0.25,
                weight_transfer_seconds=0.1,
                weight_copy_count=1,
                physical_execution=[
                    {
                        "slot": "current_student",
                        "phase": 0,
                        "logical_phase": 0,
                        "logical_slot": "initial_student",
                        "logical_identity": source,
                        "runtime_identity": runtime,
                        "publication_index": 0,
                        "replica_rank": 0,
                        "device_uuid": "card-0",
                        "requests": 4,
                        "request_ids": [str(i) for i in range(4)],
                    }
                ],
            )
            row["groups"][0].update(phase=0, slot="initial_student", requests=4)
        return summary, rows

    def save_fixture(self, path, summary, rows):
        (path / "summary.json").write_text(json.dumps(summary))
        (path / "observations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def test_staged_receipts_bind_logical_sources_to_physical_versions_and_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            summary, rows = self.staged_fixture(path)
            self.save_fixture(path, summary, rows)
            timings, memory, _ = MODULE.collect(path)
            self.assertEqual(timings[0]["weight_mode"], "staged")
            self.assertEqual(timings[0]["weight_activation_seconds_mean"], 0.25)
            self.assertEqual(timings[0]["weight_transfer_seconds_mean"], 0.1)
            self.assertEqual(memory[0]["weight_mode"], "staged")

    def test_staged_tampering_missing_equivalence_and_reference_changes_are_rejected(
        self,
    ):
        for problem in (
            "runtime",
            "fingerprint",
            "missing_check",
            "loose_tolerance",
            "reference",
            "activation",
            "server",
            "transfer",
            "observed",
            "falsified_error",
            "effective_vocab",
        ):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary)
                summary, rows = self.staged_fixture(path)
                if problem == "runtime":
                    rows[1]["physical_execution"][0]["runtime_identity"] = [
                        "current_student",
                        "/p0",
                        999,
                    ]
                elif problem == "fingerprint":
                    rows[1]["source_bindings"][0]["transfers"][0]["fingerprint"] = (
                        "wrong"
                    )
                elif problem == "missing_check":
                    summary["numerical_checks"].pop()
                elif problem == "loose_tolerance":
                    summary["numerical_checks"][0]["tv_tolerance"] = 0.5
                elif problem == "reference":
                    (path / "reference/teacher.npy").write_bytes(b"changed")
                elif problem == "activation":
                    rows[1]["source_bindings"][0]["seconds"] = 10
                    rows[1]["weight_activation_seconds"] = 10
                elif problem == "server":
                    rows[1]["source_bindings"][0]["server_status"].clear()
                elif problem == "transfer":
                    rows[1]["weight_transfer_seconds"] = 0.001
                elif problem == "observed":
                    Path(summary["numerical_checks"][0]["observed_logits"]).write_bytes(
                        b"changed"
                    )
                elif problem == "falsified_error":
                    summary["numerical_checks"][0]["max_logit_error"] = 0.01
                else:
                    summary["effective_vocab_size"] = 4
                self.save_fixture(path, summary, rows)
                with self.assertRaises(ValueError):
                    MODULE.collect(path)

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
