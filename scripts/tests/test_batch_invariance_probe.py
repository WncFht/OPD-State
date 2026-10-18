"""Audit constructed probe records without treating them as GPU evidence."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import audit_batch_invariance as AUDIT
import probe_batch_invariance as PROBE
import run_vllm_version_controls as CONTROL


def fixture(root):
    inputs = [{"prompt_id": str(i), "prompt_ids": [1, 2]} for i in range(16)]
    responses = {row["prompt_id"]: [1, 1, 1, 1] for row in inputs}
    worker = {
        "device_uuid": "GPU-card",
        "batch_flag": False,
        "kernel_overrides_initialized": False,
        "kv_regions": [{"pointer": 100, "bytes": 16}],
        "cuda_graph_enabled": False,
    }
    args = {
        "gpu": 0,
        "repeats": 2,
        "cap": 4,
        "boundary": 2,
        "batch_invariant": False,
        "graph": False,
    }
    command = {"arguments": args, "devices": {"0": "GPU-card"}}
    rows = [
        {
            "batch": size,
            "reverse": reverse,
            "repeat": repeat,
            "seconds": 1.0,
            "tokens": 64,
            "responses": responses,
            "comparison": PROBE.compare_responses(responses, responses),
        }
        for size, reverse in ((1, False), (4, False), (16, False), (16, True))
        for repeat in range(2)
    ]
    split = {
        "first": {key: [1, 1] for key in responses},
        "second": {key: [1, 1] for key in responses},
        "second_prefixes": {key: [1, 2, 1, 1] for key in responses},
        "responses": responses,
    }
    for name, value in (
        ("command.json", command),
        ("inputs.json", inputs),
        ("worker.json", worker),
        ("split.json", split),
    ):
        (root / name).write_text(json.dumps(value))
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    checks = []
    values = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
    for name in ("single", "paired", "reversed"):
        path = root / f"logits-{name}.npy"
        np.save(path, values)
        checks.append(
            {
                "name": name,
                "file": path.name,
                "sha256": AUDIT.digest(path),
                **PROBE.full_vocab_distance(values, values),
            }
        )
    summary = {
        "worker": worker,
        "prompt_ids": list(responses),
        "tokenizer_vocab_size": 3,
        "eos_token_id": 0,
        "expected_kv_layers": 1,
        "all_batch_order_replays_equal": True,
        "same_model_split": PROBE.compare_responses(responses, responses),
        "probe_prefixes": [row["prompt_ids"] for row in inputs[:2]],
        "full_vocab_checks": checks,
    }
    return summary, rows, split, worker, command


def save(root, summary, rows, split, worker, command):
    (root / "command.json").write_text(json.dumps(command))
    (root / "worker.json").write_text(json.dumps(worker))
    (root / "split.json").write_text(json.dumps(split))
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    for name, key in (
        ("inputs.json", "inputs_sha256"),
        ("worker.json", "worker_sha256"),
        ("split.json", "split_sha256"),
        ("observations.jsonl", "observations_sha256"),
    ):
        summary[key] = AUDIT.digest(root / name)
    (root / "summary.json").write_text(json.dumps(summary))


class BatchInvarianceProbeTests(unittest.TestCase):
    def test_committed_sources_are_frozen_and_changed_driver_is_rejected(self):
        def committed(command, **kwargs):
            name = command[-1].split("/", 1)[1]
            return (ROOT / "eval" / name).read_bytes()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "code"
            with patch.object(
                CONTROL.subprocess, "check_output", side_effect=committed
            ):
                hashes = CONTROL.snapshot_sources(path, "committed")
            self.assertEqual(len(hashes), 4)
            for name, value in hashes.items():
                self.assertEqual(value, PROBE.digest(ROOT / "eval" / name))
            with patch.object(
                CONTROL.subprocess, "check_output", return_value=b"changed"
            ):
                with self.assertRaisesRegex(ValueError, "driver differs"):
                    CONTROL.snapshot_sources(Path(temporary) / "rejected", "committed")
            self.assertFalse((Path(temporary) / "rejected").exists())

    def test_version_controls_cover_both_execution_modes_and_exact_fusion_endpoints(
        self,
    ):
        cases = CONTROL.cases()
        self.assertEqual(len(cases), 16)
        for graph in (False, True):
            group = [case for case in cases if case["graph"] is graph]
            self.assertEqual(len(group), 8)
            self.assertEqual(
                {case["weight"] for case in group if case["weight"] is not None},
                {0.0, 0.5, 1.0},
            )
            for case in group:
                cmd = CONTROL.command(
                    case, Path("out"), Path("old"), Path("new"), Path("prompts")
                )
                self.assertEqual(
                    cmd[0], "old" if case["version"].startswith("0.10.") else "new"
                )
                self.assertEqual("--graph" in cmd, graph)
                self.assertEqual("--batch-invariant" in cmd, case["invariant"])

    def test_fusion_reference_uses_the_same_checkpoints_states_and_token_meanings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = np.array([[0, 2], [2, 0]], dtype=np.float32)
            q = p[:, ::-1].copy()
            names = ("eager-0.15.1-p0-bi1", "eager-0.15.1-q-bi1", "dual")
            summaries = {}
            for name, values, files in zip(
                names,
                (p, q, 0.5 * (p + q)),
                (
                    {"p0": {"weights": "p"}},
                    {"q": {"weights": "q"}},
                    {"p0": {"weights": "p"}, "q": {"weights": "q"}},
                ),
                strict=True,
            ):
                (root / name).mkdir()
                np.save(root / name / "logits-single.npy", values)
                summaries[name] = {
                    "probe_prefixes": [[1], [2]],
                    "tokenizer_vocab_size": 2,
                    "tokenizer_vocab_sha256": "same",
                    "checkpoint_files_sha256": files,
                }
            case = {"graph": False, "weight": 0.5, "name": "dual"}
            self.assertTrue(CONTROL.fusion_reference(root, case, summaries)["passed"])
            np.save(
                root / "dual/logits-single.npy",
                np.array([[10, -10], [10, -10]], dtype=np.float32),
            )
            self.assertFalse(CONTROL.fusion_reference(root, case, summaries)["passed"])
            summaries["dual"]["tokenizer_vocab_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "token ID"):
                CONTROL.fusion_reference(root, case, summaries)

    def test_identity_matching_handles_reverse_order_and_length_divergence(self):
        self.assertEqual(
            PROBE.compare_responses({"a": [1, 2], "b": [3]}, {"b": [3], "a": [1, 2]})[
                "exact_responses"
            ],
            2,
        )
        self.assertEqual(
            PROBE.compare_responses({"a": [1, 2]}, {"a": [1]})["first_divergence"],
            {"a": 1},
        )
        with self.assertRaises(ValueError):
            PROBE.compare_responses({"a": [1]}, {"b": [1]})

    def test_full_vocab_distance_matches_known_distributions_and_constant_shift(self):
        uniform = np.zeros((1, 2), dtype=np.float64)
        other = np.array([[math.log(3), 0]])
        measured = PROBE.full_vocab_distance(uniform, other)
        self.assertAlmostEqual(measured["max_tv"], 0.25)
        self.assertAlmostEqual(
            measured["max_reference_observed_kl"], 0.5 * math.log(4 / 3)
        )
        self.assertAlmostEqual(PROBE.full_vocab_distance(other, other + 5)["max_tv"], 0)

    def test_actual_producer_comparisons_and_arrays_pass_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = fixture(root)
            save(root, *data)
            receipt = AUDIT.collect(root)
            self.assertTrue(receipt["batch_order_replays_equal"])
            self.assertEqual(receipt["conditions"], 8)

    def test_resigned_tampering_still_rejects_work_prefix_kernel_kv_or_numeric_errors(
        self,
    ):
        for problem in (
            "missing",
            "comparison",
            "prefix",
            "eos",
            "kernel",
            "kv",
            "logits",
        ):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                summary, rows, split, worker, command = fixture(root)
                if problem == "missing":
                    rows.pop()
                elif problem == "comparison":
                    rows[1]["comparison"] = {
                        "responses": 16,
                        "exact_responses": 0,
                        "first_divergence": {},
                    }
                elif problem == "prefix":
                    split["second_prefixes"]["0"] = [2, 1, 1, 1]
                elif problem == "eos":
                    split["first"]["0"] = [1, 0]
                elif problem == "kernel":
                    command["arguments"]["batch_invariant"] = True
                    worker["batch_flag"] = True
                elif problem == "kv":
                    summary["expected_kv_layers"] = 2
                    worker["kv_regions"].append({"pointer": 100, "bytes": 16})
                else:
                    check = summary["full_vocab_checks"][1]
                    path = root / check["file"]
                    np.save(path, np.ones((2, 3), dtype=np.float32))
                    check["sha256"] = AUDIT.digest(path)
                save(root, summary, rows, split, worker, command)
                with self.assertRaises(ValueError):
                    AUDIT.collect(root)


if __name__ == "__main__":
    unittest.main()
