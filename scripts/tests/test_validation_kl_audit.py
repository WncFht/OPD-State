"""Audit actual producer reductions with independent question-cluster inference."""

import ast
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "kl_audit", ROOT / "eval/audit_validation_kl.py"
)
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)
tree = ast.parse((ROOT / "verl/verl/trainer/ppo/validation_kl_runner.py").read_text())
node = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "summarize_kl"
)
scope = {"math": math, "defaultdict": defaultdict, "STATISTICS": AUDIT.STATISTICS}
exec(
    compile(ast.Module(body=[node], type_ignores=[]), "actual-kl-producer", "exec"),
    scope,
)


def fixture(directory, *, constant=False):
    raw, rows, accuracy = [], [], []
    for source, uid, length, value in [
        ("MATH-500", "a", 1, 1.0),
        ("MATH-500", "b", 9, 3.0),
        ("AMC23", "c", 3, 5.0),
        ("AIME25", "d", 1, 7.0),
    ]:
        if constant:
            value = 2.0
        for sample in range(16):
            record = {
                "state_source": "on_validation_states",
                "source": source,
                "uid": uid,
                "sample": sample,
                "prompt_ids": [1, 4, 3],
                "response_ids": [6] * (length - 1) + [2],
            }
            sums = {
                "student_teacher_kl": value * length,
                "teacher_student_kl": 2 * value * length,
                "student_entropy": 1.5 * length,
                "teacher_entropy": 2.5 * length,
            }
            raw.append(record)
            rows.append(
                {
                    **{
                        k: record[k]
                        for k in ("state_source", "source", "uid", "sample")
                    },
                    "response_length": length,
                    "evaluated_positions": length,
                    "record_sha256": AUDIT.raw_fingerprint(record),
                    "sums": sums,
                    "depths": {"0_255": {"tokens": length, "sums": sums.copy()}},
                }
            )
            accuracy.append(
                {
                    "data_source": source,
                    "validation_uid": uid,
                    "rollout_generation_sample": sample,
                    "rollout_response_length": length,
                    "rollout_student_version": 2,
                }
            )
    token_path = directory / "tokens.jsonl"
    token_path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    accuracy_path = directory / "accuracy.jsonl"
    accuracy_path.write_text("".join(json.dumps(r) + "\n" for r in accuracy))
    report = {
        "schema": "validation-kl-v1",
        "student_version": 2,
        "effective_vocab_size": 256,
        "scoring_temperature": 1,
        "strict_raw_sequence_kl": False,
        "scoring_started_epoch_s": 1,
        "scoring_finished_epoch_s": 2,
        "rows": rows,
        "tokens_sha256": AUDIT.digest(token_path),
        "metrics": scope["summarize_kl"](rows),
        "worker_timings": {
            name: [1.0] * 4
            for name in (
                "seconds",
                "student_forward_seconds",
                "teacher_forward_seconds",
                "head_kl_seconds",
                "head_snapshot_bytes",
            )
        },
    }
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path, token_path, accuracy_path, report, raw


class KlAuditTests(unittest.TestCase):
    def test_actual_producer_reductions_and_accuracy_binding_pass_independent_audit(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, report, _ = fixture(Path(temporary))
            result = AUDIT.audit(p, t, accuracy_path=a, replicates=1000)
            self.assertEqual(result["coverage"], "bound_to_accuracy")
            self.assertEqual(result["responses"], 64)
            self.assertEqual(result["effective_positions"], 224)
            self.assertEqual(
                result["questions"], {"MATH-500": 2, "AMC23": 1, "AIME25": 1}
            )
            prefix = "validation_kl/on_validation_states/"
            self.assertEqual(
                report["metrics"][prefix + "micro/student_teacher_kl/question_mean"],
                4.0,
            )
            self.assertAlmostEqual(
                report["metrics"][prefix + "micro/student_teacher_kl/token_weighted"],
                50 / 14,
            )
            self.assertEqual(
                report["metrics"][prefix + "macro/student_teacher_kl/question_mean"],
                14 / 3,
            )

    def test_same_question_samples_remain_one_bootstrap_cluster(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, _, _ = fixture(Path(temporary))
            first = AUDIT.audit(p, t, accuracy_path=a, replicates=1000, seed=7)
            second = AUDIT.audit(p, t, accuracy_path=a, replicates=1000, seed=7)
            self.assertEqual(first, second)
            key = "validation_kl/on_validation_states/MATH-500/student_teacher_kl/question_mean"
            self.assertEqual(first["intervals"][key], {"low": 1.0, "high": 3.0})

    def test_constant_question_effect_has_zero_interval_for_normalized_reductions(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, _, _ = fixture(Path(temporary), constant=True)
            result = AUDIT.audit(p, t, accuracy_path=a, replicates=100)
            for key, value in result["intervals"].items():
                if "/student_teacher_kl/" in key and not key.endswith(
                    "/cumulative_on_states"
                ):
                    self.assertEqual(value, {"low": 2.0, "high": 2.0}, key)

    def test_raw_token_mutation_and_nonmacro_metric_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, report, _ = fixture(Path(temporary))
            data = t.read_bytes()
            t.write_bytes(data + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                AUDIT.audit(p, t, accuracy_path=a, replicates=100)
            t.write_bytes(data)
            report["metrics"][
                "validation_kl/on_validation_states/AMC23/teacher_entropy/question_mean"
            ] += 1
            p.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "aggregate"):
                AUDIT.audit(p, t, accuracy_path=a, replicates=100)

    def test_missing_sample_accuracy_mismatch_and_false_full_coverage_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, report, raw = fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "coverage"):
                AUDIT.audit(
                    p,
                    t,
                    required_counts=AUDIT.BENCHMARKS,
                    accuracy_path=a,
                    replicates=100,
                )
            original = a.read_text()
            values = [json.loads(line) for line in original.splitlines()]
            values[0]["rollout_response_length"] += 1
            a.write_text("".join(json.dumps(r) + "\n" for r in values))
            with self.assertRaisesRegex(ValueError, "coverage"):
                AUDIT.audit(p, t, accuracy_path=a, replicates=100)
            a.write_text(original)
            report["rows"].pop()
            raw.pop()
            t.write_text("".join(json.dumps(r) + "\n" for r in raw))
            report["tokens_sha256"] = AUDIT.digest(t)
            p.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "samples"):
                AUDIT.audit(p, t, accuracy_path=a, replicates=100)

    def test_wrong_identity_depths_partial_positions_and_missing_worker_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, original, _ = fixture(Path(temporary))
            for kind in ("hash", "depth", "positions", "worker"):
                report = copy.deepcopy(original)
                if kind == "hash":
                    report["rows"][0]["record_sha256"] = "wrong"
                elif kind == "depth":
                    report["rows"][0]["depths"]["0_255"]["tokens"] = 2
                elif kind == "positions":
                    report["rows"][0]["evaluated_positions"] = 0
                else:
                    report["worker_timings"]["head_kl_seconds"].pop()
                p.write_text(json.dumps(report))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    AUDIT.audit(p, t, accuracy_path=a, replicates=100)

    def test_fixed_state_outputs_require_the_exact_declared_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            p, t, a, report, raw = fixture(directory)
            fixed = []
            for state in ("on_initial_states", "on_teacher_states"):
                record = {
                    "state_source": state,
                    "source": "MATH-500",
                    "uid": "fixed",
                    "sample": 0,
                    "prompt_ids": [1, 4],
                    "response_ids": [6, 6, 6, 2],
                    "positions": [0, 2],
                    "weights": [1, 3],
                }
                fixed.append(record)
                raw.append(record)
                report["rows"].append(
                    {
                        **{
                            k: record[k]
                            for k in ("state_source", "source", "uid", "sample")
                        },
                        "response_length": 4,
                        "evaluated_positions": 2,
                        "record_sha256": AUDIT.raw_fingerprint(record),
                        "sums": {name: 4.0 for name in AUDIT.STATISTICS},
                        "depths": {},
                    }
                )
            bank = directory / "bank.json"
            bank.write_text(
                json.dumps({"schema": "fixed-prefix-bank-v1", "records": fixed})
            )
            report["fixed_bank_sha256"] = AUDIT.digest(bank)
            report["metrics"] = scope["summarize_kl"](report["rows"])
            t.write_text("".join(json.dumps(row) + "\n" for row in raw))
            report["tokens_sha256"] = AUDIT.digest(t)
            p.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "original immutable bank"):
                AUDIT.audit(p, t, accuracy_path=a, replicates=100)
            result = AUDIT.audit(p, t, accuracy_path=a, bank_path=bank, replicates=100)
            self.assertEqual(result["bank_sha256"], AUDIT.digest(bank))
            changed = json.loads(bank.read_text())
            changed["records"][0]["prompt_ids"] = [1, 5]
            bank.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "declared common"):
                AUDIT.audit(p, t, accuracy_path=a, bank_path=bank, replicates=100)

    def test_nonfinite_timestamps_and_unsupported_sequence_claims_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, t, a, original, _ = fixture(Path(temporary))
            for field, value in (
                ("scoring_started_epoch_s", float("nan")),
                ("strict_raw_sequence_kl", True),
            ):
                report = copy.deepcopy(original)
                report[field] = value
                p.write_text(json.dumps(report))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    AUDIT.audit(p, t, accuracy_path=a, replicates=100)


if __name__ == "__main__":
    unittest.main()
