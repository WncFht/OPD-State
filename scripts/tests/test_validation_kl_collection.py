"""Test KL phase alignment, complete-evaluation guards and runnable gate plans.

Metadata fixtures do not establish real Parquet coverage or GPU model scores.
The numerical/response fixture comes from the actual producer reduction tests.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import collect_validation_kl as COLLECT
import run_validation_kl_gate as GATE
from test_validation_kl_audit import AUDIT, fixture, scope


def setup_run(root):
    run, result = root / "run", root / "result"
    run.mkdir()
    result.mkdir()
    report, tokens, accuracy, data, _ = fixture(run)
    target = run / "validation/kl"
    target.mkdir(parents=True)
    report.rename(target / "kl-step-2.json")
    tokens.rename(target / "kl-tokens-step-2.jsonl")
    accuracy.rename(run / "validation/2.jsonl")
    data["metrics"]["validation_kl/performance/scoring_seconds"] = 1.0
    (target / "kl-step-2.json").write_text(json.dumps(data))
    status = {"state": "finished", "exit_code": 0, "started": 0.0, "finished": 10.0}
    (run / "status.json").write_text(json.dumps(status))
    with (run / "gpu.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_s", "gpu", "used_mib", "utilization_pct"])
        writer.writerows(
            (epoch, gpu, used, 30)
            for epoch, used in ((0.5, 9000), (1.5, 8000), (3.5, 12000))
            for gpu in range(4)
        )
    native = {
        "numeric_annotation": 1.0,
        "final_student_version": 2,
        "evaluation_responses": 64,
        "math_summary": {"full_benchmarks": False, "samples_per_question": 16},
        "whole_card_peak_mib": {str(gpu): 12000 for gpu in range(4)},
    }
    (result / "native-acceptance.json").write_text(json.dumps(native))
    return run, result, data, status


def input_manifest(root):
    inputs = []
    for benchmark, count in GATE.BENCHMARKS.items():
        filename = root / benchmark / "test.parquet"
        filename.parent.mkdir()
        filename.write_bytes(b"metadata fixture, not real parquet")
        inputs.append(
            {"path": str(filename), "rows": count, "sha256": GATE.digest(filename)}
        )
    prepared = {
        "schema": "fixed-prefix-input-v1",
        "inputs": inputs,
        "questions": [
            {"source": benchmark, "uid": f"{benchmark}-{index}", "prompt_ids": [1, 2]}
            for benchmark in GATE.BENCHMARKS
            for index in range(30)
        ],
        "vocab_size": 256,
        "tokenizer_signature": "tokenizer-fp",
        "models": {
            "initial_student": {"fingerprint": "p0-fp"},
            "teacher": {"fingerprint": "q-fp"},
        },
    }
    path = root / "bank-input.json"
    path.write_text(json.dumps(prepared))
    return path, prepared


class KlCollectionTests(unittest.TestCase):
    def test_full_synthetic_9120_responses_and_180_bank_records_pass_without_claiming_gpu_scores(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run, result, report, _ = setup_run(Path(temporary))
            raw, rows, accuracy, fixed = [], [], [], []
            for benchmark, count in GATE.BENCHMARKS.items():
                for index in range(count):
                    for sample in range(16):
                        record = {
                            "state_source": "on_validation_states",
                            "source": benchmark,
                            "uid": f"{benchmark}-{index}",
                            "sample": sample,
                            "prompt_ids": [1, 4, 3],
                            "response_ids": [5, 2],
                        }
                        raw.append(record)
                        accuracy.append(
                            {
                                "data_source": benchmark,
                                "validation_uid": record["uid"],
                                "rollout_generation_sample": sample,
                                "rollout_response_length": 2,
                                "rollout_student_version": 2,
                            }
                        )
                for state in ("on_initial_states", "on_teacher_states"):
                    for index in range(30):
                        record = {
                            "state_source": state,
                            "source": benchmark,
                            "uid": f"{benchmark}-{index}",
                            "sample": 0,
                            "prompt_ids": [1, 4, 3],
                            "response_ids": [5, 2],
                            "positions": [0, 1],
                            "weights": [1, 1],
                        }
                        raw.append(record)
                        fixed.append(record)
            for record in raw:
                sums = {name: 0.4 for name in AUDIT.STATISTICS}
                rows.append(
                    {
                        **{
                            key: record[key]
                            for key in ("state_source", "source", "uid", "sample")
                        },
                        "response_length": 2,
                        "evaluated_positions": 2,
                        "record_sha256": AUDIT.raw_fingerprint(record),
                        "sums": sums,
                        "depths": {"0_255": {"tokens": 2, "sums": sums}}
                        if record["state_source"] == "on_validation_states"
                        else {},
                    }
                )
            bank = run / "bank.json"
            bank.write_text(
                json.dumps({"schema": "fixed-prefix-bank-v1", "records": fixed})
            )
            tokens = run / "validation/kl/kl-tokens-step-2.jsonl"
            tokens.write_text("".join(json.dumps(row) + "\n" for row in raw))
            report.update(
                rows=rows,
                tokens_sha256=AUDIT.digest(tokens),
                fixed_bank_sha256=AUDIT.digest(bank),
                metrics={
                    **scope["summarize_kl"](rows),
                    "validation_kl/performance/scoring_seconds": 1.0,
                },
            )
            (run / "validation/kl/kl-step-2.json").write_text(json.dumps(report))
            (run / "validation/2.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in accuracy)
            )
            native_path = result / "native-acceptance.json"
            native = json.loads(native_path.read_text())
            native["math_summary"]["full_benchmarks"] = True
            native["evaluation_responses"] = 9120
            native_path.write_text(json.dumps(native))
            receipt = COLLECT.collect(run, result, bank=bank, full=True, replicates=100)
            self.assertEqual(receipt["responses"], 9120)
            self.assertEqual(
                receipt["coverage"]["on_validation_states"]["evaluated_positions"],
                18240,
            )
            for state in ("on_initial_states", "on_teacher_states"):
                self.assertEqual(receipt["coverage"][state]["responses"], 90)

    def test_fixed_bank_model_prompt_and_complete_question_bindings_are_checked(self):
        for problem in (None, "model", "prompt", "duplicate"):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                path, prepared = input_manifest(root)
                bank = {
                    "schema": "fixed-prefix-bank-v1",
                    "input_sha256": GATE.digest(path),
                    "vocab_size": 256,
                    "tokenizer_signature": "tokenizer-fp",
                    "initial_student_fingerprint": "p0-fp",
                    "teacher_fingerprint": "q-fp",
                    "records": [
                        {**row, "state_source": state, "sample": 0}
                        for state in ("on_initial_states", "on_teacher_states")
                        for row in prepared["questions"]
                    ],
                }
                if problem == "model":
                    bank["teacher_fingerprint"] = "other"
                elif problem == "prompt":
                    bank["records"][0]["prompt_ids"] = [9]
                elif problem == "duplicate":
                    bank["records"][0] = bank["records"][1]
                bank_path = root / "bank.json"
                bank_path.write_text(json.dumps(bank))
                if problem is None:
                    self.assertEqual(
                        GATE.verify_bank(bank_path, prepared, path),
                        GATE.digest(bank_path),
                    )
                else:
                    with self.assertRaises(ValueError):
                        GATE.verify_bank(bank_path, prepared, path)

    def test_actual_producer_audit_and_resource_peaks_keep_scoring_separate_from_whole_run(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run, result, _, _ = setup_run(Path(temporary))
            receipt = COLLECT.collect(run, result, replicates=100)
            self.assertEqual(receipt["responses"], 64)
            self.assertTrue(receipt["all_validation_positions"])
            profile = json.loads((result / "kl-resource-profile.json").read_text())
            self.assertEqual(
                profile["scoring_sampled_peak_mib"],
                {str(gpu): 8000 for gpu in range(4)},
            )
            self.assertEqual(
                profile["whole_run_sampled_peak_mib"],
                {str(gpu): 12000 for gpu in range(4)},
            )
            self.assertTrue(profile["scoring_samples_cover_all_gpus"])
            summary = json.loads((result / "kl-summary.json").read_text())
            self.assertNotIn("rows", summary)
            self.assertEqual(
                summary["coverage"]["on_validation_states"]["evaluated_positions"], 224
            )
            native = json.loads((result / "native-acceptance.json").read_text())
            (result / "native-acceptance.json").write_text(json.dumps(native, indent=4))
            native["numeric_annotation"] = 1
            canonical = COLLECT.canonical_digest(native)
            self.assertEqual(receipt["native_acceptance_canonical_sha256"], canonical)

    def test_missing_in_phase_samples_are_explicit_and_do_not_claim_memory_coverage(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, data, status = setup_run(Path(temporary))
            path = run / "gpu.csv"
            text = path.read_text().replace("1.5,3,8000,30\n", "")
            path.write_text(text)
            profile = COLLECT.resource_profile(data, status, path)
            self.assertFalse(profile["scoring_samples_cover_all_gpus"])
            self.assertEqual(profile["scoring_sample_counts"]["3"], 0)
            self.assertIsNone(profile["scoring_sampled_peak_mib"]["3"])

    def test_partial_evidence_cannot_be_promoted_to_full_570_question_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, result, _, _ = setup_run(Path(temporary))
            with self.assertRaisesRegex(ValueError, "scope"):
                COLLECT.collect(run, result, full=True, replicates=100)
            path = result / "native-acceptance.json"
            native = json.loads(path.read_text())
            native["math_summary"]["full_benchmarks"] = True
            path.write_text(json.dumps(native))
            with self.assertRaises(ValueError):
                COLLECT.collect(run, result, full=True, replicates=100)

    def test_wrong_final_version_missing_epochs_and_changed_memory_peaks_are_rejected(
        self,
    ):
        for problem in ("version", "epochs", "peaks"):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run, result, report, status = setup_run(Path(temporary))
                if problem == "version":
                    report["student_version"] = 3
                    (run / "validation/kl/kl-step-2.json").write_text(
                        json.dumps(report)
                    )
                elif problem == "epochs":
                    status.pop("started")
                    (run / "status.json").write_text(json.dumps(status))
                else:
                    path = result / "native-acceptance.json"
                    native = json.loads(path.read_text())
                    native["whole_card_peak_mib"]["0"] = 12001
                    path.write_text(json.dumps(native))
                with self.assertRaises(ValueError):
                    COLLECT.collect(run, result, replicates=100)

    def test_full_inputs_are_forwarded_instead_of_the_bank_selected_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = input_manifest(root)
            prepared, files = GATE.input_bindings(path)
            self.assertEqual(len(prepared["questions"]), 90)
            commands = GATE.commands(
                root / "run",
                root / "result",
                root / "bank.json",
                files,
                "commit",
                Path("python"),
            )
            train = commands["train"]
            self.assertIn("--full-eval", train)
            self.assertEqual(train.count("--validation"), 3)
            for file in files.values():
                self.assertIn(file, train)
            self.assertIn("actor_rollout_ref.rollout.val_kwargs.n=16", train)
            self.assertIn("trainer.validation_kl.enabled=True", train)
            self.assertIn("trainer.validation_kl.position_chunk_size=128", train)
            self.assertIn("--full", commands["kl_audit"])

    def test_changed_input_and_subset_row_counts_fail_before_gpu_startup(self):
        for problem in ("bytes", "rows"):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path, data = input_manifest(Path(temporary))
                if problem == "bytes":
                    Path(data["inputs"][0]["path"]).write_bytes(b"changed")
                else:
                    data["inputs"][0]["rows"] = 30
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    GATE.input_bindings(path)

    def test_cli_plan_allows_pending_bank_and_never_creates_a_training_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = input_manifest(root)
            manifest = root / "train-manifest.json"
            manifest.write_text("{}")
            output = root / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "eval/run_validation_kl_gate.py"),
                    "--output",
                    str(output),
                    "--result",
                    str(root / "result"),
                    "--bank-input",
                    str(path),
                    "--bank",
                    str(root / "pending-bank.json"),
                    "--data-manifest",
                    str(manifest),
                    "--source-commit",
                    "HEAD",
                    "--plan-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            plan = json.loads((root / "run-plan.json").read_text())
            self.assertEqual(plan["state"], "planned")
            self.assertEqual(plan["bank_state"], "pending")
            self.assertEqual(plan["responses"], 9120)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
