"""Audit paired comparisons and distinguish a plateau from continuing fitting."""

import ast
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analysis", ROOT / "eval/analyze_math_sources.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def make_result(directory, correct):
    directory.mkdir()
    collector = ANALYSIS.COLLECTOR
    rows = []
    for source, count in ANALYSIS.BENCHMARKS.items():
        for index in range(count):
            raw = {"data_source": source, "input": f"question-{index}", "gts": "0"}
            rows.append(
                {
                    "data_source": source,
                    "uid": f"{source}-{index}",
                    "n": 16,
                    "correct": correct,
                    "question_sha256": collector.question_identity(raw),
                    "mean@16": correct / 16,
                    **{
                        f"pass@{k}": collector.MATH.estimate_pass_at_k(16, correct, k)
                        for k in (1, 2, 4, 8, 16)
                    },
                }
            )
    collector.write_csv(directory / "questions-step50.csv", rows)
    metrics, _ = collector.MATH.exact_math_metrics(
        [row["data_source"] for row in rows for _ in range(16)],
        [row["uid"] for row in rows for _ in range(16)],
        [int(j < correct) for _ in rows for j in range(16)],
        samples_per_question=16,
        required_counts=ANALYSIS.BENCHMARKS,
    )
    (directory / "validation-step50.json").write_text(json.dumps({"metrics": metrics}))
    (directory / "summary.json").write_text(
        json.dumps({"training_steps": 50, "full_benchmarks": True})
    )
    collector.write_csv(
        directory / "validation-timeseries.csv",
        [{"step": 50, "eval/macro/mean@16": correct / 16}],
    )


class PairedComparisonTests(unittest.TestCase):
    def test_constant_paired_effect_and_reproducible_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            make_result(base / "a", 8)
            make_result(base / "b", 12)
            result = ANALYSIS.compare(
                {"a": base / "a", "b": base / "b"}, reference="a", replicates=100
            )
            self.assertEqual(len(result["scores"]), 10)
            self.assertEqual(len(result["differences"]), 5)
            for row in result["differences"]:
                self.assertAlmostEqual(row["difference_mean@16"], 0.25)
                self.assertAlmostEqual(row["question_bootstrap_low"], 0.25)
                self.assertAlmostEqual(row["question_bootstrap_high"], 0.25)
            self.assertEqual(
                result["plateau_assessments"]["a"]["status"], "insufficient_observation"
            )

    def test_changed_question_identity_rejects_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            make_result(base / "a", 8)
            make_result(base / "b", 12)
            path = base / "b/questions-step50.csv"
            rows = ANALYSIS.csv_rows(path)
            rows[0]["question_sha256"] = "different-question"
            ANALYSIS.COLLECTOR.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "same question content"):
                ANALYSIS.compare(
                    {"a": base / "a", "b": base / "b"}, reference="a", replicates=100
                )

    def test_metric_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "a"
            make_result(directory, 8)
            path = directory / "questions-step50.csv"
            rows = ANALYSIS.csv_rows(path)
            rows[0]["pass@16"] = 0
            ANALYSIS.COLLECTOR.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "correctness counts"):
                ANALYSIS.load_questions(directory, 50)


class PlateauTests(unittest.TestCase):
    def test_actual_trainer_stop_block_keeps_last_validation_and_budget_reason(self):
        tree = ast.parse((ROOT / "verl/verl/trainer/ppo/ray_trainer.py").read_text())
        block = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.plateau_monitor is not None"
        )
        code = compile(
            ast.Module(body=[block], type_ignores=[]), "trainer-plateau-block", "exec"
        )
        for steps, falling in ((300, False), (800, True)):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as temporary:
                monitor = ANALYSIS.PLATEAU.FixedCachePlateau()

                def metrics(step):
                    return {
                        "response_length/mean": 10,
                        "opd/log_ratio_rms": 0.5 - step * 0.0004 if falling else 0.2,
                        "actor/grad_norm": 0.1,
                        "eval/macro/mean@16": 0.5,
                    }

                for step in range(1, steps):
                    monitor.add_step(step, metrics(step))
                    if step % 50 == 0:
                        monitor.observe(step, metrics(step))
                report_path = Path(temporary) / "plateau.json"
                owner = SimpleNamespace(
                    plateau_monitor=monitor,
                    global_steps=steps,
                    total_training_steps=800,
                    config=SimpleNamespace(
                        trainer=SimpleNamespace(
                            total_training_steps=800,
                            plateau_report_path=str(report_path),
                        )
                    ),
                )
                namespace = {
                    "self": owner,
                    "metrics": metrics(steps),
                    "json": json,
                    "will_validate": True,
                    "is_last_step": steps == 800,
                    "progress_bar": SimpleNamespace(total=800),
                    "val_metrics": {"final": 1},
                }
                exec(code, namespace)
                report = json.loads(report_path.read_text())
                self.assertTrue(namespace["is_last_step"])
                self.assertTrue(report["stopped"])
                self.assertEqual(report["budget_exhausted"], falling)
                if not falling:
                    self.assertEqual(namespace["last_val_metrics"], {"final": 1})
                    self.assertEqual(owner.total_training_steps, 300)
                    self.assertEqual(namespace["progress_bar"].total, 300)

    def test_streaming_monitor_matches_token_weighting_and_minimum_budget(self):
        monitor = ANALYSIS.PLATEAU.FixedCachePlateau()
        for step in range(1, 301):
            value = 1 if step % 2 else 3
            monitor.add_step(
                step,
                {
                    "response_length/mean": value,
                    "opd/log_ratio_rms": value,
                    "actor/grad_norm": 2,
                },
            )
            if step % 50 == 0:
                report = monitor.observe(step, {"eval/macro/mean@16": 0.5})
                self.assertAlmostEqual(
                    report["windows"][-1]["token_weighted_log_ratio_rms"], math.sqrt(7)
                )
                self.assertEqual(
                    report["status"],
                    "observed_plateau" if step == 300 else "insufficient_observation",
                )
        self.assertEqual(
            [row["step"] for row in report["windows"]], [150, 200, 250, 300]
        )

    def test_monitor_rejects_gaps_and_incomplete_validation_windows(self):
        monitor = ANALYSIS.PLATEAU.FixedCachePlateau()
        metrics = {
            "response_length/mean": 10,
            "opd/log_ratio_rms": 0.2,
            "actor/grad_norm": 0.1,
        }
        monitor.add_step(1, metrics)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            monitor.add_step(3, metrics)
        with self.assertRaisesRegex(ValueError, "50 new training steps"):
            monitor.observe(1, {"eval/macro/mean@16": 0.5})

    def test_stable_validation_does_not_hide_falling_training_residual(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "summary.json").write_text(json.dumps({"training_steps": 300}))
            ANALYSIS.COLLECTOR.write_csv(
                directory / "validation-timeseries.csv",
                [
                    {"step": step, "eval/macro/mean@16": 0.5}
                    for step in range(50, 301, 50)
                ],
            )
            training = [
                {
                    "step": step,
                    "response_length/mean": 100,
                    "opd/log_ratio_rms": 0.2,
                    "actor/grad_norm": 0.1,
                }
                for step in range(1, 301)
            ]
            path = directory / "training-1-300.csv"
            ANALYSIS.COLLECTOR.write_csv(path, training)
            stable = ANALYSIS.plateau_check(directory)
            self.assertEqual(stable["status"], "observed_plateau")
            for row in training[-50:]:
                row["opd/log_ratio_rms"] = 0.15
            ANALYSIS.COLLECTOR.write_csv(path, training)
            falling = ANALYSIS.plateau_check(directory)
            self.assertEqual(falling["status"], "criterion_not_met")
            self.assertGreater(falling["observed_relative_rms_span"], 0.05)


if __name__ == "__main__":
    unittest.main()
