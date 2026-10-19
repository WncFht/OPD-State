"""One schedule and publication path for all declared evaluation suites."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from opd_state.artifacts import atomic_json, inside
from opd_state.core.evaluation import due_suites


class EvaluationRunner:
    """The backend owns model resources; this runner owns triggers and results."""

    def __init__(self, evaluation, backend, directory):
        self.evaluation = evaluation
        self.backend = backend
        self.directory = Path(directory).resolve()
        self.completed = set()
        self.last_step = -1

    def due(self, step, *, final=False):
        return due_suites(self.evaluation, outer=step, final=final)

    def run(self, step, *, final=False):
        selected = self.due(step, final=final)
        if step < self.last_step:
            raise ValueError("Live evaluation cannot regress the student version")
        self.last_step = step
        metrics = {}
        for suite_name, reasons in selected.items():
            suite = self.evaluation.suites[suite_name]
            for benchmark_name in suite.benchmarks:
                identity = step, suite_name, benchmark_name
                if identity in self.completed:
                    raise ValueError(
                        "A suite/benchmark already evaluated this student version"
                    )
                benchmark = self.evaluation.benchmarks[benchmark_name]
                directory = inside(
                    self.directory, f"{suite_name}/{benchmark_name}/step-{step}"
                )
                directory.mkdir(parents=True, exist_ok=False)
                result = self.backend.evaluate(
                    suite_name=suite_name,
                    benchmark_name=benchmark_name,
                    benchmark=benchmark,
                    diagnostics=suite.diagnostics,
                    step=step,
                    directory=directory,
                )
                if result.get("student_version") != step:
                    raise ValueError("Evaluator returned a different student version")
                responses = result.get("responses")
                if type(responses) is not int or responses < 1:
                    raise ValueError(
                        "Evaluation requires a positive actual response count"
                    )
                expected = benchmark.data.expected_counts
                if (
                    expected
                    and responses
                    != sum(expected.values()) * benchmark.sampling.samples_per_case
                ):
                    raise ValueError(
                        "Evaluation response count differs from its benchmark budget"
                    )
                observed = result.get("metrics")
                if not isinstance(observed, dict) or not observed:
                    raise ValueError("Evaluation returned no metrics")
                clean = {}
                for name, value in observed.items():
                    if (
                        not isinstance(name, str)
                        or not name
                        or isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(value)
                    ):
                        raise ValueError(
                            "Evaluation metrics require names and finite numeric values"
                        )
                    clean[name] = int(value) if isinstance(value, int) else float(value)
                report = {
                    "schema": "opd-evaluation-v1",
                    "suite": suite_name,
                    "benchmark": benchmark_name,
                    "student_version": step,
                    "reasons": reasons,
                    "responses": responses,
                    "metrics": clean,
                }
                atomic_json(directory / "result.json", report)
                prefix = f"evaluation/{suite_name}/{benchmark_name}/"
                metrics.update({prefix + name: value for name, value in clean.items()})
                self.completed.add(identity)
        return metrics
