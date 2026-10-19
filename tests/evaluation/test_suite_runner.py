"""Suite triggers, scoped outputs and live model versions use one contract."""

import json
from types import SimpleNamespace

import pytest

from opd_state.core.evaluation import parse_evaluation
from opd_state.evaluation.runner import EvaluationRunner


def evaluation():
    return parse_evaluation(
        {
            "benchmarks": {
                "math": {
                    "data": {
                        "paths": ["math.parquet"],
                        "expected_counts": {"MATH-500": 1},
                    },
                    "sampling": {"samples_per_case": 2},
                },
                "hard": {
                    "data": {
                        "paths": ["hard.parquet"],
                        "expected_counts": {"AIME25": 1},
                    },
                    "sampling": {"samples_per_case": 2},
                },
            },
            "suites": {
                "periodic": {
                    "trigger": {
                        "initial": True,
                        "every_outer_steps": 2,
                        "training_end": True,
                    },
                    "benchmarks": ["math", "hard"],
                },
                "final": {"trigger": {"training_end": True}, "benchmarks": ["math"]},
            },
        }
    )


def test_multiple_suites_benchmarks_and_final_trigger_deduplicate(tmp_path):
    calls = []

    def evaluate(**kwargs):
        calls.append((kwargs["step"], kwargs["suite_name"], kwargs["benchmark_name"]))
        return {
            "student_version": kwargs["step"],
            "responses": 2,
            "metrics": {"accuracy/mean@2": 0.5},
        }

    runner = EvaluationRunner(
        evaluation(), SimpleNamespace(evaluate=evaluate), tmp_path
    )
    assert len(runner.run(0)) == 2
    assert runner.run(1) == {}
    assert len(runner.run(2)) == 2
    assert len(runner.run(4, final=True)) == 3
    assert len(calls) == len(set(calls)) == 7
    report = json.loads((tmp_path / "periodic/math/step-4/result.json").read_text())
    assert report["reasons"] == ["periodic", "training_end"]
    with pytest.raises(ValueError, match="already evaluated"):
        runner.run(4, final=True)


@pytest.mark.parametrize("fault", ["version", "responses", "nan"])
def test_invalid_backend_result_is_not_published(tmp_path, fault):
    def evaluate(**kwargs):
        return {
            "student_version": -1 if fault == "version" else kwargs["step"],
            "responses": 1 if fault == "responses" else 2,
            "metrics": {"accuracy/mean@2": float("nan") if fault == "nan" else 0.5},
        }

    runner = EvaluationRunner(
        evaluation(), SimpleNamespace(evaluate=evaluate), tmp_path
    )
    with pytest.raises(ValueError):
        runner.run(0)
    assert not list(tmp_path.rglob("result.json"))


def test_existing_output_is_not_reused(tmp_path):
    (tmp_path / "periodic/math/step-0").mkdir(parents=True)
    backend = SimpleNamespace(
        evaluate=lambda **kwargs: pytest.fail(
            "Existing output must fail before inference"
        )
    )
    with pytest.raises(FileExistsError):
        EvaluationRunner(evaluation(), backend, tmp_path).run(0)
