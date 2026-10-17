"""Compare matched questions and report a predeclared finite-budget plateau check.

Question bootstrap intervals describe sensitivity to question selection. They
do not measure variation across training seeds or establish asymptotic convergence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("collector", Path(__file__).with_name("collect_math_experiment.py"))
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)
PLATEAU_SPEC = importlib.util.spec_from_file_location(
    "plateau", Path(__file__).resolve().parents[1] / "verl/verl/trainer/ppo/plateau.py"
)
PLATEAU = importlib.util.module_from_spec(PLATEAU_SPEC)
PLATEAU_SPEC.loader.exec_module(PLATEAU)
BENCHMARKS = COLLECTOR.MATH.MATH_BENCHMARK_COUNTS
FIELDS = ("mean@16", "pass@1", "pass@2", "pass@4", "pass@8", "pass@16")


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_questions(directory, step):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    evidence = json.loads((directory / f"validation-step{step}.json").read_text())
    rows = csv_rows(directory / f"questions-step{step}.csv")
    if "questions_csv_sha256" in evidence:
        if COLLECTOR.sha256(directory / f"questions-step{step}.csv") != evidence["questions_csv_sha256"]:
            raise ValueError("Question CSV changed since collection")
    if not rows or not summary["full_benchmarks"]:
        raise ValueError("Comparison requires complete three-benchmark results")
    if any(not row.get("question_sha256") for row in rows):
        # Earlier collector revisions used ephemeral UID only. Recover stable
        # identities from their hashed raw evidence without modifying the run.
        raw_path = Path(summary["run_directory"]) / "validation" / f"{step}.jsonl"
        content = raw_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != evidence["response_jsonl_sha256"]:
            raise ValueError("Raw validation evidence changed since collection")
        mapping = {}
        for line in content.splitlines():
            raw = json.loads(line)
            key = str(raw["data_source"]), str(raw["validation_uid"])
            identity = COLLECTOR.question_identity(raw)
            if mapping.setdefault(key, identity) != identity:
                raise ValueError("Replicas with the same UID contain different questions")
        for row in rows:
            row["question_sha256"] = mapping[row["data_source"], row["uid"]]
    if Counter(row["data_source"] for row in rows) != Counter(BENCHMARKS):
        raise ValueError("Question counts do not match the full benchmark")
    if len({row["question_sha256"] for row in rows}) != len(rows):
        raise ValueError("Question identities must be unique within a run")
    result = {}
    for row in rows:
        n, correct = int(row["n"]), int(row["correct"])
        if n != 16 or not 0 <= correct <= 16:
            raise ValueError("Question correctness must use exactly 16 samples")
        values = {"mean@16": correct / 16}
        values.update({f"pass@{k}": COLLECTOR.MATH.estimate_pass_at_k(16, correct, k) for k in (1, 2, 4, 8, 16)})
        if any(
            not math.isfinite(float(row[key])) or abs(float(row[key]) - value) > 1e-10 for key, value in values.items()
        ):
            raise ValueError("Question CSV metrics disagree with its correctness counts")
        result[row["question_sha256"]] = {"data_source": row["data_source"], "correct": correct, **values}
    for source, count in BENCHMARKS.items():
        for field in FIELDS:
            mean = sum(row[field] for row in result.values() if row["data_source"] == source) / count
            if abs(mean - evidence["metrics"][f"eval/{source}/{field}"]) > 1e-10:
                raise ValueError("Question CSV aggregate disagrees with validation evidence")
    return summary, result


def plateau_check(directory, *, minimum_steps=300, points=4, accuracy_range=0.01, rms_span=0.05, grad_span=0.10):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    validations = csv_rows(directory / "validation-timeseries.csv")
    validations.sort(key=lambda row: int(row["step"]))
    settings = dict(
        minimum_steps=minimum_steps,
        points=points,
        accuracy_range=accuracy_range,
        rms_span=rms_span,
        grad_span=grad_span,
    )
    if summary["training_steps"] < minimum_steps or len(validations) < points:
        result = PLATEAU.assess_windows([], summary["training_steps"], **settings)
        result["validation_points"] = len(validations)
        return result
    training = {}
    for path in sorted(directory.glob("training-*.csv")):
        for row in csv_rows(path):
            step = int(row["step"])
            if step in training:
                raise ValueError("Duplicate training step in collected CSV files")
            training[step] = row
    if set(training) != set(range(1, int(summary["training_steps"]) + 1)):
        raise ValueError("Training evidence has missing steps")
    selected, windows = validations[-points:], []
    if any(int(b["step"]) - int(a["step"]) != 50 for a, b in zip(selected[:-1], selected[1:], strict=True)):
        raise ValueError("Plateau assessment requires validation every 50 steps")
    for val in selected:
        end = int(val["step"])
        epoch = [training[step] for step in range(end - 49, end + 1)]
        lengths = [float(row["response_length/mean"]) for row in epoch]
        rms = math.sqrt(
            sum(float(row["opd/log_ratio_rms"]) ** 2 * n for row, n in zip(epoch, lengths, strict=True)) / sum(lengths)
        )
        gradient = sum(float(row["actor/grad_norm"]) for row in epoch) / 50
        windows.append(
            {
                "step": end,
                "macro_mean@16": float(val["eval/macro/mean@16"]),
                "token_weighted_log_ratio_rms": rms,
                "mean_grad_norm": gradient,
            }
        )

    result = PLATEAU.assess_windows(windows, summary["training_steps"], **settings)
    result["validation_points"] = len(validations)
    return result


def compare(results, *, reference, step=50, replicates=5000, seed=42):
    import numpy as np

    if reference not in results or len(results) < 2 or replicates < 100:
        raise ValueError("Provide a named reference, at least two results, and >=100 bootstrap replicates")
    loaded = {name: load_questions(path, step) for name, path in results.items()}
    reference_rows = loaded[reference][1]
    identities = set(reference_rows)
    if any(set(rows) != identities for _, rows in loaded.values()):
        raise ValueError("Runs do not evaluate exactly the same question content")
    grouped = {
        source: sorted(key for key in identities if reference_rows[key]["data_source"] == source)
        for source in BENCHMARKS
    }
    rng = np.random.default_rng(seed)
    bootstrap = {source: rng.integers(0, len(keys), (replicates, len(keys))) for source, keys in grouped.items()}
    scores, differences = [], []
    for name, (summary, questions) in loaded.items():
        per_source = {}
        for source, keys in grouped.items():
            row = {"source": name, "benchmark": source, "questions": len(keys)}
            row.update({field: sum(questions[key][field] for key in keys) / len(keys) for field in FIELDS})
            scores.append(row)
            per_source[source] = row
        for aggregate in ("macro", "micro"):
            weights = {source: 1 / 3 if aggregate == "macro" else count / 570 for source, count in BENCHMARKS.items()}
            scores.append(
                {
                    "source": name,
                    "benchmark": aggregate,
                    "questions": 570,
                    **{
                        field: sum(weights[source] * row[field] for source, row in per_source.items())
                        for field in FIELDS
                    },
                }
            )
        if name == reference:
            continue
        draws = {}
        for source, keys in grouped.items():
            deltas = np.array([questions[key]["mean@16"] - reference_rows[key]["mean@16"] for key in keys])
            draws[source] = deltas[bootstrap[source]].mean(axis=1)
            low, high = np.quantile(draws[source], [0.025, 0.975])
            differences.append(
                {
                    "source": name,
                    "reference": reference,
                    "benchmark": source,
                    "difference_mean@16": float(deltas.mean()),
                    "question_bootstrap_low": float(low),
                    "question_bootstrap_high": float(high),
                }
            )
        for aggregate in ("macro", "micro"):
            weights = {source: 1 / 3 if aggregate == "macro" else count / 570 for source, count in BENCHMARKS.items()}
            combined = sum(weights[source] * values for source, values in draws.items())
            low, high = np.quantile(combined, [0.025, 0.975])
            delta = sum(
                weights[row["benchmark"]] * row["difference_mean@16"]
                for row in differences
                if row["source"] == name and row["benchmark"] in weights
            )
            differences.append(
                {
                    "source": name,
                    "reference": reference,
                    "benchmark": aggregate,
                    "difference_mean@16": delta,
                    "question_bootstrap_low": float(low),
                    "question_bootstrap_high": float(high),
                }
            )
    return {
        "analysis_source_sha256": COLLECTOR.sha256(Path(__file__).resolve()),
        "plateau_source_sha256": COLLECTOR.sha256(Path(PLATEAU_SPEC.origin)),
        "numpy_version": np.__version__,
        "step": step,
        "reference": reference,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interval_semantics": "paired question bootstrap, stratified by benchmark; not training-seed uncertainty",
        "run_provenance": {name: summary for name, (summary, _) in loaded.items()},
        "scores": scores,
        "differences": differences,
        "plateau_assessments": {name: plateau_check(path) for name, path in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = dict(value.split("=", 1) for value in args.result)
    if len(results) != len(args.result):
        raise ValueError("Result names must be unique")
    report = compare(results, reference=args.reference, step=args.step, replicates=args.replicates, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    COLLECTOR.write_csv(output / "scores.csv", report.pop("scores"))
    COLLECTOR.write_csv(output / "paired-differences.csv", report.pop("differences"))
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "plateau_assessments": report["plateau_assessments"]}))


if __name__ == "__main__":
    main()
