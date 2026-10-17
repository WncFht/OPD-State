"""Exact finite-sample math evaluation alongside veRL's native metrics.

mean@N is average correctness over N independent samples for each question.
pass@k uses the unbiased without-replacement estimator. It is deliberately not
an alias for the upstream bootstrap best@k statistic.
"""

from __future__ import annotations

import math
from collections import defaultdict

MATH_BENCHMARK_COUNTS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}


def estimate_pass_at_k(n: int, correct: int, k: int) -> float:
    if any(type(v) is not int for v in (n, correct, k)):
        raise TypeError("Sample counts and k must be integers")
    if n < 1 or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Expected n>=1, 0<=correct<=n and 1<=k<=n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod((n - correct - j) / (n - j) for j in range(k))


def exact_math_metrics(
    data_sources,
    sample_uids,
    correctness,
    *,
    samples_per_question,
    required_counts=None,
    required_sources=None,
    ks=(1, 2, 4, 8, 16),
):
    if not len(data_sources) == len(sample_uids) == len(correctness) or not len(data_sources):
        raise ValueError("Validation columns must have equal, nonzero lengths")
    grouped = defaultdict(list)
    for source, uid, value in zip(data_sources, sample_uids, correctness, strict=True):
        score = float(value)
        if score not in (0.0, 1.0):
            raise ValueError("pass@k requires binary per-sample correctness")
        grouped[(str(source), str(uid))].append(score)
    sources = {source for source, _ in grouped}
    if required_sources is not None and sources != set(required_sources):
        raise ValueError(f"Validation benchmark coverage mismatch: {sorted(sources)}")
    records, by_source = [], defaultdict(list)
    for (source, uid), values in grouped.items():
        if len(values) != samples_per_question:
            raise ValueError(f"Expected {samples_per_question} samples for {source}/{uid}, got {len(values)}")
        n, c = len(values), int(sum(values))
        row = {"data_source": source, "uid": uid, "n": n, "correct": c, f"mean@{n}": c / n}
        row.update({f"pass@{k}": estimate_pass_at_k(n, c, k) for k in ks if k <= n})
        records.append(row)
        by_source[source].append(row)
    if required_counts is not None:
        counts = {key: len(value) for key, value in by_source.items()}
        if counts != dict(required_counts):
            raise ValueError(f"Validation question counts mismatch: {counts}")
    fields = [f"mean@{samples_per_question}", *[f"pass@{k}" for k in ks if k <= samples_per_question]]
    metrics = {}
    for source, questions in by_source.items():
        metrics[f"eval/{source}/questions"] = len(questions)
        metrics[f"eval/{source}/samples_per_question"] = samples_per_question
        for field in fields:
            metrics[f"eval/{source}/{field}"] = sum(row[field] for row in questions) / len(questions)
    for field in fields:
        metrics[f"eval/macro/{field}"] = sum(metrics[f"eval/{source}/{field}"] for source in by_source) / len(by_source)
        metrics[f"eval/micro/{field}"] = sum(row[field] for row in records) / len(records)
    return metrics, records
