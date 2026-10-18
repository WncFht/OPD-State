"""Verify the engineering validation subset using only the standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BENCHMARKS = ("MATH-500", "AMC23", "AIME25")


def paths(directory):
    return [Path(directory).resolve() / f"{name}.parquet" for name in BENCHMARKS]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(path, *, per_benchmark=2):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "math-validation-subset-v1"
        or report.get("scope") != "engineering_only_not_quality"
        or report.get("per_benchmark") != per_benchmark
        or report.get("questions") != 3 * per_benchmark
    ):
        raise ValueError("Engineering validation subset protocol differs from the requested question count")
    rows = report.get("files", [])
    if [row.get("benchmark") for row in rows] != list(BENCHMARKS):
        raise ValueError("Engineering validation subset does not cover the three ordered benchmarks")
    for row, output in zip(rows, paths(path.parent), strict=True):
        if (
            row.get("selected_rows") != list(range(per_benchmark))
            or row.get("output_rows") != per_benchmark
            or row.get("input_rows", 0) < per_benchmark
            or row.get("output_file") != output.name
        ):
            raise ValueError("Engineering validation subset has inconsistent rows or output names")
        if digest(output) != row["output_sha256"] or digest(row["input_path"]) != row["input_sha256"]:
            raise ValueError("Engineering validation subset inputs or outputs changed after preparation")
    return report
