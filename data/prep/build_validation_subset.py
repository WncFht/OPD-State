"""Build deterministic engineering subsets without altering full math evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BENCHMARKS = ("MATH-500", "AMC23", "AIME25")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build(inputs, output, *, per_benchmark=2):
    import pyarrow
    import pyarrow.parquet as pq

    if type(per_benchmark) is not int or per_benchmark < 1:
        raise ValueError("Engineering validation needs a positive per-benchmark question count")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Do not replace an existing validation subset")
    tables = {}
    for filename in inputs:
        source = Path(filename).resolve()
        table = pq.read_table(source)
        if "data_source" not in table.column_names:
            raise ValueError("Validation input must identify its benchmark")
        names = set(table["data_source"].to_pylist())
        if len(names) != 1 or next(iter(names)) not in BENCHMARKS:
            raise ValueError("Each validation input must contain one supported benchmark")
        name = next(iter(names))
        if name in tables or table.num_rows < per_benchmark:
            raise ValueError("Validation benchmarks must be unique and have enough questions")
        tables[name] = source, table
    if set(tables) != set(BENCHMARKS):
        raise ValueError("Engineering validation must cover all three math benchmarks")
    output.mkdir(parents=True)
    records = []
    for name in BENCHMARKS:
        source, table = tables[name]
        target = output / f"{name}.parquet"
        pq.write_table(table.slice(0, per_benchmark), target, compression="zstd")
        records.append(
            {
                "benchmark": name,
                "input_path": str(source),
                "input_sha256": digest(source),
                "input_rows": table.num_rows,
                "selected_rows": list(range(per_benchmark)),
                "output_file": target.name,
                "output_rows": per_benchmark,
                "output_sha256": digest(target),
            }
        )
    manifest = {
        "schema": "math-validation-subset-v1",
        "scope": "engineering_only_not_quality",
        "per_benchmark": per_benchmark,
        "questions": len(BENCHMARKS) * per_benchmark,
        "pyarrow_version": pyarrow.__version__,
        "builder_sha256": digest(Path(__file__)),
        "files": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=2)
    args = parser.parse_args()
    report = build(args.input, args.output, per_benchmark=args.per_benchmark)
    print(json.dumps({"output": str(args.output.resolve()), "questions": report["questions"]}))


if __name__ == "__main__":
    main()
