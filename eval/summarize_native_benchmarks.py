"""Validate native benchmark work counts and aggregate warm/cold measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def policy_name(policy):
    kind = policy["kind"]
    if kind == "single":
        return policy["model"]
    if kind == "sequence_switch":
        return policy["first"] + "->" + policy["second"]
    if kind == "logit_fusion":
        return "fusion:" + policy["base"]
    if kind == "batch_mix":
        return "batch_mix"
    raise ValueError(f"Unknown benchmark policy: {kind}")


def collect(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    observations = [json.loads(line) for line in (run / "observations.jsonl").read_text().splitlines()]
    if len(observations) != summary["observations"]:
        raise ValueError("Benchmark observation count does not match its completion summary")
    groups = {}
    for row in observations:
        if not all(
            math.isfinite(row[k]) and row[k] > 0 for k in ["seconds", "tokens_per_second", "responses_per_second"]
        ):
            raise ValueError("Invalid native benchmark timing")
        if not row["natural_eos"] and row["tokens"] != row["batch"] * row["cap"]:
            raise ValueError("Fixed-length benchmark did not produce the requested token work")
        if sum(g["output_tokens"] for g in row["groups"]) != row["tokens"]:
            raise ValueError("Native group token counts do not cover the full responses")
        if abs(row["tokens_per_second"] - row["tokens"] / row["seconds"]) > 1e-7:
            raise ValueError("Throughput does not match the recorded work/time")
        if row["repeat"] < 0:
            continue
        key = policy_name(row["policy"]), row["batch"], row["cap"], row["natural_eos"]
        groups.setdefault(key, []).append(row)
    arguments = summary["arguments"]
    result = []
    for (policy, batch, cap, natural), rows in sorted(groups.items()):
        expected = 1 if natural else arguments["repeats"]
        if len(rows) != expected or {r["repeat"] for r in rows} != set(range(expected)):
            raise ValueError("Missing or duplicate native benchmark repeats")
        result.append(
            {
                "layout": arguments["layout"],
                "replicas": arguments["replicas"],
                "graph": arguments["graph"],
                "policy": policy,
                "batch": batch,
                "cap": cap,
                "natural_eos": natural,
                "repeats": len(rows),
                "seconds_mean": statistics.mean(r["seconds"] for r in rows),
                "seconds_min": min(r["seconds"] for r in rows),
                "seconds_max": max(r["seconds"] for r in rows),
                "tokens_mean": statistics.mean(r["tokens"] for r in rows),
                "tokens_per_second_mean": statistics.mean(r["tokens_per_second"] for r in rows),
                "responses_per_second_mean": statistics.mean(r["responses_per_second"] for r in rows),
                "prefill_tokens_mean": statistics.mean(sum(g["input_tokens"] for g in r["groups"]) for r in rows),
            }
        )
    memory = []
    for slot, spec in summary["engines"].items():
        for gpu, layout in zip(spec["gpus"], spec["layouts"], strict=True):
            memory.append(
                {
                    "layout": arguments["layout"],
                    "replicas": arguments["replicas"],
                    "slot": slot,
                    "gpu": gpu,
                    "budget": spec["budget_per_engine"],
                    "parameter_bytes": layout["parameter_bytes"],
                    "kv_bytes": sum(r["bytes"] for r in layout["kv_regions"]),
                    "cuda_graphs_captured": layout.get("cuda_graphs_captured", 0),
                }
            )
    receipt = {
        "directory": str(run.resolve()),
        "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
        "observations_sha256": hashlib.sha256((run / "observations.jsonl").read_bytes()).hexdigest(),
        "observations": len(observations),
        "cold_seconds": summary["cold_seconds"],
    }
    return result, memory, receipt


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings, memory, receipts = [], [], []
    for run in args.runs:
        a, b, c = collect(run)
        timings.extend(a)
        memory.extend(b)
        receipts.append(c)
    write_csv(args.output / "timings.csv", timings)
    write_csv(args.output / "memory.csv", memory)
    (args.output / "receipt.json").write_text(
        json.dumps(
            {
                "scope": "inference-only native benchmark; allocator KV regions, not whole-job GPU peaks",
                "inputs": receipts,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"conditions": len(timings), "engine_records": len(memory)}))


if __name__ == "__main__":
    main()
