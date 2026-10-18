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
    groups, warmups = {}, {}
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
        if abs(row["responses_per_second"] - row["batch"] / row["seconds"]) > 1e-7:
            raise ValueError("Response throughput does not match the recorded batch/time")
        executions = row.get("physical_execution")
        if summary.get("physical_receipts_required") and not executions:
            raise ValueError("Benchmark completion summary requires actual worker execution receipts")
        if executions:
            routed, request_ids = {}, set()
            for execution in executions:
                slot, phase, rank = execution["slot"], execution["phase"], execution["replica_rank"]
                physical_gpus = summary["engines"][slot]["gpus"]
                if type(rank) is not int or not 0 <= rank < len(physical_gpus):
                    raise ValueError("Physical benchmark receipt has an invalid source-local replica rank")
                expected_uuid = summary["physical_gpu_uuids"][str(physical_gpus[rank])].lower().removeprefix("gpu-")
                if execution["device_uuid"].lower().removeprefix("gpu-") != expected_uuid:
                    raise ValueError("Physical benchmark receipt used a different GPU")
                ids = execution["request_ids"]
                if len(ids) != execution["requests"] or len(set(ids)) != len(ids) or request_ids.intersection(ids):
                    raise ValueError("Physical benchmark receipts duplicate or miscount native requests")
                request_ids.update(ids)
                routed[(phase, slot)] = routed.get((phase, slot), 0) + len(ids)
            expected_routes = {(g["phase"], g["slot"]): g["requests"] for g in row["groups"]}
            if routed != expected_routes:
                raise ValueError("Physical benchmark receipts do not cover every generation group")
        key = policy_name(row["policy"]), row["batch"], row["cap"], row["natural_eos"]
        if row["repeat"] < 0:
            warmups[key] = warmups.get(key, 0) + 1
            continue
        groups.setdefault(key, []).append(row)
    arguments = summary["arguments"]
    scheduler = summary.get("scheduler", {})
    result = []
    for (policy, batch, cap, natural), rows in sorted(groups.items()):
        expected = arguments.get("natural_repeats", 1) if natural else arguments["repeats"]
        expected_warmup = int(not natural or arguments.get("natural_warmup", False))
        if warmups.get((policy, batch, cap, natural), 0) != expected_warmup:
            raise ValueError("Missing or duplicate native benchmark warmups")
        if len(rows) != expected or {r["repeat"] for r in rows} != set(range(expected)):
            raise ValueError("Missing or duplicate native benchmark repeats")
        result.append(
            {
                "layout": arguments["layout"],
                "replicas": arguments["replicas"],
                "graph": arguments["graph"],
                "budget": arguments.get("budget"),
                "max_num_seqs": scheduler.get("max_num_seqs"),
                "max_num_batched_tokens": scheduler.get("max_num_batched_tokens"),
                "chunked_prefill": scheduler.get("chunked_prefill"),
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
    telemetry = summary.get("gpu_telemetry")
    if telemetry is not None:
        observed = hashlib.sha256((run / "gpu.csv").read_bytes()).hexdigest()
        if observed != telemetry["sha256"]:
            raise ValueError("GPU telemetry content differs from its recorded summary hash")
        receipt["gpu_telemetry_sha256"] = observed
        receipt["whole_card_sampled_peak_mib"] = telemetry["sampled_peak_mib"]
        receipt["telemetry_errors"] = telemetry["errors"]
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
