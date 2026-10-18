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
    previous_binding = None
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
                logical = execution.get("logical_phase", phase), execution.get("logical_slot", slot)
                if summary.get("weight_mode") == "staged":
                    binding = row["source_bindings"][execution["publication_index"]]
                    if (
                        slot != "current_student"
                        or execution["runtime_identity"] != binding["runtime_identity"]
                        or execution["logical_identity"] != binding["source_identity"]
                        or execution["logical_slot"] != binding["source_identity"][0]
                        or binding["committed"] is not True
                    ):
                        raise ValueError("Staged physical requests are not bound to a committed checkpoint publication")
                routed[logical] = routed.get(logical, 0) + len(ids)
            expected_routes = {(g["phase"], g["slot"]): g["requests"] for g in row["groups"]}
            if routed != expected_routes:
                raise ValueError("Physical benchmark receipts do not cover every generation group")
        key = policy_name(row["policy"]), row["batch"], row["cap"], row["natural_eos"]
        if summary.get("weight_mode") == "staged":
            bindings = row["source_bindings"]
            if len(bindings) != len(row["groups"]) or any(not b["committed"] for b in bindings):
                raise ValueError("Staged benchmark did not bind every generation group")
            if row["weight_copy_count"] != sum(b["copied"] for b in bindings):
                raise ValueError("Staged benchmark miscounted checkpoint publications")
            activation = math.fsum(b["seconds"] for b in bindings)
            if not 0 <= activation <= row["seconds"] or not math.isclose(
                activation, row["weight_activation_seconds"], abs_tol=1e-7
            ):
                raise ValueError("Staged activation time disagrees with its end-to-end measurement")
            for binding in bindings:
                arguments_replicas = summary["arguments"]["replicas"]
                if (
                    binding["runtime_identity"][:2] != ["current_student", summary["model_refs"]["initial_student"][1]]
                    or type(binding["runtime_identity"][2]) is not int
                    or binding["runtime_identity"][2] < 0
                    or type(binding["copied"]) is not bool
                    or not math.isfinite(binding["seconds"])
                    or binding["seconds"] < 0
                ):
                    raise ValueError("Staged publication has an invalid mutable container identity or timing")
                expected_transfers = arguments_replicas if binding["copied"] else 0
                if (
                    len(binding["transfers"]) != expected_transfers
                    or len(binding["server_status"]) != arguments_replicas
                ):
                    raise ValueError("Staged publication is missing a sender or native server receipt")
                if any(
                    t["fingerprint"] != binding["source_identity"][2]
                    or t["parameters"] <= 0
                    or not math.isfinite(t["seconds"])
                    or t["seconds"] <= 0
                    for t in binding["transfers"]
                ):
                    raise ValueError("Staged sender loaded another or incomplete checkpoint")
                if any(
                    not s["valid"] or not s["awake"] or s["active"] or s["identity"] != binding["runtime_identity"]
                    for s in binding["server_status"]
                ):
                    raise ValueError("Staged server publication was not valid, awake and idle")
                role = binding["source_identity"][0]
                if binding["source_identity"] != summary["model_refs"].get(role):
                    raise ValueError("Staged logical source differs from the declared immutable checkpoint")
                if previous_binding is not None:
                    changed = binding["source_identity"] != previous_binding["source_identity"]
                    version, previous_version = binding["runtime_identity"][2], previous_binding["runtime_identity"][2]
                    if version < previous_version or (
                        changed and (not binding["copied"] or version <= previous_version)
                    ):
                        raise ValueError("Staged checkpoint transition reused a previous container version")
                previous_binding = binding
            transfer_seconds = math.fsum(max((t["seconds"] for t in b["transfers"]), default=0) for b in bindings)
            if (
                not math.isclose(transfer_seconds, row["weight_transfer_seconds"], abs_tol=1e-7)
                or transfer_seconds > activation
            ):
                raise ValueError("Staged weight-transfer timing does not match device-local sender receipts")
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
                "run": run.name,
                "weight_mode": summary.get("weight_mode", "resident_concurrent"),
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
                "weight_activation_seconds_mean": statistics.mean(r.get("weight_activation_seconds", 0) for r in rows),
                "weight_copy_count_mean": statistics.mean(r.get("weight_copy_count", 0) for r in rows),
                "weight_transfer_seconds_mean": statistics.mean(r.get("weight_transfer_seconds", 0) for r in rows),
                "prefill_tokens_mean": statistics.mean(sum(g["input_tokens"] for g in r["groups"]) for r in rows),
            }
        )
    memory = []
    for slot, spec in summary["engines"].items():
        for gpu, layout in zip(spec["gpus"], spec["layouts"], strict=True):
            memory.append(
                {
                    "layout": arguments["layout"],
                    "run": run.name,
                    "weight_mode": summary.get("weight_mode", "resident_concurrent"),
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
        "weight_mode": summary.get("weight_mode", "resident_concurrent"),
        "sender_cpu_cache": summary.get("sender_cpu_cache", []),
        "numerical_checks": summary.get("numerical_checks", []),
        "numerical_reference": summary.get("numerical_reference"),
    }
    if summary.get("weight_mode") == "staged":
        checks = summary.get("numerical_checks", [])
        expected_checks = {
            (phase, source)
            for phase in ("before_measurement", "after_measurement")
            for source in ("initial_student", "teacher")
        }
        if {(c["phase"], c["source"]) for c in checks} != expected_checks or len(checks) != 4:
            raise ValueError("Staged benchmark lacks before/after independent full-vocabulary reference checks")
        if any(
            c["reference_created"] is not False
            or not 0 < c["logit_tolerance"] <= 0.5
            or not 0 < c["tv_tolerance"] <= 0.05
            or not 0 <= c["max_logit_error"] <= c["logit_tolerance"]
            or not 0 <= c["max_tv"] <= c["tv_tolerance"]
            for c in checks
        ):
            raise ValueError("Staged/resident numerical equivalence was not established")
        reference = summary["numerical_reference"]
        path = Path(reference["directory"]) / "reference.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["metadata_sha256"]:
            raise ValueError("Independent resident numerical reference metadata changed")
        metadata = json.loads(path.read_text())
        if metadata["models"] != summary["model_refs"] or metadata["producer"]["weight_mode"] not in {
            "resident_serial",
            "resident_concurrent",
        }:
            raise ValueError("Staged and resident references use different checkpoint identities")
        for role, digest in metadata["arrays"].items():
            if hashlib.sha256((path.parent / f"{role}.npy").read_bytes()).hexdigest() != digest:
                raise ValueError("Independent resident numerical reference logits changed")
        import numpy as np

        for check in checks:
            observed_path = Path(check["observed_logits"])
            if hashlib.sha256(observed_path.read_bytes()).hexdigest() != check["observed_sha256"]:
                raise ValueError("Staged observed full-vocabulary logits changed")
            observed = np.load(observed_path, allow_pickle=False)
            expected = np.load(path.parent / f"{check['source']}.npy", allow_pickle=False)
            shape = check["prefix_count"], summary["effective_vocab_size"]
            if (
                observed.shape != shape
                or expected.shape != shape
                or check["vocab_size"] != shape[1]
                or shape[0] != len(metadata["prompts"])
                or shape[0] <= 0
                or metadata["vocabulary"] != shape[1]
                or not np.isfinite(observed).all()
                or not np.isfinite(expected).all()
            ):
                raise ValueError("Staged numerical comparison did not cover complete finite vocabulary logits")

            def probability(logits):
                values = np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True))
                return values / values.sum(-1, keepdims=True)

            error = float(np.abs(observed - expected).max())
            tv = float(np.abs(probability(observed) - probability(expected)).sum(-1).max() / 2)
            if not math.isclose(error, check["max_logit_error"], rel_tol=0, abs_tol=1e-7) or not math.isclose(
                tv, check["max_tv"], rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError("Staged numerical statistics disagree with independent full-vocabulary recomputation")
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
