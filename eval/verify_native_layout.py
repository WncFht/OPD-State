"""Reconcile physical worker receipts and native diagnostic reductions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from collect_math_experiment import parse_steps, sha256
from native_devices import identity_aliases, normalize_uuid


def trajectory_counts(rows):
    phases, actions = {}, Counter()
    for row in rows:
        for phase, segment in enumerate(row["segments"]):
            roles = [identity[0] for identity in segment["identities"]]
            slot = roles[0] if len(roles) == 1 else "fusion:" + roles[0]
            action = roles[0] if len(roles) == 1 else "fusion_" + roles[0]
            phases.setdefault(phase, Counter())[slot] += 1
            actions[action] += segment["end"] - segment["start"]
    return phases, actions


def check_batch(batch, expected, gpu_count, rank_devices, allowed):
    placements = batch.get("generation_placement", [])
    executions = batch.get("generation_execution", [])
    if len(placements) != len(expected) or not executions:
        raise ValueError("Missing phase placement or physical execution receipts")
    groups = {}
    for group in batch["generation"]["groups"]:
        slot = "eval:current_student" if batch["validate"] else group["slot"]
        groups.setdefault(group["phase"], Counter())[slot] += group["requests"]
    if groups != expected:
        raise ValueError("Generation groups disagree with actual trajectory stages")
    actual = {}
    ids = set()
    for receipt in executions:
        phase, slot, rank = receipt["phase"], receipt["slot"], receipt["replica_rank"]
        if phase not in expected or slot not in expected[phase] or type(rank) is not int or not 0 <= rank < gpu_count:
            raise ValueError("Worker receipt has an unexpected phase/source/replica")
        observed = normalize_uuid(receipt["device_uuid"])
        uuid = allowed.get(observed) if isinstance(allowed, dict) else observed
        if uuid not in allowed or not isinstance(receipt["pid"], int) or receipt["pid"] <= 0:
            raise ValueError("Worker receipt is outside the allowed physical GPUs or lacks a process")
        if rank in rank_devices and rank_devices[rank] != uuid:
            raise ValueError("A replica index changed its physical device")
        rank_devices[rank] = uuid
        if len(set(rank_devices.values())) != len(rank_devices):
            raise ValueError("Different replica indices unexpectedly share one physical GPU")
        request_ids = receipt["request_ids"]
        if receipt["requests"] != len(request_ids) or not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError("Physical worker request count or identities are invalid")
        if ids.intersection(request_ids):
            raise ValueError("A native request was assigned to more than one physical worker")
        ids.update(request_ids)
        key = phase, slot, rank
        if key in actual:
            raise ValueError("Duplicate source/replica batch receipt")
        actual[key] = len(request_ids)
    observations = []
    for phase, counts in expected.items():
        placement = placements[phase]
        if placement["requests"] != counts or set(placement["replicas"]) != set(counts):
            raise ValueError("Configured source counts disagree with the observed trajectories")
        layout = placement["layout"]
        if layout not in {"overlap", "split"} or (batch["validate"] and layout != "overlap"):
            raise ValueError("Invalid layout or nonstandard final student evaluation")
        occupied = set()
        for slot, count in counts.items():
            ranks = placement["replicas"][slot]
            if not ranks or len(set(ranks)) != len(ranks) or len(ranks) > min(count, gpu_count):
                raise ValueError("Source placement contains empty, duplicated or excessive replicas")
            if layout == "overlap" and ranks != list(range(min(count, gpu_count))):
                raise ValueError("Overlap baseline no longer uses its declared replica mapping")
            if layout == "split" and occupied.intersection(ranks):
                raise ValueError("Split source groups overlap on physical replicas")
            occupied.update(ranks)
            for offset, rank in enumerate(ranks):
                wanted = count // len(ranks) + (offset < count % len(ranks))
                if actual.pop((phase, slot, rank), None) != wanted:
                    raise ValueError("Actual physical request shards differ from configured placement")
                observations.append(
                    {
                        "step": batch["step"],
                        "validate": batch["validate"],
                        "phase": phase,
                        "slot": slot,
                        "replica_rank": rank,
                        "device_uuid": rank_devices[rank],
                        "requests": wanted,
                        "layout": layout,
                    }
                )
        if layout == "split" and len(occupied) != min(gpu_count, sum(counts.values())):
            raise ValueError("Split mapping left available request-bearing devices unused")
    if actual:
        raise ValueError("Physical receipts include unplanned request shards")
    return observations


def check_metrics(metrics, actions):
    total = sum(actions.values())
    if metrics.get("native_state/all/tokens") != total:
        raise ValueError("Diagnostic valid-token count differs from complete trajectories")
    fields = ["sampled_log_ratio_mean", "teacher_selected_action_probability_mean", "log_ratio_rms"]
    for source, count in actions.items():
        prefix = f"native_state/action_{source}/"
        if metrics.get(prefix + "tokens") != count:
            raise ValueError("Per-action diagnostic token counts disagree with source segments")
        for field in fields:
            if not math.isfinite(metrics[prefix + field]):
                raise ValueError("Nonfinite source-stratified diagnostic")
        probability = metrics[prefix + "teacher_selected_action_probability_mean"]
        if not 0 <= probability <= 1.00001 or metrics[prefix + "log_ratio_rms"] < 0:
            raise ValueError("Invalid teacher action probability or RMS")
    for field in fields:
        exponent = 2 if field == "log_ratio_rms" else 1
        weighted = (
            sum(
                count * metrics[f"native_state/action_{source}/{field}"] ** exponent
                for source, count in actions.items()
            )
            / total
        )
        if not math.isclose(weighted, metrics[f"native_state/all/{field}"] ** exponent, rel_tol=1e-5, abs_tol=2e-6):
            raise ValueError("Source diagnostic reductions do not reconcile with the global value")
    if not math.isclose(
        metrics["native_state/all/sampled_log_ratio_mean"],
        -metrics["opd/token_reward_mean"],
        rel_tol=1e-5,
        abs_tol=2e-6,
    ):
        raise ValueError("Native diagnostics disagree with the actual OPD token reward")


def verify(run, output, steps, allowed):
    run, output = Path(run), Path(output)
    base_path = output / "native-acceptance.json"
    base = json.loads(base_path.read_text())
    if base["status"]["exit_code"] != 0 or base["final_student_version"] != steps:
        raise ValueError("Run must pass the independent native trainer audit first")
    paths = list(run.glob("checkpoints/**/native-trajectories/manifest.json"))
    if len(paths) != 1:
        raise ValueError("Expected one native trajectory manifest")
    directory = paths[0].parent
    manifest = json.loads(paths[0].read_text())
    if not manifest.get("generation_state_metrics"):
        raise ValueError("This acceptance requires source-stratified diagnostic collection")
    logs = list((run / "logs").glob("*.log"))
    if len(logs) != 1:
        raise ValueError("Expected one trainer log")
    _, metrics = parse_steps(logs[0].read_text())
    training = {
        step: [json.loads(line) for line in (directory / f"step-{step}.jsonl").read_text().splitlines()]
        for step in range(1, steps + 1)
    }
    batches = [json.loads(line) for line in (directory / "batches.jsonl").read_text().splitlines()]
    if Counter(b["step"] for b in batches if not b["validate"]) != Counter(range(1, steps + 1)):
        raise ValueError("Diagnostic audit needs exactly one generation batch per training update")
    devices, observations = {}, []
    for batch in batches:
        if batch["validate"]:
            expected = {0: Counter({"eval:current_student": batch["metrics"]["source/response_count"]})}
        else:
            expected, actions = trajectory_counts(training[batch["step"]])
            check_metrics(metrics[batch["step"]], actions)
            if any(p["layout"] != manifest["generation_layout"] for p in batch["generation_placement"]):
                raise ValueError("Actual training layout differs from the run manifest")
        observations.extend(check_batch(batch, expected, manifest["trainer_gpu_count"], devices, allowed))
    result = {
        "scope": "Actual worker identities, request placement and sampled-action metric reductions",
        "rank_device_namespace": "nvml_via_pci" if isinstance(allowed, dict) else "provided_uuid_set",
        "base_audit_sha256": sha256(base_path),
        "batches_sha256": sha256(directory / "batches.jsonl"),
        "trainer_log_sha256": sha256(logs[0]),
        "analysis_source_sha256": sha256(Path(__file__)),
        "layout": manifest["generation_layout"],
        "rank_devices": devices,
        "observations": observations,
    }
    (output / "native-layout-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--gpu-uuids", nargs="+")
    selection.add_argument("--device-identities", type=Path)
    args = parser.parse_args()
    if args.device_identities:
        command = json.loads((args.run / "command.json").read_text())
        expected = command["environment"].get("NATIVE_DEVICE_IDENTITIES_SHA256")
        if not expected or expected != sha256(args.device_identities):
            raise ValueError("Device identity map changed since the training launch")
        allowed = identity_aliases(json.loads(args.device_identities.read_text()))
    else:
        allowed = {normalize_uuid(value) for value in args.gpu_uuids}
    result = verify(args.run, args.output, args.steps, allowed)
    if args.device_identities:
        result["device_identities_sha256"] = expected
        (args.output / "native-layout-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"layout": result["layout"], "physical_shards": len(result["observations"])}))


if __name__ == "__main__":
    main()
