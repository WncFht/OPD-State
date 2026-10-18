"""Run ten short native kernel profiles in the existing serial GPU queue."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from pathlib import Path

from native_devices import matches_device
from native_profile_metrics import trace_metrics
from probe_batch_invariance import digest
from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return [
        {
            "name": f"{'graph' if graph else 'eager'}-{role}-b{batch}-t{budget}",
            "graph": graph,
            "role": role,
            "batch": batch,
            "token_budget": budget,
        }
        for graph in (False, True)
        for role, batch, budget in (
            ("initial_student", 4, 8192),
            ("initial_student", 16, 8192),
            ("teacher", 16, 8192),
            ("fusion", 16, 8192),
            ("initial_student", 16, 64),
        )
    ]


def command(case, output, commit, prompts, python):
    result = [
        str(python),
        str(ROOT / "eval/profile_native_kernels.py"),
        "--output",
        str(output),
        "--source-commit",
        commit,
        "--prompts",
        str(prompts),
        "--role",
        case["role"],
        "--batch",
        str(case["batch"]),
        "--cap",
        "32",
        "--token-budget",
        str(case["token_budget"]),
    ]
    return result + (["--graph"] if case["graph"] else [])


def audit_artifacts(output):
    summary = json.loads((output / "summary.json").read_text())
    for name, expected in summary["artifact_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Profile artifact changed: " + name)
    trace_path = Path(summary["trace_file"])
    if digest(trace_path) != summary["trace_sha256"]:
        raise ValueError("Native profiler trace changed")
    with gzip.open(trace_path, "rt") as stream:
        trace = json.load(stream)
    steps = json.loads((output / "scheduler-steps.json").read_text())
    recomputed = trace_metrics(trace, steps)
    if recomputed != json.loads((output / "kernel-metrics.json").read_text()):
        raise ValueError("Retained native kernel metrics do not reproduce from the raw trace")
    command = json.loads((output / "command.json").read_text())
    responses = json.loads((output / "responses.json").read_text())
    worker = json.loads((output / "worker.json").read_text())
    args = command["arguments"]
    inputs = json.loads((output / "inputs.json").read_text())
    if not matches_device(worker["device_uuid"], command["devices"]["0"]):
        raise ValueError("Native profile worker is not on the requested physical GPU")
    batches = [responses["profiled"], *(values for values, _ in responses["controls"])]
    if (
        len(inputs) != args["batch"]
        or len(batches) != 3
        or any(
            len(batch) != args["batch"]
            or any(
                len(ids) != args["cap"]
                or any(type(token) is not int or not 0 <= token < command["vocab_size"] for token in ids)
                for ids in batch
            )
            for batch in batches
        )
    ):
        raise ValueError("Native profile responses changed fixed work or token support")
    equal = all(batch == batches[0] for batch in batches[1:])
    passed = (
        equal
        and recomputed["complete_attribution"]
        and all(recomputed["by_phase"][phase]["kernels"] > 0 for phase in ("prefill", "decode"))
    )
    if args["token_budget"] == 64:
        passed = passed and recomputed["by_phase"]["mixed"]["kernels"] > 0
    if summary["output_tokens_unchanged"] != equal or summary["passed"] != passed:
        raise ValueError("Native profile acceptance disagrees with raw controls and trace coverage")
    return {
        "scope": "Artifact hashes and deterministic trace re-aggregation; not a second neural execution",
        "passed": summary["passed"],
        "summary_sha256": digest(output / "summary.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", type=Path, default=ROOT.parent / "envs/opd-state/bin/python")
    parser.add_argument(
        "--prompts", type=Path, default=ROOT / "runs/native-rollout-implementation/fresh-math-data-v2/prompts.jsonl"
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not args.plan_only:
        expected = subprocess.check_output(
            ["git", "show", f"{args.source_commit}:eval/{Path(__file__).name}"], cwd=ROOT
        )
        if expected != Path(__file__).read_bytes():
            raise ValueError("Queued profiling driver differs from its requested commit")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": args.source_commit,
        "driver_sha256": digest(Path(__file__)),
        "scope": "Fixed-work native inference kernel diagnostics; profiling overhead prevents throughput conclusions",
        "cases": [],
    }

    def save():
        temp = directory / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(directory / "state.json")

    save()
    failures = []
    for case in cases():
        output = directory / case["name"]
        cmd = command(case, output, args.source_commit, args.prompts, args.python)
        item = {**case, "state": "planned", "command": cmd}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("Every physical GPU 0–3 must be idle before profiling")
        with (directory / (case["name"] + ".log")).open("w") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            item.update(state="running", pid=process.pid, started_at=time.time())
            save()
            code = process.wait()
        item.update(exit_code=code, finished_at=time.time())
        if code not in (0, 2):
            item["state"], state["state"] = "execution_failed", "failed"
            save()
            return code
        receipt = audit_artifacts(output)
        (output / "artifact-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if receipt["passed"] != (code == 0):
            raise ValueError("Native profile exit code disagrees with its saved acceptance result")
        item["state"] = "passed" if code == 0 else "failed_property"
        if code:
            failures.append(case["name"])
        save()
    state["failed_properties"] = failures
    state["state"] = "planned" if args.plan_only else "completed_with_failed_properties" if failures else "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
