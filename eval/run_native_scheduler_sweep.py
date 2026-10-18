"""Serial inference controls for replicas, KV budgets and native scheduler limits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def cases():
    base = {
        "layout": "overlap",
        "replicas": 4,
        "budget": 0.4,
        "sequences": 64,
        "batched_tokens": 8192,
        "chunked_prefill": True,
        "graph": True,
    }
    rows = [{**base, "name": f"overlap{n}", "replicas": n} for n in [1, 2, 4]]
    rows += [{**base, "name": f"split{n}", "replicas": n, "layout": "split"} for n in [2, 4]]
    rows += [{**base, "name": f"seq{n}", "sequences": n} for n in [4, 8, 16, 32]]
    rows += [{**base, "name": f"tokens{n}", "batched_tokens": n} for n in [512, 2048]]
    rows += [{**base, "name": "no-chunk", "chunked_prefill": False}]
    rows += [{**base, "name": name, "budget": value} for name, value in [("budget030", 0.3), ("budget055", 0.55)]]
    rows += [{**base, "name": "eager4", "graph": False}]
    return rows


def command(case, output, commit, python):
    result = [
        str(python),
        str(ROOT / "eval/benchmark_native_policies.py"),
        "--output",
        str(output),
        "--source-commit",
        commit,
        "--layout",
        case["layout"],
        "--replicas",
        str(case["replicas"]),
        "--budget",
        str(case["budget"]),
        "--batches",
        "64",
        "--cap",
        "128",
        "--repeats",
        "3",
        "--natural-cap",
        "4096",
        "--natural-repeats",
        "3",
        "--natural-warmup",
        "--max-num-seqs",
        str(case["sequences"]),
        "--max-num-batched-tokens",
        str(case["batched_tokens"]),
        "--policies",
        "initial_student",
        "batch_mix",
        "switch_p0_q",
    ]
    result += ["--chunked-prefill" if case["chunked_prefill"] else "--no-chunked-prefill"]
    if case["graph"]:
        result.append("--graph")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", type=Path, default=ROOT.parent / "envs/opd-state/bin/python")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    selected = [case for case in cases() if args.only is None or case["name"] in args.only]
    if not selected or (args.only and set(args.only) != {case["name"] for case in selected}):
        parser.error("Unknown scheduler sweep case")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": commit,
        "scope": "Inference-only fixed-work and natural-length observations; no training or math accuracy claims",
        "cases": [],
    }

    def save():
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "state.json")

    save()
    for case in selected:
        output = directory / case["name"]
        cmd = command(case, output, commit, args.python)
        item = {**case, "state": "planned", "command": cmd}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("GPU 0–3 are occupied; wait for the existing experiment")
        with (directory / (case["name"] + "-driver.log")).open("w") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            item.update(state="running", pid=process.pid, started_at=time.time())
            save()
            code = process.wait()
        item.update(state="finished" if code == 0 else "failed", exit_code=code, finished_at=time.time())
        save()
        if code:
            state["state"] = "failed"
            save()
            return code
        result = ROOT / "docs/experiment/results/native-rollout/scheduler-sweep" / directory.name / case["name"]
        code = subprocess.call(
            [
                sys.executable,
                str(ROOT / "eval/summarize_native_benchmarks.py"),
                "--runs",
                str(output),
                "--output",
                str(result),
            ],
            cwd=ROOT,
        )
        item["audit_exit_code"] = code
        save()
        if code:
            state["state"] = "audit_failed"
            save()
            return code
    state["state"] = "planned" if args.plan_only else "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
