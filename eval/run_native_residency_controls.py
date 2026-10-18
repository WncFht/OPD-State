"""Compare resident parallel/serial engines with staged checkpoint publication.

The reversed second block limits order confounding; two cold starts per mode
and three warm observations per workload are descriptive performance evidence,
not a stable latency confidence interval. This does not run student training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def cases():
    modes = ["resident_serial", "staged", "resident_concurrent"]
    pilots = [{"name": f"pilot-{mode}", "weight_mode": mode, "block": 0, "pilot": True} for mode in modes[:2]]
    return pilots + [
        {"name": f"{mode}-{block}", "weight_mode": mode, "block": block, "pilot": False}
        for block, order in [(1, modes), (2, list(reversed(modes)))]
        for mode in order
    ]


def command(case, output, reference, commit, python):
    pilot = case["pilot"]
    return [
        str(python),
        str(ROOT / "eval/benchmark_native_policies.py"),
        "--output",
        str(output),
        "--source-commit",
        commit,
        "--weight-mode",
        case["weight_mode"],
        "--reference-dir",
        str(reference),
        "--layout",
        "overlap",
        "--replicas",
        "1" if pilot else "4",
        "--budget",
        "0.3" if pilot else "0.4",
        "--batches",
        "4" if pilot else "64",
        "--cap",
        "16" if pilot else "128",
        "--repeats",
        "3",
        "--natural-cap",
        "64" if pilot else "4096",
        "--natural-repeats",
        "3",
        "--natural-warmup",
        "--max-num-seqs",
        "4" if pilot else "64",
        "--max-num-batched-tokens",
        "8192",
        "--chunked-prefill",
        "--graph",
        "--policies",
        "initial_student",
        "teacher",
        "switch_p0_q",
        "switch_q_p0",
        "batch_mix",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", type=Path, default=ROOT.parent / "envs/opd-state/bin/python")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": commit,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Native inference only; common 0.4 budget, model references, input prompts and sampling",
        "reference_directory": str(directory / "resident-reference"),
        "cases": [],
    }

    def save():
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "state.json")

    save()
    successful = []
    for case in cases():
        output = directory / case["name"]
        reference = directory / ("pilot-reference" if case["pilot"] else "resident-reference")
        cmd = command(case, output, reference, commit, args.python)
        item = {**case, "state": "planned", "command": cmd}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("GPU 0–3 are occupied; wait for the existing experiment")
        with (directory / f"{case['name']}-driver.log").open("w") as stream:
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
        summary = json.loads((output / "summary.json").read_text())
        if summary["observations"] != 40 or summary["weight_mode"] != case["weight_mode"]:
            state["state"] = "incomplete_workload"
            save()
            raise RuntimeError("Residency benchmark lacks its complete five-policy fixed/natural workload")
        if not case["pilot"]:
            successful.append(str(output))
        destination = ROOT / "docs/experiment/results/native-rollout/weight-residency" / directory.name / case["name"]
        code = subprocess.call(
            [
                sys.executable,
                str(ROOT / "eval/summarize_native_benchmarks.py"),
                "--runs",
                str(output),
                "--output",
                str(destination),
            ],
            cwd=ROOT,
        )
        item["audit_exit_code"] = code
        save()
        if code:
            state["state"] = "audit_failed"
            save()
            return code
    if not args.plan_only:
        destination = ROOT / "docs/experiment/results/native-rollout/weight-residency" / directory.name / "combined"
        code = subprocess.call(
            [
                sys.executable,
                str(ROOT / "eval/summarize_native_benchmarks.py"),
                "--runs",
                *successful,
                "--output",
                str(destination),
            ],
            cwd=ROOT,
        )
        state["combined_audit_exit_code"] = code
        if code:
            state["state"] = "audit_failed"
            save()
            return code
    state["state"] = "planned" if args.plan_only else "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
