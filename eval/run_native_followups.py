"""Serial GPU acceptance and ABBA layout observations after the frozen math suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def cases():
    result = []
    for policy in [
        "batch_current",
        "batch_initial",
        "teacher_current",
        "current_teacher",
        "student_teacher_fusion",
        "initial_teacher_fusion",
    ]:
        result.append(
            {
                "name": f"accept-{policy}",
                "policy": policy,
                "layout": "split",
                "steps": 2,
                "batch": 8,
                "response_cap": 64,
                "val_cap": 128,
                "verify_frozen": True,
                "parameters": {"TEACHER_FRACTION": "0.125", "SWITCH_TOKENS": "32", "TEACHER_WEIGHT": "0.5"},
            }
        )
    for index, layout in enumerate(["overlap", "split", "split", "overlap"]):
        result.append(
            {
                "name": f"observe-{index}-{layout}",
                "policy": "batch_current",
                "layout": layout,
                "steps": 5,
                "batch": 64,
                "response_cap": 4096,
                "val_cap": 4096,
                "verify_frozen": False,
                "parameters": {"TEACHER_FRACTION": "0.5"},
            }
        )
    return result


def command(case, output, commit):
    result = [
        sys.executable,
        str(ROOT / "eval/run_native_acceptance.py"),
        "--source-commit",
        commit,
        "--output",
        str(output),
        "--policy",
        case["policy"],
        "--steps",
        str(case["steps"]),
        "--batch",
        str(case["batch"]),
        "--prompt-cap",
        "1024",
        "--response-cap",
        str(case["response_cap"]),
        "--val-response-cap",
        str(case["val_cap"]),
        "--memory-budget",
        "0.55",
        "--wandb",
    ]
    for benchmark in ["MATH-500", "AMC23", "AIME25"]:
        result += ["--validation", str(ROOT.parent / "MD-OPD/data/math" / benchmark / "test.parquet")]
    return result + [
        "actor_rollout_ref.rollout.enforce_eager=False",
        "++actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=[all]",
        f"actor_rollout_ref.rollout.generation_layout={case['layout']}",
        f"actor_rollout_ref.rollout.generation_verify_frozen={case['verify_frozen']}",
        "actor_rollout_ref.rollout.generation_state_metrics=True",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    if not args.plan_only:
        source = subprocess.check_output(
            ["git", "show", commit + ":verl/verl/workers/rollout/native_pool.py"], cwd=ROOT, text=True
        )
        if "execution_audit" not in source:
            raise ValueError("Selected source predates physical worker receipts; choose the reviewed follow-up commit")
    selected = [c for c in cases() if args.only is None or c["name"] in args.only]
    if not selected or (args.only and set(args.only) != {c["name"] for c in selected}):
        parser.error("Unknown follow-up case")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": commit,
        "scope": "Short correctness checks and layout cost observations; no formal math accuracy claims",
        "data_manifest_sha256": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        "layout_order": "ABBA",
        "warmup_steps_excluded_from_observations": [1],
        "cases": [],
    }

    def save():
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "state.json")

    save()
    for case in selected:
        output = directory / case["name"]
        cmd = command(case, output, commit)
        item = {**case, "command": cmd, "state": "planned"}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("GPU 0–3 are occupied; do not start another training process")
        uuids = subprocess.check_output(
            ["nvidia-smi", "--id=0,1,2,3", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        env = {**os.environ, **case["parameters"], "NATIVE_DATA_MANIFEST": str(args.data_manifest.resolve())}
        item.update(state="running", started=time.time(), allowed_gpu_uuids=uuids)
        with (directory / (case["name"] + "-driver.log")).open("w") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item["pid"] = process.pid
            save()
            code = process.wait()
        item.update(exit_code=code, finished=time.time(), state="finished" if code == 0 else "failed")
        save()
        if code:
            state["state"] = "failed"
            save()
            return code
        result = ROOT / "docs/experiment/results/native-rollout/followups" / directory.name / case["name"]
        audits = [
            [
                sys.executable,
                str(ROOT / "eval/collect_native_acceptance.py"),
                "--run",
                str(output),
                "--output",
                str(result),
                "--steps",
                str(case["steps"]),
            ],
            [
                sys.executable,
                str(ROOT / "eval/verify_native_layout.py"),
                "--run",
                str(output),
                "--output",
                str(result),
                "--steps",
                str(case["steps"]),
                "--gpu-uuids",
                *uuids,
            ],
        ]
        for audit in audits:
            code = subprocess.call(audit, cwd=ROOT)
            item.setdefault("audit_exit_codes", []).append(code)
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
