"""Sequential native math experiments with immutable source and audited final eval16."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases():
    result = [(name, name, {}) for name in ["student_fresh", "initial_student_fresh", "teacher_fresh"]]
    result += [(name, name, {"TEACHER_WEIGHT": "0.5"}) for name in ["student_teacher_fusion", "initial_teacher_fusion"]]
    for k in [2048, 1024]:
        result += [
            (f"{name}_k{k}", name, {"SWITCH_TOKENS": str(k)})
            for name in ["teacher_current", "current_teacher", "teacher_initial", "initial_teacher"]
        ]
    result += [(name, name, {"TEACHER_FRACTION": "0.5"}) for name in ["batch_current", "batch_initial"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="Run selected named cases; default runs the complete flat matrix")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    manifest = args.data_manifest.resolve()
    selection = [c for c in cases() if args.only is None or c[0] in args.only]
    if args.only is not None and set(args.only) != {c[0] for c in selection}:
        parser.error("Unknown case in --only")
    protocol = {
        "source_commit": commit,
        "data_manifest": str(manifest),
        "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "steps": 50,
        "global_batch": 64,
        "prompt_cap": 1024,
        "response_cap": 4096,
        "eval_cap": 31744,
        "eval_questions": 570,
        "eval_samples": 16,
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "lr": 1e-6,
        "generation_memory_budget": 0.55,
        "cuda_graph": True,
        "final_eval_only": True,
        "save_checkpoint": False,
        "wandb_project": "OPD-State",
        "cases": [{"name": n, "policy": p, "parameters": e} for n, p, e in selection],
        "endpoints": "fusion lambda0/1, batch alpha0/1 and switch k0/4096 reuse the exactly reduced single baselines",
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    state = {"state": "running", "source_commit": commit, "cases": []}

    def save():
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "state.json")

    save()
    for name, policy, parameters in selection:
        output = directory / name
        command = [
            sys.executable,
            str(ROOT / "eval/run_native_acceptance.py"),
            "--source-commit",
            commit,
            "--output",
            str(output),
            "--policy",
            policy,
            "--steps",
            "50",
            "--batch",
            "64",
            "--prompt-cap",
            "1024",
            "--response-cap",
            "4096",
            "--val-response-cap",
            "31744",
            "--memory-budget",
            "0.55",
            "--full-eval",
            "--wandb",
        ]
        for benchmark in ["MATH-500", "AMC23", "AIME25"]:
            command += ["--validation", str(ROOT.parent / "MD-OPD/data/math" / benchmark / "test.parquet")]
        command += [
            "actor_rollout_ref.rollout.enforce_eager=False",
            "++actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=[all]",
        ]
        env = {**os.environ, **parameters, "NATIVE_DATA_MANIFEST": str(manifest)}
        item = {"name": name, "state": "running", "started": time.time(), "command": command, "parameters": parameters}
        state["cases"].append(item)
        with (directory / (name + "-driver.log")).open("w") as log:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            item["pid"] = child.pid
            save()
            code = child.wait()
        item.update(state="finished" if code == 0 else "failed", exit_code=code, finished=time.time())
        save()
        if code:
            state["state"] = "failed"
            save()
            return code
        audit = subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval/collect_native_acceptance.py"),
                "--run",
                str(output),
                "--output",
                str(ROOT / "docs/experiment/results/native-rollout/formal" / name),
                "--steps",
                "50",
                "--full",
            ],
            cwd=ROOT,
        )
        item["audit_exit_code"] = audit.returncode
        save()
        if audit.returncode:
            state["state"] = "audit_failed"
            save()
            return audit.returncode
    state["state"] = "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
