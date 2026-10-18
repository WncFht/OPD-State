"""Check independent native evaluation concurrency after existing GPU owners."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from native_devices import query_devices
from run_native_acceptance import gpu_rows
from verify_native_engine_limits import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return [
        {"name": "current-inherit", "policy": "student_fresh", "evaluation_sequences": None},
        {"name": "current-eval64", "policy": "student_fresh", "evaluation_sequences": 64},
        {"name": "initial-fusion-eval64", "policy": "initial_teacher_fusion", "evaluation_sequences": 64},
    ]


def command(case, output, commit, validation):
    args = [
        sys.executable,
        str(ROOT / "eval/run_native_acceptance.py"),
        "--output",
        str(output),
        "--source-commit",
        commit,
        "--policy",
        case["policy"],
        "--steps",
        "2",
        "--batch",
        "8",
        "--response-cap",
        "64",
        "--val-response-cap",
        "128",
        "--memory-budget",
        "0.55",
        "--wandb",
    ]
    for file in validation:
        args.extend(["--validation", str(file)])
    args += [
        "actor_rollout_ref.rollout.enforce_eager=False",
        "++actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=[all]",
    ]
    if case["evaluation_sequences"] is not None:
        args.append(f"actor_rollout_ref.rollout.generation_eval_max_num_seqs={case['evaluation_sequences']}")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runtime = ROOT.parent / "envs/opd-state/bin/python"
    subset = directory / "validation-subset"
    validations = [subset / f"{name}.parquet" for name in ("MATH-500", "AMC23", "AIME25")]
    default_manifest = ROOT / "runs/native-data/math-full/manifest.json"
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": commit,
        "driver_sha256": digest(Path(__file__)),
        "scope": "Three four-GPU engineering controls, not a validation-throughput or training-quality comparison",
        "cases": [],
    }

    def save():
        tmp = directory / "state.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(directory / "state.json")

    save()
    devices = {}
    if not args.plan_only:
        if not default_manifest.is_file():
            raise ValueError("Prepare the default complete native prompt pool before these controls")
        state["default_prepared_manifest_sha256"] = digest(default_manifest)
        for name in (
            Path(__file__).name,
            "verify_native_engine_limits.py",
            "native_devices.py",
            "run_native_acceptance.py",
        ):
            expected = subprocess.check_output(["git", "show", f"{commit}:eval/{name}"], cwd=ROOT)
            if expected != Path(__file__).with_name(name).read_bytes():
                raise ValueError("Queued evaluation-limit driver/helper differs from its committed version: " + name)
        preparation = [str(runtime), "data/prep/build_validation_subset.py", "--output", str(subset)]
        for name in ("MATH-500", "AMC23", "AIME25"):
            preparation.extend(["--input", str(ROOT.parent / "MD-OPD/data/math" / name / "test.parquet")])
        subprocess.run(
            preparation,
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "verl")},
            check=True,
        )
        devices = query_devices()
        (directory / "devices.json").write_text(json.dumps(devices, indent=2) + "\n")
    for case in cases():
        output = directory / case["name"]
        cmd = command(case, output, commit, validations)
        item = {**case, "command": cmd, "state": "planned"}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        while any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "waiting_for_idle_gpu_0_3"
            save()
            time.sleep(10)
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        # Exercise the default prepared-data entry, including its pre-Ray binding.
        env.pop("TRAIN_FILES", None)
        env.pop("NATIVE_DATA_MANIFEST", None)
        with (directory / (case["name"] + ".log")).open("w") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item.update(state="running", pid=process.pid, started_at=time.time())
            save()
            code = process.wait()
        item.update(exit_code=code, finished_at=time.time())
        if code:
            item["state"], state["state"] = "failed", "failed"
            save()
            return code
        result = ROOT / "docs/experiment/results/native-rollout/eval-concurrency" / case["name"]
        subprocess.run(
            [
                sys.executable,
                "eval/collect_native_acceptance.py",
                "--run",
                str(output),
                "--output",
                str(result),
                "--steps",
                "2",
            ],
            cwd=ROOT,
            check=True,
        )
        report = verify(output, devices, training_sequences=8, evaluation_sequences=case["evaluation_sequences"] or 8)
        if report["prepared_data_manifest_sha256"] != state["default_prepared_manifest_sha256"]:
            raise ValueError("Engineering control did not use the default complete prepared prompt pool")
        (result / "engine-limits.json").write_text(json.dumps(report, indent=2) + "\n")
        subprocess.run(
            [str(ROOT / "node_modules/.bin/prettier"), "--write", *map(str, result.glob("*.json"))],
            cwd=ROOT,
            check=True,
        )
        auth = {
            **env,
            "CUDA_VISIBLE_DEVICES": "",
            "NETRC": str(ROOT.parent / ".secrets/netrc"),
            "WANDB_CONFIG_DIR": str(ROOT.parent / "config/wandb"),
            "WANDB_BASE_URL": "https://api.wandb.ai",
        }
        auth.pop("WANDB_API_KEY", None)
        subprocess.run(
            [str(runtime), "eval/verify_wandb_math.py", "--result", str(result), "--engineering"],
            cwd=ROOT,
            env=auth,
            check=True,
        )
        item["state"] = "passed"
        save()
    state["state"] = "planned" if args.plan_only else "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
