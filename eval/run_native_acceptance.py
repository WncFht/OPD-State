"""Run a source-frozen native trainer check and record whole-card telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path


def gpu_rows():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0,1,2,3",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [[int(x.strip()) for x in row.split(",")] for row in output.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="student_fresh")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt-cap", type=int, default=512)
    parser.add_argument("--response-cap", type=int, default=64)
    parser.add_argument("--val-response-cap", type=int, default=128)
    parser.add_argument("--memory-budget", type=float, default=0.15)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--source-commit", help="Freeze a committed verl tree instead of the current worktree")
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = run / "source"
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", (args.source_commit or "HEAD") + "^{commit}"], cwd=root, text=True
    ).strip()
    if args.source_commit:
        source.mkdir()
        archive = subprocess.Popen(["git", "archive", commit, "verl"], cwd=root, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(source)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait() != 0:
            raise RuntimeError("Could not freeze the selected source commit")
    else:
        shutil.copytree(root / "verl", source / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "base_commit": commit,
        "source_kind": "git_archive" if args.source_commit else "worktree_copy",
        "files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "verl").rglob("*"))
            if p.is_file()
        },
    }
    (run / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Ray creates Unix sockets below this directory; the shared workspace path
    # exceeds AF_UNIX's 107-byte limit. Only Ray's small runtime files go here.
    ray_dir = Path("/tmp") / ("opdns-" + uuid.uuid4().hex[:12])
    settings = {
        "STEPS": str(args.steps),
        "MINI_BATCH_SIZE": str(args.batch),
        "MAX_PROMPT_LENGTH": str(args.prompt_cap),
        "MAX_RESP_LENGTH": str(args.response_cap),
        "MAX_VAL_RESP_LENGTH": str(args.val_response_cap),
        "GPU_MEMORY_UTILIZATION": str(args.memory_budget),
        "TRAINER_LOGGER": "['console','wandb']" if args.wandb else "['console']",
        "VAL_FILES": f"[{args.validation.resolve()}]",
        "OPD_ROOT": str(run),
        "EXPERIMENT_NAME": run.name,
        "RAY_TMPDIR": str(ray_dir),
        "VERL_ROOT": str(source / "verl"),
        "WANDB_RUN_GROUP": "math-native-rollout-sources",
        "OPD_SOURCE_COMMIT": commit,
    }
    for name in ["SWITCH_TOKENS", "TEACHER_FRACTION"]:
        if name in os.environ:
            settings[name] = os.environ[name]
    command = [
        "bash",
        str(root / "recipe/math_native.sh"),
        args.policy,
        "trainer.math_eval_require_full=False",
        "data.val_batch_size=6",
        f"+trainer.run_source_manifest={run / 'source-manifest.json'}",
        *overrides,
    ]
    provenance = {"command": command, "environment": settings, "gpu_before": gpu_rows()}
    (run / "command.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.time()
    with (run / "launcher.log").open("w") as log, (run / "gpu.csv").open("w", newline="") as telemetry:
        writer = csv.writer(telemetry, lineterminator="\n")
        writer.writerow(["elapsed_s", "gpu", "used_mib", "utilization_pct"])
        child = subprocess.Popen(
            command, env={**os.environ, **settings}, cwd=root, stdout=log, stderr=subprocess.STDOUT
        )
        (run / "status.json").write_text(json.dumps({"pid": child.pid, "state": "running", "started": started}) + "\n")
        while True:
            for row in gpu_rows():
                writer.writerow([round(time.time() - started, 3), *row])
            telemetry.flush()
            try:
                code = child.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
    status = {
        "state": "finished" if code == 0 else "failed",
        "exit_code": code,
        "elapsed_s": time.time() - started,
        "gpu_after": gpu_rows(),
        "source_manifest": str(run / "source-manifest.json"),
    }
    (run / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
