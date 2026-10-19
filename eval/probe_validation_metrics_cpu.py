"""Run validation CPU checks and full-vocabulary numerical reference probes."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "eval/probe_validation_metrics_cpu.py",
    "src/opd_state/kernels/divergence.py",
    "src/opd_state/core/records.py",
    "src/opd_state/evaluation/prefixes.py",
    "src/opd_state/evaluation/kl_metrics.py",
    "src/opd_state/evaluation/rollout.py",
    "src/opd_state/integrations/verl/validation_data.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/ppo/rollout_coordinator.py",
    "verl/tests/opd/test_native_validation_on_cpu.py",
    "verl/tests/opd/test_validation_metrics_runtime.py",
    "verl/tests/opd/test_rollout_coordinator_runtime.py",
    "src/opd_state/integrations/verl/validation.py",
    "src/opd_state/kernels/causal_scoring.py",
    "verl/verl/workers/fsdp_teacher_worker.py",
    "verl/verl/single_controller/ray/base.py",
    "verl/verl/trainer/config/ppo_trainer.yaml",
    "verl/tests/opd/test_validation_scoring_runtime.py",
    "verl/tests/opd/test_validation_runner_runtime.py",
    "eval/probe_validation_fsdp_cpu.py",
    "scripts/tests/test_math_snapshots.py",
    "eval/collect_math_experiment.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "verl")]
    import torch

    from opd_state.kernels.divergence import full_vocab_divergences

    torch.set_num_threads(2)
    args.logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "verl")])}
    commands = [
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            "verl/tests/opd",
            "-p",
            "test_native_*_on_cpu.py",
            "-v",
        ],
        [sys.executable, "-m", "unittest", "discover", "-s", "verl/tests/opd", "-p", "test_*_runtime.py", "-v"],
    ]
    checks = []
    for index, command in enumerate(commands):
        log = args.logs / f"checks-{index}.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        count = re.search(r"Ran (\d+) tests?", log.read_text())
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "tests": int(count[1]) if count else None,
                "seconds": time.perf_counter() - started,
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        )
        if result.returncode or count is None:
            raise RuntimeError(f"Validation CPU checks failed: {log}")

    generator = torch.Generator().manual_seed(42)
    student = torch.randn((4, 151936), generator=generator).bfloat16()
    teacher = torch.randn((4, 151936), generator=generator).bfloat16()
    student[:, 151665:] = teacher[:, 151665:] = float("nan")
    logp = student[:, :151665].double().log_softmax(-1)
    logq = teacher[:, :151665].double().log_softmax(-1)
    reference = {
        "student_teacher_kl": (logp.exp() * (logp - logq)).sum(-1),
        "teacher_student_kl": (logq.exp() * (logq - logp)).sum(-1),
        "student_entropy": -(logp.exp() * logp).sum(-1),
        "teacher_entropy": -(logq.exp() * logq).sum(-1),
    }
    stress = []
    for chunk in (1024, 8192, 151665):
        actual = full_vocab_divergences(student, teacher, vocab_size=151665, chunk_size=chunk)
        errors = {key: float((actual[key].double() - value).abs().max()) for key, value in reference.items()}
        if max(errors.values()) > 1e-5:
            raise RuntimeError(f"Full-vocabulary FP32 error exceeds the 1e-5 absolute bound: {errors}")
        stress.append({"chunk_size": chunk, "max_absolute_errors": errors})
    receipt = {
        "schema": "validation-metrics-cpu-v2",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_kind": "worktree",
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checks": checks,
        "full_vocabulary_numeric_stress": {
            "states": 4,
            "output_width": 151936,
            "effective_vocab": 151665,
            "dtype": "BF16 inputs / FP32 reduction / FP64 reference",
            "nan_head_padding_ignored": True,
            "results": stress,
        },
        "gpu_generation_or_scoring_verified": False,
        "kl_valid_hook_connected": True,
        "validation_rollout_hook_connected": True,
        "frozen_formal_matrix_changed": False,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(args.output), "checks": checks}))


if __name__ == "__main__":
    main()
