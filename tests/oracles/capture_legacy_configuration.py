"""Capture the real pre-migration shell outputs without importing its trainer.

Run in the original source checkout with the training Python. These fixtures
are an independent migration reference, not defaults for the new frontend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value, substitutions):
    if isinstance(value, dict):
        return {key: normalize(item, substitutions) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item, substitutions) for item in value]
    if isinstance(value, str):
        for source, target in substitutions:
            value = value.replace(source, target)
    return value


def capture(output):
    source_names = [
        "recipe/math_native.sh",
        "recipe/math_baseline.sh",
        "recipe/opd.sh",
        "recipe/_common.sh",
    ]
    config_root = ROOT / "verl/verl/trainer/config"
    sources = {name: digest(ROOT / name) for name in source_names}
    sources.update(
        {str(p.relative_to(ROOT)): digest(p) for p in config_root.rglob("*.yaml")}
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    cases = [
        (name, name, {}, [])
        for name in ("student_fresh", "initial_student_fresh", "teacher_fresh")
    ]
    cases += [
        (name, name, {"TEACHER_WEIGHT": "0.5"}, [])
        for name in ("student_teacher_fusion", "initial_teacher_fusion")
    ]
    for boundary in (1024, 2048):
        cases += [
            (f"{name}_{boundary}", name, {"SWITCH_TOKENS": str(boundary)}, [])
            for name in (
                "teacher_current",
                "current_teacher",
                "teacher_initial",
                "initial_teacher",
            )
        ]
    cases += [
        (name, name, {"TEACHER_FRACTION": "0.5"}, [])
        for name in ("batch_current", "batch_initial")
    ]
    cases.append(
        (
            "validation_fast",
            "student_fresh",
            {},
            [
                "+experiment=validation_fast",
                "data.val_batch_size=64",
                "ray_kwargs.ray_init.object_store_memory=8589934592",
            ],
        )
    )
    cases.append(
        (
            "formal_graph",
            "student_fresh",
            {},
            [
                "actor_rollout_ref.rollout.enforce_eager=False",
                "++actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]",
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=[all]",
            ],
        )
    )
    captured = {}
    with tempfile.TemporaryDirectory(prefix="opd-config-baseline-") as scratch:
        directory = Path(scratch)
        binaries = directory / "bin"
        binaries.mkdir()
        recorder = binaries / "python3"
        recorder.write_text(
            "#!" + sys.executable + "\nimport sys,os,json\nfrom pathlib import Path\n"
            "if sys.argv[1:2]==['-c']:\n os.execv("
            + repr(sys.executable)
            + ", ["
            + repr(sys.executable)
            + ",*sys.argv[1:]])\n"
            "Path(os.environ['OPD_CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        recorder.chmod(0o755)
        prepared = directory / "prepared"
        prepared.mkdir()
        (prepared / "manifest.json").write_text("{}\n")
        for name, policy, parameters, overrides in cases:
            args_file = directory / "argv.json"
            env = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "LANG")
                if key in os.environ
            }
            env.update(
                {
                    "OPD_TRAIN_PYTHON": str(recorder),
                    "OPD_DRY_RUN": "1",
                    "OPD_ROOT": str(directory / "run"),
                    "EXPERIMENT_NAME": "reference",
                    "MODEL_BASE": str(directory / "models"),
                    "DATA_BASE": str(directory / "data"),
                    "NATIVE_DATA_MANIFEST": str(prepared / "manifest.json"),
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "N_GPUS_PER_NODE": "4",
                    "OPD_CAPTURE_ARGS": str(args_file),
                    **parameters,
                }
            )
            subprocess.run(
                ["bash", "recipe/math_native.sh", policy, *overrides],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            argv = json.loads(args_file.read_text())
            native_overrides = [
                arg for arg in argv if "=" in arg and not arg.startswith("--")
            ]
            with initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
                config = compose(config_name="ppo_trainer", overrides=native_overrides)
            OmegaConf.resolve(config)
            normalized = normalize(
                OmegaConf.to_container(config, resolve=True),
                [
                    (str(directory), "/reference"),
                    (str(ROOT), "/reference/repository"),
                    (commit, "BASELINE_COMMIT"),
                ],
            )
            captured[name] = normalized
    # Store exact deltas from one full reference, reducing repeated backend data.
    base = captured.pop("student_fresh")

    def differences(old, new, prefix=""):
        result = {}
        if isinstance(old, dict) and isinstance(new, dict) and set(old) == set(new):
            for key in old:
                result.update(
                    differences(
                        old[key], new[key], prefix + "." + key if prefix else key
                    )
                )
        elif old != new:
            result[prefix] = new
        return result

    artifact = {
        "schema_version": 1,
        "source_commit": commit,
        "source_sha256": sources,
        "capture_driver_sha256": digest(Path(__file__)),
        "scope": "Real shell argv plus actual native Hydra composition; no Ray, model, or CUDA execution",
        "base_name": "student_fresh",
        "base": base,
        "cases": {
            "student_fresh": {},
            **{name: differences(base, cfg) for name, cfg in captured.items()},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(f"Captured {len(artifact['cases'])} complete configurations at {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output)
