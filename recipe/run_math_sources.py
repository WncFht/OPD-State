"""Serialize the paired math experiments on GPUs 0--3 with auditable subprocesses.

The teacher run may already be active. Wait for its exact process, require its
success marker and independently audit its results before acquiring the GPUs.
An unsuccessful fusion probe does not discard the independent source baselines.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / "envs/opd-state/bin/python"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class Suite:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.revision + "^{commit}"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.env = {
            **os.environ,
            "OPD_SOURCE_COMMIT": self.revision,
            "OPD_FREEZE_SOURCE": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        self.state = {
            "source_commit": self.revision,
            "stages": [],
            "phase": "initializing",
        }
        self.snapshot = self.directory / "tooling"
        self.snapshot.mkdir(exist_ok=True)
        archive = self.directory / "tooling.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", self.revision, "verl", "data/prep", "eval"],
                cwd=ROOT,
                stdout=stream,
                check=True,
            )
        subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(self.snapshot)], check=True
        )
        archive.unlink()

    def save(self, phase):
        self.state.update(
            phase=phase, updated_at=datetime.now().astimezone().isoformat()
        )
        temporary = self.directory / "status.json.new"
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.directory / "status.json")
        print(phase, flush=True)

    def command(self, label, argv, *, env=None, optional=False):
        stage = {
            "name": label,
            "argv": list(map(str, argv)),
            "log": str(self.directory / f"{label}.log"),
        }
        self.state["stages"].append(stage)
        self.save(label)
        with Path(stage["log"]).open("w") as stream:
            result = subprocess.run(
                stage["argv"],
                cwd=ROOT,
                env={**self.env, **(env or {})},
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        stage["exit_code"] = result.returncode
        self.save(label + ("_passed" if result.returncode == 0 else "_failed"))
        if result.returncode and not optional:
            raise RuntimeError(f"{label} failed; inspect {stage['log']}")
        return result.returncode == 0

    def recipe(self, name):
        for filename in dict.fromkeys(
            (name, "math_baseline.sh", "opd.sh", "_common.sh")
        ):
            path = ROOT / "recipe" / filename
            committed = subprocess.check_output(
                ["git", "show", f"{self.revision}:recipe/{filename}"], cwd=ROOT
            )
            if path.read_bytes() != committed:
                raise RuntimeError(f"Recipe changed after suite was pinned: {path}")
        return str(ROOT / "recipe" / name)

    def collect(self, run, steps):
        output = (
            ROOT
            / "docs/experiment/results"
            / f"{datetime.now(TIMEZONE):%Y-%m-%d}-{run.name}"
        )
        self.command(
            "collect-" + run.name,
            [
                PYTHON,
                self.snapshot / "eval/collect_math_experiment.py",
                "--run",
                run,
                "--output",
                output,
                "--steps",
                str(steps),
                "--interval",
                "50",
            ],
        )
        self.state.setdefault("collected_results", {})[run.name] = str(output)
        return output

    def train(self, label, script, *, steps=50, extra=(), env=None, plateau=False):
        name = (
            f"math-{label}-{steps}steps-eval16-{datetime.now(TIMEZONE):%Y%m%d-%H%M%S}"
        )
        run = ROOT / "runs/rollout-sources" / name
        if plateau:
            extra = (
                *extra,
                "+trainer.plateau_min_steps=300",
                "+trainer.plateau_report_path=" + str(run / "plateau.json"),
            )
        self.command(
            label,
            ["bash", self.recipe(script), *extra],
            env={
                "STEPS": str(steps),
                "TOTAL_EPOCHS": str((steps + 49) // 50),
                "EXPERIMENT_NAME": name,
                "OPD_ROOT": str(run),
                **(env or {}),
            },
        )
        self.state[label + "_maximum_steps"] = steps
        if plateau:
            report = json.loads((run / "plateau.json").read_text())
            if not report["stopped"]:
                raise RuntimeError(
                    "Long run exited without a final plateau/budget report"
                )
            steps = report["training_steps"]
            self.state[label + "_plateau_report"] = report
        self.collect(run, steps)
        self.state[label + "_run"] = str(run)
        return run

    def analyze(self, comparison, label):
        arguments = []
        for name, run in comparison.items():
            arguments.extend(
                ["--result", name + "=" + self.state["collected_results"][run.name]]
            )
        self.command(
            label,
            [
                PYTHON,
                self.snapshot / "eval/analyze_math_sources.py",
                *arguments,
                "--reference",
                "student-online",
                "--output",
                self.directory / label,
            ],
        )

    def run(self):
        pid = self.args.wait_pid
        start = process_start(pid)
        self.save("waiting_for_teacher")
        while start is not None and process_start(pid) == start:
            time.sleep(15)
        if (
            "TEACHER_TRAIN_FINISHED"
            not in Path(self.args.teacher_status_log).read_text().splitlines()
        ):
            raise RuntimeError("Teacher pipeline exited without its success marker")
        teacher = Path(self.args.teacher_run).resolve()
        self.collect(teacher, 50)
        # Do not overlap the isolated performance measurement or cache generation
        # with the preceding Ray cluster's delayed GPU teardown.
        self.save("waiting_for_gpu_release")
        for _ in range(24):
            used = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=0,1,2,3",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=0,1,2,3",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if (
                all(int(value) < 2048 for value in used.splitlines())
                and not processes.strip()
            ):
                break
            time.sleep(5)
        else:
            raise RuntimeError("GPUs 0--3 were not released after the teacher pipeline")
        benchmark = self.command(
            "fusion-gpu-benchmark",
            [
                PYTHON,
                self.snapshot / "data/prep/benchmark_logit_fusion.py",
                "--student",
                ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B",
                "--teacher",
                ROOT.parent / "models/JustRL-DeepSeek-1.5B",
                "--prompts",
                ROOT / "runs/rollout-sources/paired-schedule/prompts.jsonl",
                "--output",
                self.directory / "fusion-gpu-benchmark.json",
                "--graph-attention",
                "flash-kv",
                "--backends",
                "dynamic,cuda_graph_flash",
            ],
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": str(self.snapshot / "verl"),
            },
            optional=True,
        )
        fusion_ready = False
        if benchmark:
            if not json.loads(
                (self.directory / "fusion-gpu-benchmark.json").read_text()
            )["timing_isolated"]:
                raise RuntimeError(
                    "Fusion benchmark unexpectedly shared GPU 0 with another process"
                )
            smoke_name = "fusion-two-step-" + self.directory.name
            smoke = ROOT / "runs/rollout-sources" / smoke_name
            passed = self.command(
                "fusion-two-step",
                [
                    "bash",
                    self.recipe("math_fusion.sh"),
                    "data.train_max_samples=8",
                    "trainer.math_eval_require_full=False",
                ],
                env={
                    "STEPS": "2",
                    "TOTAL_EPOCHS": "2",
                    "MINI_BATCH_SIZE": "8",
                    "MAX_RESP_LENGTH": "64",
                    "MAX_VAL_RESP_LENGTH": "128",
                    "VAL_FILES": f"[{ROOT}/runs/rollout-sources/validation-smoke.parquet]",
                    "TRAINER_LOGGER": "['console']",
                    "EXPERIMENT_NAME": smoke_name,
                    "OPD_ROOT": str(smoke),
                    "FUSION_BACKEND": "cuda_graph_flash",
                },
                optional=True,
            )
            if passed:
                # Check actual versioned rollouts and the student-only evaluation
                # through the same independent collector used for real results.
                passed = self.command(
                    "collect-fusion-smoke",
                    [
                        PYTHON,
                        self.snapshot / "eval/collect_math_experiment.py",
                        "--run",
                        smoke,
                        "--output",
                        self.directory / "fusion-smoke-results",
                        "--steps",
                        "2",
                        "--interval",
                        "2",
                        "--smoke",
                    ],
                    optional=True,
                )
                if passed:
                    import re

                    log = next((smoke / "logs").glob("*.log")).read_text()
                    versions = [
                        float(v)
                        for v in re.findall(r"fusion/student_version:([0-9.]+)", log)
                    ]
                    fusion_ready = versions == [0.0, 1.0]
                    self.state["fusion_smoke_versions"] = versions
        self.state["fusion_ready"] = fusion_ready
        cache = ROOT / "runs/rollout-sources/student-cache"
        if not (cache / "manifest.json").exists():
            self.command(
                "prepare-student-cache",
                [
                    "bash",
                    self.recipe("prepare_math_cache.sh"),
                    "cached_initial_student",
                    cache,
                ],
            )
        cache_meta = json.loads((cache / "manifest.json").read_text())
        if (
            cache_meta["source"] != "cached_initial_student"
            or cache_meta["records"] != 3200
        ):
            raise ValueError("Initial-student cache does not match the paired schedule")
        semi = self.train(
            "semi-opd", "math_cached.sh", extra=("cached_initial_student", str(cache))
        )
        online = self.train("student-online", "math_online.sh")
        comparison = {"teacher": teacher, "semi-opd": semi, "student-online": online}
        self.analyze(comparison, "analyze-three-sources")
        if fusion_ready:
            fusion = self.train(
                "fusion-lambda05",
                "math_fusion.sh",
                env={"FUSION_BACKEND": "cuda_graph_flash", "FUSION_LAMBDA": "0.5"},
            )
            comparison["fusion-lambda05"] = fusion
            self.analyze(comparison, "analyze-fusion")
        if self.args.long_steps:
            long_run = self.train(
                "semi-opd-long",
                "math_cached.sh",
                steps=self.args.long_steps,
                extra=("cached_initial_student", str(cache)),
                plateau=True,
            )
            comparison["semi-opd-long"] = long_run
            self.analyze(comparison, "analyze-long-run")
        self.save(
            "runs_finished" if fusion_ready else "sources_finished_fusion_needs_repair"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--teacher-status-log", required=True)
    parser.add_argument(
        "--long-steps",
        type=int,
        default=800,
        help="Maximum long-run budget; plateau stopping is enabled from step 300",
    )
    args = parser.parse_args()
    if args.long_steps and (args.long_steps < 300 or args.long_steps % 50):
        raise ValueError(
            "Long-run budget must be >=300 and a multiple of 50, or zero to defer it"
        )
    suite = Suite(args)
    try:
        suite.run()
    except BaseException as error:
        suite.state["error"] = f"{type(error).__name__}: {error}"
        suite.save("failed")
        raise


if __name__ == "__main__":
    main()
