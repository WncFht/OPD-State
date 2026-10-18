"""Summarize logged stage costs or sample whole-device GPU usage during a run.

Actor allocator peaks exclude other processes. GPU samples are instantaneous
device readings, starting when this monitor launches, not historical peaks.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plots", Path(__file__).with_name("plot_math_experiment.py"))
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    phase_rows, memory_rows, cases, inputs = [], [], {}, {}
    for name, directory in PLOTS.RUNS.items():
        directory = args.results / directory
        path = directory / "training-1-50.csv"
        rows = list(csv.DictReader(path.open()))
        assert [int(row["step"]) for row in rows] == list(range(1, 51))
        inputs[str(path)] = digest(path)
        sums = {
            key: sum(float(row.get(f"timing_s/{key}") or 0) for row in rows)
            for key in (
                "gen",
                "old_log_prob",
                "opd_teacher",
                "adv",
                "update_actor",
                "update_weights",
                "step",
                "testing",
            )
        }
        # Teacher scoring is nested inside `adv`; count it exactly once.
        phases = {
            "generate_or_read_cache": sums["gen"],
            "student_log_prob": sums["old_log_prob"],
            "teacher_score": sums["opd_teacher"],
            "advantage_excluding_teacher": sums["adv"] - sums["opd_teacher"],
            "actor_update": sums["update_actor"],
            "weight_sync": sums["update_weights"],
        }
        phases["other_training_overhead"] = sums["step"] - sum(phases.values())
        assert all(math.isfinite(x) and x >= -1e-5 for x in phases.values())
        assert abs(sum(phases.values()) - sums["step"]) < 1e-5
        for phase, seconds in phases.items():
            phase_rows.append(
                {"source": name, "phase": phase, "total_seconds": seconds, "seconds_per_step": seconds / 50}
            )
        memory = {
            "source": name,
            "actor_reported_max_allocated_GiB": max(float(r["actor/perf/max_memory_allocated_gb"]) for r in rows),
            "actor_reported_max_reserved_GiB": max(float(r["actor/perf/max_memory_reserved_gb"]) for r in rows),
        }
        memory_rows.append(memory)
        cache = directory / "cache-manifest.json"
        preparation = None
        if cache.exists():
            preparation = json.loads(cache.read_text())["seconds"]
            inputs[str(cache)] = digest(cache)
        cases[name] = {
            "steps": 50,
            "cache_preparation_seconds": preparation,
            "training_seconds": sums["step"],
            "validation_seconds": sums["testing"],
            "phases_seconds": phases,
            "actor_memory": memory,
            "validation_fraction_of_training_plus_validation": sums["testing"] / (sums["step"] + sums["testing"]),
        }
    models = {}
    for name in ("DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B"):
        directory = ROOT.parent / "models" / name
        config = directory / "config.json"
        c = json.loads(config.read_text())
        assert c["torch_dtype"] == "bfloat16" and not c.get("use_sliding_window", False)
        kv = 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * (c["hidden_size"] // c["num_attention_heads"]) * 2
        models[name] = {
            "checkpoint_file_GiB": sum(p.stat().st_size for p in directory.glob("*.safetensors")) / 2**30,
            "BF16_KV_bytes_per_token": kv,
            "single_model_KV_16_sequences_5120_tokens_GiB": 16 * 5120 * kv / 2**30,
            "scope": "Config-derived KV storage estimate; excludes runtime workspaces, allocator and CUDA contexts",
        }
        inputs[str(config)] = digest(config)
    report = {
        "profiler_sha256": digest(Path(__file__)),
        "inputs_sha256": inputs,
        "runs": cases,
        "model_estimates": models,
        "memory_scope": (
            "Maximum reported actor PyTorch allocator high-water mark, reduced by max across ranks; "
            "not whole-device/run peaks"
        ),
        "timing_scope": (
            "Logged training stage wall times; teacher is inside adv; setup/hash checks are not fully timed"
        ),
    }
    if args.gpu_samples:
        raw = args.gpu_samples.read_bytes()
        raw = raw[: raw.rfind(b"\n") + 1]
        samples = list(csv.DictReader(io.StringIO(raw.decode())))
        assert samples
        groups = {}
        for gpu in sorted({r["gpu"] for r in samples}):
            subset = [r for r in samples if r["gpu"] == gpu]
            groups[gpu] = {
                "samples": len(subset),
                "max_memory_used_GiB": max(float(r["memory_used_MiB"]) for r in subset) / 1024,
                "mean_gpu_utilization_percent": sum(float(r["gpu_utilization_percent"]) for r in subset) / len(subset),
            }
        report["whole_gpu_observation"] = {
            "path": str(args.gpu_samples),
            "observed_bytes": len(raw),
            "prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "start_utc": samples[0]["utc"],
            "end_utc": samples[-1]["utc"],
            "gpus": groups,
            "scope": "Sampled whole-device usage only during the recorded window; short peaks can be missed",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "training-phases.csv", phase_rows)
    save_csv(args.output / "actor-memory.csv", memory_rows)
    report["outputs_sha256"] = {
        name: digest(args.output / name) for name in ("training-phases.csv", "actor-memory.csv")
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "training_min": r["training_seconds"] / 60,
                    "validation_min": r["validation_seconds"] / 60,
                    "memory": r["actor_memory"],
                }
                for name, r in cases.items()
            }
        )
    )


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def phase_hint(log):
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 131072))
        text = stream.read().decode(errors="replace")
    steps = list(re.finditer(r"step:(\d+) - ", text))
    validations = list(re.finditer(r"test_gen_batch meta info:[^\n]*'global_steps': (\d+)", text))
    step = steps[-1][1] if steps else ""
    if validations and (not steps or validations[-1].start() > steps[-1].start()):
        return step, "validation", validations[-1][1]
    return step, "training_or_transition" if steps else "unknown", ""


def monitor(args):
    start_id = process_start(args.pid)
    if start_id is None:
        raise ValueError("The watched process is not live")
    if args.interval < 0.5 or set(args.gpus.split(",")) - {"0", "1", "2", "3"}:
        raise ValueError("Use GPUs 0-3 and a polling interval of at least 0.5 seconds")
    logs = list((args.run / "logs").glob("*.log"))
    assert len(logs) == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".meta.json").exists():
        raise FileExistsError("Use a fresh resource monitor output path")
    metadata = {
        "monitor_pid": os.getpid(),
        "watched_pid": args.pid,
        "watched_start_id": start_id,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": args.interval,
        "gpus": args.gpus,
        "monitor_source_sha256": digest(Path(__file__)),
        "phase_hint_scope": "Coarse log hint only; not a substage memory attribution",
    }
    (args.output.with_suffix(".meta.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    command = [
        "nvidia-smi",
        f"--id={args.gpus}",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    with args.output.open("x", newline="", buffering=1) as stream:
        fields = [
            "utc",
            "gpu",
            "memory_used_MiB",
            "memory_total_MiB",
            "gpu_utilization_percent",
            "power_W",
            "last_logged_step",
            "phase_hint",
            "validation_step_hint",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        while process_start(args.pid) == start_id:
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            except subprocess.SubprocessError as error:
                print(json.dumps({"utc": timestamp, "sample_error": type(error).__name__}), flush=True)
                time.sleep(args.interval)
                continue
            step, phase, validation = phase_hint(logs[0])
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                gpu, used, total, utilization, power = row
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [timestamp, gpu, used, total, utilization, power, step, phase, validation],
                            strict=True,
                        )
                    )
                )
            time.sleep(args.interval)
    print("Watched process ended or its PID identity changed; GPU monitor stopped", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--results", type=Path, default=ROOT / "docs/experiment/results")
    summary.add_argument("--output", type=Path, default=ROOT / "docs/experiment/results/resources")
    summary.add_argument("--gpu-samples", type=Path)
    watch = commands.add_parser("monitor")
    watch.add_argument("--pid", type=int, required=True)
    watch.add_argument("--run", type=Path, required=True)
    watch.add_argument("--output", type=Path, required=True)
    watch.add_argument("--interval", type=float, default=2)
    watch.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    summarize(args) if args.command == "summarize" else monitor(args)


if __name__ == "__main__":
    main()
