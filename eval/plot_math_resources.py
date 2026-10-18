"""Compare four audited experiments using logged costs and sampled GPU telemetry.

Semantic ledger: each hue is one rollout source, matching the accuracy figures.
Filled/hollow bars denote training/validation; hatching denotes cache preparation.
Timeline line styles identify physical GPUs 0--3; gray spans identify validation.
Geometric ledger: time is elapsed wall minutes, memory is whole-device GiB,
utilization is sampled GPU busy percent. Sample maxima are not continuous peaks.
Offline generation has no historical memory trace and is never assigned zero.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PLOTS = module("plots", "plot_math_experiment.py")
COSTS = module("costs", "analyze_math_rollout_costs.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "docs/experiment/results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/figures/resources")
    parser.add_argument("--capture-capacity", action="store_true", help="Create a fresh GPU 0--3 capacity receipt")
    args = parser.parse_args()
    resources = args.results / "resources"
    capacity_path = resources / "device-capacity.json"
    if args.capture_capacity:
        if capacity_path.exists():
            raise FileExistsError("Keep the existing device receipt for reproducible historical figures")
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0,1,2,3",
        ]
        output = subprocess.check_output(command, text=True)
        capacity_path.write_text(
            json.dumps(
                {
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "command": command,
                    "gpus": [
                        {"index": int(r[0]), "name": r[1].strip(), "memory_total_MiB": float(r[2])}
                        for r in csv.reader(output.splitlines())
                    ],
                    "scope": "Current physical GPU capacity; historical usage comes from archived W&B samples",
                },
                indent=2,
            )
            + "\n"
        )
    inputs = {}

    def read_json(path):
        inputs[str(path)] = sha256(path)
        return json.loads(path.read_text())

    capacity = read_json(capacity_path)
    totals = {r["index"]: r["memory_total_MiB"] / 1024 for r in capacity["gpus"]}
    assert set(totals) == set(range(4)) and max(totals.values()) == min(totals.values())
    device_gib = totals[0]
    timing = read_json(resources / "summary.json")
    telemetry = read_json(resources / "wandb-gpu/summary.json")
    comparisons, gpu_statistics, timeline, settings = [], [], [], {}
    traces = {}
    for arm, directory in PLOTS.RUNS.items():
        local = args.results / directory
        summary = read_json(local / "summary.json")
        training_path = local / "training-1-50.csv"
        inputs[str(training_path)] = sha256(training_path)
        assert inputs[str(training_path)] == telemetry["runs"][arm]["local_training_csv_sha256"]
        with training_path.open() as stream:
            training = list(csv.DictReader(stream))
        assert [int(r["step"]) for r in training] == list(range(1, 51))
        log = next((Path(summary["run_directory"]) / "logs").glob("*.log"))
        inputs[str(log)] = sha256(log)
        assert inputs[str(log)] == summary["training_log_sha256"]
        cfg = COSTS.logged_config(log.read_text())
        actor = cfg["actor_rollout_ref"]["actor"]
        rollout = cfg["actor_rollout_ref"]["rollout"]
        settings[arm] = {
            "source_commit": cfg["trainer"]["source_commit"],
            "global_batch_size": cfg["data"]["train_batch_size"],
            "ppo_epochs": actor["ppo_epochs"],
            "actor_param_offload": actor["fsdp_config"]["param_offload"],
            "actor_optimizer_offload": actor["fsdp_config"]["optimizer_offload"],
            "teacher_param_offload": cfg["distillation"]["fsdp_teacher"]["fsdp_config"]["param_offload"],
            "teacher_micro_batch_per_gpu": cfg["distillation"]["fsdp_teacher"]["micro_batch_size_per_gpu"],
            "actor_token_budget_per_gpu": actor["ppo_max_token_len_per_gpu"],
            "rollout": {
                k: rollout[k]
                for k in (
                    "n",
                    "prompt_length",
                    "response_length",
                    "val_kwargs",
                    "max_num_seqs",
                    "max_model_len",
                    "max_num_batched_tokens",
                    "gpu_memory_utilization",
                    "enforce_eager",
                    "calculate_log_probs",
                    "free_cache_engine",
                    "tensor_model_parallel_size",
                    "data_parallel_size",
                    "pipeline_model_parallel_size",
                )
            },
            "fusion": rollout.get("logit_fusion"),
        }
        assert all(
            settings[arm][k] for k in ("actor_param_offload", "actor_optimizer_offload", "teacher_param_offload")
        )
        meta = telemetry["runs"][arm]
        boundaries = meta["phase_boundaries"]
        observed = []
        for relative, expected in meta["outputs_sha256"].items():
            path = resources / "wandb-gpu" / relative
            inputs[str(path)] = sha256(path)
            assert inputs[str(path)] == expected
            with path.open() as stream:
                observed.extend(csv.DictReader(stream))
        normalized = []
        for r in sorted(observed, key=lambda r: float(r["timestamp"])):
            timestamp = float(r["timestamp"])
            if not boundaries["training_start_estimate"] <= timestamp <= boundaries["validation_end_estimate"]:
                continue
            row = {
                "source": arm,
                "elapsed_minutes": (timestamp - boundaries["training_start_estimate"]) / 60,
                "phase": r["phase_hint"],
            }
            valid = False
            for g in range(4):
                memory, busy = r.get(f"gpu{g}.memoryAllocatedBytes"), r.get(f"gpu{g}.gpu")
                row[f"gpu{g}_memory_GiB"] = float(memory) / 2**30 if memory else ""
                row[f"gpu{g}_busy_percent"] = float(busy) if busy else ""
                if memory:
                    percent = r.get(f"gpu{g}.memoryAllocated")
                    assert 0 <= float(memory) / 2**30 <= device_gib
                    if percent and float(percent) > 0:
                        assert abs(float(memory) / 2**30 / (float(percent) / 100) - device_gib) < 0.02
                if busy:
                    assert math.isfinite(float(busy)) and 0 <= float(busy) <= 100
                valid |= bool(memory or busy)
            if valid:
                normalized.append(row)
        assert normalized
        traces[arm] = normalized
        timeline.extend(normalized)
        for phase in ("training", "validation"):
            for g in range(4):
                subset = [r for r in normalized if r["phase"] == phase]
                memory = [r[f"gpu{g}_memory_GiB"] for r in subset if r[f"gpu{g}_memory_GiB"] != ""]
                busy = [r[f"gpu{g}_busy_percent"] for r in subset if r[f"gpu{g}_busy_percent"] != ""]
                assert memory and busy
                gpu_statistics.append(
                    {
                        "source": arm,
                        "phase": phase,
                        "gpu": g,
                        "memory_samples": len(memory),
                        "memory_mean_GiB": statistics.mean(memory),
                        "memory_max_GiB": max(memory),
                        "memory_max_percent": max(memory) / device_gib * 100,
                        "gpu_busy_mean_percent": statistics.mean(busy),
                    }
                )
        t = timing["runs"][arm]
        tokens = int(sum(float(r["response_length/mean"]) * 64 for r in training))
        preparation = t["cache_preparation_seconds"] or 0
        generation = t["phases_seconds"]["generate_or_read_cache"]
        gen_seconds = preparation if preparation else generation
        phase_memory = {
            phase: max(r["memory_max_GiB"] for r in gpu_statistics if r["source"] == arm and r["phase"] == phase)
            for phase in ("training", "validation")
        }
        for phase in phase_memory:
            expected = max(r["max_used_GiB"] for r in meta["sampled_gpu_statistics"][phase].values())
            assert abs(phase_memory[phase] - expected) < 1e-8
        comparisons.append(
            {
                "source": arm,
                "response_tokens": tokens,
                "preparation_minutes": preparation / 60,
                "training_minutes": t["training_seconds"] / 60,
                "validation_minutes": t["validation_seconds"] / 60,
                "preparation_training_validation_minutes": (
                    preparation + t["training_seconds"] + t["validation_seconds"]
                )
                / 60,
                "generation_or_preparation_tokens_per_second": tokens / gen_seconds,
                "training_loop_response_tokens_per_second": tokens / t["training_seconds"],
                "training_sampled_max_GiB": phase_memory["training"],
                "validation_sampled_max_GiB": phase_memory["validation"],
                "maximum_sampled_memory_percent": max(phase_memory.values()) / device_gib * 100,
                "actor_allocated_peak_GiB": t["actor_memory"]["actor_reported_max_allocated_GiB"],
                "actor_reserved_peak_GiB": t["actor_memory"]["actor_reported_max_reserved_GiB"],
            }
        )
        if preparation:
            cache = read_json(local / "cache-manifest.json")
            assert cache["response_tokens"] == tokens and cache["records"] == 3200
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "comparison.csv", comparisons)
    save_csv(args.output / "gpu-phase-statistics.csv", gpu_statistics)
    for start in range(0, len(timeline), 900):
        save_csv(
            args.output / f"timeline-{start + 1}-{min(start + 900, len(timeline))}.csv", timeline[start : start + 900]
        )
    skill = Path(os.environ.get("FIGURE_VIZ_HOME", ROOT.parent / ".agents/skills/figure-viz"))
    sys.path.insert(0, str(skill / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fv.apply_rc("paper", fontsize=11)
    colors = dict(zip(PLOTS.RUNS, fv.series_colors("paper", 4), strict=True))
    labels = ["Online student", "Teacher cache", "Semi-OPD", "Fusion 0.5"]
    neutral = fv.NEUTRALS["baseline"]
    audits = {}

    def finish(fig, stem, caption, *, bottom):
        fig.tight_layout(rect=(0, bottom, 1, 1), w_pad=2)
        fig.text(0.5, 0.009, caption, ha="center", color=neutral, fontsize="small")
        audits[stem] = fv.audit(fig)
        fig.savefig(args.output / f"{stem}.png", dpi=240, metadata={"Description": caption})
        fig.savefig(args.output / f"{stem}.pdf", metadata={"Creator": Path(__file__).name, "Subject": caption})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    arm_colors = [colors[r["source"]] for r in comparisons]
    prep_values = [r["preparation_minutes"] for r in comparisons]
    train_values = [r["training_minutes"] for r in comparisons]
    axes[0].barh(
        range(4),
        prep_values,
        color=[matplotlib.colors.to_rgba(c, 0.3) for c in arm_colors],
        edgecolor=arm_colors,
        hatch="///",
        height=0.55,
        label="Preparation (time only)",
    )
    axes[0].barh(range(4), train_values, left=prep_values, color=arm_colors, height=0.55, label="Training")
    axes[0].barh(
        range(4),
        [r["validation_minutes"] for r in comparisons],
        left=[p + t for p, t in zip(prep_values, train_values, strict=True)],
        facecolor="none",
        edgecolor=arm_colors,
        height=0.55,
        label="Validation",
    )
    for offset, phase, filled in ((-0.17, "training", True), (0.17, "validation", False)):
        axes[1].barh(
            [i + offset for i in range(4)],
            [r[f"{phase}_sampled_max_GiB"] for r in comparisons],
            height=0.28,
            color=arm_colors if filled else "none",
            edgecolor=arm_colors,
            label=phase.title(),
        )
    for i, row in enumerate(comparisons):
        prep, train, val = (row[k] for k in ("preparation_minutes", "training_minutes", "validation_minutes"))
        axes[0].text(prep + train + val + 2, i, f"{prep + train + val:.1f}", va="center", fontsize="small")
        for offset, phase, filled in ((-0.17, "training", True), (0.17, "validation", False)):
            value = row[f"{phase}_sampled_max_GiB"]
            axes[1].text(value + 1, i + offset, f"{value:.1f}", va="center", fontsize="small")
    fv.mark_neutral(axes[1].axvline(device_gib, color=neutral, ls=":", lw=1))
    for ax in axes:
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.6, -0.6)
        ax.set_ylabel("Rollout source")
        fv.apply_frame(ax, "open")
        fv.apply_grid(ax, "y-only", axis="x")
    axes[0].set(xlabel="Logged phase time (minutes)", xlim=(0, 170))
    axes[1].set(xlabel="Whole-GPU memory (GiB)", xlim=(0, 90))
    axes[0].set_title("(a) Preparation + 50 updates + eval16", loc="left")
    axes[1].set_title("(b) Highest sampled device usage", loc="left")
    handles = [
        Patch(
            facecolor=matplotlib.colors.to_rgba(neutral, 0.3),
            edgecolor=neutral,
            hatch="///",
            label="Preparation (time only)",
        ),
        Patch(facecolor=neutral, label="Training"),
        Patch(facecolor="none", edgecolor=neutral, label="Validation"),
        Line2D([], [], color=neutral, ls=":", label=f"Device capacity: {device_gib:.2f} GiB"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2)
    finish(
        fig,
        "resource-comparison",
        "Logged phases exclude untimed setup; memory = sampled GPU 0--3 maxima.",
        bottom=0.23,
    )

    fv.apply_rc("paper", fontsize=12.5)
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 6.4), sharex=True, sharey="row")
    styles = ("-", "--", ":", "-.")
    for col, (arm, label) in enumerate(zip(PLOTS.RUNS, labels, strict=True)):
        data = traces[arm]
        b = telemetry["runs"][arm]["phase_boundaries"]
        start = (b["validation_start_estimate"] - b["training_start_estimate"]) / 60
        end = (b["validation_end_estimate"] - b["training_start_estimate"]) / 60
        for row, metric in enumerate(("memory_GiB", "busy_percent")):
            ax = axes[row, col]
            fv.mark_neutral(ax.axvspan(start, end, color=fv.NEUTRALS["vmist"], alpha=0.45, lw=0))
            for g in range(4):
                selected = [r for r in data if r[f"gpu{g}_{metric}"] != ""]
                ax.plot(
                    [r["elapsed_minutes"] for r in selected],
                    [r[f"gpu{g}_{metric}"] for r in selected],
                    color=colors[arm],
                    ls=styles[g],
                    lw=0.8,
                )
            ax.set_xlim(0, 150)
            ax.set_xticks((0, 50, 100, 150))
            fv.apply_frame(ax, "open")
            fv.apply_grid(ax, "y-only")
            if row == 0:
                ax.set_ylim(0, 85)
                fv.mark_neutral(ax.axhline(device_gib, color=neutral, ls=":", lw=0.8))
                ax.set_title(f"({chr(97 + col)}) {label}", loc="left")
            else:
                ax.set_ylim(0, 105)
                ax.set_xlabel("Elapsed minutes")
    axes[0, 0].set_ylabel("Whole-GPU memory (GiB)")
    axes[1, 0].set_ylabel("GPU busy (%)")
    handles = [Line2D([], [], color=neutral, ls=styles[g], label=f"GPU {g}") for g in range(4)]
    handles += [Patch(facecolor=colors[arm], label=label) for arm, label in zip(PLOTS.RUNS, labels, strict=True)]
    handles += [Patch(facecolor=fv.NEUTRALS["vmist"], label="Validation interval")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.047), ncol=3)
    finish(
        fig,
        "gpu-usage-timeline",
        "Sampled device usage; gray spans: eval16; memory dotted reference: device capacity.",
        bottom=0.26,
    )
    report = {
        "plotter_sha256": sha256(Path(__file__)),
        "inputs_sha256": inputs,
        "helper_sources_sha256": {p: sha256(Path(p)) for p in (PLOTS.__file__, COSTS.__file__)},
        "style_sha256": sha256(skill / "assets/mpl/fvstyle.py"),
        "matplotlib_version": matplotlib.__version__,
        "device_capacity_GiB": device_gib,
        "runtime_settings": settings,
        "scope": (
            "Observed GPU samples and logged phase times; "
            "excludes unrecorded initialization and offline generation memory"
        ),
        "audit": audits,
        "outputs_sha256": {p.name: sha256(p) for p in args.output.iterdir() if p.suffix in {".csv", ".png", ".pdf"}},
    }
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"comparison": comparisons, "audit": audits}))
    if any(audits.values()):
        raise SystemExit("Resource figure audit reported issues")


if __name__ == "__main__":
    main()
