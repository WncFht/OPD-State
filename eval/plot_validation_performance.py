"""Plot actual paired validation benchmarks, excluding warmup from comparisons.

Semantic ledger: each row is a measured resource configuration. Teal means a
numerically accepted KL scorer; coral marks a rejected scorer. Generation
colors identify concurrency settings, not model quality. Dots are repeats.
Geometric ledger: horizontal positions are seconds or GiB; rows are categorical.
Allocator peaks belong to the scoring process and exclude whole-card overhead.
Generation measurements exclude actor training, grading and KL. Natural-length
throughput uses the actual generated-token count, never a configured token cap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--generation-gpu", type=Path)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper", fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    scoring = json.loads(args.scoring.read_text())
    if scoring["state"] not in ("passed", "completed_with_rejected_candidates"):
        raise ValueError("Only completed scoring measurements may be plotted")
    measurements = [row for row in scoring["measurements"] if not row["warmup"]]
    cases = list(dict.fromkeys(row["case"] for row in measurements))
    rows = []
    for case in cases:
        repeats = [row for row in measurements if row["case"] == case]
        rows.append(
            {
                "case": case,
                "seconds_mean": statistics.mean(row["seconds"] for row in repeats),
                "seconds_min": min(row["seconds"] for row in repeats),
                "seconds_max": max(row["seconds"] for row in repeats),
                "allocated_gib": max(max(row["allocated_bytes_by_rank"]) for row in repeats) / 2**30,
                "reserved_gib": max(max(row["reserved_bytes_by_rank"]) for row in repeats) / 2**30,
                "max_paired_mean_error": max(row["max_paired_mean_error"] for row in repeats),
                "accepted": all(row["passed"] for row in repeats),
            }
        )
    labels = [
        row["case"].replace("legacy-input", "Original order").replace("balanced", "Balanced").replace("-full", "/full")
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for accepted, label, color in ((True, "Accepted", fv.HUES["vteal"]), (False, "Rejected", fv.HUES["vcoral"])):
        indices = [i for i, row in enumerate(rows) if row["accepted"] == accepted]
        for ax, metric in zip(axes, ("seconds_mean", "allocated_gib"), strict=True):
            ax.barh(indices, [rows[i][metric] for i in indices], color=color, height=0.6, label=label)
    for index, row in enumerate(rows):
        marker = axes[0].plot(
            [row["seconds_min"], row["seconds_max"]], [index, index], color=fv.NEUTRALS["vink"], marker="|", lw=1
        )[0]
        fv.mark_neutral(marker)
        axes[0].text(row["seconds_max"] + 0.5, index, f"{row['seconds_mean']:.2f}", va="center")
        fv.mark_neutral(axes[1].plot(row["reserved_gib"], index, "|", color=fv.NEUTRALS["vink"], ms=11)[0])
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(row["seconds_max"] for row in rows) * 1.15)
    axes[0].set_xlabel("Four-rank scoring wall time (s)")
    axes[0].set_ylabel("Configuration")
    axes[1].set_xlabel("Peak allocated / reserved marker (GiB)")
    for ax in axes:
        fv.apply_frame(ax, "open")
        fv.apply_grid(ax, "y-only", axis="x")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.14, top=0.90, wspace=0.15)
    fig.text(
        0.5,
        0.025,
        "104 frozen responses; full vocabulary; 2 repeats; coral cases fail the 1e-5 paired error limit.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    outputs = []
    # The right panel intentionally shares the configuration labels on the left.
    audits = {"scoring": fv.audit(fig, scoped_ignore=[("missing-axis-label", axes[1])])}
    for suffix in ("png", "pdf"):
        path = args.output / f"scoring.{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(fig)
    with (args.output / "scoring.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    inputs = {str(args.scoring): digest(args.scoring)}
    if args.generation:
        generation = json.loads(args.generation.read_text())
        if generation["state"] != "passed":
            raise ValueError("Only completed generation benchmarks may be plotted")
        inputs[str(args.generation)] = digest(args.generation)
        if args.generation_gpu is None:
            raise ValueError("Generation plots require the measured whole-card GPU CSV")
        inputs[str(args.generation_gpu)] = digest(args.generation_gpu)
        with args.generation_gpu.open() as stream:
            telemetry = list(csv.DictReader(stream))
        measured = [r for r in generation["measurements"] if not r["warmup"]]
        fig, panels = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        ax, memory_ax = panels
        fixed_x, fixed_y, natural_x, natural_y = [], [], [], []
        memory_peaks = []
        for index, profile in enumerate(generation["profiles"]):
            selected = [r for r in measured if r["profile"] == profile["name"]]
            natural = next(r for r in selected if not r["fixed_length"])
            fixed = [
                sum(x["output_positions"] for x in r["ranks"]) / r["api_seconds"] for r in selected if r["fixed_length"]
            ]
            throughput = sum(x["output_positions"] for x in natural["ranks"]) / natural["api_seconds"]
            fixed_x.extend(fixed)
            fixed_y.extend([index] * len(fixed))
            natural_x.append(throughput)
            natural_y.append(index)
            samples = [
                float(r["used_mib"]) / 1024
                for r in telemetry
                if natural["started_epoch_s"] <= float(r["epoch_s"]) <= natural["finished_epoch_s"]
            ]
            if not samples:
                raise ValueError("Natural generation has no aligned whole-card memory samples")
            memory_peaks.append(max(samples))
        ax.plot(fixed_x, fixed_y, "o", ls="none", color=fv.HUES["vteal"], label="Fixed 128 tokens")
        ax.plot(natural_x, natural_y, "D", ls="none", color=fv.HUES["vorange"], label="Natural EOS")
        ax.set_yticks(range(len(generation["profiles"])), [p["name"] for p in generation["profiles"]])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel("Four-replica output tokens / API second")
        ax.set_ylabel("Configuration")
        ax.legend(frameon=False, loc="lower right")
        fv.mark_neutral(memory_ax.barh(natural_y, memory_peaks, color=fv.NEUTRALS["baseline"]))
        fv.mark_neutral(memory_ax.axvline(81559 / 1024, ls="--", color=fv.NEUTRALS["vink"], lw=1))
        memory_ax.set_xlim(0, 81559 / 1024 * 1.05)
        memory_ax.set_xlabel("Natural-phase whole-card peak (GiB)")
        for index, peak in enumerate(memory_peaks):
            memory_ax.text(peak + 0.8, index, f"{peak:.1f}", va="center")
        for panel in panels:
            fv.apply_frame(panel, "open")
            fv.apply_grid(panel, "y-only", axis="x")
        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.23, top=0.97, wspace=0.16)
        fig.text(
            0.5,
            0.025,
            "512 requests; same model/prompts/seeds; natural outputs may differ across batch shapes.",
            ha="center",
            color=fv.NEUTRALS["baseline"],
            fontsize=9,
        )
        audits["generation"] = fv.audit(fig, scoped_ignore=[("missing-axis-label", memory_ax)])
        for suffix in ("png", "pdf"):
            path = args.output / f"generation.{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(path)
        plt.close(fig)
    receipt = {
        "inputs_sha256": inputs,
        "driver_sha256": digest(Path(__file__)),
        "style_sha256": digest(args.style / "assets/mpl/fvstyle.py"),
        "outputs_sha256": {p.name: digest(p) for p in [*outputs, args.output / "scoring.csv"]},
        "matplotlib_version": matplotlib.__version__,
        "audits": audits,
        "scope": "Paired scoring and separate inference API benchmarks, not full trainer speedup",
    }
    (args.output / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
