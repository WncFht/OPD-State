"""Render the unified rollout-source experiment from audited local evidence.

Semantic ledger: each color denotes one trajectory source in every panel;
each line point is one optimizer step, each accuracy bar is one completed
step-50 student evaluation. Gray references are separate evaluation-only runs.
Geometric ledger: x=optimizer steps or sample count k; y units are explicit
per panel. Accuracy bars start at zero. No interpolated accuracy trajectory,
training-seed band, or full-vocabulary KL is invented. The signed log-ratio is
the negative logged token reward, not a conditional KL on off-policy actions.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Online student": "2026-10-18-math-student-online-50steps-eval16-20261018-044021",
    "Teacher rollout": "2026-10-18-math-teacher-cache-50steps-eval16-20261018-005033",
    "Semi-OPD": "2026-10-18-math-semi-opd-50steps-eval16-20261018-025524",
    "Fusion 0.5": "2026-10-18-math-fusion-lambda05-50steps-eval16-20261018-065746",
}
BENCHMARKS = ("MATH-500", "AMC23", "AIME25", "macro")
TRAINING_METRICS = {
    "signed_log_ratio": "opd/token_reward_mean",
    "token_reward": "opd/token_reward_mean",
    "log_ratio_rms": "opd/log_ratio_rms",
    "grad_norm": "actor/grad_norm",
    "entropy": "actor/entropy",
    "actor_loss": "actor/loss",
    "response_tokens": "response_length/mean",
    "cap_fraction": "response_length/clip_ratio",
    "sequence_reward": "critic/rewards/mean",
    "actor_update_seconds": "timing_s/update_actor",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_evidence(results):
    training, accuracy, sources = [], [], {}
    for arm, name in RUNS.items():
        directory = results / name
        path = directory / "training-1-50.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in rows] == list(range(1, 51)), name
        validation = json.loads((directory / "validation-step50.json").read_text())
        assert validation["questions"] == 570 and validation["responses"] == 9120
        assert sha256(directory / "questions-step50.csv") == validation["questions_csv_sha256"]
        for row in rows:
            point = {"source": arm, "step": int(row["step"])}
            for metric, key in TRAINING_METRICS.items():
                value = float(row[key])
                assert math.isfinite(value), (name, key)
                point[metric] = -value if metric == "signed_log_ratio" else value
            training.append(point)
        metrics = validation["metrics"]
        for benchmark in BENCHMARKS:
            accuracy.append(
                {
                    "source": arm,
                    "step": 50,
                    "benchmark": benchmark,
                    "mean@16": metrics[f"eval/{benchmark}/mean@16"],
                    **{f"pass@{k}": metrics[f"eval/{benchmark}/pass@{k}"] for k in (1, 2, 4, 8, 16)},
                }
            )
        for file in (path, directory / "validation-step50.json", directory / "questions-step50.csv"):
            sources[str(file.relative_to(ROOT))] = sha256(file)
    reference_path = results / "reference-accuracy.json"
    references = json.loads(reference_path.read_text())["runs"]
    sources[str(reference_path.relative_to(ROOT))] = sha256(reference_path)
    return training, accuracy, references, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "docs/experiment/results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/figures")
    args = parser.parse_args()
    skill = Path(os.environ.get("FIGURE_VIZ_HOME", ROOT.parent / ".agents/skills/figure-viz"))
    if not (skill / "assets/mpl/fvstyle.py").is_file():
        raise FileNotFoundError("Set FIGURE_VIZ_HOME to the figure-viz skill directory")
    sys.path.insert(0, str(skill / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fv.apply_rc("paper", fontsize=7.6)
    training, accuracy, references, sources = load_evidence(args.results.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "training-plotted.csv", training)
    write_csv(args.output / "accuracy-plotted.csv", accuracy)
    colors = dict(zip(RUNS, fv.series_colors("paper", 4), strict=True))
    audits = {}
    neutral = fv.NEUTRALS["baseline"]
    ink = fv.NEUTRALS["vink"]
    handles = [Line2D([], [], color=colors[arm], lw=1.5, label=arm) for arm in RUNS]

    def save(fig, stem, caption, *, legend=handles, bottom=0.13):
        for ax in fig.axes:
            fv.apply_frame(ax, "open")
            fv.apply_grid(ax, "y-only")
            ax.tick_params(labelsize="small")
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=1.8, w_pad=1.8)
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2)
        fig.text(0.5, 0.009, caption, ha="center", color=neutral, fontsize="small")
        audits[stem] = fv.audit(fig)
        metadata = {"Creator": "eval/plot_math_experiment.py", "Subject": caption}
        fig.savefig(args.output / f"{stem}.png", dpi=240, metadata={"Description": caption})
        fig.savefig(args.output / f"{stem}.pdf", metadata=metadata)
        plt.close(fig)

    def training_panels(specs, stem, caption):
        fig, axes = plt.subplots(len(specs) // 2, 2, sharex=True, figsize=(5.5, 2.05 * (len(specs) // 2) + 0.55))
        for index, (ax, (metric, title, unit)) in enumerate(zip(axes.flat, specs, strict=True)):
            for arm in RUNS:
                rows = [row for row in training if row["source"] == arm]
                ax.plot(
                    [row["step"] for row in rows], [row[metric] for row in rows], color=colors[arm], lw=1, label=arm
                )
            ax.set_title(f"({chr(97 + index)}) {title}", loc="left")
            ax.set_ylabel(unit)
            ax.set_xlim(1, 50)
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if index >= len(specs) - 2:
                ax.set_xlabel("Optimizer step")
            if metric in {"signed_log_ratio", "token_reward", "actor_loss", "sequence_reward"}:
                fv.mark_neutral(ax.axhline(0, color=neutral, ls=":", lw=0.6))
        save(fig, stem, caption, bottom=0.12 if len(specs) == 6 else 0.16)

    training_panels(
        [
            ("signed_log_ratio", "Sampled log-ratio (KL proxy)", "Signed ratio (nats/token)"),
            ("token_reward", "Teacher token reward", "Reward (nats/token)"),
            ("log_ratio_rms", "Log-ratio RMS", "RMS (nats/token)"),
            ("grad_norm", "Gradient norm (pre-clip)", "L2 norm"),
            ("entropy", "Student entropy on source states", "Entropy (nats/token)"),
            ("actor_loss", "Actor surrogate loss", "token mean"),
        ],
        "training-optimization",
        "Raw steps; source-dependent diagnostics; signed log-ratio is not a full KL.",
    )
    training_panels(
        [
            ("response_tokens", "Training response length", "tokens / response"),
            ("cap_fraction", "Responses reaching 4K cap", "fraction"),
            ("sequence_reward", "Summed teacher reward", "nats / response"),
            ("actor_update_seconds", "Actor update time", "seconds / step"),
        ],
        "training-rollouts",
        "Raw steps; update time excludes generation, cache preparation and validation.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.9))
    for index, (ax, benchmark) in enumerate(zip(axes.flat, BENCHMARKS, strict=True)):
        for j, arm in enumerate(RUNS):
            score = next(r["mean@16"] for r in accuracy if r["source"] == arm and r["benchmark"] == benchmark)
            ax.bar(j, score * 100, width=0.65, color=colors[arm], label=arm)
            ax.text(j, score * 100 + 1.5, f"{score * 100:.1f}", ha="center", fontsize="small")
        for role, style, color in (("initial_student", "--", neutral), ("teacher", ":", ink)):
            score = references[role]["metrics"][f"eval/{benchmark}/mean@16"]
            fv.mark_neutral(ax.axhline(score * 100, color=color, ls=style, lw=1))
        ax.set_title(f"({chr(97 + index)}) {benchmark}", loc="left")
        ax.set_xticks(range(4), ("Online", "Teacher", "Semi", "Fusion"))
        ax.set_ylim(0, 100)
        ax.set_ylabel("mean@16 (%)")
    reference_handles = [
        Line2D([], [], color=neutral, ls="--", label="Initial student (external)"),
        Line2D([], [], color=ink, ls=":", label="Teacher (external)"),
    ]
    save(
        fig,
        "accuracy-step50",
        "Step-50 student evaluations; dashed/dotted lines: user-provided reference runs.",
        legend=handles + reference_handles,
        bottom=0.22,
    )

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8))
    for index, (ax, benchmark) in enumerate(zip(axes.flat, BENCHMARKS, strict=True)):
        for arm in RUNS:
            row = next(r for r in accuracy if r["source"] == arm and r["benchmark"] == benchmark)
            ks = (1, 2, 4, 8, 16)
            ax.plot(ks, [row[f"pass@{k}"] * 100 for k in ks], color=colors[arm], marker="o", ms=2, label=arm)
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 2, 4, 8, 16), ("1", "2", "4", "8", "16"))
        ax.minorticks_off()
        ax.set_ylim(0, 100)
        ax.set_title(f"({chr(97 + index)}) {benchmark}", loc="left")
        ax.set_xlabel("Samples k")
        ax.set_ylabel("pass@k (%)")
    save(fig, "pass-at-k", "Step 50; unbiased without-replacement pass@k estimated from 16 responses.", bottom=0.16)
    report = {
        "plotter_sha256": sha256(Path(__file__)),
        "style_sha256": sha256(skill / "assets/mpl/fvstyle.py"),
        "matplotlib_version": matplotlib.__version__,
        "sources": sources,
        "colors": colors,
        "training_rows": len(training),
        "accuracy_rows": len(accuracy),
        "audit": audits,
        "outputs": {p.name: sha256(p) for p in args.output.iterdir() if p.suffix in {".png", ".pdf", ".csv"}},
        "metric_semantics": {
            "signed_log_ratio": "-opd/token_reward_mean; sampled log(student/teacher), not full conditional KL",
            "grad_norm": "actor/grad_norm before clipping; one recorded training run per source",
            "accuracy": (
                "completed step-50 evaluations only; external references are not paired local step-0 measurements"
            ),
        },
    }
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": len(audits), "audit": audits}, ensure_ascii=False))
    if any(audits.values()):
        raise SystemExit("Figure audit reported issues; inspect figures and audit.json")


if __name__ == "__main__":
    main()
