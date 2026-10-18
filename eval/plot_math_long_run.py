"""Plot a reproducible prefix of fixed-cache training and audited evaluations.

Semantic ledger: coral always means the single Semi-OPD run. A training point
is one logged update; an accuracy point is a fully collected eval16 checkpoint.
Gray horizontal references are independent evaluation-only model runs.
Geometric ledger: x=logged update; each 50-update span replays the same cache.
RMS/gradient/length/accuracy have separate y units. Accuracy uses a zoomed axis.
The sampled RMS on fixed actions is not a full conditional KL.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collector", Path(__file__).with_name("collect_math_experiment.py"))
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_stats(rows):
    tokens = sum(row["response_length/mean"] for row in rows)
    return {
        "token_weighted_rms": math.sqrt(
            sum(row["opd/log_ratio_rms"] ** 2 * row["response_length/mean"] for row in rows) / tokens
        ),
        "mean_grad_norm": sum(row["actor/grad_norm"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--snapshot", type=Path, required=True, help="Collected snapshot covering all completed validations"
    )
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/figures/long")
    args = parser.parse_args()
    if args.through_step < 50:
        raise ValueError("Plot long-run progress after at least one complete eval16 checkpoint")
    logs = list((args.run / "logs").glob("*.log"))
    assert len(logs) == 1
    observed = logs[0].read_bytes()
    observed = observed[: observed.rfind(b"\n") + 1]
    target = list(re.finditer(rb"step:" + str(args.through_step).encode() + rb" - [^\n]*\n", observed))
    assert len(target) == 1, "The requested complete training row must exist exactly once"
    observed = observed[: target[0].end()]
    _, data = COLLECTOR.parse_steps(observed.decode(), through_step=args.through_step)
    assert set(data) == set(range(1, args.through_step + 1))
    training = [data[step] for step in range(1, args.through_step + 1)]
    for row in training:
        assert row["source/cache_hit_fraction"] == 1
        assert row["source/cache_read_count"] == row["step"] * 64
        if row["step"] > 50:
            assert row["response_length/mean"] == data[row["step"] - 50]["response_length/mean"]
    summary = json.loads((args.snapshot / "summary.json").read_text())
    assert Path(summary["run_directory"]).resolve() == args.run.resolve()
    assert summary["full_benchmarks"] and summary["samples_per_question"] == 16
    assert summary["source_commit"] == (args.run / "source/COMMIT").read_text().strip()
    inputs = {str(args.snapshot / "summary.json"): sha256(args.snapshot / "summary.json")}
    accuracy = []
    for step in range(50, args.through_step + 1, 50):
        record_path = args.snapshot / f"validation-step{step}.json"
        record = json.loads(record_path.read_text())
        assert record["questions"] == 570 and record["responses"] == 9120
        question_path = args.snapshot / f"questions-step{step}.csv"
        assert sha256(question_path) == record["questions_csv_sha256"]
        inputs[str(record_path)] = sha256(record_path)
        inputs[str(question_path)] = sha256(question_path)
        for key, value in record["metrics"].items():
            assert abs(data[step][key] - value) < 1e-8, (step, key)
        accuracy.append({"step": step, "macro_mean@16": record["metrics"]["eval/macro/mean@16"]})
    references_path = ROOT / "docs/experiment/results/reference-accuracy.json"
    references = json.loads(references_path.read_text())["runs"]
    inputs[str(references_path)] = sha256(references_path)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ("step", "opd/log_ratio_rms", "actor/grad_norm", "response_length/mean", "opd/token_reward_mean")
    COLLECTOR.write_csv(args.output / "training-plotted.csv", [{key: row[key] for key in fields} for row in training])
    if accuracy:
        COLLECTOR.write_csv(args.output / "accuracy-plotted.csv", accuracy)
    skill = Path(os.environ.get("FIGURE_VIZ_HOME", ROOT.parent / ".agents/skills/figure-viz"))
    sys.path.insert(0, str(skill / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fv.apply_rc("paper", fontsize=7.6)
    color, gray, ink = fv.HUES["vcoral"], fv.NEUTRALS["baseline"], fv.NEUTRALS["vink"]
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(5.5, 4.9))
    specs = [
        ("opd/log_ratio_rms", "(a) Fixed-cache log-ratio RMS", "RMS (nats/token)"),
        ("actor/grad_norm", "(b) Gradient norm (pre-clip)", "L2 norm"),
        ("response_length/mean", "(c) Repeated batch response length", "tokens / response"),
    ]
    for ax, (key, title, unit) in zip(axes.flat, specs, strict=False):
        ax.plot([row["step"] for row in training], [row[key] for row in training], color=color, lw=1, label="Semi-OPD")
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
    ax = axes[1, 1]
    if accuracy:
        ax.plot(
            [r["step"] for r in accuracy],
            [r["macro_mean@16"] * 100 for r in accuracy],
            color=color,
            marker="o",
            ms=3.5,
            label="Semi-OPD",
        )
    values = [r["macro_mean@16"] * 100 for r in accuracy]
    for role, style, c in (("initial_student", "--", gray), ("teacher", ":", ink)):
        value = references[role]["metrics"]["eval/macro/mean@16"] * 100
        values.append(value)
        fv.mark_neutral(ax.axhline(value, color=c, ls=style, lw=1))
    margin = max(1, (max(values) - min(values)) * 0.15)
    ax.set_ylim(max(0, math.floor(min(values) - margin)), min(100, math.ceil(max(values) + margin)))
    ax.set_title("(d) Completed eval16 checkpoints", loc="left")
    ax.set_ylabel("macro mean@16 (%)")
    for ax in axes.flat:
        ax.set_xlim(1, max(args.through_step, 50))
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        for boundary in range(50, args.through_step, 50):
            fv.mark_neutral(ax.axvline(boundary + 0.5, color=fv.NEUTRALS["vmist"], ls=":", lw=0.7))
        fv.apply_frame(ax, "open")
        fv.apply_grid(ax, "y-only")
    for ax in axes[-1]:
        ax.set_xlabel("Logged optimizer step")
    fig.tight_layout(rect=(0, 0.18, 1, 1), h_pad=1.8, w_pad=1.8)
    handles = [
        Line2D([], [], color=color, label="Semi-OPD"),
        Line2D([], [], color=gray, ls="--", label="Initial student (external)"),
        Line2D([], [], color=ink, ls=":", label="Teacher (external)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2)
    caption = f"Logged through {args.through_step}; vertical markers: cache passes; accuracy: completed eval16 only."
    fig.text(0.5, 0.009, caption, ha="center", color=gray, fontsize="small")
    issues = fv.audit(fig)
    fig.savefig(args.output / "long-training-progress.png", dpi=240, metadata={"Description": caption})
    fig.savefig(
        args.output / "long-training-progress.pdf", metadata={"Creator": Path(__file__).name, "Subject": caption}
    )
    plt.close(fig)
    cycles = [training[start : start + 50] for start in range(0, len(training), 50)]
    paired_count = min(len(cycle) for cycle in cycles)
    paired = [
        {"cache_pass": i + 1, "matched_batches": paired_count, **window_stats(cycle[:paired_count])}
        for i, cycle in enumerate(cycles)
    ]
    report = {
        "plotter_sha256": sha256(Path(__file__)),
        "collector_sha256": sha256(Path(COLLECTOR.__file__)),
        "style_sha256": sha256(skill / "assets/mpl/fvstyle.py"),
        "matplotlib_version": matplotlib.__version__,
        "source_commit": summary["source_commit"],
        "logged_through_step": args.through_step,
        "log_path": str(logs[0]),
        "observed_log_bytes": len(observed),
        "observed_log_prefix_sha256": hashlib.sha256(observed).hexdigest(),
        "inputs_sha256": inputs,
        "completed_validation_steps": [r["step"] for r in accuracy],
        "all_cached_lengths_repeat_exactly": True,
        "paired_prefix_batch_statistics": paired,
        "paired_rms_decreases_vs_first_pass": [
            {
                "cache_pass": i + 1,
                "matched_batches": paired_count,
                "lower_rms_batches": sum(
                    cycle[j]["opd/log_ratio_rms"] < cycles[0][j]["opd/log_ratio_rms"] for j in range(paired_count)
                ),
            }
            for i, cycle in enumerate(cycles[1:], start=1)
        ],
        "audit": issues,
        "outputs_sha256": {p.name: sha256(p) for p in args.output.iterdir() if p.suffix in {".csv", ".png", ".pdf"}},
    }
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"paired_statistics": paired, "audit": issues}, ensure_ascii=False))
    if issues:
        raise SystemExit("Long-run figure audit reported issues")


if __name__ == "__main__":
    main()
