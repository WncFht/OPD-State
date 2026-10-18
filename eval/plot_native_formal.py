"""Plot completed native formal runs without mixing legacy cached rollouts.

Semantic ledger: one color is one training rollout policy throughout all four
figures. Training points are raw optimizer steps, accuracy points are the final
student's complete evaluation, and pass@k is recomputed from question counts.
Gray dashed/dotted series are external initial-student/teacher evaluations,
not local step-zero measurements. No training-seed uncertainty is estimated.
Geometric ledger: training x is optimizer step, accuracy x is categorical policy,
pass x is sample count k, and resource x is policy or physical GPU index. Units
are explicit; accuracy/pass axes zoom independently per benchmark. Resource
peaks cover the whole run at the recorder's sampling rate, not allocator peaks.
The training log-ratio RMS is a sampled-action diagnostic, not full-vocab KL.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from plot_math_experiment import (
    BENCHMARKS,
    reference_metric,
    sha256,
    write_csv,
    zoom_limits,
)

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 2, 4, 8, 16)
COUNTS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}
METRICS = {
    "log_ratio_rms": ("opd/log_ratio_rms", "Sampled log-ratio RMS", "nats/token"),
    "token_reward": ("opd/token_reward_mean", "Teacher token reward", "nats/token"),
    "grad_norm": ("actor/grad_norm", "Gradient norm before clipping", "L2 norm"),
    "response_length": (
        "response_length/mean",
        "Mean training response length",
        "tokens/response",
    ),
    "entropy": ("actor/entropy", "Student entropy on source states", "nats/token"),
    "step_seconds": ("timing_s/step", "Training step time", "seconds/step"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_evidence(specifications, reference_path):
    training, accuracy, resources, identities, hashes, protocols = (
        [],
        [],
        [],
        {},
        {},
        [],
    )
    question_sets = []

    def read(path):
        hashes[str(path)] = sha256(path)
        return json.loads(path.read_text())

    for label, directory in specifications:
        summary = read(directory / "summary.json")
        native = read(directory / "native-acceptance.json")
        remote = read(directory / "wandb-verification.json")
        validation = read(directory / "validation-step50.json")
        require(
            summary == native["math_summary"], f"{label}: collector summaries disagree"
        )
        require(
            summary["training_steps"] == 50
            and summary["samples_per_question"] == 16
            and summary["full_benchmarks"] is True
            and summary["checkpoints_saved"] is False
            and summary["validation_steps"] == [50],
            f"{label}: expected complete 50-step, no-checkpoint eval16 protocol",
        )
        require(
            native["status"]["state"] == "finished"
            and native["status"]["exit_code"] == 0
            and native["trainer_gpu_count"] == 4
            and native["final_student_version"] == 50
            and native["evaluation_responses"] == 9120,
            f"{label}: native GPU acceptance is incomplete",
        )
        require(
            remote["state"] == "finished"
            and remote["source_commit"] == summary["source_commit"]
            and remote["history_training_rows"] == 50
            and remote["steps"] == list(range(1, 51))
            and remote["validation_steps"] == [50]
            and remote["finite_loss_and_grad"] is True
            and remote["save_freq"] == -1,
            f"{label}: W&B evidence is incomplete or mismatched",
        )
        for filename in (
            "summary.json",
            "native-acceptance.json",
            "training-1-50.csv",
            "validation-step50.json",
        ):
            require(
                remote["local_evidence_sha256"][filename]
                == sha256(directory / filename),
                f"{label}: W&B-bound input changed: {filename}",
            )
        binding = remote["native_policy_verification"]
        settings = remote["settings"]
        require(
            binding["manifest_sha256"] == native["native_manifest_sha256"]
            and binding["policy"] == settings["rollout_policy"]
            and settings["rollout_source"] == "native_policy"
            and settings["generation_backend"] == "vllm_native",
            f"{label}: native policy binding is inconsistent",
        )
        # Only the rollout policy differs in this comparison. Keep the other
        # remotely verified settings and frozen source identity in the ledger.
        protocols.append(
            {
                "source_commit": summary["source_commit"],
                **{k: v for k, v in settings.items() if k != "rollout_policy"},
            }
        )
        identities[label] = {
            "run": summary["run_name"],
            "source_commit": summary["source_commit"],
            "policy": binding["policy"],
            "wandb_url": remote["url"],
        }
        path = directory / "training-1-50.csv"
        hashes[str(path)] = sha256(path)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        require(
            [int(row["step"]) for row in rows] == list(range(1, 51)),
            f"{label}: missing/duplicate steps",
        )
        for row in rows:
            point = {"source": label, "step": int(row["step"])}
            for name, (key, _, _) in METRICS.items():
                value = float(row[key])
                require(math.isfinite(value), f"{label}: non-finite {key}")
                point[name] = value
            training.append(point)
        path = directory / "questions-step50.csv"
        hashes[str(path)] = sha256(path)
        require(
            hashes[str(path)] == validation["questions_csv_sha256"],
            f"{label}: question CSV changed",
        )
        with path.open() as stream:
            questions = list(csv.DictReader(stream))
        require(
            validation["questions"] == 570 and validation["responses"] == 9120,
            f"{label}: partial evaluation",
        )
        keys = {(row["data_source"], row["question_sha256"]) for row in questions}
        require(
            len(questions) == len(keys) == 570,
            f"{label}: question identity is duplicated or incomplete",
        )
        question_sets.append(keys)
        for benchmark in COUNTS:
            selected = [row for row in questions if row["data_source"] == benchmark]
            require(
                len(selected) == COUNTS[benchmark],
                f"{label}: wrong benchmark size: {benchmark}",
            )
            counts = [(int(row["n"]), int(row["correct"])) for row in selected]
            require(
                all(n == 16 and 0 <= c <= n for n, c in counts),
                f"{label}: invalid per-question counts",
            )
            point = {
                "source": label,
                "benchmark": benchmark,
                "questions": len(selected),
            }
            for k in KS:
                score = math.fsum(
                    1 - math.comb(n - c, k) / math.comb(n, k) for n, c in counts
                ) / len(counts)
                key = f"eval/{benchmark}/pass@{k}"
                require(
                    math.isclose(
                        score, validation["metrics"][key], abs_tol=1e-12, rel_tol=0
                    )
                    and math.isclose(
                        score, remote["metrics"][key], abs_tol=1e-12, rel_tol=0
                    ),
                    f"{label}: recomputed {key} disagrees with validation/W&B",
                )
                point[f"pass@{k}"] = score
            point["mean@16"] = point["pass@1"]
            accuracy.append(point)
        scores = [row for row in accuracy if row["source"] == label]
        accuracy.append(
            {
                "source": label,
                "benchmark": "macro",
                "questions": 570,
                **{
                    f"pass@{k}": math.fsum(row[f"pass@{k}"] for row in scores) / 3
                    for k in KS
                },
                "mean@16": math.fsum(row["mean@16"] for row in scores) / 3,
            }
        )
        resources.append(
            {
                "source": label,
                "training_minutes": math.fsum(
                    float(row["timing_s/step"]) for row in rows
                )
                / 60,
                "final_validation_minutes": float(rows[-1]["timing_s/testing"]) / 60,
                **{
                    f"gpu_{gpu}_peak_gib": native["whole_card_peak_mib"][str(gpu)]
                    / 1024
                    for gpu in range(4)
                },
            }
        )
    require(
        all(p == protocols[0] for p in protocols),
        "Runs have different recorded settings or source commits",
    )
    require(
        all(q == question_sets[0] for q in question_sets),
        "Runs do not evaluate the same question identities",
    )
    references = read(reference_path)["runs"]
    for role in ("initial_student", "teacher"):
        for benchmark in BENCHMARKS:
            for k in KS:
                value = reference_metric(references, role, benchmark, k)
                require(
                    math.isfinite(value) and 0 <= value <= 1,
                    f"Invalid external reference: {role}/{benchmark}/{k}",
                )
    return training, accuracy, resources, references, hashes, identities, protocols[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", action="append", required=True, metavar="LABEL=DIRECTORY"
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT / "docs/experiment/results/reference-accuracy.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = [
        (value.split("=", 1)[0], Path(value.split("=", 1)[1]).resolve())
        for value in args.result
    ]
    require(
        1 <= len(specs) <= 4 and len({label for label, _ in specs}) == len(specs),
        "Use one to four unique labels",
    )
    training, accuracy, resources, references, inputs, identities, protocol = (
        load_evidence(specs, args.references.resolve())
    )
    skill = Path(
        os.environ.get(
            "FIGURE_VIZ_HOME", ROOT.parent / "dotfiles/.agents/skills/figure-viz"
        )
    )
    sys.path.insert(0, str(skill / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fv.apply_rc("paper", fontsize=7.6)
    args.output.mkdir(parents=True, exist_ok=True)
    colors = dict(
        zip(
            (name for name, _ in specs),
            fv.series_colors("paper", len(specs)),
            strict=True,
        )
    )
    neutral, ink = fv.NEUTRALS["baseline"], fv.NEUTRALS["vink"]
    handles = [
        Line2D([], [], color=color, marker="o", ms=3, lw=1, label=name)
        for name, color in colors.items()
    ]
    reference_styles = {"initial_student": ("--", neutral), "teacher": (":", ink)}
    reference_handles = [
        Line2D(
            [],
            [],
            color=color,
            ls=style,
            label=f"{role.replace('_', ' ').capitalize()} (external)",
        )
        for role, (style, color) in reference_styles.items()
    ]
    audits, axes, outputs = {}, {}, []

    def save(
        fig, stem, caption, with_references=False, bottom=0.15, independent_y=False
    ):
        for ax in fig.axes:
            fv.apply_frame(ax, "open")
            fv.apply_grid(ax, "y-only")
            ax.tick_params(labelsize="small")
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=1.7, w_pad=1.7)
        legend = handles + reference_handles if with_references else handles
        fig.legend(
            handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3
        )
        fig.text(0.5, 0.012, caption, ha="center", fontsize="small", color=neutral)
        ignored = (
            [("facet-ylim-mismatch", ax) for ax in fig.axes] if independent_y else []
        )
        audits[stem] = fv.audit(fig, scoped_ignore=ignored)
        axes[stem] = [
            {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())} for ax in fig.axes
        ]
        for suffix in ("png", "pdf"):
            path = args.output / f"{stem}.{suffix}"
            metadata = (
                {"Description": caption} if suffix == "png" else {"Subject": caption}
            )
            fig.savefig(path, dpi=240, metadata=metadata)
            outputs.append(path)
        plt.close(fig)

    fig, panels = plt.subplots(3, 2, sharex=True, figsize=(5.5, 6.5))
    for index, (ax, (name, (_, title, unit))) in enumerate(
        zip(panels.flat, METRICS.items(), strict=True)
    ):
        for label, color in colors.items():
            selected = [row for row in training if row["source"] == label]
            ax.plot(
                [row["step"] for row in selected],
                [row[name] for row in selected],
                color=color,
                lw=1,
                label=label,
            )
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left")
        ax.set_ylabel(unit)
        ax.set_xlim(1, 50)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        if index >= 4:
            ax.set_xlabel("Optimizer step")
        if name == "token_reward":
            fv.mark_neutral(ax.axhline(0, color=neutral, lw=0.6, ls=":"))
    save(
        fig,
        "training",
        "Raw steps; one training seed; log-ratio RMS is not full-vocabulary KL.",
        bottom=0.105,
        independent_y=True,
    )

    fig, panels = plt.subplots(2, 2, figsize=(5.5, 4.8))
    for index, (ax, benchmark) in enumerate(zip(panels.flat, BENCHMARKS, strict=True)):
        selected = [row for row in accuracy if row["benchmark"] == benchmark]
        values = [100 * row["mean@16"] for row in selected]
        values += [
            100 * reference_metric(references, role, benchmark, 1)
            for role in reference_styles
        ]
        limits = zoom_limits(values)
        for position, row in enumerate(selected):
            score = 100 * row["mean@16"]
            ax.plot(
                position,
                score,
                marker="o",
                color=colors[row["source"]],
                ls="none",
                label=row["source"],
            )
            ax.annotate(
                f"{score:.2f}",
                (position, score),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
            )
        for role, (style, color) in reference_styles.items():
            fv.mark_neutral(
                ax.axhline(
                    100 * reference_metric(references, role, benchmark, 1),
                    ls=style,
                    color=color,
                    lw=1,
                )
            )
        ax.set_title(f"({chr(97 + index)}) {benchmark}", loc="left")
        ax.set_xticks(range(len(specs)), colors)
        ax.set_xlim(-0.5, len(specs) - 0.5)
        ax.set_ylim(*limits)
        ax.set_ylabel("mean@16 (%)")
        ax.yaxis.set_major_locator(MaxNLocator(4))
    save(
        fig,
        "accuracy",
        "Final student50; zoomed axes; external references are separate runs.",
        True,
        0.19,
        True,
    )

    fig, panels = plt.subplots(2, 2, figsize=(5.5, 4.8))
    for index, (ax, benchmark) in enumerate(zip(panels.flat, BENCHMARKS, strict=True)):
        values = []
        for row in (row for row in accuracy if row["benchmark"] == benchmark):
            scores = [100 * row[f"pass@{k}"] for k in KS]
            ax.plot(
                KS,
                scores,
                color=colors[row["source"]],
                marker="o",
                ms=3,
                lw=1,
                label=row["source"],
            )
            values.extend(scores)
        for role, (style, color) in reference_styles.items():
            scores = [
                100 * reference_metric(references, role, benchmark, k) for k in KS
            ]
            fv.mark_neutral(ax.plot(KS, scores, color=color, ls=style, lw=1)[0])
            values.extend(scores)
        ax.set_title(f"({chr(97 + index)}) {benchmark}", loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, KS)
        ax.set_ylim(*zoom_limits(values))
        ax.set_ylabel("Exact pass@k (%)")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("Samples k")
    save(
        fig,
        "pass-at-k",
        "Question-count estimator; one training seed; independently zoomed axes.",
        True,
        0.19,
        True,
    )

    fig, panels = plt.subplots(1, 3, figsize=(5.5, 2.7))
    for ax, key, title in zip(
        panels[:2],
        ("training_minutes", "final_validation_minutes"),
        ("(a) 50 training steps", "(b) Final validation"),
        strict=True,
    ):
        for position, row in enumerate(resources):
            ax.plot(
                position,
                row[key],
                marker="o",
                ls="none",
                color=colors[row["source"]],
                label=row["source"],
            )
        ax.set_xticks(range(len(specs)), colors, rotation=30, ha="right")
        ax.set_ylabel("minutes")
        ax.set_title(title, loc="left")
    ax = panels[2]
    for row in resources:
        ax.plot(
            range(4),
            [row[f"gpu_{gpu}_peak_gib"] for gpu in range(4)],
            color=colors[row["source"]],
            marker="o",
            ms=3,
            lw=1,
            label=row["source"],
        )
    ax.set_title("(c) Whole-run GPU peaks", loc="left")
    ax.set_xticks(range(4))
    ax.set_xlabel("Physical GPU")
    ax.set_ylabel("GiB, sampled whole card")
    save(
        fig,
        "resources",
        "Observed single runs; GPU peaks include initialization, training and validation.",
        bottom=0.26,
        independent_y=True,
    )

    reference_rows = [
        {
            "source": role,
            "benchmark": benchmark,
            **{
                f"pass@{k}": reference_metric(references, role, benchmark, k)
                for k in KS
            },
        }
        for role in reference_styles
        for benchmark in BENCHMARKS
    ]
    for stem, rows in (
        ("training", training),
        ("accuracy", accuracy),
        ("resources", resources),
        ("references", reference_rows),
    ):
        path = args.output / f"{stem}-plotted.csv"
        write_csv(path, rows)
        outputs.append(path)
    report = {
        "schema": "native-formal-figures-v1",
        "command": [sys.executable, "eval/plot_native_formal.py", *sys.argv[1:]],
        "plotter_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("plot_math_experiment.py")),
        "style_sha256": sha256(skill / "assets/mpl/fvstyle.py"),
        "matplotlib_version": matplotlib.__version__,
        "inputs": inputs,
        "runs": identities,
        "common_recorded_protocol": protocol,
        "colors": colors,
        "training_rows": len(training),
        "accuracy_rows": len(accuracy),
        "question_count": 570,
        "responses_per_run": 9120,
        "axis_limits": axes,
        "audit": audits,
        "audit_scope": (
            "Accuracy/pass axes zoom independently by benchmark; training panels have different metrics; "
            "resource panels separate training and final-validation durations"
        ),
        "limitations": [
            "One training seed per policy; no training-seed confidence bands or convergence claim",
            "Training sampled-action log-ratio RMS is not full-vocabulary KL",
            "External initial-student/teacher references are not paired local step-zero measurements",
            "Whole-card sampled GPU peaks are not kernel/allocator peaks or free KV capacity",
        ],
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (args.output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"figures": len(audits), "audit": audits}))
    require(
        not any(audits.values()),
        "Figure audit reported issues; inspect output and audit.json",
    )


if __name__ == "__main__":
    main()
