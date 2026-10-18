"""Plot paired exact-prefix diagnostics from complete native training traces.

Ledger: each series compares Current with one candidate on the same 3,200
prompt/visit/sample identities. Left x is optimizer step; y is mean common
response-prefix length in tokens. Right x is that length on a log1p scale;
y is its empirical cumulative probability. Colors identify candidate policies
in both panels. These are lexical diagnostics with role-dependent sampling
seeds, not semantic coverage, KL, or a causal estimate of source effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "initial_student_fresh": "Initial student",
    "teacher_fresh": "Teacher",
    "student_teacher_fusion": "Current/teacher fusion",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return int(row["step"]), row["prompt_id"], int(row["visit"]), int(row["sample"])


def verify_raw(summary, rows):
    """Use vectorized first mismatch, independently of the production counter."""
    indexed = {identity(row): row for row in rows}
    roots, checked = [], 0
    for side in ("reference", "candidate"):
        run = Path(summary[side]["run"])
        manifests = list(run.glob("checkpoints/**/native-trajectories/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Raw verification requires exactly one trajectory manifest")
        folder = manifests[0].parent
        for name, expected in summary[side]["files_sha256"].items():
            path = run / name if name in {"command.json", "source-manifest.json"} else folder / name
            if digest(path) != expected:
                raise ValueError(f"Raw prefix evidence changed: {path}")
            checked += 1
        roots.append(folder)
    seen = set()
    for step in range(1, summary["steps"] + 1):
        batches = [
            [json.loads(line) for line in (root / f"step-{step}.jsonl").read_text().splitlines()] for root in roots
        ]
        left = {(r["request"]["prompt_id"], r["request"]["visit"], r["request"]["sample"]): r for r in batches[0]}
        if len(left) != 64 or len(batches[1]) != 64:
            raise ValueError("Raw prefix evidence has an incomplete or repeated global batch")
        for other in batches[1]:
            request = other["request"]
            key = request["prompt_id"], request["visit"], request["sample"]
            row_key = (step, *key)
            original, recorded = left[key], indexed[row_key]
            if row_key in seen or original["request"]["prompt_ids"] != request["prompt_ids"]:
                raise ValueError("Raw prefix evidence repeats or changes a paired prompt")
            seen.add(row_key)
            a, b = np.asarray(original["token_ids"]), np.asarray(other["token_ids"])
            limit = min(len(a), len(b))
            mismatch = np.flatnonzero(a[:limit] != b[:limit])
            common = int(mismatch[0]) if mismatch.size else limit
            shared = min(common + 1, limit)
            expected = {
                "reference_length": len(a),
                "candidate_length": len(b),
                "common_response_prefix_tokens": common,
                "shared_sampled_states": shared,
            }
            if any(int(recorded[name]) != value for name, value in expected.items()):
                raise ValueError("Paired CSV disagrees with independent raw-token prefix counting")
            expected_ratio = shared / (len(a) + len(b) - shared)
            if not math.isclose(float(recorded["sampled_state_jaccard"]), expected_ratio, abs_tol=1e-15):
                raise ValueError("Paired CSV Jaccard disagrees with independently counted raw states")
    if seen != set(indexed):
        raise ValueError("Raw prefix verification did not cover every plotted pair")
    return {"responses_recounted": len(seen), "raw_files_hash_verified": checked, "algorithm": "NumPy first mismatch"}


def load(directory, *, check_raw):
    result, hashes, raw_audits = {}, {}, {}
    for case in CASES:
        summary_path, csv_path = directory / case / "summary.json", directory / case / "paired-prefixes.csv"
        summary = json.loads(summary_path.read_text())
        if digest(csv_path) != summary["paired_csv_sha256"]:
            raise ValueError("Prefix CSV does not match its analysis receipt")
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if summary["steps"] != 50 or len(rows) != 3200 or len({identity(row) for row in rows}) != 3200:
            raise ValueError("Plot requires all fifty steps and 3,200 distinct matched responses")
        values = [int(row["common_response_prefix_tokens"]) for row in rows]
        if any(value < 0 for value in values) or not math.isclose(
            sum(values) / len(values), summary["paired"]["mean_common_response_prefix_tokens"], abs_tol=1e-12
        ):
            raise ValueError("Prefix lengths disagree with the summary")
        for step in range(1, 51):
            selected = [value for row, value in zip(rows, values, strict=True) if int(row["step"]) == step]
            if len(selected) != 64 or not math.isclose(
                sum(selected) / 64, summary["by_step"][str(step)]["mean_common_response_prefix_tokens"], abs_tol=1e-12
            ):
                raise ValueError("Per-step prefix lengths disagree with the complete global batch")
        if check_raw:
            raw_audits[case] = verify_raw(summary, rows)
        for path in (summary_path, csv_path):
            hashes[str(path)] = digest(path)
        result[case] = (summary, np.asarray(values))
    return result, hashes, raw_audits


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-raw", action="store_true", help="Recount every pair from original token records")
    args = parser.parse_args()
    result, hashes, raw_audits = load(args.input.resolve(), check_raw=args.verify_raw)
    skill = Path(os.environ.get("FIGURE_VIZ_HOME", ROOT.parent / "dotfiles/.agents/skills/figure-viz"))
    sys.path.insert(0, str(skill / "assets/mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper", fontsize=8)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45))
    step_rows, cdf_rows = [], []
    for (case, label), color in zip(CASES.items(), fv.series_colors("paper", 4)[1:], strict=True):
        summary, values = result[case]
        steps = list(range(1, 51))
        means = [summary["by_step"][str(step)]["mean_common_response_prefix_tokens"] for step in steps]
        axes[0].plot(steps, means, label=label, color=color, lw=1)
        step_rows.extend(
            {"case": case, "step": step, "mean_prefix_tokens": mean} for step, mean in zip(steps, means, strict=True)
        )
        unique, counts = np.unique(values, return_counts=True)
        cdf = np.cumsum(counts) / len(values)
        axes[1].step(unique, cdf * 100, where="post", label=label, color=color, lw=1.4)
        cdf_rows.extend(
            {"case": case, "prefix_tokens": int(x), "responses": int(n), "cdf": float(y)}
            for x, n, y in zip(unique, counts, cdf, strict=True)
        )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Mean common prefix (tokens)",
        title="64 paired prompts per step",
        xlim=(1, 50),
        ylim=(0, None),
    )
    axes[1].set_xscale("function", functions=(np.log1p, np.expm1))
    axes[1].set(
        xlabel="Common prefix (tokens; log1p scale)",
        ylabel="Cumulative responses (%)",
        title="All 3,200 paired prompts",
        xlim=(0, 80),
        ylim=(0, 100),
    )
    axes[1].set_xticks([0, 2, 8, 32, 80], labels=["0", "2", "8", "32", "80"])
    for ax in axes:
        fv.apply_frame(ax, "open")
        fv.apply_grid(ax, "y-only")
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3)
    caption = "Current reference; role-dependent seeds; exact tokens, not semantic coverage."
    fig.text(0.5, 0.012, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize=8)
    audit = fv.audit(fig, scoped_ignore=[("facet-ylim-mismatch", ax) for ax in axes])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = args.output / f"training-prefixes.{suffix}"
        fig.savefig(path, dpi=240, metadata={"Description" if suffix == "png" else "Subject": caption})
        paths.append(path)
    plt.close(fig)
    for name, rows in (("per-step.csv", step_rows), ("ecdf.csv", cdf_rows)):
        path = args.output / name
        write_csv(path, rows)
        paths.append(path)
    receipt = {
        "scope": "Exact paired training prefixes, not semantic coverage or KL",
        "source_sha256": digest(Path(__file__)),
        "style_sha256": digest(skill / "assets/mpl/fvstyle.py"),
        "input_sha256": hashes,
        "output_sha256": {path.name: digest(path) for path in paths},
        "raw_verification": raw_audits,
        "audit": audit,
        "audit_exemption": "Different y quantities: token mean and cumulative percentage",
        "comparison_limit": "Role-dependent request seeds; no common-random-number or source-only causal claim",
    }
    (args.output / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"audit": audit, "raw_verification": raw_audits}))
    if audit:
        raise ValueError("Figure audit reported issues; inspect the rendered figure and receipt")


if __name__ == "__main__":
    main()
