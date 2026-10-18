"""GPU placement changes throughput of the same 50/50 trajectory mixture.

Ledger: each bar is one physical layout at global batch64/cap128, CUDA graphs,
identical prompt IDs and exact 50/50 whole-trajectory routing. The bar is the
mean of two warm repeats; whiskers show the observed min/max, not a confidence
interval. x is thousands of completed response tokens per wall-clock second.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    home = Path(
        os.environ.get("FIGURE_VIZ_HOME", Path(__file__).resolve().parents[2] / "dotfiles/.agents/skills/figure-viz")
    )
    sys.path.insert(0, str(home / "assets/mpl"))
    import fvstyle as fv
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(args.input.open()))
    layout = [
        ("overlap", "1", "1 GPU: two models"),
        ("overlap", "4", "4 GPUs: two models / GPU"),
        ("split", "4", "4 GPUs: two GPUs / model"),
    ]
    selected = [
        next(
            r
            for r in rows
            if r["policy"] == "batch_mix"
            and r["batch"] == "64"
            and r["cap"] == "128"
            and r["natural_eos"] == "False"
            and r["layout"] == name
            and r["replicas"] == replicas
        )
        for name, replicas, _ in layout
    ]
    current = fv.apply_rc("paper")
    colors = fv.series_colors("paper", 3)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    values = [float(r["tokens_per_second_mean"]) / 1000 for r in selected]
    for i, (row, value) in enumerate(zip(selected, values, strict=True)):
        lo = value - float(row["tokens_mean"]) / float(row["seconds_max"]) / 1000
        hi = float(row["tokens_mean"]) / float(row["seconds_min"]) / 1000 - value
        ax.barh(i, value, height=0.55, color=colors[i])
        ax.hlines(i, value - max(lo, 0), value + max(hi, 0), color=fv.HUES["vgray"], linewidth=1.2)
        ax.vlines([value - max(lo, 0), value + max(hi, 0)], i - 0.06, i + 0.06, color=fv.HUES["vgray"], linewidth=1.2)
        ax.text(value + 0.4, i, f"{value:.2f}", va="center", color=fv.HUES["vgray"])
    ax.set_yticks(range(3), [x[2] for x in layout])
    ax.invert_yaxis()
    ax.set_xlim(0, 18)
    ax.set_xlabel("Response throughput (thousand token/s)")
    ax.set_ylabel("Physical model placement")
    ax.set_title("Same 50/50 mixture, different GPU placement")
    fv.apply_frame(ax, current.frame)
    fv.apply_grid(ax, current.grid)
    fig.tight_layout()
    fig.canvas.draw()
    # Category ticks name the bars; whisker components are not extra series.
    problems = fv.audit(fig, ignore=["legend-coverage", "palette-overflow"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fv.save(fig, str(args.output.with_suffix(suffix)))
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "ledger": __doc__,
                "rows": selected,
                "audit": problems,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"audits": problems, "thousand_tokens_per_second": values}))


if __name__ == "__main__":
    main()
