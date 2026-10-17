"""A finite-budget diagnostic for fixed-cache OPD, shared by trainer and analysis.

An observed plateau does not establish a zero gradient, alignment with the
teacher, or an asymptotic limit. Thresholds must be declared before the run.
"""

from __future__ import annotations

import math


def assess_windows(
    windows, training_steps, *, minimum_steps=300, points=4, accuracy_range=0.01, rms_span=0.05, grad_span=0.10
):
    if minimum_steps < 300 or minimum_steps % 50 or points < 4:
        raise ValueError("Plateau assessment requires >=300 steps and >=4 validation points")
    selected = windows[-points:]
    result = {
        "criterion": {
            "minimum_steps": minimum_steps,
            "validation_points": points,
            "macro_mean_range": accuracy_range,
            "relative_rms_span": rms_span,
            "relative_gradient_span": grad_span,
        },
        "training_steps": training_steps,
        "validation_points": len(windows),
        "windows": selected,
        "status": "insufficient_observation",
    }
    if training_steps < minimum_steps or len(windows) < points:
        return result
    if any(b["step"] - a["step"] != 50 for a, b in zip(selected[:-1], selected[1:], strict=True)):
        raise ValueError("Plateau assessment requires validation every 50 steps")
    if selected[-1]["step"] != training_steps:
        raise ValueError("Plateau assessment must include the current validation step")
    values = [
        float(row[key])
        for row in selected
        for key in ("macro_mean@16", "token_weighted_log_ratio_rms", "mean_grad_norm")
    ]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Plateau observations must be finite and nonnegative")

    def relative_span(values):
        return (max(values) - min(values)) / max(sum(values) / len(values), 1e-12)

    accuracy = [row["macro_mean@16"] for row in selected]
    observed_accuracy_range = max(accuracy) - min(accuracy)
    observed_rms_span = relative_span([row["token_weighted_log_ratio_rms"] for row in selected])
    observed_grad_span = relative_span([row["mean_grad_norm"] for row in selected])
    passed = (
        observed_accuracy_range <= accuracy_range and observed_rms_span <= rms_span and observed_grad_span <= grad_span
    )
    result.update(
        status="observed_plateau" if passed else "criterion_not_met",
        observed_accuracy_range=observed_accuracy_range,
        observed_relative_rms_span=observed_rms_span,
        observed_relative_gradient_span=observed_grad_span,
    )
    return result


class FixedCachePlateau:
    def __init__(self, minimum_steps=300):
        assess_windows([], 0, minimum_steps=minimum_steps)
        self.minimum_steps = minimum_steps
        self.windows = []
        self.step = 0
        self.tokens = self.squares = self.gradients = self.count = 0

    def add_step(self, step, metrics):
        if step != self.step + 1:
            raise ValueError("Plateau monitor requires contiguous training steps from 1")
        n, rms, gradient = [
            float(metrics[key]) for key in ("response_length/mean", "opd/log_ratio_rms", "actor/grad_norm")
        ]
        if not all(math.isfinite(value) and value >= 0 for value in (n, rms, gradient)) or n == 0:
            raise ValueError("Plateau training metrics must be finite with nonempty responses")
        self.tokens += n
        self.squares += rms * rms * n
        self.gradients += gradient
        self.count += 1
        self.step = step

    def observe(self, step, metrics):
        if step != self.step or step % 50 or self.count != 50:
            raise ValueError("Plateau observations must follow exactly 50 new training steps")
        score = float(metrics["eval/macro/mean@16"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Plateau validation requires a finite macro mean@16 in [0,1]")
        self.windows.append(
            {
                "step": step,
                "macro_mean@16": score,
                "token_weighted_log_ratio_rms": math.sqrt(self.squares / self.tokens),
                "mean_grad_norm": self.gradients / self.count,
            }
        )
        self.tokens = self.squares = self.gradients = self.count = 0
        return assess_windows(self.windows, step, minimum_steps=self.minimum_steps)
