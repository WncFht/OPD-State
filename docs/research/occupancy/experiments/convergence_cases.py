"""Exact convergence checks for six rollout sources in a shared-logit tree.

Root action 1 reaches L, action 0 reaches R. At the second step action 1 is
successful. Teacher success is 0.66, initial student success is 0.1. Every
student conditional probability shares the same logit. All expectations are
enumerated; no LLM, GPU, or random Monte Carlo results are reported.
"""

import hashlib
from pathlib import Path

import numpy as np
from exact_checks import logit, sigmoid

TEACHER = np.array([0.3, 0.9, 0.1])  # root, R, L
INITIAL = 0.1
MIX = 0.5
CASES = (
    "initial_student",
    "teacher",
    "initial_fusion",
    "online_fusion",
    "batch_online",
    "student_then_teacher",
    "teacher_then_student",
    "online_student",
    "batch_initial",
    "initial_then_teacher",
    "teacher_then_initial",
)


def path_distribution(probabilities):
    root, right, left = probabilities
    return np.array(
        [
            [(1 - root) * (1 - right), (1 - root) * right],
            [root * (1 - left), root * left],
        ]
    )


def behavior(name, z):
    current = float(sigmoid(z))
    old = np.full(3, INITIAL)
    student = np.full(3, current)
    if name == "initial_student":
        return path_distribution(old)
    if name == "teacher":
        return path_distribution(TEACHER)
    if name == "initial_fusion":
        return path_distribution(sigmoid((1 - MIX) * logit(old) + MIX * logit(TEACHER)))
    if name == "online_fusion":
        return path_distribution(sigmoid((1 - MIX) * z + MIX * logit(TEACHER)))
    if name in {"batch_initial", "batch_online"}:
        base = old if name == "batch_initial" else student
        return (1 - MIX) * path_distribution(base) + MIX * path_distribution(TEACHER)
    if name in {"student_then_teacher", "initial_then_teacher"}:
        root = INITIAL if name == "initial_then_teacher" else current
        return path_distribution([root, TEACHER[1], TEACHER[2]])
    if name in {"teacher_then_student", "teacher_then_initial"}:
        suffix = INITIAL if name == "teacher_then_initial" else current
        return path_distribution([TEACHER[0], suffix, suffix])
    if name == "online_student":
        return path_distribution(student)
    raise ValueError(name)


def gradient(name, z, raw=False):
    """Compute score expectation with frozen data, not total dF(theta,theta)."""
    p = float(sigmoid(z))
    measure = behavior(name, z)
    if not raw:
        weights = np.array([1, measure[0].sum(), measure[1].sum()])
        return float(np.sum(weights * p * (1 - p) * (z - logit(TEACHER))))
    value = 0.0
    for first in (0, 1):
        for second in (0, 1):
            for action, q in ((first, TEACHER[0]), (second, TEACHER[first + 1])):
                error = np.log(p / q) if action else np.log((1 - p) / (1 - q))
                value += measure[first, second] * error * (action - p)
    return float(value)


def bisect(f, low, high):
    f_low = f(low)
    assert f_low * f(high) < 0, (low, high)
    for _ in range(70):
        middle = (low + high) / 2
        if f_low * f(middle) <= 0:
            high = middle
        else:
            low, f_low = middle, f(middle)
    return (low + high) / 2


def roots_in_interval(f, low=-12, high=12):
    grid = np.linspace(low, high, 1201)
    values = [f(z) for z in grid]
    roots = [float(z) for z, value in zip(grid, values, strict=True) if value == 0]
    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] < 0:
            roots.append(bisect(f, grid[i], grid[i + 1]))
    return sorted(set(round(z, 13) for z in roots))


def static_square_loss(name, z):
    p = float(sigmoid(z))
    measure = behavior(name, z)
    value = 0.0
    for first in (0, 1):
        for second in (0, 1):
            for action, q in ((first, TEACHER[0]), (second, TEACHER[first + 1])):
                error = np.log(p / q) if action else np.log((1 - p) / (1 - q))
                value += 0.5 * measure[first, second] * error**2
    return value


def categorical_natural_flow_check():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.7, 0.2, 0.1])
    b = np.array([0.1, 0.8, 0.1])
    error = np.log(p / q)
    c = float(b @ error)
    dp = -b * error + p * c
    derivative = float(-np.sum(q / p * dp))
    identity = float(np.sum(b * (q / p - 1) * error))
    assert abs(dp.sum()) < 1e-14
    assert abs(derivative - identity) < 1e-14 and derivative < 0
    h = 1e-6

    def f(x):
        return float(np.sum(q * np.log(q / x)))

    numeric = (f(p + h * dp) - f(p - h * dp)) / (2 * h)
    assert abs(numeric - derivative) < 1e-8
    return {
        "lyapunov_derivative": derivative,
        "finite_difference_error": abs(numeric - derivative),
    }


def raw_tabular_gradient(z, q, b):
    p = np.exp(z - z.max())
    p /= p.sum()
    error = np.log(p / q)
    return b * error - p * float(b @ error)


def online_teacher_linearization_check():
    q = np.array([0.2, 0.3, 0.5])
    z = np.log(q)
    fisher = np.diag(q) - np.outer(q, q)
    errors = []
    for mixing in (0.0, 0.25, 0.75, 1.0):

        def vector(zz):
            blend = np.exp((1 - mixing) * zz + mixing * np.log(q))
            blend /= blend.sum()
            return raw_tabular_gradient(zz, q, blend)

        h = 1e-5
        derivative = np.column_stack(
            [(vector(z + h * e) - vector(z - h * e)) / (2 * h) for e in np.eye(3)]
        )
        errors.append(float(np.max(np.abs(derivative - fisher))))
    assert max(errors) < 1e-8
    return errors


def positive_feedback_check():
    roots = {}
    for mixing in (0.0, 0.2, 0.5):

        def f(z):
            return z - 3 * np.tanh((1 - mixing) * z / 2)

        found = roots_in_interval(f)
        expected = 3 if mixing < 1 / 3 else 1
        assert len(found) == expected
        roots[str(mixing)] = [
            {
                "z": z,
                "p": float(sigmoid(z)),
                "restoring_slope": float(
                    1 - 1.5 * (1 - mixing) / np.cosh((1 - mixing) * z / 2) ** 2
                ),
            }
            for z in found
        ]
    return roots


def tree_teacher_linearization_check():
    """Check the occupancy-dependent Jacobian with independent state logits."""
    q = TEACHER
    expected = np.diag(np.array([1, 1 - q[0], q[0]]) * q * (1 - q))
    errors = {}
    for name in (
        "online_student",
        "online_fusion",
        "batch_online",
        "student_then_teacher",
        "teacher_then_student",
        "teacher",
    ):

        def vector(z):
            p = sigmoid(z)
            if name == "online_fusion":
                measure = path_distribution(sigmoid((1 - MIX) * z + MIX * logit(q)))
            elif name == "batch_online":
                measure = (1 - MIX) * path_distribution(p) + MIX * path_distribution(q)
            elif name == "student_then_teacher":
                measure = path_distribution([p[0], q[1], q[2]])
            elif name == "teacher_then_student":
                measure = path_distribution([q[0], p[1], p[2]])
            else:
                measure = path_distribution(q if name == "teacher" else p)
            g = np.zeros(3)
            for i in (0, 1):
                for j in (0, 1):
                    for state, action in ((0, i), (i + 1, j)):
                        e = (
                            np.log(p[state] / q[state])
                            if action
                            else np.log((1 - p[state]) / (1 - q[state]))
                        )
                        g[state] += measure[i, j] * e * (action - p[state])
            return g

        z = logit(q)
        h = 1e-5
        derivative = np.column_stack(
            [(vector(z + h * e) - vector(z - h * e)) / (2 * h) for e in np.eye(3)]
        )
        errors[name] = float(np.max(np.abs(derivative - expected)))
    assert max(errors.values()) < 1e-8
    return errors


def run_checks():
    cases = {}
    for name in CASES:
        entry = {}
        for raw, label in ((False, "conditional_kl"), (True, "raw_sampled")):

            def f(z):
                return gradient(name, z, raw=raw)

            roots = roots_in_interval(f)
            assert len(roots) == 1, (name, label, roots)
            root = roots[0]
            slope = (f(root + 1e-5) - f(root - 1e-5)) / 2e-5
            assert slope > 0 and abs(f(root)) < 1e-11
            final_errors = []
            for start in (0.1, 0.5, 0.9):
                z = float(logit(start))
                for _ in range(1800):
                    z -= 0.05 * f(z)
                final_errors.append(abs(z - root))
            assert max(final_errors) < 1e-9, (name, label, final_errors)
            entry[label] = {
                "z": root,
                "p": float(sigmoid(root)),
                "gradient": f(root),
                "derivative": slope,
                "max_logit_error_after_1800_steps": max(final_errors),
                "roots_on_minus12_plus12": len(roots),
            }
        cases[name] = entry
    # The closed form is independent of the last action generator for H=2.
    assert (
        cases["teacher"]["conditional_kl"]["p"]
        == cases["teacher_then_student"]["conditional_kl"]["p"]
    )
    assert (
        cases["online_student"]["conditional_kl"]["p"]
        == cases["student_then_teacher"]["conditional_kl"]["p"]
    )
    for name in ("initial_student", "teacher", "initial_fusion", "batch_initial"):
        for z in (-1.0, 0.2, 1.5):
            h = 1e-5
            derivative = (
                static_square_loss(name, z + h) - static_square_loss(name, z - h)
            ) / (2 * h)
            assert abs(derivative - gradient(name, z, raw=True)) < 1e-8
    target = cases["online_student"]["conditional_kl"]["z"]
    handoff_errors = []
    for entry in cases.values():
        for label in ("conditional_kl", "raw_sampled"):
            z = entry[label]["z"]
            for _ in range(1800):
                z -= 0.05 * gradient("online_student", z)
            handoff_errors.append(abs(z - target))
    assert max(handoff_errors) < 1e-9
    return {
        "metadata": {
            "device": "cpu",
            "dtype": "float64",
            "expectations": "exact enumeration",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "step_size": 0.05,
            "steps": 1800,
            "teacher_root_right_left": TEACHER.tolist(),
            "initial": INITIAL,
            "mix": MIX,
            "raw_root_search_interval": [-12, 12],
        },
        "cases": cases,
        "handoff_max_logit_error": max(handoff_errors),
        "natural_gradient": categorical_natural_flow_check(),
        "online_teacher_linearization_errors": online_teacher_linearization_check(),
        "tree_teacher_linearization_errors": tree_teacher_linearization_check(),
        "positive_feedback_fusion_roots": positive_feedback_check(),
    }


if __name__ == "__main__":
    import json

    result = run_checks()
    output = Path(__file__).resolve().parents[1] / "results/convergence-cases.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
