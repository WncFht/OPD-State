"""Audit final math runs or completed validation prefixes of a live run.

Raw responses remain in the ignored run directory. This collector needs only the
standard library; it imports the exact estimator directly from its source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("math_eval", ROOT / "verl/verl/trainer/ppo/math_eval.py")
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_identity(row):
    content = {key: row[key] for key in ("data_source", "input", "gts")}
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_steps(text, *, through_step=None):
    # Match Path.read_text() universal-newline behavior while hashing the
    # original bytes separately, including tqdm carriage returns.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if through_step is not None:
        # A live writer can be partway through a later training row.
        text = text[: text.rfind("\n") + 1]
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    rows = {}
    for match in re.finditer(r"step:(\d+) - ([^\n]*)", text):
        step = int(match[1])
        if through_step is not None and step > through_step:
            continue
        values = {key: float(value) for key, value in re.findall(r"([\w/@.()-]+):(-?[0-9.eE+\-]+)", match[2])}
        if not values or not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Invalid training metrics at step {step}")
        if step in rows:
            raise ValueError(f"Duplicate training log row at step {step}")
        rows[step] = {"step": step, **values}
    return text, rows


def read_steps(log):
    return parse_steps(log.read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        # Git normalizes tracked text to LF; hashes must survive a fresh checkout.
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(run, output, *, steps, interval, require_full=True, snapshot=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if steps < 1 or interval < 1 or (snapshot and steps % interval):
        raise ValueError("Snapshot steps must be positive completed validation intervals")
    if snapshot and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory for each immutable snapshot")
    logs = list((run / "logs").glob("*.log"))
    if len(logs) != 1:
        raise ValueError(f"Expected exactly one training log under {run}/logs")
    observed_log = logs[0].read_bytes()
    if snapshot:
        observed_log = observed_log[: observed_log.rfind(b"\n") + 1]
    text, training = parse_steps(observed_log.decode(), through_step=steps if snapshot else None)
    if set(training) != set(range(1, steps + 1)):
        raise ValueError(f"Training log does not contain exactly steps 1..{steps}")
    if not snapshot and "Final validation metrics:" not in text:
        raise ValueError("Run has no final validation success marker")
    if list(run.glob("checkpoints/**/global_step_*")):
        raise ValueError("Run unexpectedly contains saved checkpoints")
    revision = (run / "source/COMMIT").read_text().strip()
    required = set(range(interval, steps + 1, interval)) | {steps}
    paths = {int(path.stem): path for path in (run / "validation").glob("*.jsonl")}
    if snapshot:
        # Later JSONL files may still be open; only committed training rows
        # through the requested validation step authorize reading responses.
        paths = {step: path for step, path in paths.items() if step <= steps}
    if set(paths) != required:
        raise ValueError(f"Validation steps mismatch: {sorted(paths)} != {sorted(required)}")
    output.mkdir(parents=True, exist_ok=True)
    validations, timeseries = {}, []
    for step, path in sorted(paths.items()):
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        if any(int(row["step"]) != step for row in raw):
            raise ValueError(f"Validation JSONL contains the wrong step: {path}")
        metrics, questions = MATH.exact_math_metrics(
            [row["data_source"] for row in raw],
            [row["validation_uid"] for row in raw],
            [row["score"] for row in raw],
            samples_per_question=16,
            required_sources=MATH.MATH_BENCHMARK_COUNTS,
            required_counts=MATH.MATH_BENCHMARK_COUNTS if require_full else None,
        )
        identities = {}
        for row in raw:
            key = str(row["data_source"]), str(row["validation_uid"])
            identity = question_identity(row)
            if identities.setdefault(key, identity) != identity:
                raise ValueError(f"Replicas with the same validation UID have different questions: {key}")
        for question in questions:
            question["question_sha256"] = identities[question["data_source"], question["uid"]]
        if len({row["question_sha256"] for row in questions}) != len(questions):
            raise ValueError("Validation contains duplicate question content under different UIDs")
        for key, value in metrics.items():
            if key not in training[step] or abs(value - training[step][key]) > 1e-8:
                raise ValueError(f"Recomputed metric disagrees with trainer: step={step} {key}")
        write_csv(output / f"questions-step{step}.csv", questions)
        validations[str(step)] = {
            "responses": len(raw),
            "questions": len(questions),
            "response_jsonl_sha256": sha256(path),
            "questions_csv_sha256": sha256(output / f"questions-step{step}.csv"),
            "metrics": metrics,
        }
        timeseries.append({"step": step, **metrics})
    # Keep each artifact below 1,000 lines, including multi-epoch experiments.
    train_rows = [training[step] for step in range(1, steps + 1)]
    for start in range(0, len(train_rows), 900):
        write_csv(output / f"training-{start + 1}-{min(start + 900, steps)}.csv", train_rows[start : start + 900])
    write_csv(output / "validation-timeseries.csv", timeseries)
    urls = re.findall(r"https://wandb.ai/[^\s\x1b]+/runs/[a-z0-9]+", text)
    summary = {
        "run_name": run.name,
        "run_directory": str(run),
        "source_commit": revision,
        "training_steps": steps,
        "samples_per_question": 16,
        "full_benchmarks": require_full,
        "checkpoints_saved": False,
        "wandb_url": urls[-1] if urls else None,
        "training_log_sha256": hashlib.sha256(observed_log).hexdigest(),
        "training_log_observed_bytes": len(observed_log),
        "collection_scope": "through_step_snapshot" if snapshot else "final_validation",
        "final_validation_marker_verified": not snapshot,
        "collector_source_sha256": sha256(Path(__file__).resolve()),
        "validations": validations,
    }
    # One JSON per validation keeps long-run summaries bounded too.
    for step, record in validations.items():
        (output / f"validation-step{step}.json").write_text(json.dumps(record, indent=2) + "\n")
    summary["validation_steps"] = sorted(map(int, validations))
    del summary["validations"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Allow fewer questions, still requiring all three labels")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Audit only through --steps; require its complete validation but not run termination",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.run,
                args.output,
                steps=args.steps,
                interval=args.interval,
                require_full=not args.smoke,
                snapshot=args.snapshot,
            )
        )
    )


if __name__ == "__main__":
    main()
