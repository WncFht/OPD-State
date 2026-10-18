"""Collect audited KL coverage and phase-specific sampled GPU resources.

Native acceptance must run first. This consumes completed local evidence and
does not generate responses, change model weights, or initialize CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_validation_kl import BENCHMARKS, audit, digest


def canonical_digest(value):
    """Ignore JSON whitespace/key order and equivalent integer-number spelling."""

    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, list):
            return [normalize(value) for value in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return hashlib.sha256(
        json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resource_profile(report, status, gpu_path):
    """Align recorded two-second whole-card samples with the scorer's epochs."""
    started = status.get("started")
    begin, end = report["scoring_started_epoch_s"], report["scoring_finished_epoch_s"]
    if (
        status.get("state") != "finished"
        or status.get("exit_code") != 0
        or started is None
        or not all(math.isfinite(value) for value in (started, begin, end, status["finished"]))
        or not started <= begin < end <= status["finished"]
    ):
        raise ValueError("KL resource alignment requires a successful run and preserved scoring/run epochs")
    first, last = begin - started, end - started
    whole, scoring = defaultdict(list), defaultdict(list)
    with Path(gpu_path).open() as stream:
        for row in csv.DictReader(stream):
            gpu, used, elapsed = int(row["gpu"]), int(row["used_mib"]), float(row["elapsed_s"])
            if gpu not in range(4) or used < 0 or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("KL resource telemetry contains an invalid physical GPU, memory or time")
            whole[gpu].append(used)
            if first <= elapsed <= last:
                scoring[gpu].append(used)
    if set(whole) != set(range(4)):
        raise ValueError("KL resource telemetry does not cover all four physical GPUs")
    return {
        "scope": "Two-second sampled whole-card usage, not allocator/kernel peaks or KV capacity",
        "run_started_epoch_s": started,
        "scoring_epoch_s": [begin, end],
        "scoring_interval_elapsed_s": [first, last],
        "scoring_wall_seconds": report["metrics"]["validation_kl/performance/scoring_seconds"],
        "gpu_csv_sha256": digest(gpu_path),
        "whole_run_sampled_peak_mib": {str(gpu): max(whole[gpu]) for gpu in range(4)},
        "scoring_sample_counts": {str(gpu): len(scoring.get(gpu, ())) for gpu in range(4)},
        "scoring_sampled_peak_mib": {str(gpu): max(scoring[gpu]) if scoring.get(gpu) else None for gpu in range(4)},
        "scoring_samples_cover_all_gpus": all(scoring.get(gpu) for gpu in range(4)),
        "worker_timings": report["worker_timings"],
        "timing_scope": (
            "Worker API wall times include management/communication; teacher forward excludes its head callback"
        ),
    }


def collect(run, output, *, bank=None, full=False, replicates=5000):
    run, output = Path(run).resolve(), Path(output).resolve()
    native_path = output / "native-acceptance.json"
    native = json.loads(native_path.read_text())
    step = native["final_student_version"]
    if native["math_summary"]["full_benchmarks"] is not full or native["math_summary"]["samples_per_question"] != 16:
        raise ValueError("KL collection scope must match native full/engineering eval16 acceptance")
    report_path = run / "validation/kl" / f"kl-step-{step}.json"
    tokens_path = run / "validation/kl" / f"kl-tokens-step-{step}.jsonl"
    report = json.loads(report_path.read_text())
    if report["student_version"] != step:
        raise ValueError("KL and native acceptance refer to different final student versions")
    independent = audit(
        report_path,
        tokens_path,
        accuracy_path=run / "validation" / f"{step}.jsonl",
        bank_path=bank,
        required_counts=BENCHMARKS if full else None,
        samples=16,
        workers=4,
        replicates=replicates,
    )
    main = [row for row in report["rows"] if row["state_source"] == "on_validation_states"]
    if len(main) != native["evaluation_responses"] or (full and len(main) != 9120):
        raise ValueError("KL response coverage differs from the completed accuracy evaluation")
    status_path, gpu_path = run / "status.json", run / "gpu.csv"
    status = json.loads(status_path.read_text())
    resources = resource_profile(report, status, gpu_path)
    if resources["whole_run_sampled_peak_mib"] != native["whole_card_peak_mib"]:
        raise ValueError("KL resource telemetry disagrees with the native acceptance peaks")
    lengths = defaultdict(list)
    positions = defaultdict(int)
    for row in report["rows"]:
        lengths[row["state_source"]].append(row["response_length"])
        positions[row["state_source"]] += row["evaluated_positions"]
    coverage = {
        source: {
            "responses": len(values),
            "response_tokens": sum(values),
            "evaluated_positions": positions[source],
            "min_length": min(values),
            "max_length": max(values),
            "mean_length": math.fsum(values) / len(values),
        }
        for source, values in lengths.items()
    }
    summary = {key: value for key, value in report.items() if key != "rows"}
    summary.update(
        scope="All effective validation positions; fixed-bank positions are a separate state-source diagnostic",
        coverage=coverage,
        original_report_sha256=digest(report_path),
        original_report_path=str(report_path),
    )
    receipt = {
        "schema": "validation-kl-gate-collection-v1",
        "scope": "Complete evidence/coverage audit; no independent neural-model forward recomputation",
        "student_version": step,
        "full_benchmarks": full,
        "responses": len(main),
        "all_validation_positions": coverage["on_validation_states"]["evaluated_positions"]
        == coverage["on_validation_states"]["response_tokens"],
        "effective_vocab_size": report["effective_vocab_size"],
        "coverage": coverage,
        "scoring_samples_cover_all_gpus": resources["scoring_samples_cover_all_gpus"],
        "source_sha256": digest(Path(__file__)),
        "audit_source_sha256": digest(Path(__file__).with_name("audit_validation_kl.py")),
        "inputs_sha256": {str(path): digest(path) for path in (report_path, tokens_path, status_path, gpu_path)},
        "native_acceptance_path": str(native_path),
        "native_acceptance_canonical_sha256": canonical_digest(native),
    }
    if bank is not None:
        receipt["fixed_bank_sha256"] = digest(bank)
    for filename, value in (
        ("kl-independent-audit.json", independent),
        ("kl-resource-profile.json", resources),
        ("kl-summary.json", summary),
        ("kl-collection.json", receipt),
    ):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    result = collect(args.run, args.output, bank=args.bank, full=args.full, replicates=args.replicates)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("student_version", "full_benchmarks", "responses", "all_validation_positions")
            }
        )
    )


if __name__ == "__main__":
    main()
