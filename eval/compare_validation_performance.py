"""Compare completed full gates with explicit workload and training controls.

Whole-gate times are observations, not fixed-checkpoint causal measurements.
The separate fixed-token and fixed-work generation benchmarks isolate changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(baseline, optimized, workflow, archived):
    baseline, optimized = Path(baseline), Path(optimized)
    old_audit = read(baseline / "performance-audit.json")
    old_native, new_native = [read(path / "native-acceptance.json") for path in (baseline, optimized)]
    native = new_native
    if native["math_summary"]["validation_steps"] != [2] or not native["math_summary"]["full_benchmarks"]:
        raise ValueError("Expected the completed two-update, full eval16 engineering gate")
    run = Path(native["math_summary"]["run_directory"])
    report_path = run / "validation/kl/kl-step-2.json"
    baseline_report_path = Path(old_native["math_summary"]["run_directory"]) / "validation/kl/kl-step-2.json"
    report = read(report_path)
    status = read(run / "status.json")
    start, end = read(workflow)["full_started_epoch_s"], read(archived)["finished_epoch_s"]
    if (
        read(archived)["state"] != "passed"
        or status["exit_code"]
        or not start <= status["started"] < status["finished"] <= end
    ):
        raise ValueError("Full workflow timestamps require successful training and independent audits")
    trace = next(run.glob("checkpoints/**/native-trajectories/batches.jsonl"))
    batches = [row for row in map(json.loads, trace.read_text().splitlines()) if row["validate"]]
    responses = sum(row["metrics"]["source/response_count"] for row in batches)
    positions = sum(row["metrics"]["source/output_tokens"] for row in batches)
    if (
        responses != 9120
        or positions != report["metrics"]["validation_kl/on_validation_states/micro/evaluated_positions"]
    ):
        raise ValueError("Main generation and all-position KL coverage disagree")
    with (optimized / "training-1-2.csv").open() as stream:
        new_training = list(csv.DictReader(stream))
    with (baseline / "training-1-2.csv").open() as stream:
        old_training = list(csv.DictReader(stream))
    generation = sum(row["generation"]["seconds"] for row in batches)
    scoring = report["metrics"]["validation_kl/performance/scoring_seconds"]
    validation = float(new_training[-1]["timing_s/testing"])
    elapsed = end - start
    phases = {
        "native_validation_generation": generation,
        "kl_scoring_rpc": scoring,
        "other_validation_residual": validation - generation - scoring,
        "outside_validation_including_gate": elapsed - validation,
    }
    if min(phases.values()) < 0 or abs(sum(phases.values()) - elapsed) > 1e-6:
        raise ValueError("Mutually exclusive full-gate phase accounting failed")
    controls = []
    for old, new, old_row, new_row in zip(
        old_native["training"], new_native["training"], old_training, new_training, strict=True
    ):
        controls.append(
            {
                "step": new["step"],
                "same_training_trace_sha256": old["trace_sha256"] == new["trace_sha256"],
                "loss_absolute_difference": abs(float(old_row["actor/loss"]) - float(new_row["actor/loss"])),
                "gradient_norm_absolute_difference": abs(
                    float(old_row["actor/grad_norm"]) - float(new_row["actor/grad_norm"])
                ),
                "learning_rate": float(new_row["actor/lr"]),
            }
        )

    def bank(data):
        return {
            (r["state_source"], r["source"], r["uid"], r["sample"]): r
            for r in data["rows"]
            if r["state_source"] != "on_validation_states"
        }

    a, b = bank(read(baseline_report_path)), bank(report)
    if len(a) != 180 or a.keys() != b.keys() or any(a[k]["record_sha256"] != b[k]["record_sha256"] for k in a):
        raise ValueError("Fixed-bank controls differ between full gates")
    errors = {
        stat: max(
            abs(a[k]["sums"][stat] / a[k]["response_length"] - b[k]["sums"][stat] / b[k]["response_length"]) for k in a
        )
        for stat in ("student_teacher_kl", "teacher_student_kl", "student_entropy", "teacher_entropy")
    }
    old = {
        "label": "Baseline",
        "responses": 9120,
        "generated_positions": old_audit["generation"]["response_positions"],
        "phases_s": old_audit["phases_s"],
        "gate_elapsed_s": old_audit["gate_elapsed_s"],
        "validation_elapsed_s": old_audit["validation_timer_s"],
        "whole_card_peak_mib": old_native["whole_card_peak_mib"],
    }
    new = {
        "label": "Fast",
        "responses": responses,
        "generated_positions": positions,
        "phases_s": phases,
        "gate_elapsed_s": elapsed,
        "validation_elapsed_s": validation,
        "whole_card_peak_mib": new_native["whole_card_peak_mib"],
        "validation_batches": len(batches),
        "generation_tokens_per_second": positions / generation,
        "main_positions_per_scoring_second": positions / scoring,
        "validation_api_timings": {
            k: float(v) for k, v in new_training[-1].items() if k.startswith("validation_performance/") and v
        },
    }
    sources = [
        baseline / "performance-audit.json",
        baseline / "native-acceptance.json",
        baseline / "kl-summary.json",
        baseline / "training-1-2.csv",
        optimized / "native-acceptance.json",
        optimized / "kl-summary.json",
        optimized / "training-1-2.csv",
        run / "status.json",
        trace,
        Path(workflow),
        Path(archived),
        report_path,
        baseline_report_path,
    ]
    return {
        "scope": "Observed complete gates; generated lengths differ; final checkpoints were not reused",
        "runs": [old, new],
        "training_controls": controls,
        "same_final_checkpoint_reused": False,
        "fixed_bank_normalized_statistic_max_errors": errors,
        "observed_gate_wall_ratio": old["gate_elapsed_s"] / elapsed,
        "observed_validation_wall_ratio": old_audit["validation_timer_s"] / validation,
        "inputs_sha256": {str(path): digest(path) for path in sources},
        "driver_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.optimized, args.workflow, args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"observed_gate_wall_ratio": result["observed_gate_wall_ratio"], "optimized": result["runs"][1]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
