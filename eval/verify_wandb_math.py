"""Verify a finished W&B math run or a completed validation prefix.

Credentials use the normal SDK environment/netrc flow. Reports contain selected
experiment settings and metrics, never API keys or the complete environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def verify(directory, api, *, snapshot=False, engineering=False, kl_report=None):
    directory = Path(directory)
    local = json.loads((directory / "summary.json").read_text())
    is_snapshot = local.get("collection_scope") == "through_step_snapshot"
    if snapshot != is_snapshot:
        raise ValueError("Snapshot evidence requires explicit snapshot verification and vice versa")
    evidence_paths = [directory / "summary.json"]
    parts = urlparse(local["wandb_url"]).path.strip("/").split("/")
    if len(parts) != 4 or parts[1] != "OPD-State" or parts[2] != "runs":
        raise ValueError("Expected an OPD-State W&B run URL")
    run = api.run("/".join((parts[0], parts[1], parts[3])))
    trainer, rollout = run.config["trainer"], run.config["actor_rollout_ref"]["rollout"]
    native_policy = rollout.get("policy")
    native_binding = None
    if native_policy is not None:
        acceptance_path = directory / "native-acceptance.json"
        if not acceptance_path.is_file() or not local.get("run_directory"):
            raise ValueError("Native W&B verification requires the local native acceptance evidence")
        manifests = list(Path(local["run_directory"]).glob("checkpoints/**/native-trajectories/manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Native W&B verification requires one generation manifest")
        manifest_path = manifests[0]
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        acceptance = json.loads(acceptance_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if manifest_sha != acceptance["native_manifest_sha256"] or manifest["policy"] != native_policy:
            raise ValueError("W&B native rollout policy disagrees with the audited generation manifest")
        if rollout.get("generation_backend") != "vllm_native":
            raise ValueError("W&B native rollout policy does not use the audited native backend")
        evidence_paths.extend([acceptance_path, manifest_path])
        native_binding = {"policy": native_policy, "manifest_sha256": manifest_sha}
    observed_steps = list(
        run.scan_history(keys=["training/global_step", "actor/loss", "actor/grad_norm"], page_size=100)
    )
    steps = [
        row for row in observed_steps if not snapshot or int(row["training/global_step"]) <= local["training_steps"]
    ]
    if (not snapshot and run.state != "finished") or [int(row["training/global_step"]) for row in steps] != list(
        range(1, local["training_steps"] + 1)
    ):
        raise ValueError("W&B run is unfinished or has missing training steps")
    if not all(math.isfinite(float(row[key])) for row in steps for key in ("actor/loss", "actor/grad_norm")):
        raise ValueError("W&B training metrics contain nonfinite loss or gradients")
    training_csv = directory / f"training-1-{local['training_steps']}.csv"
    with training_csv.open() as stream:
        local_steps = list(csv.DictReader(stream))
    if [int(row["step"]) for row in local_steps] != list(range(1, local["training_steps"] + 1)):
        raise ValueError("Local training evidence has missing or duplicate steps")
    for remote_row, local_row in zip(steps, local_steps, strict=True):
        for key in ("actor/loss", "actor/grad_norm"):
            expected = float(local_row[key])
            if not math.isfinite(expected) or abs(float(remote_row[key]) - expected) > 1e-8:
                raise ValueError(f"W&B training history disagrees with local {key}")
    evidence_paths.append(training_csv)
    if trainer["source_commit"] != local["source_commit"] or trainer["save_freq"] != -1:
        raise ValueError("W&B source revision or no-checkpoint setting disagrees with the experiment")
    if rollout["val_kwargs"]["n"] != 16 or trainer["math_eval_samples"] != 16:
        raise ValueError("W&B run does not use the full three-benchmark eval16 protocol")
    if engineering:
        if trainer["math_eval_require_full"] or local.get("full_benchmarks") or native_policy is None:
            raise ValueError("Engineering verification requires an explicitly partial native evaluation")
    elif not trainer["math_eval_require_full"]:
        raise ValueError("Partial evaluation requires explicit engineering verification")
    # Archived W&B history can return every step with nulls for keys that were
    # not logged there. Live history may instead omit those rows entirely.
    observed_validations = [
        row
        for row in run.scan_history(keys=["training/global_step", "eval/macro/mean@16"], page_size=100)
        if row.get("eval/macro/mean@16") is not None
    ]
    validations = [
        row
        for row in observed_validations
        if not snapshot or int(row["training/global_step"]) <= local["training_steps"]
    ]
    if [int(row["training/global_step"]) for row in validations] != local["validation_steps"]:
        raise ValueError("W&B validation steps disagree with local evidence")
    for row in validations:
        step = int(row["training/global_step"])
        evidence_path = directory / f"validation-step{step}.json"
        evidence_paths.append(evidence_path)
        evidence = json.loads(evidence_path.read_text())
        actual = float(row["eval/macro/mean@16"])
        expected = float(evidence["metrics"]["eval/macro/mean@16"])
        if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 1e-8:
            raise ValueError(f"W&B validation history disagrees with raw responses at step {step}")
    final = json.loads((directory / f"validation-step{local['training_steps']}.json").read_text())
    metrics_by_step = {}
    if snapshot:
        # The summary may already describe a later validation. Request the exact
        # metric row at each audited step, without declaring the run finished.
        detail_rows = list(run.scan_history(keys=["training/global_step", *final["metrics"]], page_size=100))
        detail_rows = [
            row
            for row in detail_rows
            if row.get("eval/macro/mean@16") is not None and int(row["training/global_step"]) <= local["training_steps"]
        ]
        if [int(row["training/global_step"]) for row in detail_rows] != local["validation_steps"]:
            raise ValueError("W&B snapshot lacks complete validation metric rows")
        for row in detail_rows:
            step = int(row["training/global_step"])
            evidence = json.loads((directory / f"validation-step{step}.json").read_text())
            metrics_by_step[str(step)] = {}
            for key, expected in evidence["metrics"].items():
                actual = row.get(key)
                if (
                    actual is None
                    or not math.isfinite(float(actual))
                    or not math.isfinite(float(expected))
                    or abs(float(actual) - expected) > 1e-8
                ):
                    raise ValueError(f"W&B snapshot metric disagrees at step {step}: {key}")
                metrics_by_step[str(step)][key] = actual
        latest = metrics_by_step[str(local["training_steps"])]
    else:
        latest = run.summary
    remote = {}
    for key, value in final["metrics"].items():
        actual = latest.get(key)
        if (
            actual is None
            or not math.isfinite(float(actual))
            or not math.isfinite(float(value))
            or abs(float(actual) - value) > 1e-8
        ):
            raise ValueError(f"W&B summary disagrees with independently recomputed metric: {key}")
        remote[key] = actual
    kl_verification = None
    if kl_report is not None:
        kl_report = Path(kl_report)
        kl = json.loads(kl_report.read_text())
        settings = trainer.get("validation_kl", {})
        if (
            not settings.get("enabled")
            or kl["student_version"] != local["training_steps"]
            or kl["position_chunk_size"] != settings.get("position_chunk_size")
            or kl["vocab_chunk_size"] != settings.get("vocab_chunk_size")
        ):
            raise ValueError("W&B KL configuration or final student version differs from local scoring")
        coverage_key = "validation_kl/on_validation_states/micro/responses"
        history = [
            row
            for row in run.scan_history(keys=["training/global_step", *kl["metrics"]], page_size=100)
            if row.get(coverage_key) is not None and int(row["training/global_step"]) == kl["student_version"]
        ]
        if len(history) != 1 or any(
            not math.isfinite(float(history[0].get(key, float("nan")))) or abs(float(history[0][key]) - value) > 1e-8
            for key, value in kl["metrics"].items()
        ):
            raise ValueError("W&B KL metrics disagree with the exact final validation history row")
        evidence_paths.append(kl_report)
        kl_verification = {
            "student_version": kl["student_version"],
            "metrics": kl["metrics"],
            "report_sha256": hashlib.sha256(kl_report.read_bytes()).hexdigest(),
        }
    report = {
        "id": run.id,
        "state": run.state,
        "url": run.url,
        "source_commit": trainer["source_commit"],
        "history_training_rows": len(steps),
        "steps": [int(row["training/global_step"]) for row in steps],
        "finite_loss_and_grad": True,
        "save_freq": trainer["save_freq"],
        "samples_per_question": rollout["val_kwargs"]["n"],
        "validation_steps": local["validation_steps"],
        "metrics": remote,
        "verification_scope": "completed_validation_prefix" if snapshot else "finished_run",
        "observed_remote_training_rows": len(observed_steps),
        "observed_remote_validation_steps": [int(row["training/global_step"]) for row in observed_validations],
        "validation_metrics_by_step": metrics_by_step,
        "local_evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_paths},
        "native_policy_verification": native_binding,
        "quality_scope": "engineering_only" if engineering else "full_three_benchmark_eval16",
        "kl_verification": kl_verification,
        "settings": {
            "student_model": run.config["actor_rollout_ref"]["model"]["path"],
            "teacher_model": run.config["distillation"]["fsdp_teacher"]["model_path"],
            "rollout_source": "native_policy"
            if native_policy is not None
            else trainer.get("rollout_source", "student_online"),
            "rollout_policy": native_policy,
            "generation_backend": rollout.get("generation_backend"),
            "train_files": run.config["data"]["train_files"],
            "val_files": run.config["data"]["val_files"],
            "seed": run.config["data"]["seed"],
            "train_batch_size": run.config["data"]["train_batch_size"],
            "prompt_cap": rollout["prompt_length"],
            "training_response_cap": rollout["response_length"],
            "validation": rollout["val_kwargs"],
        },
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return report


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--engineering", action="store_true", help="Verify an explicitly partial native engineering run"
    )
    parser.add_argument(
        "--kl-report", type=Path, help="Also verify KL metrics at the exact final validation history row"
    )
    parser.add_argument(
        "--snapshot", action="store_true", help="Verify completed history without requiring a finished run"
    )
    args = parser.parse_args()
    report = verify(
        args.result,
        wandb.Api(timeout=30),
        snapshot=args.snapshot,
        engineering=args.engineering,
        kl_report=args.kl_report,
    )
    path = Path(args.result) / ("wandb-snapshot-verification.json" if args.snapshot else "wandb-verification.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("id", "state", "history_training_rows", "validation_steps")}))


if __name__ == "__main__":
    main()
