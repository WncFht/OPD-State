"""Verify a finished W&B math run against independently collected local evidence.

Credentials use the normal SDK environment/netrc flow. Reports contain selected
experiment settings and metrics, never API keys or the complete environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def verify(directory, api):
    directory = Path(directory)
    local = json.loads((directory / "summary.json").read_text())
    parts = urlparse(local["wandb_url"]).path.strip("/").split("/")
    if len(parts) != 4 or parts[1] != "OPD-State" or parts[2] != "runs":
        raise ValueError("Expected an OPD-State W&B run URL")
    run = api.run("/".join((parts[0], parts[1], parts[3])))
    trainer, rollout = run.config["trainer"], run.config["actor_rollout_ref"]["rollout"]
    steps = list(run.scan_history(keys=["training/global_step", "actor/loss", "actor/grad_norm"], page_size=100))
    if run.state != "finished" or [int(row["training/global_step"]) for row in steps] != list(
        range(1, local["training_steps"] + 1)
    ):
        raise ValueError("W&B run is unfinished or has missing training steps")
    if not all(math.isfinite(float(row[key])) for row in steps for key in ("actor/loss", "actor/grad_norm")):
        raise ValueError("W&B training metrics contain nonfinite loss or gradients")
    if trainer["source_commit"] != local["source_commit"] or trainer["save_freq"] != -1:
        raise ValueError("W&B source revision or no-checkpoint setting disagrees with the experiment")
    if rollout["val_kwargs"]["n"] != 16 or trainer["math_eval_samples"] != 16 or not trainer["math_eval_require_full"]:
        raise ValueError("W&B run does not use the full three-benchmark eval16 protocol")
    validations = list(run.scan_history(keys=["training/global_step", "eval/macro/mean@16"], page_size=100))
    if [int(row["training/global_step"]) for row in validations] != local["validation_steps"]:
        raise ValueError("W&B validation steps disagree with local evidence")
    for row in validations:
        step = int(row["training/global_step"])
        evidence = json.loads((directory / f"validation-step{step}.json").read_text())
        if abs(float(row["eval/macro/mean@16"]) - evidence["metrics"]["eval/macro/mean@16"]) > 1e-8:
            raise ValueError(f"W&B validation history disagrees with raw responses at step {step}")
    final = json.loads((directory / f"validation-step{local['training_steps']}.json").read_text())
    remote = {}
    for key, value in final["metrics"].items():
        actual = run.summary.get(key)
        if actual is None or not math.isfinite(float(actual)) or abs(float(actual) - value) > 1e-8:
            raise ValueError(f"W&B summary disagrees with independently recomputed metric: {key}")
        remote[key] = actual
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
        "settings": {
            "student_model": run.config["actor_rollout_ref"]["model"]["path"],
            "teacher_model": run.config["distillation"]["fsdp_teacher"]["model_path"],
            "rollout_source": trainer.get("rollout_source", "student_online"),
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
    args = parser.parse_args()
    report = verify(args.result, wandb.Api(timeout=30))
    path = Path(args.result) / "wandb-verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("id", "state", "history_training_rows", "validation_steps")}))


if __name__ == "__main__":
    main()
