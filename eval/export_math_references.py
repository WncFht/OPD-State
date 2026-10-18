"""Export the two user-selected, evaluation-only W&B reference runs.

Only selected model/configuration fields and numeric evaluation metrics leave
the API response. Credentials and the complete configuration are never saved.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}
REFERENCES = {
    "teacher": ("u87buult", "JustRL-DeepSeek-1.5B"),
    "initial_student": ("6uorrk2k", "DeepSeek-R1-Distill-Qwen-1.5B"),
}


def export_reference(api, role, run_id, model_name):
    run = api.run(f"wncfht/One-Shot-OPD/{run_id}")
    config = run.config
    model = config["actor_rollout_ref"]["model"]["path"]
    sampling = config["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert run.state == "finished", (run_id, run.state)
    assert Path(model).name == model_name, (role, model)
    assert config["trainer"]["val_only"] is True
    expected = {"n": 16, "temperature": 0.7, "top_p": 0.9, "max_tokens": 31744, "do_sample": True}
    for key, value in expected.items():
        assert sampling[key] == value, (run_id, key, sampling[key])
    files = config["data"]["val_files"]
    assert {Path(p).parent.name for p in files} == set(BENCHMARKS)
    assert config["data"].get("val_max_samples", -1) == -1
    metrics = {}
    for benchmark in BENCHMARKS:
        key = f"val-core/{benchmark}/acc/mean@16"
        value = float(run.summary[key])
        assert math.isfinite(value) and 0 <= value <= 1
        metrics[f"eval/{benchmark}/mean@16"] = value
        for k in (2, 4, 8, 16):
            prefix = "val-core" if k == 16 else "val-aux"
            source_key = f"{prefix}/{benchmark}/acc/pass@{k}"
            if source_key in run.summary:
                metrics[f"eval/{benchmark}/pass@{k}"] = float(run.summary[source_key])
    scores = [metrics[f"eval/{b}/mean@16"] for b in BENCHMARKS]
    metrics["eval/macro/mean@16"] = sum(scores) / len(scores)
    metrics["eval/micro/mean@16"] = sum(metrics[f"eval/{b}/mean@16"] * n for b, n in BENCHMARKS.items()) / sum(
        BENCHMARKS.values()
    )
    return {
        "role": role,
        "id": run_id,
        "name": run.name,
        "state": run.state,
        "source_url": f"https://forge.coreweave.com/wandb/wncfht/One-Shot-OPD/runs/{run_id}",
        "api_url": f"https://wandb.ai/wncfht/One-Shot-OPD/runs/{run_id}",
        "model_name": model_name,
        "model_path": model,
        "val_only": True,
        "sampling": {key: sampling[key] for key in (*expected, "top_k")},
        "validation_files": files,
        "val_max_samples": config["data"].get("val_max_samples", -1),
        "metrics": metrics,
        "evidence_scope": "Remote summary and configuration; no raw responses or dataset/weight hashes verified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import wandb

    api = wandb.Api(timeout=40)
    runs = {role: export_reference(api, role, *values) for role, values in REFERENCES.items()}
    report = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {role: {"name": r["name"], "model": r["model_name"], "metrics": r["metrics"]} for role, r in runs.items()}
        )
    )


if __name__ == "__main__":
    main()
