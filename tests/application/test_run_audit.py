"""Synthetic producer files must satisfy independent run completion checks."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import opd_state.evaluation.math as estimator
from opd_state.artifacts import atomic_json, file_digest
from recipes.application import run_prepared
from tools.verify import verify_prepared, verify_run


def lines(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def produce(root):
    cfg, manifest, inputs = verify_prepared(root)
    directory = root / "native-trajectories"
    directory.mkdir()
    pool = json.loads(Path(cfg.data.train_files[0]).read_text())
    token_data = json.loads((root / "data/selection.json").read_text())
    rows_by_id = {row["row_id"]: row for row in token_data["train"]}
    policy = OmegaConf.to_container(cfg.actor_rollout_ref.rollout.policy, resolve=True)
    atomic_json(
        directory / "manifest.json",
        {
            "policy": policy,
            "trainer_gpu_count": 4,
            "prepared_data_manifest_sha256": file_digest(cfg.recipe.prepared_manifest),
        },
    )
    batches, metrics = [], []
    for step in (1, 2):
        trajectories = []
        for raw in pool:
            row = rows_by_id[raw["opd_prompt_id"]]
            ids = [int(x) for x in row["rendered_input"].split(",")]
            trajectories.append(
                {
                    "request": {
                        "prompt_id": "train:" + row["generation_id"],
                        "prompt_ids": ids,
                        "visit": step - 1,
                        "sample": 0,
                        "row_id": row["row_id"],
                    },
                    "token_ids": [7, 15],
                    "route": 0,
                    "termination": "eos",
                    "segments": [
                        {
                            "start": 0,
                            "end": 2,
                            "identities": [
                                [
                                    "current_student",
                                    inputs["models"]["student"]["path"],
                                    step - 1,
                                ]
                            ],
                            "teacher_weight": None,
                        }
                    ],
                }
            )
        lines(directory / f"step-{step}.jsonl", trajectories)
        source_metrics = {
            "source/response_count": len(trajectories),
            "source/output_tokens": 2 * len(trajectories),
            "source/unique_prompts": len(trajectories),
        }
        batches.append(
            {
                "validate": False,
                "step": step,
                "identities": [
                    ["current_student", inputs["models"]["student"]["path"], step - 1]
                ],
                "metrics": source_metrics,
            }
        )
        metrics.append(
            {
                "step": step,
                "data": {
                    "actor/loss": 0.1,
                    "actor/grad_norm": 0.2,
                    "timing_s/update_actor": 0.01,
                    **source_metrics,
                },
            }
        )
    validation = []
    for index, row in enumerate(token_data["validation"], 4):
        for sample in range(16):
            correct = int(sample % 2 == 0)
            validation.append(
                {
                    "step": 2,
                    "rollout_student_version": 2,
                    "data_source": row["source"],
                    "input": row["rendered_input"],
                    "gts": str(index),
                    "output": str(index if correct else -1),
                    "score": correct,
                    "validation_uid": str(index),
                    "rollout_generation_prompt_id": "eval:" + row["generation_id"],
                    "rollout_generation_sample": sample,
                }
            )
    lines(root / "validation/2.jsonl", validation)
    batches.append(
        {
            "validate": True,
            "step": 2,
            "identities": [["current_student", inputs["models"]["student"]["path"], 2]],
        }
    )
    lines(directory / "batches.jsonl", batches)
    # Use the existing producer's estimator; verification uses its own binomial formula.
    computed, _ = estimator.exact_math_metrics(
        [r["data_source"] for r in validation],
        [r["validation_uid"] for r in validation],
        [r["score"] for r in validation],
        samples_per_question=16,
        required_counts={"MATH-500": 1, "AMC23": 1, "AIME25": 1},
    )
    metrics[-1]["data"].update(computed)
    lines(root / "logs/metrics.jsonl", metrics)
    atomic_json(
        root / "train-exit.json", {"run_id": manifest["run_id"], "return_code": 0}
    )
    atomic_json(
        root / "resources.json",
        {"run_id": manifest["run_id"], "gpu_count": 4, "fixture": True},
    )


def test_complete_artifact_fixture_and_single_execution(prepared):
    root = prepared["root"]

    def runner(*args, **kwargs):
        produce(root)
        return 0

    result = run_prepared(
        root,
        resource_check=lambda cfg: {"gpu_count": 4, "fixture": True},
        command_runner=runner,
    )
    assert result["updates"] == 2
    assert result["validation"]["2"]["responses"] == 48
    assert result["validation"]["2"]["metrics"]["eval/macro/mean@16"] == 0.5
    assert json.loads((root / "status.json").read_text())["state"] == "completed"
    assert verify_run(root) == result
    assert not Path(prepared["manifest"]["scratch"]).exists()
    with pytest.raises(ValueError, match="unexecuted"):
        run_prepared(root)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_update",
        "duplicate_update",
        "nan_gradient",
        "stale_actor",
        "foreign_prompt",
        "partial_validation",
        "wrong_validation_version",
        "wrong_reference",
        "duplicate_sample",
        "wrong_metric",
        "wrong_resource",
        "duplicate_receipt",
        "wrong_receipt_tokens",
        "unrequested_kl",
    ],
)
def test_corrupt_or_incomplete_producer_evidence_is_rejected(prepared, mutation):
    root = prepared["root"]
    produce(root)
    if mutation in {"duplicate_receipt", "wrong_receipt_tokens"}:
        path = root / "native-trajectories/batches.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if mutation == "duplicate_receipt":
            rows.append(rows[0])
        else:
            rows[0]["metrics"]["source/output_tokens"] += 1
        lines(path, rows)
    elif mutation == "unrequested_kl":
        path = root / "validation/kl/kl-step-2.json"
        path.parent.mkdir()
        atomic_json(path, {})
    elif mutation in {
        "missing_update",
        "duplicate_update",
        "nan_gradient",
        "wrong_metric",
    }:
        path = root / "logs/metrics.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if mutation == "missing_update":
            rows.pop(0)
        elif mutation == "duplicate_update":
            rows.append(rows[0])
        elif mutation == "nan_gradient":
            rows[0]["data"]["actor/grad_norm"] = None
        else:
            rows[-1]["data"]["eval/macro/mean@16"] = 0.75
        lines(path, rows)
    elif mutation in {"stale_actor", "foreign_prompt"}:
        path = root / "native-trajectories/step-1.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if mutation == "stale_actor":
            rows[0]["segments"][0]["identities"][0][2] = 12
        else:
            rows[0]["request"]["prompt_ids"] = [1, 9]
        lines(path, rows)
    elif mutation == "wrong_resource":
        path = root / "resources.json"
        value = json.loads(path.read_text())
        value["gpu_count"] = 8
        atomic_json(path, value)
    else:
        path = root / "validation/2.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if mutation == "partial_validation":
            rows.pop()
        elif mutation == "wrong_validation_version":
            rows[0]["rollout_student_version"] = 0
        elif mutation == "wrong_reference":
            rows[0]["gts"] = "wrong"
        else:
            rows[1]["rollout_generation_sample"] = 0
        lines(path, rows)
    with pytest.raises(ValueError):
        verify_run(root)
