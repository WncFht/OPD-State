"""Recompute resource estimates and summarize the isolated GPU experiments."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs/research/full-vocab/results"
SOURCES = ROOT / "runs/full-vocab-research-20261019/sources"


def summarize_training(prefix):
    files = sorted(RESULTS.glob(prefix + "-rank*.json"))
    if not files:
        return None
    ranks = [json.loads(p.read_text()) for p in files]
    hashes = {r["input_sha256"] for r in ranks}
    assert len(hashes) == 1
    steady = [max(r["updates"][i]["seconds"] for r in ranks) for i in [1, 2]]
    elapsed = statistics.median(steady)
    return {
        "ranks": len(ranks),
        "input_sha256": hashes.pop(),
        "response_tokens": ranks[0]["response_tokens_global"],
        "steady_seconds": steady,
        "median_seconds": elapsed,
        "response_tokens_per_second": ranks[0]["response_tokens_global"] / elapsed,
        "peak_allocated_gib": max(
            s["peak_allocated_gib"] for r in ranks for s in r["updates"]
        ),
        "peak_reserved_gib": max(
            s["peak_reserved_gib"] for r in ranks for s in r["updates"]
        ),
        "rank_parameter_sample_max_difference": max(
            r["rank_parameter_sample_max_difference"] for r in ranks
        ),
        "teacher_has_gradient": any(r["teacher_has_gradient"] for r in ranks),
        "eight_gpu_seconds_model": [elapsed / 2, elapsed / 1.5],
        "eight_gpu_assumption": "same workload, 1.5x to 2x speedup; not measured",
    }


def main():
    params, hidden, vocab, physical_vocab = 1777088000, 1536, 151665, 151936
    estimates = {}
    for response in [4096, 8192]:
        n = 64 * response
        estimates[str(response)] = {
            "global_fp32_teacher_logprobs_gib": 4 * n * vocab / 2**30,
            "global_bf16_teacher_hidden_gib": 2 * n * hidden / 2**30,
            "four_rank_bf16_hidden_gib": 2 * n * hidden / 4 / 2**30,
            "eight_rank_bf16_hidden_gib": 2 * n * hidden / 8 / 2**30,
        }
    memory = {}
    for name, ps, pt in [
        ("1.5B_1.5B", params, params),
        ("1.5B_7B", params, 7e9),
        ("7B_7B", 7e9, 7e9),
        ("1.5B_32B", params, 32e9),
    ]:
        total = 16 * ps + 2 * pt
        memory[name] = {
            "replicated_static_gib": total / 2**30,
            "ideal_fully_sharded_4_gib": total / 4 / 2**30,
            "ideal_fully_sharded_8_gib": total / 8 / 2**30,
        }
    prefixes = [
        "train-4gpu-b64-4096",
        "train-4gpu-b64-4096-dense",
        "train-4gpu-b64-8192",
        "train-4gpu-b64-8192-strict",
    ]
    training = {p: summarize_training(p) for p in prefixes}
    source_rows = []
    if SOURCES.exists():
        versions = json.loads((SOURCES / "versions.json").read_text())
        for key, meta in versions.items():
            files = []
            for path in sorted((SOURCES / key).rglob("*")):
                if path.is_file():
                    files.append(
                        {
                            "path": str(path.relative_to(SOURCES / key)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
            source_rows.append(
                {
                    "name": key,
                    "repository": meta["repo"],
                    "commit": meta["sha"],
                    "files": files,
                }
            )
    rows = list(
        csv.DictReader(
            (
                ROOT
                / "docs/experiment/results/native-rollout/formal/student_fresh/training-1-50.csv"
            ).open()
        )
    )[1:49]
    metrics = [
        "source/output_tokens",
        "timing_s/native_generate",
        "timing_s/native_score_and_pack",
        "timing_s/old_log_prob",
        "timing_s/opd_teacher",
        "timing_s/update_actor",
        "timing_s/update_weights",
        "timing_s/step",
    ]
    historical = {
        k: statistics.median(float(r[k]) for r in rows if r.get(k)) for k in metrics
    }
    result = {
        "date": "2026-10-19",
        "scope": "GPU training prototype and estimates; no verl or rollout integration",
        "model_parameters_each": params,
        "hidden": hidden,
        "effective_vocab": vocab,
        "teacher_bf16_head_gib": 2 * physical_vocab * hidden / 2**30,
        "teacher_bf16_weights_gib": 2 * params / 2**30,
        "payload_estimates": estimates,
        "static_memory_estimates": memory,
        "static_memory_formula": "(16 * student_params + 2 * teacher_params) / DP; excludes activation, gather, workspace, runtime, KV",
        "training": training,
        "historical_sampled_training_steps_2_to_49_medians": historical,
        "sources": source_rows,
        "artifacts": [
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(RESULTS.iterdir())
            if p.is_file() and p.name != "summary.json"
        ],
        "delivery_sources": [
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(Path(__file__).parent.glob("*.py"))
        ],
    }
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["payload_estimates", "static_memory_estimates", "training"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
