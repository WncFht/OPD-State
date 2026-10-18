"""Audit first-pass rollout work, request timing, batching, and prompt duplication.

This is a retrospective CPU audit, not a controlled GPU throughput benchmark.
Preparation time includes process startup, engine construction, output assembly,
database writes and model checksums; it is not a pure decode measurement.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONLINE = "math-student-online-50steps-eval16-20261018-044021"
SEMI = "math-semi-opd-50steps-eval16-20261018-025524"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logged_config(text):
    """Read the actual printed configuration without importing training libraries."""
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    lines = [re.sub(r"^\(TaskRunner pid=\d+\) ", "", line) for line in clean.splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith("{'actor_rollout_ref':"))
    for end in range(start, len(lines)):
        if lines[end].rstrip().endswith("}}"):
            try:
                result = ast.literal_eval("\n".join(lines[start : end + 1]))
            except (ValueError, SyntaxError):
                continue
            return result
    raise ValueError("No complete printed experiment configuration")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/rollout-sources")
    parser.add_argument("--results", type=Path, default=ROOT / "docs/experiment/results")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/results/rollout-cost-audit")
    args = parser.parse_args()
    inputs = {}

    def read_json(path):
        inputs[str(path)] = sha256(path)
        return json.loads(path.read_text())

    schedule = args.runs / "paired-schedule"
    manifest = read_json(schedule / "manifest.json")
    path = schedule / "prompts.jsonl"
    inputs[str(path)] = sha256(path)
    assert inputs[str(path)] == manifest["prompts_sha256"]
    path = schedule / "ordered.parquet"
    inputs[str(path)] = sha256(path)
    assert inputs[str(path)] == manifest["ordered_parquet_sha256"]
    records = [json.loads(line) for line in (schedule / "prompts.jsonl").read_text().splitlines()]
    assert len(records) == 3200 == len(set(manifest["source_indices"]))
    assert len({r["prompt_id"] for r in records}) == 3200
    # Content identity is the entire serialized chat prompt, not the row ID.
    by_content = defaultdict(list)
    for ordinal, record in enumerate(records):
        by_content[record["prompt_sha256"]].append(ordinal)
    duplicates = [
        {
            "prompt_sha256": key,
            "first_ordinal": positions[0],
            "second_ordinal": positions[1],
            "first_step": positions[0] // 64 + 1,
            "second_step": positions[1] // 64 + 1,
            "first_source_index": manifest["source_indices"][positions[0]],
            "second_source_index": manifest["source_indices"][positions[1]],
        }
        for key, positions in by_content.items()
        if len(positions) > 1
    ]
    assert max(map(len, by_content.values())) == 2
    arms, rows_by_arm, configs = {}, {}, {}
    for arm, name in (("online", ONLINE), ("semi", SEMI)):
        result = args.results / f"2026-10-18-{name}"
        summary = read_json(result / "summary.json")
        csv_path = result / "training-1-50.csv"
        inputs[str(csv_path)] = sha256(csv_path)
        rows = list(csv.DictReader(csv_path.open()))
        assert [int(r["step"]) for r in rows] == list(range(1, 51))
        rows_by_arm[arm] = rows
        log = next((args.runs / name / "logs").glob("*.log"))
        inputs[str(log)] = sha256(log)
        assert inputs[str(log)] == summary["training_log_sha256"]
        text = log.read_text()
        cfg = logged_config(text)
        configs[arm] = cfg
        assert cfg["trainer"]["source_commit"] == summary["source_commit"]
        assert cfg["trainer"]["total_epochs"] == 1
        assert cfg["trainer"]["total_training_steps"] == 50
        assert cfg["data"]["train_batch_size"] == 64 and not cfg["data"]["shuffle"]
        assert cfg["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1
        assert cfg["actor_rollout_ref"]["rollout"]["n"] == 1
        for relative in (
            "trainer/ppo/ray_trainer.py",
            "experimental/agent_loop/agent_loop.py",
            "workers/rollout/vllm_rollout/vllm_async_server.py",
            "workers/engine_workers.py",
        ):
            path = args.runs / name / "source/verl/verl" / relative
            inputs[str(path)] = sha256(path)
        arms[arm] = {
            "responses": 3200,
            "response_tokens": int(sum(float(r["response_length/mean"]) * 64 for r in rows)),
            "training_seconds": sum(float(r["timing_s/step"]) for r in rows),
            "validation_seconds": sum(float(r.get("timing_s/testing") or 0) for r in rows),
            "generation_or_cache_read_seconds": sum(float(r["timing_s/gen"]) for r in rows),
            "weight_sync_seconds": sum(float(r.get("timing_s/update_weights") or 0) for r in rows),
            "logged_weight_sync_calls": sum(bool(r.get("timing_s/update_weights")) for r in rows),
            "rollout_settings": {
                key: cfg["actor_rollout_ref"]["rollout"][key]
                for key in (
                    "calculate_log_probs",
                    "enforce_eager",
                    "gpu_memory_utilization",
                    "max_model_len",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "free_cache_engine",
                    "load_format",
                )
            },
            "observed_engine_init_log_lines": text.count("Initializing a V1 LLM engine"),
        }
    assert configs["online"]["data"]["train_files"] == configs["semi"]["data"]["train_files"]
    rollout = configs["online"]["actor_rollout_ref"]["rollout"]
    assert all(
        rollout[k] == 1 for k in ("tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size")
    )
    assert configs["online"]["trainer"]["n_gpus_per_node"] == 4
    cache = read_json(args.runs / "student-cache/manifest.json")
    assert cache["response_tokens"] == arms["semi"]["response_tokens"]
    assert cache["schedule_parquet_sha256"] == manifest["ordered_parquet_sha256"]
    cache_source = args.runs / f"preparation-source-{cache['generator_commit']}" / "data/prep/build_rollout_cache.py"
    inputs[str(cache_source)] = sha256(cache_source)
    launcher = ROOT / "recipe/prepare_math_cache.sh"
    frozen_launcher = subprocess.check_output(
        ["git", "show", f"{cache['generator_commit']}:recipe/prepare_math_cache.sh"], cwd=ROOT
    )
    assert launcher.read_bytes() == frozen_launcher
    assert b"--gpus 4 --batch-size 64 --response-cap 4096" in frozen_launcher
    inputs[str(launcher)] = sha256(launcher)
    llm_call = next(
        node
        for node in ast.walk(ast.parse(cache_source.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
    )
    cache_engine = {}
    for keyword in llm_call.keywords:
        try:
            cache_engine[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            pass
    shard_sizes, wave_lengths = [], []
    ordinals = []
    for rank in range(4):
        path = args.runs / f"student-cache/shard-{rank}.jsonl"
        inputs[str(path)] = sha256(path)
        shard = [json.loads(line) for line in path.read_text().splitlines()]
        shard_sizes.append(len(shard))
        for r in shard:
            assert r["prompt_id"] == records[r["ordinal"]]["prompt_id"]
            ordinals.append(r["ordinal"])
        wave_lengths.append([max(len(r["response_ids"]) for r in shard[i : i + 64]) for i in range(0, len(shard), 64)])
    assert sorted(ordinals) == list(range(3200)) and shard_sizes == [800] * 4
    rows = rows_by_arm["online"]
    request_max = sum(float(r["timing_s/agent_loop/generate_sequences/max"]) for r in rows)
    chain = [
        sum(float(r[f"timing_s/agent_loop/slowest/{k}"]) for k in ("generate_sequences", "tool_calls", "compute_score"))
        for r in rows
    ]
    residual = [float(r["timing_s/gen"]) - c for r, c in zip(rows, chain, strict=True)]
    assert min(residual) >= 0
    preparation = cache["seconds"]
    online, semi = arms["online"], arms["semi"]
    total_difference = online["training_seconds"] - (preparation + semi["training_seconds"])
    generation_difference = (
        online["generation_or_cache_read_seconds"] - preparation - semi["generation_or_cache_read_seconds"]
    )
    sync_difference = online["weight_sync_seconds"] - semi["weight_sync_seconds"]
    report = {
        "analyzer_sha256": sha256(Path(__file__)),
        "inputs_sha256": inputs,
        "prompt_audit": {
            "records": 3200,
            "unique_source_rows": 3200,
            "unique_prompt_ids": 3200,
            "unique_prompt_contents": len(by_content),
            "duplicate_content_records": 3200 - len(by_content),
            "duplicate_content_groups": len(duplicates),
            "within_batch_content_duplicates": sum(
                len(b) - len({r["prompt_sha256"] for r in b}) for i in range(0, 3200, 64) if (b := records[i : i + 64])
            ),
            "sampler_repeats_source_rows": False,
            "ppo_epochs": 1,
        },
        "arms": arms,
        "offline_preparation": {
            "seconds_including_startup_database_and_hashes": preparation,
            "engine_literal_settings_from_frozen_source": cache_engine,
            "shard_records": shard_sizes,
            "requests_per_gpu_batch": 64,
            "waves_per_gpu": len(wave_lengths[0]),
            "max_response_length_each_wave_per_gpu": wave_lengths,
            "phase_output_tokens_per_second": semi["response_tokens"] / preparation,
            "max_model_len": cache["prompt_cap"] + cache["response_cap"],
            "generation_logprobs": None,
            "batch_size_provenance": "prepare_math_cache.sh passes --gpus 4 --batch-size 64; shard totals checked",
        },
        "online_generation": {
            "requests_per_global_training_batch": 64,
            "average_available_requests_per_gpu": 16,
            "waves": 50,
            "request_latency_max_sum_seconds": request_max,
            "slowest_request_chain_sum_seconds": sum(chain),
            "generation_wall_minus_slowest_request_chain_seconds": sum(residual),
            "median_generation_wall_seconds": statistics.median(float(r["timing_s/gen"]) for r in rows),
            "median_max_request_latency_seconds": statistics.median(
                float(r["timing_s/agent_loop/generate_sequences/max"]) for r in rows
            ),
            "phase_output_tokens_per_second": online["response_tokens"] / online["generation_or_cache_read_seconds"],
            "scope": (
                "Request latency includes routing, engine queue/prefill/decode/logprobs and response return; "
                "not GPU-kernel-only time"
            ),
        },
        "first_pass_difference": {
            "total_seconds": total_difference,
            "generation_and_preparation_seconds": generation_difference,
            "generation_and_preparation_fraction": generation_difference / total_difference,
            "weight_sync_seconds": sync_difference,
            "remaining_scoring_update_overhead_seconds": total_difference - generation_difference - sync_difference,
            "preparation_plus_training_speed_ratio": online["training_seconds"]
            / (preparation + semi["training_seconds"]),
        },
        "causal_scope": (
            "Batching is a concrete configuration difference consistent with the wave counts. "
            "Its isolated effect versus logprobs, engine budget, routing and model weights has not been "
            "measured in a controlled GPU ablation. No rollout-count savings in this first 50-step pass."
        ),
    }
    assert all(math.isfinite(v) for v in report["first_pass_difference"].values())
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "duplicate-prompt-pairs.csv", duplicates)
    report["outputs_sha256"] = {"duplicate-prompt-pairs.csv": sha256(args.output / "duplicate-prompt-pairs.csv")}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"prompt_audit": report["prompt_audit"], "difference": report["first_pass_difference"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
