"""Build the paired prompt schedule and immutable fixed-source trajectories."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path


def prepare_schedule(args):
    import numpy as np
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.trainer.ppo.fixed_rollout import digest_file, digest_json, tokenizer_signature
    from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.tokenizer import hf_tokenizer

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Paired schedule already exists: {root}")
    config = OmegaConf.create(json.loads(Path(args.baseline_config).read_text()))
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    data = config.data
    assert data.shuffle and data.seed == 42 and data.train_batch_size == args.batch_size
    dataset = create_rl_dataset(data.train_files, data, tokenizer, None, max_samples=-1, is_train=True)
    dataset.dataframe = dataset.dataframe.add_column("opd_filtered_index", list(range(len(dataset))))
    loader = StatefulDataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=create_rl_sampler(data, dataset),
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    selected = []
    for i, batch in enumerate(loader):
        selected.extend(int(v) for v in batch["opd_filtered_index"])
        if i + 1 == args.steps:
            break
    if len(selected) != args.steps * args.batch_size:
        raise ValueError("Dataset is too small for the paired schedule")
    frame = dataset.dataframe.select(selected)
    source_hash = digest_json(
        {
            "files": {str(p): digest_file(p) for p in data.train_files},
            "tokenizer": tokenizer_signature(tokenizer),
            "seed": data.seed,
        }
    )
    keys = [digest_json({"dataset": source_hash, "index": i}) for i in selected]
    frame = frame.add_column("opd_prompt_id", keys)
    frame.to_parquet(str(root / "ordered.parquet"))
    prompts = []
    for key, row in zip(keys, frame, strict=True):
        raw = row[data.prompt_key]
        token_ids = tokenizer.apply_chat_template(
            raw, add_generation_prompt=True, tokenize=True, **dict(data.get("apply_chat_template_kwargs", {}))
        )
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
        prompts.append({"prompt_id": key, "prompt_sha256": digest_json(raw), "prompt_ids": token_ids})
    (root / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prompts))
    manifest = {
        "schema": 1,
        "records": len(prompts),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": data.seed,
        "original_filtered_rows": len(dataset),
        "source_hash": source_hash,
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "ordered_parquet_sha256": digest_file(root / "ordered.parquet"),
        "prompts_sha256": digest_file(root / "prompts.jsonl"),
        "apply_chat_template_kwargs": dict(data.get("apply_chat_template_kwargs", {})),
        "prompt_cap": int(data.max_prompt_length),
        "source_indices": selected,
    }
    # Each batch must reproduce the original baseline's observed prompt lengths.
    if args.baseline_log:
        import re

        text = Path(args.baseline_log).read_text()
        observed = []
        for match in re.finditer(r"step:(\d+) - ([^\n]*)", text):
            fields = dict(re.findall(r"([\w/.-]+):(-?[0-9.eE+-]+)", match[2]))
            observed.append(fields)
        for i in range(args.steps):
            lengths = [len(p["prompt_ids"]) for p in prompts[i * args.batch_size : (i + 1) * args.batch_size]]
            for suffix, value in [("mean", np.mean(lengths)), ("min", min(lengths)), ("max", max(lengths))]:
                if abs(float(observed[i][f"prompt_length/{suffix}"]) - value) > 1e-6:
                    raise ValueError(f"Baseline prompt-length mismatch in batch {i}, {suffix}")
        manifest["baseline_batch_prompt_length_checks"] = args.steps
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_indices"}), flush=True)


def generate_worker(rank, records, options):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    import torch
    from vllm import LLM, SamplingParams

    torch.set_num_threads(4)
    llm = LLM(
        model=options["model"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=options["prompt_cap"] + options["response_cap"],
        max_num_seqs=options["batch_size"],
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        enable_prefix_caching=True,
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        seed=options["seed"] + rank,
    )
    llm.collective_rpc("monkey_patch_model", kwargs={"vocab_size": options["vocab_size"], "banned_token_ids": []})
    shard = Path(options["output"]) / f"shard-{rank}.jsonl"
    with shard.open("w") as stream:
        for start in range(0, len(records), options["batch_size"]):
            batch = records[start : start + options["batch_size"]]
            params = [
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=options["response_cap"],
                    stop_token_ids=[options["eos"]],
                    seed=options["seed"] + item["ordinal"],
                    repetition_penalty=1.0,
                )
                for item in batch
            ]
            outputs = llm.generate([{"prompt_token_ids": item["prompt_ids"]} for item in batch], params, use_tqdm=False)
            for item, result in zip(batch, outputs, strict=True):
                response = result.outputs[0]
                record = {
                    **item,
                    "response_ids": list(response.token_ids),
                    "finish_reason": response.finish_reason,
                    "stop_reason": response.stop_reason,
                }
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"CACHE rank={rank} completed={min(start + len(batch), len(records))}/{len(records)}", flush=True)
    del llm


def generate_cache(args):
    from verl.trainer.ppo.fixed_rollout import digest_file, model_signature, tokenizer_signature
    from verl.utils.tokenizer import hf_tokenizer

    root, schedule = Path(args.output).resolve(), Path(args.schedule).resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Frozen cache already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schedule_meta = json.loads((schedule / "manifest.json").read_text())
    if digest_file(schedule / "prompts.jsonl") != schedule_meta["prompts_sha256"]:
        raise ValueError("Paired schedule checksum mismatch")
    tokenizer = hf_tokenizer(args.student)
    teacher_tokenizer = hf_tokenizer(args.teacher)
    if (
        tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
        or tokenizer.eos_token_id != teacher_tokenizer.eos_token_id
    ):
        raise ValueError("Student and teacher token identities differ")
    if tokenizer_signature(tokenizer) != schedule_meta["tokenizer_signature"]:
        raise ValueError("Schedule tokenizer does not match the student")
    records = [json.loads(line) for line in (schedule / "prompts.jsonl").read_text().splitlines()]
    for i, record in enumerate(records):
        record["ordinal"] = i
    if args.limit:
        records = records[: args.limit]
    context = multiprocessing.get_context("spawn")
    options = {
        "model": str(Path(args.teacher if args.source == "cached_teacher" else args.student).resolve()),
        "prompt_cap": schedule_meta["prompt_cap"],
        "response_cap": args.response_cap,
        "batch_size": args.batch_size,
        "output": str(root),
        "seed": args.seed,
        "vocab_size": len(tokenizer),
        "eos": tokenizer.eos_token_id,
    }
    processes = [
        context.Process(target=generate_worker, args=(rank, records[rank :: args.gpus], options))
        for rank in range(args.gpus)
    ]
    started = time.monotonic()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError(f"Cache workers failed: {[p.exitcode for p in processes]}")
    database = root / "trajectories.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE trajectories (prompt_id TEXT PRIMARY KEY, prompt_sha256 TEXT NOT NULL, "
        "prompt_ids TEXT NOT NULL, response_ids TEXT NOT NULL, finish_reason TEXT NOT NULL)"
    )
    total, total_tokens = 0, 0
    for rank in range(args.gpus):
        for line in (root / f"shard-{rank}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r["response_ids"] or len(r["response_ids"]) > args.response_cap:
                raise ValueError("Invalid generated response length")
            connection.execute(
                "INSERT INTO trajectories VALUES (?,?,?,?,?)",
                (
                    r["prompt_id"],
                    r["prompt_sha256"],
                    json.dumps(r["prompt_ids"]),
                    json.dumps(r["response_ids"]),
                    r["finish_reason"],
                ),
            )
            total += 1
            total_tokens += len(r["response_ids"])
    connection.commit()
    connection.close()
    if total != len(records):
        raise ValueError("Generated cache has missing records")
    manifest = {
        "schema": 1,
        "source": args.source,
        "records": total,
        "response_tokens": total_tokens,
        "database_sha256": digest_file(database),
        "schedule_parquet_sha256": schedule_meta["ordered_parquet_sha256"],
        "student_signature": model_signature(args.student),
        "teacher_signature": model_signature(args.teacher),
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "prompt_cap": options["prompt_cap"],
        "response_cap": args.response_cap,
        "temperature": 1.0,
        "top_p": 1.0,
        "apply_chat_template_kwargs": schedule_meta["apply_chat_template_kwargs"],
        "seed": args.seed,
        "seconds": time.monotonic() - started,
        "generator_commit": args.generator_commit,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--baseline-config", required=True)
    schedule.add_argument("--baseline-log")
    schedule.add_argument("--output", required=True)
    schedule.add_argument("--steps", type=int, default=50)
    schedule.add_argument("--batch-size", type=int, default=64)
    cache = sub.add_parser("generate")
    cache.add_argument("--source", choices=["cached_teacher", "cached_initial_student"], required=True)
    cache.add_argument("--schedule", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--student", required=True)
    cache.add_argument("--teacher", required=True)
    cache.add_argument("--response-cap", type=int, default=4096)
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--seed", type=int, default=42)
    cache.add_argument("--gpus", type=int, default=4, choices=range(1, 5))
    cache.add_argument("--limit", type=int)
    cache.add_argument("--generator-commit", help="Commit of the frozen preparation source")
    args = parser.parse_args()
    if args.command == "schedule":
        prepare_schedule(args)
    else:
        generate_cache(args)


if __name__ == "__main__":
    main()
