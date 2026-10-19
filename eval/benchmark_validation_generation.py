"""Four-replica native vLLM validation scheduling/memory controls.

Offline LLM.generate isolates the inference API. The trainer integration is
measured separately; these numbers exclude its FSDP, reward and RPC packing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

from benchmark_validation_scoring import digest, sample_memory

ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    {"name": "seq8-mem55-token32k", "sequences": 8, "budget": 0.55, "batched_tokens": 32768},
    {"name": "seq64-mem55-token32k", "sequences": 64, "budget": 0.55, "batched_tokens": 32768},
    {"name": "seq128-mem55-token32k", "sequences": 128, "budget": 0.55, "batched_tokens": 32768},
    {"name": "seq128-mem70-token32k", "sequences": 128, "budget": 0.70, "batched_tokens": 32768},
    {"name": "seq64-mem70-token8k", "sequences": 64, "budget": 0.70, "batched_tokens": 8192},
]


def requests(prepared, prompt_count=32, samples=16):
    groups = {
        source: sorted(
            [row for row in prepared["questions"] if row["source"] == source], key=lambda row: row["selection_key"]
        )
        for source in ("MATH-500", "AMC23", "AIME25")
    }
    sources = list(groups)
    selected = []
    for index in range(prompt_count):
        source = sources[index % len(sources)]
        offset = index // len(sources)
        if offset >= len(groups[source]):
            raise ValueError("Prepared question set is too small")
        row = groups[source][offset]
        for sample in range(samples):
            key = f"validation-perf/42/{source}/{row['uid']}/{sample}"
            selected.append(
                {
                    "key": key,
                    "prompt_ids": row["prompt_ids"],
                    "seed": int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"),
                }
            )
    return selected


def clean_metrics(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_metrics(item) for item in value]
    return value


def worker(rank, ordinal, profile, prepared, connection, barrier, directory):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(ordinal),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    os.environ["OPD_VALIDATION_BENCH_ID"] = str(directory)
    from vllm import LLM, SamplingParams

    model = prepared["models"]["initial_student"]
    started = time.perf_counter()
    llm = LLM(
        model=model["path"],
        tokenizer=model["path"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=32768,
        max_num_seqs=profile["sequences"],
        max_num_batched_tokens=profile["batched_tokens"],
        gpu_memory_utilization=profile["budget"],
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=False,
        compilation_config={
            "custom_ops": ["all"],
            "cudagraph_capture_sizes": [
                value for value in (1, 2, 4, 8, 16, 32, 64, 128) if value <= profile["sequences"]
            ],
        },
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    try:
        source = llm.collective_rpc(
            "configure_native_source",
            args=(
                "initial_student",
                model["path"],
                model["fingerprint"],
                "",
                None,
                True,
                False,
            ),
        )[0]
        layout = llm.collective_rpc("native_model_layout")[0]
        frozen = llm.collective_rpc("native_source_status")[0]["frozen_digests"]
        connection.send(
            {
                "state": "ready",
                "rank": rank,
                "cold_seconds": time.perf_counter() - started,
                "source": source,
                "layout": layout,
                "frozen_digests": frozen,
            }
        )
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                after = llm.collective_rpc("native_source_status")[0]["frozen_digests"]
                if after != frozen:
                    raise ValueError("Frozen inference weights changed")
                connection.send({"state": "closed", "rank": rank, "frozen_equal": True})
                break
            rows = command["requests"]
            params = [
                SamplingParams(
                    max_tokens=command["cap"],
                    temperature=0.7,
                    top_p=0.9,
                    top_k=-1,
                    seed=row["seed"],
                    ignore_eos=command["fixed_length"],
                )
                for row in rows
            ]
            llm.reset_prefix_cache()
            llm.collective_rpc("native_cuda_memory", kwargs={"reset": True})
            before = clean_metrics([dataclasses.asdict(metric) for metric in llm.get_metrics()])
            connection.send({"state": "phase_ready", "rank": rank})
            barrier.wait()
            tick = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": row["prompt_ids"]} for row in rows], sampling_params=params, use_tqdm=False
            )
            seconds = time.perf_counter() - tick
            if len(outputs) != len(rows):
                raise ValueError("Inference response count changed")
            records = []
            for row, output in zip(rows, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise ValueError("Validation response expansion happened twice")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                if (
                    not tokens
                    or len(tokens) > command["cap"]
                    or (command["fixed_length"] and len(tokens) != command["cap"])
                ):
                    raise ValueError("Native generation violated its output budget")
                records.append(
                    {
                        "key": row["key"],
                        "response_ids": tokens,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    }
                )
            filename = f"{command['name']}-rank{rank}.jsonl"
            path = Path(directory) / filename
            path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
            connection.send(
                {
                    "state": "completed",
                    "rank": rank,
                    "seconds": seconds,
                    "responses": len(records),
                    "output_positions": sum(len(row["response_ids"]) for row in records),
                    "eos": sum(row["finish_reason"] == "stop" for row in records),
                    "length": sum(row["finish_reason"] == "length" for row in records),
                    "input_positions": sum(len(row["prompt_ids"]) for row in rows),
                    "memory": llm.collective_rpc("native_cuda_memory")[0],
                    "metrics_before": before,
                    "metrics_after": clean_metrics([dataclasses.asdict(metric) for metric in llm.get_metrics()]),
                    "responses_file": filename,
                    "responses_sha256": digest(path),
                }
            )
    finally:
        llm.llm_engine.engine_core.shutdown()
        connection.close()


def receive(connection, process):
    while not connection.poll(1):
        if not process.is_alive():
            raise RuntimeError(f"Generation worker {process.pid} exited {process.exitcode}")
    return connection.recv()


def cleanup_owned_workers(directories):
    """Only terminate processes with this run's unique inherited ownership tag."""
    import psutil

    targets = []
    for process in psutil.process_iter():
        try:
            if process.environ().get("OPD_VALIDATION_BENCH_ID") in directories:
                targets.append((process, process.create_time()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    live = []
    for process, birth in targets:
        try:
            if process.create_time() == birth:
                process.terminate()
                live.append(process)
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(live, timeout=10)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(remaining, timeout=10)
    return [process.pid for process in live]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--natural-cap", type=int, default=8192)
    parser.add_argument("--fixed-cap", type=int, default=128)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.prompt_count, args.samples, args.natural_cap, args.fixed_cap) < 1 or args.natural_cap > 31744:
        parser.error("Invalid validation workload dimensions")
    profiles = [p for p in PROFILES if args.profiles is None or p["name"] in args.profiles]
    if not profiles or (args.profiles is not None and set(args.profiles) != {p["name"] for p in profiles}):
        parser.error("Unknown generation profile")
    prepared = json.loads(args.input.read_text())
    workload = requests(prepared, args.prompt_count, args.samples)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = {
        "state": "planned",
        "scope": "Offline native vLLM inference; no actor updates, grader or KL",
        "input_sha256": digest(args.input),
        "profiles": profiles,
        "responses": len(workload),
        "driver_sha256": digest(Path(__file__)),
        "measurements": [],
        "engines": [],
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.plan_only:
        return
    from native_devices import matches_device, query_devices
    from run_native_acceptance import gpu_rows

    from verl.trainer.ppo.fixed_rollout import model_signature

    if any(used > 128 for _, used, _ in gpu_rows()):
        raise ValueError("Reserved GPU 0–3 are occupied")
    devices = query_devices()
    model = prepared["models"]["initial_student"]
    if model_signature(model["path"]) != model["fingerprint"]:
        raise ValueError("Prepared inference checkpoint changed")
    result["devices"] = devices
    stopped = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(directory, stopped), daemon=True)
    sampler.start()
    context = mp.get_context("spawn")
    processes, connections = [], []
    owned_directories = set()
    try:
        for profile in profiles:
            case_directory = directory / profile["name"]
            case_directory.mkdir()
            owned_directories.add(str(case_directory))
            barrier = context.Barrier(5, timeout=180)
            processes, connections = [], []
            for rank in range(4):
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker,
                    args=(rank, devices[rank]["cuda_ordinal"], profile, prepared, child, barrier, str(case_directory)),
                )
                process.start()
                child.close()
                processes.append(process)
                connections.append(parent)
                receipt = receive(parent, process)
                execution = receipt["source"]["worker_execution"]
                if (
                    not matches_device(execution["device_uuid"], devices[rank])
                    or execution["max_num_seqs"] != profile["sequences"]
                    or execution["max_num_batched_tokens"] != profile["batched_tokens"]
                    or execution["gpu_memory_utilization"] != profile["budget"]
                ):
                    raise ValueError("Actual engine device or limits differ from profile")
                result["engines"].append({"profile": profile["name"], **receipt})
                result["state"] = "initializing"
                (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            phases = [
                ("warmup-fixed", True, True, args.fixed_cap),
                ("fixed-0", True, False, args.fixed_cap),
                ("fixed-1", True, False, args.fixed_cap),
                ("natural", False, False, args.natural_cap),
            ]
            for name, fixed, warmup, cap in phases:
                for rank, connection in enumerate(connections):
                    connection.send(
                        {
                            "kind": "generate",
                            "name": name,
                            "requests": workload[rank::4],
                            "cap": cap,
                            "fixed_length": fixed,
                        }
                    )
                # Detect failed workers before entering a collective barrier.
                # A child can fail while resetting metrics or preparing inputs.
                for connection, process in zip(connections, processes, strict=True):
                    if receive(connection, process).get("state") != "phase_ready":
                        raise ValueError("Generation worker did not prepare this phase")
                barrier.wait()
                started = time.time()
                tick = time.perf_counter()
                rows = [
                    receive(connection, process) for connection, process in zip(connections, processes, strict=True)
                ]
                entry = {
                    "profile": profile["name"],
                    "phase": name,
                    "warmup": warmup,
                    "fixed_length": fixed,
                    "cap": cap,
                    "started_epoch_s": started,
                    "finished_epoch_s": time.time(),
                    "dispatch_wall_seconds": time.perf_counter() - tick,
                    "api_seconds": max(row["seconds"] for row in rows),
                    "ranks": rows,
                }
                if sum(row["responses"] for row in rows) != len(workload):
                    raise ValueError("Global validation response count changed")
                result["measurements"].append(entry)
                result["state"] = "running"
                (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            for connection in connections:
                connection.send({"kind": "close"})
            result.setdefault("closures", []).extend(receive(c, p) for c, p in zip(connections, processes, strict=True))
            for process in processes:
                process.join(timeout=60)
                if process.is_alive() or process.exitcode:
                    raise ValueError("Native worker did not close cleanly")
            for connection in connections:
                connection.close()
            processes, connections = [], []
        result["state"] = "passed"
    except BaseException as error:
        result.update(state="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=15)
        result["cleaned_owned_pids"] = cleanup_owned_workers(owned_directories)
        stopped.set()
        sampler.join(timeout=5)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
