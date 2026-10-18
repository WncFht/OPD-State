"""Bounded native-vLLM experiment for switched and mixed inference trajectories.

This does not connect to Ray or mutate a training job. Two owned process groups
host independent engines; raw token calls go over a pipe once per source/stage.
Run in the training environment, on an explicitly selected physical GPU 0--3.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from rollout_composition import Call, Completion, Mixture, Request, Single, Switch, generate

ROOT = Path(__file__).resolve().parents[1]


def reset_idle_engine(llm):
    """V1 0.10 wrappers return None, despite LLM's bool return annotation.

    This probe disables prefix caching. An APC-enabled production adapter still
    needs a positive scheduler acknowledgement before publishing new weights.
    """
    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError("Do not replace model weights while requests remain active")
    result = llm.reset_prefix_cache()
    if result is False:
        raise RuntimeError("Prefix-cache reset explicitly failed")
    return result


def load_native_weights(worker, weights):
    """Only called in the owned in-process engine, with CPU data already local."""
    import torch

    native = worker.model_runner.model
    with torch.no_grad():
        loaded = native.load_weights(weights.items())
    missing = set(dict(native.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"Native loader left parameters unfilled: {sorted(missing)[:5]}")
    return {"loaded_parameters": len(loaded)}


def gpu_snapshot(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    total, used, busy = map(int, result.stdout.strip().split(","))
    return {"time": time.time(), "total_mib": total, "used_mib": used, "busy_percent": busy}


def engine_worker(connection, model, path, gpu, memory_fraction, max_context, vocab_size, eos, deterministic):
    os.setsid()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="0" if deterministic else "1",
        VLLM_LOGGING_LEVEL="WARNING",
        TOKENIZERS_PARALLELISM="false",
    )
    try:
        from vllm import LLM, SamplingParams

        started = time.perf_counter()
        llm = LLM(
            model=path,
            tokenizer=path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            gpu_memory_utilization=memory_fraction,
            enforce_eager=True,
            enable_sleep_mode=True,
            enable_prefix_caching=False,
            max_model_len=max_context,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            disable_log_stats=True,
            disable_custom_all_reduce=True,
            seed=42,
        )
        connection.send(
            {"ready": True, "model": model, "cold_start_seconds": time.perf_counter() - started, "pid": os.getpid()}
        )
        allowed = list(range(vocab_size))
        active_model = model
        cpu_weights = {}
        while True:
            message = connection.recv()
            if message["op"] == "close":
                break
            if message["op"] == "activate":
                if not deterministic:
                    raise RuntimeError("Bulk CPU weights stay local to the owned in-process test engine")
                from safetensors.torch import load_file

                target = message["path"]
                before = time.perf_counter()
                cached = target in cpu_weights
                if not cached:
                    weights = {}
                    for file in sorted(Path(target).glob("*.safetensors")):
                        weights.update(load_file(file, device="cpu"))
                    if not weights:
                        raise RuntimeError("No dense safetensor weights found")
                    cpu_weights[target] = weights
                cpu_seconds = time.perf_counter() - before
                active_model = None
                before = time.perf_counter()
                loaded = llm.collective_rpc(load_native_weights, kwargs={"weights": cpu_weights[target]})
                import torch

                torch.cuda.synchronize()
                copy_seconds = time.perf_counter() - before
                reset_result = reset_idle_engine(llm)
                active_model = message["source_name"]
                connection.send(
                    {
                        "cpu_load_seconds": cpu_seconds,
                        "copy_seconds": copy_seconds,
                        "cpu_cache_hit": cached,
                        "loaded": loaded,
                        "prefix_reset_api_return": reset_result,
                        "prefix_caching_enabled": False,
                    }
                )
                continue
            if message["op"] in {"sleep", "wake"}:
                before = time.perf_counter()
                if message["op"] == "sleep":
                    llm.sleep(level=1)
                else:
                    llm.wake_up()
                connection.send({"seconds": time.perf_counter() - before})
                continue
            calls, controls = message["calls"], message.get("controls", {})
            params = [
                SamplingParams(
                    temperature=controls.get("temperature", 0.0),
                    top_p=1,
                    top_k=-1,
                    max_tokens=c.max_tokens,
                    seed=c.seed,
                    logprobs=5 if controls.get("diagnostic_logprobs") else None,
                    allowed_token_ids=[eos] if controls.get("force_eos") else allowed,
                )
                for c in calls
            ]
            before = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": list(c.prompt_ids)} for c in calls],
                params,
                use_tqdm=False,
            )
            records = []
            for call, output in zip(calls, outputs, strict=True):
                choice = output.outputs[0]
                records.append(
                    Completion(
                        call.key,
                        active_model,
                        tuple(output.prompt_token_ids),
                        tuple(choice.token_ids),
                        choice.finish_reason,
                        choice.stop_reason,
                    )
                )
            diagnostics = None
            if controls.get("diagnostic_logprobs"):
                diagnostics = [
                    [{token: value.logprob for token, value in row.items()} for row in output.outputs[0].logprobs]
                    for output in outputs
                ]
            connection.send(
                {"records": records, "seconds": time.perf_counter() - before, "diagnostic_logprobs": diagnostics}
            )
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class Engine:
    def __init__(self, model, path, args, vocab_size, eos):
        parent, child = mp.get_context("spawn").Pipe()
        self.connection = parent
        self.process = mp.get_context("spawn").Process(
            target=engine_worker,
            args=(
                child,
                model,
                str(path),
                args.gpu,
                args.memory_fraction,
                args.context,
                vocab_size,
                eos,
                args.mode == "correctness",
            ),
        )
        self.process.start()
        child.close()
        self.lock = threading.Lock()
        try:
            self.ready = self.receive(180)
        except BaseException:
            self.close()
            raise

    def receive(self, timeout=120):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Owned inference process {self.process.pid} exceeded {timeout} seconds")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, op, **kwargs):
        with self.lock:
            self.connection.send({"op": op, **kwargs})
            return self.receive()

    def close(self):
        # Only this worker's process group is owned. Never stop Ray/global jobs.
        if self.process.is_alive():
            try:
                self.connection.send({"op": "close"})
                self.process.join(8)
            except (BrokenPipeError, EOFError):
                pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.process.join(3)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.connection.close()


class Backend:
    def __init__(self, engines, trace_path):
        self.engines = engines
        self.trace_path = trace_path
        self.controls = {}
        self.trace = []

    async def generate(self, model, calls):
        result = await asyncio.to_thread(
            self.engines[model].call, "generate", calls=calls, controls=self.controls.get(model, {})
        )
        entry = {
            "model": model,
            "calls": [asdict(c) for c in calls],
            "outputs": [asdict(r) for r in result["records"]],
            "engine_seconds": result["seconds"],
        }
        self.trace.append(entry)
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        return result["records"]


async def experiment(args, requests, engines, vocab_size, eos):
    backend = Backend(engines, args.output / "raw-calls.jsonl")
    cases, checks = [], []

    def record_check(record):
        checks.append(record)
        pending = args.output / "checks.pending.json"
        pending.write_text(json.dumps(checks, indent=2) + "\n")
        pending.replace(args.output / "checks.json")

    async def run(name, policy, rows=requests, cap=None, concurrent=False):
        backend.trace = []
        result, stats = await generate(
            rows,
            policy,
            backend,
            cap=cap or args.cap,
            seed=42,
            vocab_size=vocab_size,
            eos_token_id=eos,
            context_limits={name: args.context for name in engines},
            concurrent_groups=concurrent,
        )
        record = {
            "name": name,
            "policy": asdict(policy),
            "batch": len(rows),
            "cap": cap or args.cap,
            "concurrent_groups": concurrent,
            "tokens": sum(len(r.token_ids) for r in result),
            "input_tokens": sum(g["input_tokens"] for g in stats["groups"]),
            **stats,
            "trajectories": [asdict(r) for r in result],
        }
        record["output_tokens_per_second"] = record["tokens"] / stats["seconds"]
        cases.append(record)
        with (args.output / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({k: record[k] for k in ("name", "batch", "tokens", "seconds")}), flush=True)
        return result, list(backend.trace)

    async def check_replay(name, trace):
        for entry in trace:
            calls = [Call(tuple(c["key"]), tuple(c["prompt_ids"]), c["max_tokens"], c["seed"]) for c in entry["calls"]]
            actual = await backend.generate(entry["model"], calls)
            expected = [tuple(r["token_ids"]) for r in entry["outputs"]]
            assert [r.token_ids for r in actual] == expected, f"Native deterministic replay failed: {name}"
        record_check({"check": name + "/native_group_replay", "passed": True, "groups": len(trace)})

    async def benchmark():
        for size in [2, 8, 16]:
            rows = requests[:size]
            for repeat in range(args.repeats):
                for name, policy, concurrent in [
                    ("single", Single("student"), False),
                    ("mixture-serial", Mixture((Single("teacher"), Single("student")), (1, 1)), False),
                    ("mixture-concurrent", Mixture((Single("teacher"), Single("student")), (1, 1)), True),
                    ("switch", Switch(Single("teacher"), Single("student"), args.cap // 2), False),
                ]:
                    await run(f"benchmark/b{size}/{repeat}/{name}", policy, rows, concurrent=concurrent)

    for model in engines:
        await run(f"warmup/{model}", Single(model), requests[:2], cap=8)
    if args.mode == "throughput":
        await benchmark()
        return {
            "cases": cases,
            "checks": [],
            "lifecycle": [],
            "all_checks_passed": True,
            "scope": "Throughput only; bitwise repeatability is not asserted for multiprocessing scheduling.",
        }
    references = {}
    for model in engines:
        references[model], _ = await run(f"single/{model}", Single(model))
    for first, second in [("teacher", "student"), ("student", "teacher")]:
        for boundary in [0, 1, args.cap // 2, args.cap, args.cap + 5]:
            rows, trace = await run(
                f"switch/{first}-{second}/k{boundary}", Switch(Single(first), Single(second), boundary)
            )
            if boundary in {0, args.cap, args.cap + 5}:
                target = second if boundary == 0 else first
                assert [r.token_ids for r in rows] == [r.token_ids for r in references[target]]
                record_check({"check": f"endpoint/{first}-{second}/{boundary}", "passed": True})
            else:
                await check_replay(f"switch/{first}-{second}/{boundary}", trace)

    for teacher_fraction in [0, 0.25, 0.5, 0.75, 1]:
        policy = Mixture((Single("teacher"), Single("student")), (teacher_fraction, 1 - teacher_fraction))
        rows, trace = await run(f"mixture/teacher-{teacher_fraction}", policy)
        await check_replay(f"mixture/{teacher_fraction}", trace)
        record_check(
            {
                "check": f"mixture/{teacher_fraction}/original_order",
                "passed": [r.request.key for r in rows] == [r.key for r in requests],
                "teacher_responses": sum(r.route == 0 for r in rows),
            }
        )
    nested = Mixture((Single("teacher"), Switch(Single("student"), Single("teacher"), args.cap // 2)), (1, 1))
    _, trace = await run("mixture/pure-teacher-and-switched", nested)
    await check_replay("nested", trace)

    # Native EOS at exactly the stage boundary must prevent the second call.
    backend.controls["teacher"] = {"force_eos": True}
    rows, trace = await run("termination/eos-at-k1", Switch(Single("teacher"), Single("student"), 1))
    assert len(trace) == 1 and all(r.token_ids == (eos,) and r.response_mask == (1,) for r in rows)
    record_check(
        {
            "check": "native_eos_at_boundary_no_continuation",
            "passed": True,
            "native_finish_reasons": sorted({r.segments[0].finish_reason for r in rows}),
        }
    )
    backend.controls.clear()

    # Same-model raw continuation versus uninterrupted native generation.
    direct = references["student"]
    prefix_calls = [Call(r.key, r.prompt_ids, args.cap // 2, 42) for r in requests]
    first = await backend.generate("student", prefix_calls)
    continuation_calls = [
        Call(r.key, r.prompt_ids + out.token_ids, args.cap - len(out.token_ids), 42)
        for r, out in zip(requests, first, strict=True)
        if out.finish_reason == "length"
    ]
    second = await backend.generate("student", continuation_calls) if continuation_calls else []
    suffixes = {r.key: r.token_ids for r in second}
    combined = [out.token_ids + suffixes.get(out.key, ()) for out in first]
    record_check(
        {
            "check": "native_same_model_split_vs_uninterrupted",
            "passed": combined == [r.token_ids for r in direct],
            "responses": len(combined),
        }
    )

    for model in engines:
        backend.controls[model] = {"temperature": 1.0}
    a, _ = await run("sampling/fixed-seed-first", nested)
    b, _ = await run("sampling/fixed-seed-repeat", nested)
    record_check(
        {"check": "fixed_seed_sampling_replay", "passed": [r.token_ids for r in a] == [r.token_ids for r in b]}
    )
    backend.controls.clear()

    if args.long_cap:
        long_rows = requests[:2]
        for name, policy in [
            ("student", Single("student")),
            ("teacher", Single("teacher")),
            ("teacher-student", Switch(Single("teacher"), Single("student"), args.long_cap // 2)),
            ("student-teacher", Switch(Single("student"), Single("teacher"), args.long_cap // 2)),
            ("batch-half", Mixture((Single("teacher"), Single("student")), (1, 1))),
        ]:
            rows, _ = await run(f"long-context/{name}", policy, long_rows, cap=args.long_cap)
            record_check(
                {
                    "check": f"long-context/{name}/budget-and-source-contract",
                    "passed": all(0 < len(r.token_ids) <= args.long_cap for r in rows),
                }
            )

    # Paired, interleaved runs keep the shared-load comparison local in time.
    await benchmark()
    lifecycle = []
    for model in engines:
        sleep = await asyncio.to_thread(engines[model].call, "sleep")
        sleeping = gpu_snapshot(args.gpu)
        wake = await asyncio.to_thread(engines[model].call, "wake")
        awakened = gpu_snapshot(args.gpu)
        after, _ = await run(f"lifecycle/{model}/after-wake", Single(model))
        equal = [r.token_ids for r in after] == [r.token_ids for r in references[model]]
        record_check({"check": f"{model}/sleep1-wake-reference", "passed": equal})
        lifecycle.append(
            {
                "model": model,
                "sleep_seconds": sleep["seconds"],
                "wake_seconds": wake["seconds"],
                "whole_gpu_sleeping": sleeping,
                "whole_gpu_awakened": awakened,
            }
        )
    weight_switches = []
    if args.weight_switch:
        await asyncio.to_thread(engines["teacher"].call, "sleep")
        # Route only the selected role to one owned native engine at a time.
        backend.engines = {name: engines["student"] for name in engines}
        for _ in range(2):
            for target in ["teacher", "student"]:
                transfer = await asyncio.to_thread(
                    engines["student"].call,
                    "activate",
                    source_name=target,
                    path=args.model_paths[target],
                )
                after, _ = await run(f"weight-switch/{target}/{len(weight_switches)}", Single(target))
                equal = [r.token_ids for r in after] == [r.token_ids for r in references[target]]
                record_check({"check": f"weight-switch/{target}/{len(weight_switches)}/reference", "passed": equal})
                weight_switches.append({"target": target, **transfer})
        backend.engines = engines
        await asyncio.to_thread(engines["teacher"].call, "wake")
        after, _ = await run("weight-switch/frozen-teacher-reference", Single("teacher"))
        record_check(
            {
                "check": "weight-switch/other-teacher-engine-unchanged",
                "passed": [r.token_ids for r in after] == [r.token_ids for r in references["teacher"]],
            }
        )
    return {
        "cases": cases,
        "checks": checks,
        "lifecycle": lifecycle,
        "weight_switches": weight_switches,
        "all_checks_passed": all(c["passed"] for c in checks),
    }


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned probe received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mode", choices=["correctness", "throughput"], default="correctness")
    parser.add_argument("--long-cap", type=int, default=0)
    parser.add_argument("--weight-switch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not 0 < args.memory_fraction <= 0.12 or args.cap < 2 or args.repeats < 1:
        raise ValueError("This bounded probe requires fraction <= .12, cap >= 2 and repeats >= 1")
    before = gpu_snapshot(args.gpu)
    required = 2 * args.memory_fraction * before["total_mib"] + 4096
    if before["total_mib"] - before["used_mib"] < required:
        raise RuntimeError("Insufficient joint engine headroom; do not change or stop other jobs")
    from transformers import AutoTokenizer

    model_paths = {
        "student": ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B",
        "teacher": ROOT.parent / "models/JustRL-DeepSeek-1.5B",
    }
    args.model_paths = {name: str(path) for name, path in model_paths.items()}
    tokenizer, teacher_tokenizer = [AutoTokenizer.from_pretrained(p) for p in model_paths.values()]
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    assert tokenizer.eos_token_id == teacher_tokenizer.eos_token_id
    schedule = ROOT / "runs/rollout-sources/paired-schedule/prompts.jsonl"
    rows = [json.loads(line) for line in schedule.read_text().splitlines()[:16]]
    requests = [Request(str(r["prompt_id"]), tuple(r["prompt_ids"])) for r in rows]
    assert max(len(r.prompt_ids) for r in requests) + args.cap <= args.context
    metadata = {
        "gpu": args.gpu,
        "baseline": before,
        "memory_fraction_per_engine": args.memory_fraction,
        "max_model_len": args.context,
        "response_cap": args.cap,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 2048,
        "enforce_eager": True,
        "prefix_caching": False,
        "shared_gpu_load": True,
        "mode": args.mode,
        "long_response_cap": args.long_cap,
        "weight_switch_test": args.weight_switch,
        "vllm_enable_v1_multiprocessing": args.mode != "correctness",
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "models": {k: str(v.resolve()) for k, v in model_paths.items()},
        "schedule_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), ROOT / "eval/rollout_composition.py"]
        },
    }
    engines, samples, stop = {}, [], threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(gpu_snapshot(args.gpu))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for name, path in model_paths.items():
            engines[name] = Engine(name, path, args, len(tokenizer), tokenizer.eos_token_id)
            metadata.setdefault("cold_starts", {})[name] = engines[name].ready
            print(json.dumps({"engine_ready": name, **engines[name].ready}), flush=True)
        result = asyncio.run(experiment(args, requests, engines, len(tokenizer), tokenizer.eos_token_id))
        result["metadata"] = metadata
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["all_checks_passed"]:
            raise RuntimeError("One or more native checks failed; inspect summary.json before interpreting timings")
    finally:
        for engine in engines.values():
            engine.close()
        metadata["after_cleanup"] = gpu_snapshot(args.gpu)
        stop.set()
        thread.join(12)
        (args.output / "gpu.json").write_text(json.dumps(samples, indent=2) + "\n")
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
