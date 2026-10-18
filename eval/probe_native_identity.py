"""Bounded GPU check of production native source ownership and token execution.

One frozen source is owned by each child process. This probes the real worker
extension and production executor, not FSDP or trainer integration. No HF model
forward is used. Only physical GPUs 0--3 may be selected.
"""

import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNED_ENGINES = []


def frozen_write_probe(llm):
    before = llm.collective_rpc("native_source_status")
    try:
        llm.collective_rpc("update_weights_from_ipc", kwargs={"global_steps": 1})
    except Exception as error:
        if "Frozen native" not in str(error):
            raise
    else:
        raise RuntimeError("Frozen source accepted an actor weight stream")
    assert llm.collective_rpc("native_source_status") == before
    return {"rejected_actor_write": True, "role": before[0]["role"]}


def snapshot(gpu):
    row = (
        subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .split(",")
    )
    return {"used_mib": int(row[0]), "total_mib": int(row[1]), "time": time.time()}


def child_body(role, gpu, output):
    os.setsid()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    sys.path.insert(0, str(ROOT / "verl"))
    from vllm import LLM, SamplingParams

    from verl.trainer.ppo.fixed_rollout import digest_json, model_signature
    from verl.workers.rollout.native_executor import Completion, Request, generate
    from verl.workers.rollout.policy import ModelRef, compile_policy

    paths = {
        "initial_student": ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B",
        "teacher": ROOT.parent / "models/JustRL-DeepSeek-1.5B",
    }
    path = str(paths[role].resolve())
    fingerprint = digest_json(model_signature(path))
    reference = ModelRef(role, path, fingerprint=fingerprint)
    t0 = time.perf_counter()
    llm = LLM(
        model=path,
        dtype="bfloat16",
        enforce_eager=True,
        enable_sleep_mode=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.10,
        max_model_len=5120,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        tensor_parallel_size=1,
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    OWNED_ENGINES.append(llm)
    cold = time.perf_counter() - t0
    registered = llm.collective_rpc(
        "configure_native_source", kwargs={"role": role, "path": path, "fingerprint": fingerprint}
    )
    assert len(registered) == 1 and registered[0]["role"] == role
    llm.collective_rpc("monkey_patch_model", kwargs={"vocab_size": 151665})
    raw = [
        json.loads(line)
        for line in (ROOT / "runs/rollout-sources/paired-schedule/prompts.jsonl").read_text().splitlines()[:2]
    ]
    requests = [Request(str(r["prompt_id"]), tuple(r["prompt_ids"])) for r in raw]

    class Backend:
        def __init__(self, force_eos=False):
            self.force_eos = force_eos

        async def generate(self, stage, calls, refs):
            assert stage.models == (role,) and refs[0].identity == reference.identity
            params = [
                SamplingParams(
                    max_tokens=c.max_tokens,
                    temperature=0,
                    top_p=1,
                    seed=c.seed,
                    allowed_token_ids=[151643] if self.force_eos else None,
                )
                for c in calls
            ]
            outputs = llm.generate([{"prompt_token_ids": list(c.prompt_ids)} for c in calls], params, use_tqdm=False)
            return [
                Completion(
                    c.request_id,
                    c.prompt_ids,
                    tuple(o.outputs[0].token_ids),
                    (reference.identity,),
                    o.outputs[0].finish_reason,
                    o.outputs[0].stop_reason,
                )
                for c, o in zip(calls, outputs, strict=True)
            ]

        async def abort_and_drain(self, ids):
            assert not llm.llm_engine.has_unfinished_requests()
            llm.collective_rpc("native_request_barrier")
            return True

    async def run(cap, force_eos=False):
        plan = compile_policy({"kind": "single", "model": role}, cap=cap, batch_size=2, seed=42, batch_id=0)
        return await generate(
            requests,
            plan,
            Backend(force_eos),
            refs={role: reference},
            context_limits={role: 5120},
            vocab_size=151665,
            eos_token_id=151643,
            seed=42,
        )

    before, _ = asyncio.run(run(32))
    denied = frozen_write_probe(llm)
    llm.reset_prefix_cache()
    after, _ = asyncio.run(run(32))
    assert [r.token_ids for r in before] == [r.token_ids for r in after]
    awake = snapshot(gpu)
    llm.sleep(level=1)
    asleep = snapshot(gpu)
    llm.wake_up()
    llm.reset_prefix_cache()
    restored, _ = asyncio.run(run(32))
    assert [r.token_ids for r in before] == [r.token_ids for r in restored]
    eos, _ = asyncio.run(run(1, True))
    assert all(r.token_ids == (151643,) and r.response_mask == (1,) and r.termination == "eos" for r in eos)
    assert all(r.segments[0].finish_reason == "length" for r in eos)
    result = {
        "role": role,
        "path": path,
        "fingerprint": fingerprint,
        "parameter_count": registered[0]["parameter_count"],
        "cold_start_seconds": cold,
        "denied": denied,
        "frozen_probe_unchanged": True,
        "sleep_wake_restored_output": True,
        "eos_equals_pad_preserved": True,
        "native_eos_at_length_limit_reason": "length",
        "awake": awake,
        "asleep": asleep,
        "sleep_released_mib": awake["used_mib"] - asleep["used_mib"],
        "response_sha256": [hashlib.sha256(json.dumps(r.token_ids).encode()).hexdigest() for r in before],
    }
    # Match the production EngineCore process layout, and close it explicitly
    # before interpreter teardown of CUDA allocator state in the probe process.
    llm.llm_engine.engine_core.shutdown()
    OWNED_ENGINES.clear()
    llm = None
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    result["engine_shutdown_called"] = True
    result["vllm_enable_v1_multiprocessing"] = True
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": role, "sleep_released_mib": result["sleep_released_mib"]}), flush=True)


def child(role, gpu, output):
    try:
        child_body(role, gpu, output)
    except BaseException:
        # multiprocessing joins non-daemon EngineCore children before printing
        # the original exception. Print it now and close only our own engines.
        traceback.print_exc()
        raise
    finally:
        for engine in OWNED_ENGINES:
            try:
                engine.llm_engine.engine_core.shutdown()
            except Exception:
                traceback.print_exc()
        OWNED_ENGINES.clear()
        for process in mp.active_children():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = snapshot(args.gpu)
    if before["total_mib"] - before["used_mib"] < 16000:
        raise RuntimeError("Insufficient free memory; do not change another job")
    for role in ["initial_student", "teacher"]:
        p = mp.get_context("spawn").Process(target=child, args=(role, args.gpu, str(args.output / f"{role}.json")))
        p.start()
        p.join()
        if p.exitcode:
            raise RuntimeError(f"Native {role} identity probe failed: {p.exitcode}")
    receipt = {
        "scope": "Native worker plus production executor GPU checks, not trainer/FSDP acceptance",
        "gpu": args.gpu,
        "before": before,
        "after": snapshot(args.gpu),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
