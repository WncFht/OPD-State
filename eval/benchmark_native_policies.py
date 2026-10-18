"""Inference-only policy/layout benchmark using production native RPC execution."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from run_native_acceptance import gpu_rows
from verify_native_layout import normalize_uuid

ROOT = Path(__file__).resolve().parents[1]


def validate_options(args):
    if args.layout == "split" and args.replicas == 1:
        raise ValueError("split layout needs at least two GPUs")
    if not args.batches or min(args.batches) < 1 or args.repeats < 1 or not 0 < args.cap <= 4096:
        raise ValueError("Positive batch/repeat counts and a response cap within 4096 are required")
    if not 0 <= args.natural_cap <= 4096 or not 0 < args.budget < 1:
        raise ValueError("Invalid natural response cap or joint GPU memory budget")
    if args.natural_repeats < 1:
        raise ValueError("Natural-length observations need at least one timed repeat")
    if args.max_num_seqs is None:
        args.max_num_seqs = max(args.batches)
    if args.max_num_seqs < 1 or args.max_num_batched_tokens < 1:
        raise ValueError("Native scheduler capacities must be positive")
    if not args.chunked_prefill and args.max_num_batched_tokens < 5120:
        raise ValueError("Without chunked prefill the token budget must cover the 5120-token context")


def check_execution(executions, engines, gpu_uuids):
    if not executions:
        raise ValueError("Benchmark source lacks physical native execution receipts")
    for record in executions:
        gpus = engines[record["slot"]]["gpus"]
        rank = record["replica_rank"]
        if (
            type(rank) is not int
            or not 0 <= rank < len(gpus)
            or normalize_uuid(record["device_uuid"]) != normalize_uuid(gpu_uuids[gpus[rank]])
        ):
            raise ValueError("Benchmark local replica rank does not match its assigned physical GPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=["overlap", "split"], default="overlap")
    parser.add_argument("--replicas", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--natural-cap", type=int, default=4096)
    parser.add_argument("--natural-repeats", type=int, default=1)
    parser.add_argument("--natural-warmup", action="store_true")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["initial_student", "teacher", "switch_p0_q", "switch_q_p0", "batch_mix", "fusion"],
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        validate_options(args)
    except ValueError as error:
        parser.error(str(error))
    if args.layout == "split" and args.policies is not None and "fusion" in args.policies:
        parser.error("Paired fusion uses the overlap model layout")
    if any(used > 128 for gpu, used, _ in gpu_rows() if gpu < args.replicas):
        raise RuntimeError("Requested benchmark GPUs are occupied; wait for the existing job")
    args.output.mkdir(parents=True, exist_ok=False)
    from native_source_snapshot import freeze

    source = freeze(ROOT, args.output, Path(__file__), commit=args.source_commit)
    if "execution_audit" not in (source / "verl/workers/rollout/native_pool.py").read_text():
        raise ValueError("Benchmark requires a source with physical execution receipts")
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VERL_USE_EXTERNAL_PLUGINS="none",
        RAY_ADDRESS="local",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        RAY_TMPDIR="/tmp/opdbm-" + uuid.uuid4().hex[:10],
    )
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from verl.trainer.ppo.fixed_rollout import model_signature
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.native_executor import Request, generate
    from verl.workers.rollout.native_pool import NativeRayBackend
    from verl.workers.rollout.policy import ModelRef, compile_policy
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    paths = {
        r: str(ROOT.parent / "models" / p)
        for r, p in [("initial_student", "DeepSeek-R1-Distill-Qwen-1.5B"), ("teacher", "JustRL-DeepSeek-1.5B")]
    }
    tok = hf_tokenizer(paths["initial_student"])
    refs = {r: ModelRef(r, p, fingerprint=model_signature(p)) for r, p in paths.items()}
    prompt_bytes = (ROOT / "runs/native-rollout-implementation/fresh-math-data-v2/prompts.jsonl").read_bytes()
    records = [json.loads(line) for line in prompt_bytes.splitlines()[: max(args.batches)]]
    if len(records) < max(args.batches):
        raise ValueError("Benchmark prompt pool is smaller than the requested global batch")
    with initialize_config_dir(config_dir=str(source / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer")
    with open_dict(cfg):
        cfg.actor_rollout_ref.model.path = paths["initial_student"]
        rcfg = cfg.actor_rollout_ref.rollout
        rcfg.name = "vllm"
        rcfg.policy = {"kind": "single", "model": "initial_student"}
        rcfg.tensor_model_parallel_size = 1
        rcfg.max_model_len = 5120
        rcfg.prompt_length = 1024
        rcfg.response_length = 4096
        rcfg.max_num_batched_tokens = args.max_num_batched_tokens
        rcfg.max_num_seqs = args.max_num_seqs
        rcfg.enable_chunked_prefill = args.chunked_prefill
        rcfg.load_format = "auto"
        rcfg.enforce_eager = not args.graph
        rcfg.generation_purpose = "training"
        rcfg.generation_sync_namespace = uuid.uuid4().hex
        rcfg.cudagraph_capture_sizes = sorted(
            {n for n in [1, 2, 4, 8, 16, 32, 64, args.max_num_seqs] if n <= args.max_num_seqs}
        )
        rcfg.engine_kwargs.vllm = {"compilation_config": {"custom_ops": ["all"]}}
        cfg.trainer.n_gpus_per_node = 4
    ray.init(num_cpus=16, num_gpus=4, object_store_memory=2 << 30, include_dashboard=False)
    servers = []
    specs = {}
    cold = {}
    gpu_uuids = {
        int(row.split(",")[0]): row.split(",")[1].strip()
        for row in subprocess.check_output(
            ["nvidia-smi", "--id=0,1,2,3", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        ).splitlines()
    }
    stop_sampling = threading.Event()
    sampling_errors = []

    def sample_memory():
        with (args.output / "gpu.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["unix_s", "gpu", "used_mib", "utilization_pct"])
            while not stop_sampling.is_set():
                try:
                    now = time.time()
                    writer.writerows([now, *row] for row in gpu_rows())
                    stream.flush()
                except Exception as error:
                    sampling_errors.append({"unix_s": time.time(), "error": str(error)})
                stop_sampling.wait(2)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    class Pool:
        handles = {}
        awake = set()

        async def prepare(self, slots):
            for slot in slots:
                if slot not in self.awake:
                    await asyncio.gather(*(h.wake_up.remote() for h in self.handles[slot]))
                    self.awake.add(slot)
                await asyncio.gather(*(h.reopen_native_generation.remote() for h in self.handles[slot]))

        async def sleep_all(self):
            await asyncio.gather(*(h.drain_native_requests.remote() for s in self.awake for h in self.handles[s]))
            await asyncio.gather(*(h.sleep.remote() for s in self.awake for h in self.handles[s]))
            self.awake.clear()

        async def abort_and_drain(self):
            return all(
                await asyncio.gather(
                    *(h.drain_native_requests.remote(abort=True) for hs in self.handles.values() for h in hs)
                )
            )

        async def reset(self):
            await asyncio.gather(*(h.clear_kv_cache.remote() for s in self.awake for h in self.handles[s]))

    pool = Pool()
    actor_cls = ray.remote(vLLMHttpServer)

    def create(slot, role, gpus, budget, fusion=None):
        t = time.perf_counter()
        handles = []
        layouts = []
        for gpu in gpus:
            c = OmegaConf.create(OmegaConf.to_container(rcfg, resolve=True))
            c.generation_role = role
            c.generation_fingerprint = refs[role].fingerprint
            c.gpu_memory_utilization = budget
            c.generation_fusion = fusion
            m = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True))
            m.path = paths[role]
            h = actor_cls.options(num_cpus=1, max_concurrency=128).remote(
                config=c,
                model_config=m,
                rollout_mode=RolloutMode.COLOCATED,
                workers=[],
                replica_rank=len(handles),
                node_rank=0,
                gpus_per_node=1,
                nnodes=1,
                cuda_visible_devices=str(gpu),
            )
            handles.append(h)
            servers.append(h)
            # Start one engine at a time: vLLM 0.10 chooses its TCPStore port
            # before binding it, so concurrent cold starts can select the same
            # ephemeral port. Warm-call timing remains parallel below.
            ray.get(h.launch_server.remote())
            layouts.append(ray.get(h.collective_rpc.remote("native_model_layout")))
            ray.get(h.sleep.remote())
        pool.handles[slot] = handles
        specs[slot] = {"gpus": list(gpus), "budget_per_engine": budget, "layouts": [x[0] for x in layouts]}
        cold[slot] = time.perf_counter() - t

    rows = []

    def run(policy, batch, cap, rep, natural):
        asyncio.run(pool.reset())
        for hs in pool.handles.values():
            ray.get(
                [
                    h.collective_rpc.remote(
                        "monkey_patch_model",
                        kwargs={"vocab_size": len(tok), "banned_token_ids": [] if natural else [tok.eos_token_id]},
                    )
                    for h in hs
                ]
            )
        requests = [Request(r["prompt_id"], tuple(r["prompt_ids"]), sample=max(rep, 0)) for r in records[:batch]]
        plan = compile_policy(policy, cap=cap, batch_size=batch, seed=42, batch_id=rep + 1)
        backend = NativeRayBackend(
            pool, {"temperature": 1 if natural else 0, "top_p": 1, "top_k": -1, "logprobs": False}
        )
        started_at = time.time()
        output, timing = asyncio.run(
            generate(
                requests,
                plan,
                backend,
                refs=refs,
                context_limits={r: 5120 for r in refs},
                vocab_size=len(tok),
                eos_token_id=tok.eos_token_id,
                seed=42,
                concurrent_groups=True,
            )
        )
        count = sum(len(o.token_ids) for o in output)
        check_execution(backend.execution_audit, specs, gpu_uuids)
        row = {
            "policy": policy,
            "batch": batch,
            "cap": cap,
            "repeat": rep,
            "sample_index": max(rep, 0),
            "natural_eos": natural,
            "tokens": count,
            "seconds": timing["seconds"],
            "tokens_per_second": count / timing["seconds"],
            "responses_per_second": batch / timing["seconds"],
            "groups": timing["groups"],
            "started_at": started_at,
            "finished_at": time.time(),
            "source_local_placement": backend.placement_audit,
            "physical_execution": backend.execution_audit,
        }
        rows.append(row)
        (args.output / "observations.jsonl").open("a").write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in {"groups", "source_local_placement", "physical_execution"}}
            ),
            flush=True,
        )

    try:
        all_gpus = list(range(args.replicas))
        p_gpus = all_gpus if args.layout == "overlap" else all_gpus[: args.replicas // 2]
        q_gpus = all_gpus if args.layout == "overlap" else all_gpus[args.replicas // 2 :]
        fraction = args.budget / 2 if args.layout == "overlap" else args.budget
        create("initial_student", "initial_student", p_gpus, fraction)
        create("teacher", "teacher", q_gpus, fraction)
        single = [{"kind": "single", "model": r} for r in paths]
        policies = single + [
            {"kind": "sequence_switch", "first": a, "second": b, "boundary": {"kind": "budget_fraction", "value": 0.5}}
            for a, b in [("initial_student", "teacher"), ("teacher", "initial_student")]
        ]
        policies += [
            {
                "kind": "batch_mix",
                "assignment": "exact_count",
                "routing_unit": "trajectory",
                "components": [{"policy": {"kind": "single", "model": r}, "weight": 0.5} for r in paths],
            }
        ]
        names = ["initial_student", "teacher", "switch_p0_q", "switch_q_p0", "batch_mix"]
        for name, policy in zip(names, policies, strict=True):
            if args.policies is not None and name not in args.policies:
                continue
            for batch in args.batches:
                for rep in range(-1, args.repeats):
                    run(policy, batch, args.cap, rep, False)
            if args.natural_cap:
                for rep in range(-1 if args.natural_warmup else 0, args.natural_repeats):
                    run(policy, max(args.batches), args.natural_cap, rep, True)
        asyncio.run(pool.sleep_all())
        if args.layout == "overlap" and (args.policies is None or "fusion" in args.policies):
            fusion = {
                "teacher_path": paths["teacher"],
                "teacher_fingerprint": refs["teacher"].fingerprint,
                "teacher_weight": 0.5,
            }
            create("fusion:initial_student", "initial_student", all_gpus, args.budget, fusion)
            policy = {"kind": "logit_fusion", "base": "initial_student", "teacher_weight": 0.5}
            for batch in args.batches:
                for rep in range(-1, args.repeats):
                    run(policy, batch, args.cap, rep, False)
            if args.natural_cap:
                for rep in range(-1 if args.natural_warmup else 0, args.natural_repeats):
                    run(policy, max(args.batches), args.natural_cap, rep, True)
        summary = {
            "scope": "Inference-only native RPCs/executor; fixed-length runs mask EOS, natural runs retain EOS",
            "arguments": vars(args) | {"output": str(args.output)},
            "engines": specs,
            "cold_seconds": cold,
            "model_refs": {r: ref.identity for r, ref in refs.items()},
            "observations": len(rows),
            "scheduler": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "chunked_prefill": args.chunked_prefill,
            },
            "physical_gpu_uuids": gpu_uuids,
            "physical_receipts_required": True,
            "prompt_manifest_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        }
    finally:
        for h in servers:
            ray.kill(h, no_restart=True)
        ray.shutdown()
        stop_sampling.set()
        sampler.join(timeout=10)
        if sampler.is_alive():
            raise RuntimeError("Native benchmark GPU telemetry did not finish")
    with (args.output / "gpu.csv").open() as stream:
        telemetry = list(csv.DictReader(stream))
    summary["gpu_telemetry"] = {
        "interval_seconds": 2,
        "errors": sampling_errors,
        "sha256": hashlib.sha256((args.output / "gpu.csv").read_bytes()).hexdigest(),
        "sampled_peak_mib": {
            str(gpu): max(int(row["used_mib"]) for row in telemetry if int(row["gpu"]) == gpu) for gpu in range(4)
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
