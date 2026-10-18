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
    mode = getattr(args, "weight_mode", "resident_concurrent")
    if mode != "resident_concurrent" and args.layout != "overlap":
        raise ValueError("Weight-residency controls require the common overlap GPU layout")
    if mode == "staged" and getattr(args, "reference_dir", None) is None:
        raise ValueError("Staged weight benchmarks require an independent resident logits reference")


def benchmark_server_class():
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class BenchmarkServer(vLLMHttpServer):
        async def probe_full_logits(self, prompts, identity, vocabulary):
            import numpy as np

            async def one(index, prompt):
                request_id = "weight-reference-" + uuid.uuid4().hex + "-" + str(index)
                self.native_state.admit(request_id, tuple(identity))
                try:
                    stream = self.engine.generate(
                        TokensPrompt(prompt_token_ids=prompt),
                        SamplingParams(max_tokens=1, temperature=0, logprobs=vocabulary),
                        request_id,
                    )
                    final = None
                    async for final in stream:
                        pass
                    scores = final.outputs[0].logprobs[0]
                    return np.array([scores[token].logprob for token in range(vocabulary)], dtype=np.float32)
                finally:
                    self.native_state.complete(request_id)

            output = await asyncio.gather(*(one(i, prompt) for i, prompt in enumerate(prompts)))
            await self.check_native_frozen()
            return output

    return BenchmarkServer


def weight_sender_class():
    class CachedWeightSender:
        def __init__(self, gpu, paths, fingerprints):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            import torch
            from safetensors import safe_open

            from verl.trainer.ppo.fixed_rollout import model_signature

            started = time.perf_counter()
            self.cache = {}
            self.fingerprints = fingerprints
            for role, path in paths.items():
                if model_signature(path) != fingerprints[role]:
                    raise RuntimeError("Frozen weight checkpoint changed before CPU-cache preparation")
                weights = {}
                for file in sorted(Path(path).glob("*.safetensors")):
                    with safe_open(str(file), framework="pt", device="cpu") as reader:
                        for name in reader.keys():
                            if name in weights:
                                raise RuntimeError("Duplicate checkpoint tensor in cached sender")
                            weights[name] = reader.get_tensor(name).pin_memory()
                if not weights or model_signature(path) != fingerprints[role]:
                    raise RuntimeError("Frozen weight checkpoint changed while preparing the CPU cache")
                self.cache[role] = weights
            signatures = [
                {name: (tuple(t.shape), t.dtype) for name, t in weights.items()} for weights in self.cache.values()
            ]
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise RuntimeError("Staged checkpoints have incompatible parameter names/shapes/dtypes")
            self.info = {
                "pid": os.getpid(),
                "physical_gpu": gpu,
                "cpu_cache_seconds": time.perf_counter() - started,
                "parameter_counts": {role: len(weights) for role, weights in self.cache.items()},
                "cpu_pinned_bytes": sum(
                    t.numel() * t.element_size() for weights in self.cache.values() for t in weights.values()
                ),
                "fingerprints": fingerprints,
            }
            torch.cuda.set_device(0)

        def describe(self):
            return self.info

        def publish(self, endpoint, role):
            import torch

            from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

            started = time.perf_counter()

            def weights():
                for name, tensor in self.cache[role].items():
                    device_tensor = tensor.to("cuda:0", non_blocking=True)
                    # Ordinary buckets synchronize before IPC. The large-weight
                    # branch sends a direct tensor handle instead, so finish
                    # that tensor's pinned-host copy before handing it off.
                    if device_tensor.nbytes > 128 << 20:
                        torch.cuda.current_stream().synchronize()
                    yield name, device_tensor

            asyncio.run(BucketedWeightSender(endpoint, bucket_size_mb=128).async_send_weights(weights()))
            torch.cuda.synchronize()
            return {
                "seconds": time.perf_counter() - started,
                "parameters": len(self.cache[role]),
                "fingerprint": self.fingerprints[role],
                "pid": os.getpid(),
            }

    return CachedWeightSender


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
        "--weight-mode", choices=["resident_concurrent", "resident_serial", "staged"], default="resident_concurrent"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Create/reuse resident full-vocabulary numerical references; required for staged",
    )
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
    if args.weight_mode != "resident_concurrent" and args.policies is not None and "fusion" in args.policies:
        parser.error(
            "Weight-residency controls compare single, switched and mixed policies; fusion is a separate paired engine"
        )
    if args.reference_dir is not None:
        args.reference_dir = args.reference_dir.resolve()
    if args.weight_mode == "staged" and not (args.reference_dir / "reference.json").is_file():
        parser.error("Run a resident reference benchmark before the staged benchmark")
    if any(used > 128 for gpu, used, _ in gpu_rows() if gpu < args.replicas):
        raise RuntimeError("Requested benchmark GPUs are occupied; wait for the existing job")
    args.output.mkdir(parents=True, exist_ok=False)
    from native_source_snapshot import freeze

    source = freeze(ROOT, args.output, Path(__file__), commit=args.source_commit)
    if args.weight_mode == "staged" and not (source / "verl/workers/rollout/native_staged.py").is_file():
        raise ValueError("Selected source commit does not contain the staged benchmark adapter")
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
    (args.output / "command.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "driver_pid": os.getpid(),
                "started_at": time.time(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(
                    (args.output / "source-manifest.json").read_bytes()
                ).hexdigest(),
                "environment": {key: os.environ[key] for key in ("RAY_TMPDIR", "CUDA_VISIBLE_DEVICES")},
            },
            indent=2,
        )
        + "\n"
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
        if args.reference_dir is not None:
            rcfg.logprobs_mode = "raw_logits"
            rcfg.engine_kwargs.vllm.max_logprobs = len(tok)
        cfg.trainer.n_gpus_per_node = 4
    ray.init(num_cpus=16, num_gpus=4, object_store_memory=2 << 30, include_dashboard=False)
    servers = []
    senders = []
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
    actor_cls = ray.remote(benchmark_server_class())

    def create(slot, role, gpus, budget, fusion=None):
        t = time.perf_counter()
        handles = []
        layouts = []
        for gpu in gpus:
            c = OmegaConf.create(OmegaConf.to_container(rcfg, resolve=True))
            c.generation_role = role
            c.generation_fingerprint = refs[role].fingerprint if role != "current_student" else None
            c.gpu_memory_utilization = budget
            c.generation_fusion = fusion
            m = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True))
            c.preserve_source_weights = role == "current_student"
            m.path = paths[role] if role != "current_student" else paths["initial_student"]
            h = actor_cls.options(num_cpus=1, max_concurrency=128).remote(
                config=c,
                model_config=m,
                rollout_mode=RolloutMode.HYBRID if role == "current_student" else RolloutMode.COLOCATED,
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
    active_checkpoint = None
    container_version = -1
    sender_cache = []
    numerical_checks = []

    async def completed(refs):
        results = await asyncio.gather(*refs, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]
        return results

    async def activate(ref):
        nonlocal active_checkpoint, container_version
        started = time.perf_counter()
        handles = pool.handles["current_student"]
        status = await completed([h.native_source_status.remote() for h in handles])
        await completed([h.wake_up.remote() for h, state in zip(handles, status, strict=True) if not state["awake"]])
        pool.awake.add("current_student")
        copied = active_checkpoint != ref.identity
        transfers = []
        if copied:
            acknowledgments = await completed([h.drain_native_requests.remote() for h in handles])
            if not all(value is True for value in acknowledgments):
                raise RuntimeError("Staged native servers did not acknowledge drain before publication")
            container_version += 1
            try:
                await completed([h.begin_native_update.remote(container_version) for h in handles])
                receivers = [
                    h.collective_rpc.remote(
                        "update_weights_from_ipc", kwargs={"global_steps": container_version, "use_shm": False}
                    )
                    for h in handles
                ]
                job = ray.get_runtime_context().get_job_id()
                from verl.workers.rollout.sync_endpoint import weight_sync_endpoint

                # All device-local senders run concurrently. Neither CPU cache
                # loading nor disk I/O occurs in this timed activation path.
                transfers = await completed(
                    [
                        sender.publish.remote(
                            weight_sync_endpoint(job, rank, 0, rcfg.generation_sync_namespace), ref.role
                        )
                        for rank, sender in enumerate(senders)
                    ]
                )
                await completed(receivers)
                await completed([h.clear_kv_cache.remote() for h in handles])
                await completed([h.commit_native_update.remote(container_version) for h in handles])
            except BaseException:
                active_checkpoint = None
                await completed([h.fail_native_update.remote() for h in handles])
                raise
            active_checkpoint = ref.identity
        runtime = ModelRef("current_student", paths["initial_student"], version=container_version)
        status = await completed([h.native_source_status.remote() for h in handles])
        if any(
            not state["valid"] or not state["awake"] or state["active"] or tuple(state["identity"]) != runtime.identity
            for state in status
        ):
            raise RuntimeError("Staged publication left an invalid, busy or mismatched mutable native container")
        if any(transfer["fingerprint"] != ref.fingerprint for transfer in transfers):
            raise RuntimeError("Staged sender streamed another checkpoint")
        return {
            "source_identity": ref.identity,
            "runtime_ref": runtime,
            "committed": True,
            "copied": copied,
            "seconds": time.perf_counter() - started,
            "transfers": transfers,
            "server_status": status,
        }

    def numerical_gate(phase):
        if args.reference_dir is None:
            return
        import numpy as np

        prompts = [row["prompt_ids"] for row in records[:2]]
        metadata_path = args.reference_dir / "reference.json"
        binding = {
            "models": {role: list(ref.identity) for role, ref in refs.items()},
            "vocabulary": len(tok),
            "prompts": prompts,
            "logits_mode": "raw_logits",
        }
        existing = metadata_path.is_file()
        if not existing and args.reference_dir.exists() and any(args.reference_dir.iterdir()):
            raise RuntimeError("Resident reference directory is incomplete; preserve it and use a new output directory")
        metadata = (
            json.loads(metadata_path.read_text())
            if existing
            else {
                **binding,
                "arrays": {},
                "producer": {
                    "weight_mode": args.weight_mode,
                    "source_manifest_sha256": hashlib.sha256(
                        (args.output / "source-manifest.json").read_bytes()
                    ).hexdigest(),
                    "graph": args.graph,
                    "max_num_seqs": args.max_num_seqs,
                    "joint_memory_budget": args.budget,
                },
            }
        )
        if {key: metadata.get(key) for key in binding} != binding:
            raise RuntimeError(
                "Resident numerical reference has different checkpoint identities, vocabulary or prefixes"
            )
        args.reference_dir.mkdir(parents=True, exist_ok=True)
        for role, ref in refs.items():
            if args.weight_mode == "staged":
                receipt = asyncio.run(activate(ref))
                identity = receipt["runtime_ref"].identity
                handle = pool.handles["current_student"][0]
            else:
                asyncio.run(pool.prepare([role]))
                identity, handle = ref.identity, pool.handles[role][0]
            ray.get(handle.clear_kv_cache.remote())
            ray.get(
                handle.collective_rpc.remote(
                    "monkey_patch_model", kwargs={"vocab_size": len(tok), "banned_token_ids": []}
                )
            )
            values = np.stack(ray.get(handle.probe_full_logits.remote(prompts, identity, len(tok))))
            path = args.reference_dir / f"{role}.npy"
            created = not existing
            if created:
                np.save(path, values)
                metadata["arrays"][role] = hashlib.sha256(path.read_bytes()).hexdigest()
                expected = values
            else:
                if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["arrays"][role]:
                    raise RuntimeError("Resident reference logits changed after collection")
                expected = np.load(path, allow_pickle=False)
            if (
                values.shape != expected.shape
                or values.shape != (len(prompts), len(tok))
                or not np.isfinite(values).all()
                or not np.isfinite(expected).all()
            ):
                raise RuntimeError("Native numerical reference lacks finite complete-vocabulary logits")

            def probability(logits):
                exponent = np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True))
                return exponent / exponent.sum(-1, keepdims=True)

            error = float(np.abs(values - expected).max())
            tv = float(np.abs(probability(values) - probability(expected)).sum(-1).max() / 2)
            if error > 0.5 or tv > 0.05:
                raise RuntimeError(f"Staged/resident full-vocabulary numerical bound failed: {error=}, {tv=}")
            observed_path = args.output / "numerical" / f"{phase}-{role}.npy"
            observed_path.parent.mkdir(exist_ok=True)
            np.save(observed_path, values)
            numerical_checks.append(
                {
                    "phase": phase,
                    "source": role,
                    "reference_created": created,
                    "max_logit_error": error,
                    "max_tv": tv,
                    "logit_tolerance": 0.5,
                    "tv_tolerance": 0.05,
                    "runtime_identity": identity,
                    "prefix_count": len(prompts),
                    "vocab_size": len(tok),
                    "observed_logits": str(observed_path),
                    "observed_sha256": hashlib.sha256(observed_path.read_bytes()).hexdigest(),
                }
            )
        initial = np.load(args.reference_dir / "initial_student.npy", allow_pickle=False)
        teacher = np.load(args.reference_dir / "teacher.npy", allow_pickle=False)
        if np.max(np.abs(initial - teacher)) <= 0.1:
            raise RuntimeError("The two different resident checkpoints produced indistinguishable reference logits")
        if not existing:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

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
        if args.weight_mode == "staged":
            from verl.workers.rollout.native_staged import StagedNativeBackend

            backend = StagedNativeBackend(backend, activate, paths["initial_student"])
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
                concurrent_groups=args.weight_mode == "resident_concurrent",
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
            "weight_mode": args.weight_mode,
            "source_bindings": getattr(backend, "bindings", []),
        }
        row["weight_activation_seconds"] = sum(binding["seconds"] for binding in row["source_bindings"])
        row["weight_copy_count"] = sum(binding["copied"] for binding in row["source_bindings"])
        row["weight_transfer_seconds"] = sum(
            max((transfer["seconds"] for transfer in binding["transfers"]), default=0)
            for binding in row["source_bindings"]
        )
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
        if args.weight_mode == "staged":
            create("current_student", "current_student", all_gpus, args.budget)
            sender_cls = ray.remote(weight_sender_class())
            senders.extend(
                sender_cls.options(num_cpus=1).remote(gpu, paths, {role: ref.fingerprint for role, ref in refs.items()})
                for gpu in all_gpus
            )
            sender_cache = ray.get([sender.describe.remote() for sender in senders])
        else:
            create("initial_student", "initial_student", p_gpus, fraction)
            create("teacher", "teacher", q_gpus, fraction)
        numerical_gate("before_measurement")
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
        numerical_gate("after_measurement")
        if (
            args.weight_mode == "resident_concurrent"
            and args.layout == "overlap"
            and (args.policies is None or "fusion" in args.policies)
        ):
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
            "arguments": vars(args)
            | {
                "output": str(args.output),
                "reference_dir": str(args.reference_dir) if args.reference_dir is not None else None,
            },
            "engines": specs,
            "cold_seconds": cold,
            "model_refs": {r: ref.identity for r, ref in refs.items()},
            "effective_vocab_size": len(tok),
            "observations": len(rows),
            "scheduler": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "chunked_prefill": args.chunked_prefill,
            },
            "physical_gpu_uuids": gpu_uuids,
            "physical_receipts_required": True,
            "prompt_manifest_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "weight_mode": args.weight_mode,
            "sender_cpu_cache": sender_cache,
            "numerical_checks": numerical_checks,
            "numerical_reference": {
                "directory": str(args.reference_dir),
                "metadata_sha256": hashlib.sha256((args.reference_dir / "reference.json").read_bytes()).hexdigest(),
            }
            if args.reference_dir is not None
            else None,
            "weight_cost_scope": (
                "Cached immutable CPU checkpoint to CUDA IPC, cache reset and commit; no FSDP export or training"
            ),
        }
    finally:
        try:
            for h in servers + senders:
                try:
                    ray.kill(h, no_restart=True)
                except ray.exceptions.RayError:
                    pass
        finally:
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
