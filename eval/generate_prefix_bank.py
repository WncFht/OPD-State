"""Generate one immutable p0/q prefix bank using production native vLLM RPCs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from benchmark_native_policies import check_execution
from native_devices import query_devices
from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def bank_server_class():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class BankServer(vLLMHttpServer):
        async def execution_identity(self):
            return {**self.native_worker_execution, "replica_rank": self.replica_rank}

    return BankServer


def engine_configs(rollout, model_config, role, descriptor):
    """Bind the native protocol switch and exact frozen identity together."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(rollout, resolve=True))
    cfg.generation_role, cfg.generation_fingerprint = role, descriptor["fingerprint"]
    cfg.policy = {"kind": "single", "model": role}
    model = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    model.path = descriptor["path"]
    return cfg, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    prepared = json.loads(raw)
    if prepared.get("schema") != "fixed-prefix-input-v1" or not prepared.get("questions"):
        parser.error("A nonempty prepared prefix-bank input is required")
    if not 0 < args.budget < 0.9 or prepared["prompt_cap"] + prepared["response_cap"] > 5120:
        parser.error("Prefix-bank inference needs a valid memory budget and the accepted 5120-token context")
    if any(used > 128 for _, used, _ in gpu_rows()):
        raise RuntimeError("Wait for the serial GPU 0–3 slot before generating a fixed bank")
    physical = query_devices()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from native_source_snapshot import freeze

    source = freeze(ROOT, args.output, Path(__file__), commit=args.source_commit)
    if not (source / "verl/trainer/ppo/prefix_bank.py").is_file():
        raise ValueError("Select a committed native source containing the bank assembler")
    (args.output / "input.json").write_bytes(raw)
    sys.path.insert(0, str(source))
    os.environ.update(
        PYTHONPATH=str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        CUDA_VISIBLE_DEVICES=",".join(str(physical[gpu]["cuda_ordinal"]) for gpu in range(4)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VERL_USE_EXTERNAL_PLUGINS="none",
        RAY_ADDRESS="local",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        RAY_TMPDIR="/tmp/opdbank-" + uuid.uuid4().hex[:10],
    )
    (args.output / "command.json").write_text(
        json.dumps(
            {
                "arguments": vars(args) | {"input": str(args.input.resolve()), "output": str(args.output)},
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "environment": {
                    key: os.environ[key] for key in ("RAY_TMPDIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
                },
                "driver_dependencies_sha256": {
                    name: hashlib.sha256((ROOT / "eval" / name).read_bytes()).hexdigest()
                    for name in ("generate_prefix_bank.py", "native_devices.py", "benchmark_native_policies.py")
                },
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "device-identities.json").write_text(json.dumps(physical, indent=2) + "\n")
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from verl.trainer.ppo.fixed_rollout import model_signature, tokenizer_signature
    from verl.trainer.ppo.prefix_bank import assemble
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.native_executor import Request, generate
    from verl.workers.rollout.native_pool import NativeRayBackend
    from verl.workers.rollout.policy import ModelRef, compile_policy
    from verl.workers.rollout.replica import RolloutMode

    models = prepared["models"]
    tokenizer = hf_tokenizer(models["initial_student"]["path"])
    if tokenizer_signature(tokenizer) != prepared["tokenizer_signature"] or len(tokenizer) != prepared["vocab_size"]:
        raise ValueError("Prepared prefix-bank tokenizer changed")
    for model in models.values():
        if model_signature(model["path"]) != model["fingerprint"]:
            raise ValueError("Prepared prefix-bank model weights changed")
    refs = {role: ModelRef(role, model["path"], fingerprint=model["fingerprint"]) for role, model in models.items()}
    with initialize_config_dir(config_dir=str(source / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    with open_dict(config):
        config.actor_rollout_ref.model.path = models["initial_student"]["path"]
        rollout = config.actor_rollout_ref.rollout
        rollout.name = "vllm"
        rollout.tensor_model_parallel_size = 1
        rollout.max_model_len = prepared["prompt_cap"] + prepared["response_cap"]
        rollout.prompt_length, rollout.response_length = prepared["prompt_cap"], prepared["response_cap"]
        rollout.max_num_seqs = 32
        rollout.max_num_batched_tokens = 8192
        rollout.enable_chunked_prefill = True
        rollout.load_format = "auto"
        rollout.enforce_eager = not args.graph
        rollout.gpu_memory_utilization = args.budget
        rollout.generation_purpose = "training"
        rollout.generation_sync_namespace = uuid.uuid4().hex
        rollout.generation_verify_frozen = True
        rollout.cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32]
        rollout.engine_kwargs.vllm = {"compilation_config": {"custom_ops": ["all"]}}

    class Pool:
        def __init__(self):
            self.handles = {}
            self.awake = set()

        async def prepare(self, placements):
            desired = {(slot, rank) for slot, ranks in placements.items() for rank in ranks}
            stale = self.awake - desired
            await asyncio.gather(*(self.handles[slot][rank].drain_native_requests.remote() for slot, rank in stale))
            await asyncio.gather(*(self.handles[slot][rank].sleep.remote() for slot, rank in stale))
            await asyncio.gather(*(self.handles[slot][rank].wake_up.remote() for slot, rank in desired - self.awake))
            self.awake = desired
            await asyncio.gather(
                *(self.handles[slot][rank].reopen_native_generation.remote() for slot, rank in desired)
            )

        async def sleep_all(self):
            await asyncio.gather(
                *(self.handles[slot][rank].drain_native_requests.remote() for slot, rank in self.awake)
            )
            await asyncio.gather(*(self.handles[slot][rank].sleep.remote() for slot, rank in self.awake))
            self.awake.clear()

        async def abort_and_drain(self):
            return all(
                await asyncio.gather(
                    *(h.drain_native_requests.remote(abort=True) for handles in self.handles.values() for h in handles)
                )
            )

    ray.init(num_cpus=16, num_gpus=4, object_store_memory=2 << 30, include_dashboard=False)
    pool, specs, cold, traces = Pool(), {}, {}, {}
    servers = []
    samples, sampling_errors = [], []
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.is_set():
            try:
                samples.append({"unix_s": time.time(), "gpus": gpu_rows()})
            except Exception as error:
                sampling_errors.append(str(error))
            stop_sampling.wait(2)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    cls = ray.remote(bank_server_class())
    try:
        for role, model in models.items():
            started = time.perf_counter()
            handles = []
            layouts = []
            for gpu in range(4):
                cfg, model_cfg = engine_configs(rollout, config.actor_rollout_ref.model, role, model)
                handle = cls.options(num_cpus=1, max_concurrency=128).remote(
                    config=cfg,
                    model_config=model_cfg,
                    rollout_mode=RolloutMode.COLOCATED,
                    workers=[],
                    replica_rank=gpu,
                    node_rank=0,
                    gpus_per_node=1,
                    nnodes=1,
                    cuda_visible_devices=str(physical[gpu]["cuda_ordinal"]),
                )
                servers.append(handle)
                handles.append(handle)
                # Bind the TCPStore before constructing the next engine.
                ray.get(handle.launch_server.remote())
                execution = ray.get(handle.execution_identity.remote())
                with (args.output / "engine-initialization.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"slot": role, "physical_gpu": gpu, "expected": physical[gpu], "observed": execution}
                        )
                        + "\n"
                    )
                check_execution([{"slot": role, **execution}], {role: {"gpus": [0, 1, 2, 3]}}, physical)
                layouts.extend(ray.get(handle.collective_rpc.remote("native_model_layout")))
                ray.get(handle.sleep.remote())
            pool.handles[role] = handles
            specs[role] = {"gpus": [0, 1, 2, 3], "budget_per_engine": args.budget, "layouts": layouts}
            cold[role] = time.perf_counter() - started

        requests = [Request(row["uid"], tuple(row["prompt_ids"]), row_id=row["uid"]) for row in prepared["questions"]]
        trajectories = {}
        for role in models:
            policy = {"kind": "single", "model": role}
            plan = compile_policy(
                policy,
                cap=prepared["response_cap"],
                batch_size=len(requests),
                seed=prepared["selection_seed"],
                batch_id=0,
            )
            backend = NativeRayBackend(pool, {**prepared["sampling"], "logprobs": False})
            output, timing = asyncio.run(
                generate(
                    requests,
                    plan,
                    backend,
                    refs=refs,
                    context_limits={name: rollout.max_model_len for name in models},
                    vocab_size=prepared["vocab_size"],
                    eos_token_id=prepared["eos_token_id"],
                    seed=prepared["selection_seed"],
                )
            )
            traces[role] = {
                "timing": timing,
                "execution": backend.execution_audit,
                "placement": backend.placement_audit,
                "frozen_verifications": backend.verifications,
            }
            (args.output / f"{role}-execution.json").write_text(json.dumps(traces[role], indent=2) + "\n")
            with (args.output / f"{role}-trajectories.jsonl").open("x") as stream:
                for row in output:
                    stream.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")
            check_execution(backend.execution_audit, specs, physical)
            if not backend.verifications:
                raise RuntimeError("Fixed prefix-bank generation lacks frozen parameter verification")
            trajectories[role] = output
            asyncio.run(pool.sleep_all())
        stop_sampling.set()
        sampler.join(timeout=10)
        provenance = {
            "source_manifest_sha256": hashlib.sha256((args.output / "source-manifest.json").read_bytes()).hexdigest(),
            "driver_sha256": json.loads((args.output / "command.json").read_text())["driver_dependencies_sha256"][
                "generate_prefix_bank.py"
            ],
            "physical_gpu_uuids": {gpu: row["nvml_uuid"] for gpu, row in physical.items()},
            "physical_gpu_identities": physical,
            "driver_dependencies_sha256": json.loads((args.output / "command.json").read_text())[
                "driver_dependencies_sha256"
            ],
            "cold_seconds": cold,
            "engines": specs,
            "generation": traces,
            "graph": args.graph,
            "whole_card_samples": list(samples),
            "sampling_errors": list(sampling_errors),
        }
        bank = assemble(
            prepared, trajectories, input_sha256=hashlib.sha256(raw).hexdigest(), generation_provenance=provenance
        )
        with (args.output / "prefix-bank.json").open("x") as stream:
            json.dump(bank, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "bank": str(args.output / "prefix-bank.json"),
                    "responses": len(bank["records"]),
                    "prefixes": sum(len(r["positions"]) for r in bank["records"]),
                }
            ),
            flush=True,
        )
    finally:
        try:
            asyncio.run(pool.sleep_all())
        finally:
            stop_sampling.set()
            sampler.join(timeout=10)
            try:
                for server in servers:
                    try:
                        ray.kill(server, no_restart=True)
                    except Exception:
                        pass
            finally:
                ray.shutdown()


if __name__ == "__main__":
    main()
