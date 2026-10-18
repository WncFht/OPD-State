"""Native source engines colocated on existing actor GPU placements.

The plain actor engine is also the evaluation engine. Frozen engines have their
own process/weights and weight-preserving sleep; they never receive actor IPC.
The configured memory fraction is a joint budget for concurrent train sources.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict

from .native_executor import Completion


class NativeModelPool:
    def __init__(self, config, main_manager):
        self.config = config
        self.replicas = {"current_student": main_manager.get_replicas()}
        self.handles = {"current_student": main_manager.server_handles}
        self.awake = set()
        self.cold_start = {}

    async def initialize(self):
        import time

        from omegaconf import OmegaConf

        from verl.workers.rollout.replica import RolloutMode, get_rollout_replica_class

        # Main startup reserves its own KV budget. Sleep it before constructing
        # auxiliaries so the sum of cold-start allocations cannot exceed budget.
        await self.sleep_all()
        rollout = self.config.actor_rollout_ref.rollout
        replica_cls = get_rollout_replica_class("vllm")
        for role, descriptor in rollout.generation_models.items():
            if role == "current_student":
                continue
            start = time.perf_counter()
            rcfg = OmegaConf.create(OmegaConf.to_container(rollout, resolve=True))
            rcfg.generation_role = role
            rcfg.generation_fingerprint = descriptor.fingerprint
            rcfg.load_format = "auto"
            rcfg.gpu_memory_utilization = descriptor.memory_fraction
            rcfg.max_model_len = int(rollout.prompt_length + rollout.response_length)
            rcfg.max_num_batched_tokens = min(int(rollout.max_num_batched_tokens), rcfg.max_model_len)
            mcfg = copy.deepcopy(self.config.actor_rollout_ref.model)
            mcfg.path = descriptor.path
            # Bind the checkpoint config and canonical tokenizer independently.
            for key in ["local_path", "hf_config", "generation_config", "tokenizer", "processor"]:
                if key in mcfg:
                    mcfg[key] = None
            if "hf_config_path" in mcfg:
                mcfg.hf_config_path = descriptor.path
            if "local_hf_config_path" in mcfg:
                mcfg.local_hf_config_path = None
            if "tokenizer_path" in mcfg:
                mcfg.tokenizer_path = self.config.actor_rollout_ref.model.path

            async def launch(main, rcfg=rcfg, mcfg=mcfg, role=role):
                replica = replica_cls(
                    replica_rank=main.replica_rank,
                    config=rcfg,
                    model_config=mcfg,
                    gpus_per_node=main.gpus_per_node,
                    name_suffix=f"native_{role}",
                )
                # Reuse handles strictly for node/device placement, not weights.
                replica.rollout_mode = RolloutMode.COLOCATED
                replica.workers = list(main.workers)
                await replica.launch_servers()
                await replica.sleep()
                return replica

            replicas = await asyncio.gather(*(launch(main) for main in self.replicas["current_student"]))
            self.replicas[role] = replicas
            self.handles[role] = [r._server_handle for r in replicas]
            self.cold_start[role] = time.perf_counter() - start

    async def prepare(self, slots):
        unknown = set(slots) - self.handles.keys()
        if unknown:
            raise ValueError(f"Native model pool has no initialized slot: {sorted(unknown)}")
        # Current weights are woken by the checkpoint writer, not blindly here:
        # sleep2 discarded them and waking allocations alone is insufficient.
        for slot in slots:
            if slot != "current_student" and slot not in self.awake:
                await asyncio.gather(*(r.wake_up() for r in self.replicas[slot]))
                self.awake.add(slot)
            await asyncio.gather(*(h.reopen_native_generation.remote() for h in self.handles[slot]))

    async def sleep_all(self):
        async def one(replica):
            status = await replica._server_handle.native_source_status.remote()
            if status["awake"]:
                await replica.sleep()

        await asyncio.gather(*(one(r) for replicas in self.replicas.values() for r in replicas))
        self.awake.clear()

    async def abort_and_drain(self):
        results = await asyncio.gather(
            *(h.drain_native_requests.remote(abort=True) for handles in self.handles.values() for h in handles)
        )
        return all(r is True for r in results)


class NativeRayBackend:
    """One batch RPC per source/replica with shielded native completion futures."""

    def __init__(self, pool, sampling_params):
        self.pool, self.sampling_params = pool, sampling_params
        self.pending = set()

    async def prepare_phase(self, stages):
        await self.pool.prepare({stage.slot for stage, _ in stages})

    async def generate(self, stage, calls, refs):
        if len(stage.models) != 1:
            raise NotImplementedError("Native paired fusion has not been registered in this pool")
        handles = self.pool.handles[stage.slot]
        shards = [[] for _ in handles]
        for i, call in enumerate(calls):
            shards[i % len(handles)].append((i, call))

        async def fetch(ref, rows):
            records = await ref
            if len(records) != len(rows):
                raise RuntimeError("Native server changed replica request count")
            return [
                (
                    i,
                    Completion(
                        row["request_id"],
                        tuple(row["prompt_ids"]),
                        tuple(row["token_ids"]),
                        (tuple(row["identity"]),),
                        row["finish_reason"],
                        row["stop_reason"],
                        tuple(row["log_probs"]) if row["log_probs"] is not None else None,
                    ),
                )
                for (i, _), row in zip(rows, records, strict=True)
            ]

        tasks = []
        for handle, rows in zip(handles, shards, strict=True):
            if rows:
                ref = handle.generate_native_batch.remote(
                    [asdict(call) for _, call in rows], refs[0].identity, dict(self.sampling_params)
                )
                task = asyncio.create_task(fetch(ref, rows))
                self.pending.add(task)
                task.add_done_callback(self.pending.discard)
                tasks.append(task)
        # Cancelling a driver coroutine must not pretend its Ray calls stopped.
        batches = await asyncio.shield(asyncio.gather(*tasks))
        ordered = sorted((row for rows in batches for row in rows), key=lambda x: x[0])
        return [row for _, row in ordered]

    async def abort_and_drain(self, ids):
        del ids  # Engines are dedicated to this source pool; drain all of them.
        ack = await self.pool.abort_and_drain()
        if self.pending:
            await asyncio.gather(*tuple(self.pending), return_exceptions=True)
        return ack
