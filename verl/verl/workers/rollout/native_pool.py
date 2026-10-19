"""Native source engines colocated on existing actor GPU placements.

Training and pure-student evaluation have independent engines. Frozen engines
have their own process/weights and weight-preserving sleep; they never receive actor IPC.
The configured memory fraction is a joint budget for concurrent train sources.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict

from opd_state.rollout.executor import Completion
from .placement import place_groups
from opd_state.core.policy import Fusion, parse_policy


class NativeModelPool:
    def __init__(self, config, main_manager):
        self.config = config
        self.eval_slot = "eval:current_student"
        self.replicas = {self.eval_slot: main_manager.get_replicas()}
        self.handles = {self.eval_slot: main_manager.server_handles}
        self.awake = set()
        self.cold_start = {}
        self.actor_slots = {"current_student", self.eval_slot}
        self.fusion_config = None
        self.layout = config.actor_rollout_ref.rollout.get("generation_layout", "overlap")

    async def initialize(self):
        import time

        from omegaconf import OmegaConf

        from verl.workers.rollout.replica import RolloutMode, get_rollout_replica_class

        # Main startup reserves its own KV budget. Sleep it before constructing
        # auxiliaries so the sum of cold-start allocations cannot exceed budget.
        await self.sleep_all()
        rollout = self.config.actor_rollout_ref.rollout
        replica_cls = get_rollout_replica_class("vllm")
        policy = parse_policy(OmegaConf.to_container(rollout.policy, resolve=True), int(rollout.response_length))
        if isinstance(policy, Fusion):
            teacher = rollout.generation_models.teacher
            self.fusion_config = {
                "teacher_path": teacher.path,
                "teacher_fingerprint": teacher.fingerprint,
                "teacher_weight": policy.teacher_weight,
            }
            slot = f"fusion:{policy.base}"
            if policy.base == "current_student":
                self.actor_slots.add(slot)
            sources = [
                (
                    slot,
                    policy.base,
                    {
                        "path": self.config.actor_rollout_ref.model.path,
                        "fingerprint": None
                        if policy.base == "current_student"
                        else rollout.generation_models.initial_student.fingerprint,
                        "memory_fraction": self.config.trainer.native_generation_memory_budget,
                    },
                )
            ]
        else:
            sources = [(role, role, dict(d)) for role, d in rollout.generation_models.items()]
            if "current_student" in self.config.trainer.native_required_models:
                sources.insert(
                    0,
                    (
                        "current_student",
                        "current_student",
                        {
                            "path": self.config.actor_rollout_ref.model.path,
                            "fingerprint": None,
                            "memory_fraction": self.config.trainer.native_generation_memory_budget
                            / len(self.config.trainer.native_required_models),
                        },
                    ),
                )
        for slot, role, descriptor in sources:
            start = time.perf_counter()
            rcfg = OmegaConf.create(OmegaConf.to_container(rollout, resolve=True))
            rcfg.generation_role = role
            rcfg.generation_purpose = "training"
            rcfg.generation_fingerprint = descriptor["fingerprint"]
            rcfg.generation_fusion = self.fusion_config
            rcfg.preserve_source_weights = self.fusion_config is not None
            rcfg.load_format = "dummy" if role == "current_student" and self.fusion_config is None else "auto"
            rcfg.gpu_memory_utilization = descriptor["memory_fraction"]
            rcfg.max_model_len = int(rollout.prompt_length + rollout.response_length)
            rcfg.max_num_batched_tokens = min(int(rollout.max_num_batched_tokens), rcfg.max_model_len)
            mcfg = copy.deepcopy(self.config.actor_rollout_ref.model)
            mcfg.path = descriptor["path"]
            # Bind the checkpoint config and canonical tokenizer independently.
            for key in ["local_path", "hf_config", "generation_config", "tokenizer", "processor"]:
                if key in mcfg:
                    mcfg[key] = None
            if "hf_config_path" in mcfg:
                mcfg.hf_config_path = descriptor["path"]
            if "local_hf_config_path" in mcfg:
                mcfg.local_hf_config_path = None
            if "tokenizer_path" in mcfg:
                mcfg.tokenizer_path = self.config.actor_rollout_ref.model.path

            async def launch(main, rcfg=rcfg, mcfg=mcfg, slot=slot):
                replica = replica_cls(
                    replica_rank=main.replica_rank,
                    config=rcfg,
                    model_config=mcfg,
                    gpus_per_node=main.gpus_per_node,
                    name_suffix=f"native_{slot.replace(':', '_')}",
                )
                # Reuse handles strictly for node/device placement, not weights.
                replica.rollout_mode = RolloutMode.HYBRID if slot in self.actor_slots else RolloutMode.COLOCATED
                replica.workers = list(main.workers)
                await replica.launch_servers()
                await replica.sleep()
                return replica

            replicas = await asyncio.gather(*(launch(main) for main in self.replicas[self.eval_slot]))
            self.replicas[slot] = replicas
            self.handles[slot] = [r._server_handle for r in replicas]
            self.cold_start[slot] = time.perf_counter() - start

    async def prepare(self, slots):
        unknown = set(slots) - self.handles.keys()
        if unknown:
            raise ValueError(f"Native model pool has no initialized slot: {sorted(unknown)}")
        # Current weights are woken by the checkpoint writer, not blindly here:
        # sleep2 discarded them and waking allocations alone is insufficient.
        selected = slots if isinstance(slots, dict) else {s: tuple(range(len(self.handles[s]))) for s in slots}
        if any(
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(self.handles[slot]) for i in indices)
            for slot, indices in selected.items()
        ):
            raise ValueError("Native placement needs distinct valid physical replica indices for every source")

        async def one(slot, index, replica):
            status = await replica._server_handle.native_source_status.remote()
            if index not in selected[slot]:
                if status["awake"]:
                    await replica.sleep()
                return
            if not status["awake"]:
                if slot in self.actor_slots:
                    raise RuntimeError("Selected current replica needs a fresh actor publication before generation")
                await replica.wake_up()
            await replica._server_handle.reopen_native_generation.remote()

        await asyncio.gather(*(one(s, i, r) for s in selected for i, r in enumerate(self.replicas[s])))
        self.awake.update(selected)

    def describe(self):
        return {
            slot: {
                "replicas": len(replicas),
                "role": replicas[0].config.generation_role,
                "purpose": replicas[0].config.generation_purpose,
                "context_limit": replicas[0].config.max_model_len,
                "max_num_batched_tokens": replicas[0].config.max_num_batched_tokens,
                "max_num_seqs": replicas[0].config.max_num_seqs,
                "gpu_memory_utilization": replicas[0].config.gpu_memory_utilization,
                "cold_start_seconds": self.cold_start.get(slot),
            }
            for slot, replicas in self.replicas.items()
        }

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

    def __init__(self, pool, sampling_params, *, validate=False):
        self.pool, self.sampling_params = pool, sampling_params
        self.validate = validate
        self.pending = set()
        self.verifications = []
        self.placements = {}
        self.placement_audit = []
        self.execution_audit = []

    async def prepare_phase(self, stages):
        counts = {}
        for stage, count in stages:
            slot = self.slot(stage)
            counts[slot] = counts.get(slot, 0) + count
        if not counts:
            self.placements = {}
            return
        sizes = {len(self.pool.handles[slot]) for slot in counts}
        if len(sizes) != 1:
            raise ValueError("Native placement requires consistent physical replica indices")
        layout = "overlap" if self.validate else getattr(self.pool, "layout", "overlap")
        self.placements = place_groups(counts, sizes.pop(), layout)
        await self.pool.prepare(self.placements)
        self.placement_audit.append({"layout": layout, "requests": counts, "replicas": self.placements})

    def slot(self, stage):
        if self.validate:
            if stage.models != ("current_student",):
                raise ValueError("Evaluation must use the pure current student engine")
            return self.pool.eval_slot
        return stage.slot

    async def generate(self, stage, calls, refs):
        handles = self.pool.handles[self.slot(stage)]
        shards = [[] for _ in handles]
        selected = self.placements.get(self.slot(stage), tuple(range(len(handles))))
        for i, call in enumerate(calls):
            shards[selected[i % len(selected)]].append((i, call))

        phase = len(self.placement_audit) - 1

        async def fetch(ref, rows, replica_index):
            records = await ref
            if len(records) != len(rows):
                raise RuntimeError("Native server changed replica request count")
            execution = records[0].get("execution")
            if (
                not execution
                or execution.get("replica_rank") != replica_index
                or not execution.get("device_uuid")
                or any(r.get("execution") != execution for r in records)
            ):
                raise RuntimeError("Native responses lack matching physical worker/replica receipts")
            self.execution_audit.append(
                {
                    "slot": self.slot(stage),
                    "phase": phase,
                    "requests": len(rows),
                    "request_ids": [row["request_id"] for row in records],
                    **execution,
                }
            )
            if records and records[0].get("frozen_digests"):
                self.verifications.append({"slot": stage.slot, "frozen_digests": records[0]["frozen_digests"]})
            return [
                (
                    i,
                    Completion(
                        row["request_id"],
                        tuple(row["prompt_ids"]),
                        tuple(row["token_ids"]),
                        tuple(tuple(i) for i in row.get("identities", [row["identity"]])),
                        row["finish_reason"],
                        row["stop_reason"],
                        tuple(row["log_probs"]) if row["log_probs"] is not None else None,
                        row.get("teacher_weight"),
                    ),
                )
                for (i, _), row in zip(rows, records, strict=True)
            ]

        tasks = []
        for replica_index, (handle, rows) in enumerate(zip(handles, shards, strict=True)):
            if rows:
                ref = handle.generate_native_batch.remote(
                    [asdict(call) for _, call in rows],
                    refs[0].identity,
                    dict(self.sampling_params),
                    **(
                        {"expected_companion": refs[1].identity, "teacher_weight": stage.teacher_weight}
                        if len(stage.models) == 2
                        else {}
                    ),
                )
                task = asyncio.create_task(fetch(ref, rows, replica_index))
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
