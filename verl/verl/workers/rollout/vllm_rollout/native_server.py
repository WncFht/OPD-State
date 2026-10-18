"""Strict source identity and request lifecycle for the existing native server.

No generation happens in this mixin: all token production remains in the
server's original vLLM path. Legacy configurations bypass these opt-in guards.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..lifecycle import GenerationState


class NativePolicyServerMixin:
    async def initialize_native_source(self):
        self.native_state = None
        self._native_generating_tasks = {}
        if self.config.get("policy") is None:
            return
        role = self.config.get("generation_role", "current_student")
        path = str(Path(self.model_config.path).resolve())
        fingerprint = self.config.get("generation_fingerprint")
        summaries = await self.engine.collective_rpc(
            "configure_native_source",
            kwargs={
                "role": role,
                "path": path,
                "fingerprint": fingerprint,
                "sync_namespace": self.config.get("generation_sync_namespace", ""),
                "fusion": self.config.get("generation_fusion"),
                "verify_frozen": self.config.get("generation_verify_frozen", False),
                "verify_addresses": not self.config.get("enforce_eager", True),
            },
        )
        if len(summaries) != 1 or summaries[0]["path"] != path or summaries[0]["role"] != role:
            raise RuntimeError("Native source initialization requires the matching TP=1 worker")
        self.native_state = GenerationState(role, path, fingerprint)
        self.native_fusion = self.config.get("generation_fusion")
        if summaries[0].get("fusion") != self.native_fusion:
            raise RuntimeError("Native worker did not bind the declared fusion")
        companion = summaries[0].get("companion")
        self.native_companion = tuple(companion) if companion is not None else None
        self.native_frozen_digests = summaries[0].get("frozen_digests", {})
        self.native_parameter_addresses = summaries[0].get("parameter_addresses")

    async def generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        priority=0,
        kv_transfer_params=None,
    ):
        state = getattr(self, "native_state", None)
        if state is not None:
            state.admit(request_id, state.identity)
            self._native_generating_tasks[request_id] = asyncio.current_task()
        try:
            output = await self._generate_one(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                kv_transfer_params=kv_transfer_params,
            )
            if state is not None:
                output.extra_fields["generation_identity"] = state.identity
            return output
        except BaseException:
            if state is not None:
                state.close_admission()
                state.drain_pending = True
                # AsyncLLM.abort returns after the abort message is submitted.
                # The pool additionally waits for the generating coroutines.
                await asyncio.shield(self.engine.abort(request_id))
            raise
        finally:
            if state is not None:
                self._native_generating_tasks.pop(request_id)
                state.complete(request_id)

    async def generate_native_batch(
        self, calls, expected_identity, sampling_params, expected_companion=None, teacher_weight=None
    ):
        state = getattr(self, "native_state", None)
        if state is None or state.identity != tuple(expected_identity):
            raise RuntimeError("Native batch targeted the wrong source or actor version")
        if (
            tuple(expected_companion) if expected_companion is not None else None
        ) != self.native_companion or teacher_weight != (
            self.native_fusion["teacher_weight"] if self.native_fusion is not None else None
        ):
            raise RuntimeError("Native batch targeted the wrong companion or fusion coefficient")
        if any(
            c["max_tokens"] < 1 or len(c["prompt_ids"]) + c["max_tokens"] > self.config.max_model_len for c in calls
        ):
            raise ValueError("Native batch does not fit the exact stage/context budget")

        async def one(call):
            params = {**sampling_params, "seed": call["seed"], "max_tokens": call["max_tokens"]}
            output = await self.generate(
                request_id=call["request_id"], prompt_ids=call["prompt_ids"], sampling_params=params
            )
            return {
                "request_id": call["request_id"],
                "prompt_ids": call["prompt_ids"],
                "token_ids": output.token_ids,
                "log_probs": output.log_probs,
                "identity": output.extra_fields["generation_identity"],
                "identities": (output.extra_fields["generation_identity"],)
                + ((self.native_companion,) if self.native_companion is not None else ()),
                "teacher_weight": teacher_weight,
                "finish_reason": output.extra_fields["native_finish_reason"],
                "stop_reason": output.extra_fields["native_stop_reason"],
                "num_preempted": output.num_preempted,
            }

        tasks = [asyncio.create_task(one(call)) for call in calls]
        try:
            rows = await asyncio.gather(*tasks)
            if self.native_frozen_digests or self.native_parameter_addresses is not None:
                await self.check_native_frozen()
                for row in rows:
                    row["frozen_digests"] = dict(self.native_frozen_digests)
                    row["parameter_addresses"] = self.native_parameter_addresses
            return rows
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def check_native_frozen(self, summaries=None):
        if not self.native_frozen_digests and self.native_parameter_addresses is None:
            return
        if summaries is None:
            summaries = await self.engine.collective_rpc("native_source_status")
        if len(summaries) != 1 or summaries[0].get("frozen_digests") != self.native_frozen_digests:
            self.native_state.fail_update()
            raise RuntimeError("Native frozen model parameters changed")
        if (
            self.native_parameter_addresses is not None
            and summaries[0].get("parameter_addresses") != self.native_parameter_addresses
        ):
            self.native_state.fail_update()
            raise RuntimeError("Native parameter addresses changed after graph capture")

    async def begin_native_update(self, version):
        if getattr(self, "native_state", None) is not None:
            self.native_state.begin_update(version)
            summaries = await self.engine.collective_rpc("native_source_status")
            if len(summaries) != 1:
                raise RuntimeError("Native actor publication requires exactly one inference worker")
            self._native_begin_write_serial = summaries[0]["write_serial"]

    async def commit_native_update(self, version):
        if getattr(self, "native_state", None) is None:
            return
        summaries = await self.engine.collective_rpc("native_source_status")
        await self.check_native_frozen(summaries)
        if (
            len(summaries) != 1
            or summaries[0]["actor_version"] != version
            or summaries[0]["write_serial"] <= self._native_begin_write_serial
        ):
            raise RuntimeError("Native worker did not finish the complete actor weight stream")
        barriers = await self.engine.collective_rpc("native_request_barrier")
        if not barriers or not all(b.get("native_device_idle") is True for b in barriers):
            raise RuntimeError("Native weight copies have not completed on the device")
        # The sender explicitly resets prefix/KV metadata after all weights load.
        self.native_state.commit_update(version, cache_reset=True)

    async def fail_native_update(self):
        if getattr(self, "native_state", None) is not None:
            self.native_state.fail_update()

    async def native_source_status(self):
        state = getattr(self, "native_state", None)
        if state is None:
            raise RuntimeError("Native policy was not enabled")
        return {
            "identity": state.identity if state.valid else None,
            "active": len(state.active),
            "awake": state.awake,
            "valid": state.valid,
            "pending_version": state.pending_version,
            "drain_pending": state.drain_pending,
            "source_role": state.role,
            "source_path": state.path,
            "sync_namespace": self.config.get("generation_sync_namespace", ""),
            "fusion": self.native_fusion,
            "purpose": self.config.get("generation_purpose", "evaluation"),
            "verified_frozen_digests": self.native_frozen_digests,
            "parameter_addresses": self.native_parameter_addresses,
        }

    async def drain_native_requests(self, *, abort=False, timeout=60):
        state = getattr(self, "native_state", None)
        if state is None:
            raise RuntimeError("Native policy was not enabled")
        state.close_admission()
        state.drain_pending = True
        deadline = time.monotonic() + timeout

        async def bounded(awaitable):
            try:
                return await asyncio.wait_for(awaitable, timeout=max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError as error:
                raise TimeoutError("Native drain operation did not acknowledge before its deadline") from error

        if abort:
            active = tuple(state.active)
            tasks = [self._native_generating_tasks[rid] for rid in active if rid in self._native_generating_tasks]
            await bounded(asyncio.gather(*(self.engine.abort(request_id) for request_id in active)))
            # V1 0.10 abort removes frontend states without waking the per-request
            # output queue. Cancel and join the actual iterators as well; a Ray
            # future alone does not acknowledge native completion.
            for task in tasks:
                task.cancel()
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=max(0, deadline - time.monotonic()))
                if pending:
                    raise TimeoutError("Native iterator cancellation did not drain; engine must not be reused")
        while state.active:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native generating coroutines did not drain; engine must not be reused")
            await asyncio.sleep(0.01)
        while self.engine.output_processor.request_states:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native output processor did not drain")
            await asyncio.sleep(0.01)
        barriers = await bounded(self.engine.collective_rpc("native_request_barrier"))
        if not barriers or not all(b.get("native_device_idle") is True for b in barriers):
            raise RuntimeError("Native device barrier did not acknowledge completion")
        # Resetting the actual scheduler cache is also a native idle check. V1
        # may return None on success; explicit False is the failure value.
        reset_result = await bounded(self._reset_prefix_cache())
        if reset_result is False:
            raise RuntimeError("Native prefix cache reset rejected the drained state")
        state.drain_pending = False
        return True

    async def reopen_native_generation(self):
        self.native_state.reopen()
