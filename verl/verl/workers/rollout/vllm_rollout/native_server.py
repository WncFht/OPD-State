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
            },
        )
        if len(summaries) != 1 or summaries[0]["path"] != path or summaries[0]["role"] != role:
            raise RuntimeError("Native source initialization requires the matching TP=1 worker")
        self.native_state = GenerationState(role, path, fingerprint)

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
                # AsyncLLM.abort returns after the abort message is submitted.
                # The pool additionally waits for the generating coroutines.
                await asyncio.shield(self.engine.abort(request_id))
            raise
        finally:
            if state is not None:
                state.complete(request_id)

    async def generate_native_batch(self, calls, expected_identity, sampling_params):
        state = getattr(self, "native_state", None)
        if state is None or state.identity != tuple(expected_identity):
            raise RuntimeError("Native batch targeted the wrong source or actor version")
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
                "finish_reason": output.extra_fields["native_finish_reason"],
                "stop_reason": output.extra_fields["native_stop_reason"],
                "num_preempted": output.num_preempted,
            }

        tasks = [asyncio.create_task(one(call)) for call in calls]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        }

    async def drain_native_requests(self, *, abort=False, timeout=60):
        state = getattr(self, "native_state", None)
        if state is None:
            raise RuntimeError("Native policy was not enabled")
        state.close_admission()
        if abort:
            await asyncio.gather(*(self.engine.abort(request_id) for request_id in tuple(state.active)))
        deadline = time.monotonic() + timeout
        while state.active:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native generating coroutines did not drain; engine must not be reused")
            await asyncio.sleep(0.01)
        while self.engine.output_processor.request_states:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native output processor did not drain")
            await asyncio.sleep(0.01)
        barriers = await self.engine.collective_rpc("native_request_barrier")
        if not barriers or not all(b.get("native_device_idle") is True for b in barriers):
            raise RuntimeError("Native device barrier did not acknowledge completion")
        # Resetting the actual scheduler cache is also a native idle check. V1
        # may return None on success; explicit False is the failure value.
        reset_result = await self._reset_prefix_cache()
        if reset_result is False:
            raise RuntimeError("Native prefix cache reset rejected the drained state")
        return True

    async def reopen_native_generation(self):
        self.native_state.reopen()
