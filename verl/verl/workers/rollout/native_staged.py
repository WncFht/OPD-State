"""Experimental inference benchmark adapter for one mutable native model slot.

This is not selected by the trainer. Immutable p0/q references remain logical
checkpoint identities; a separate publication receipt binds them to a mutable
current-student container/version. Only that container accepts complete native
weight streams. Frozen native workers are never made writable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from opd_state.core.policy import ModelRef, Stage


class StagedNativeBackend:
    def __init__(self, delegate, activate, container_path):
        self.delegate, self.activate, self.container_path = delegate, activate, container_path
        self.lock = asyncio.Lock()
        self.publications = set()
        self.bindings = []
        self.last = None
        self.phase = -1

    @property
    def execution_audit(self):
        return self.delegate.execution_audit

    @property
    def placement_audit(self):
        return self.delegate.placement_audit

    @property
    def verifications(self):
        return self.delegate.verifications

    async def prepare_phase(self, stages):
        # The shared container cannot be activated for multiple source groups
        # here. Prepare its real placement inside the serialized transaction.
        if any(len(stage.models) != 1 or stage.models[0] == "current_student" for stage, _ in stages):
            raise ValueError("Staged benchmark accepts only single frozen-checkpoint stages")
        self.phase += 1

    async def generate(self, stage, calls, refs):
        if (
            len(stage.models) != 1
            or len(refs) != 1
            or refs[0].role != stage.models[0]
            or refs[0].role == "current_student"
        ):
            raise ValueError("Staged benchmark requires one matching immutable checkpoint reference")
        async with self.lock:
            task = asyncio.create_task(self.activate(refs[0]))
            self.publications.add(task)
            try:
                receipt = await asyncio.shield(task)
            finally:
                if task.done():
                    self.publications.discard(task)
            runtime = receipt.get("runtime_ref")
            if (
                receipt.get("source_identity") != refs[0].identity
                or receipt.get("committed") is not True
                or not isinstance(runtime, ModelRef)
                or runtime.role != "current_student"
                or runtime.path != self.container_path
            ):
                raise RuntimeError(
                    "Staged publication did not bind the requested checkpoint to its committed container"
                )
            if self.last is not None:
                previous_source, previous_version = self.last
                if runtime.version < previous_version or (
                    refs[0].identity != previous_source and runtime.version <= previous_version
                ):
                    raise RuntimeError("Staged checkpoint changes require a new committed container version")
            self.last = refs[0].identity, runtime.version
            self.bindings.append(
                {**{k: v for k, v in receipt.items() if k != "runtime_ref"}, "runtime_identity": runtime.identity}
            )
            physical = Stage(("current_student",), stage.max_tokens)
            await self.delegate.prepare_phase(((physical, len(calls)),))
            first_execution = len(self.delegate.execution_audit)
            started = time.perf_counter()
            outputs = await self.delegate.generate(physical, calls, (runtime,))
            self.bindings[-1]["native_generate_seconds"] = time.perf_counter() - started
            if any(output.identities != (runtime.identity,) for output in outputs):
                raise RuntimeError("Staged native output came from a different mutable container/version")
            for execution in self.delegate.execution_audit[first_execution:]:
                execution.update(
                    logical_slot=stage.slot,
                    logical_phase=self.phase,
                    logical_identity=refs[0].identity,
                    runtime_identity=runtime.identity,
                    publication_index=len(self.bindings) - 1,
                )
            return [replace(output, identities=(refs[0].identity,)) for output in outputs]

    async def abort_and_drain(self, ids):
        # Cancelling asyncio.to_thread does not stop CUDA IPC. Wait for every
        # in-flight publisher before asking the native server to acknowledge
        # drain; the owner must abandon the benchmark if publication failed.
        if self.publications:
            await asyncio.shield(asyncio.gather(*tuple(self.publications), return_exceptions=True))
            self.publications.clear()
        return await self.delegate.abort_and_drain(ids)
