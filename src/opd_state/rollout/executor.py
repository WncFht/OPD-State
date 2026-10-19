"""Execute a global flat policy against a token-only native inference backend.

Source engines are grouped at stage boundaries, never called through a per-token
RPC. The backend owns native request cancellation and must acknowledge drain
before a failed batch can be followed by a weight update or another activation.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass

from opd_state.core.policy import BatchPlan, ModelRef, Stage, stable_seed


@dataclass(frozen=True)
class Request:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    visit: int = 0
    sample: int = 0
    row_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("A stable nonempty prompt identity is required")
        if any(type(v) is not int or v < 0 for v in (self.visit, self.sample)):
            raise ValueError("Visit and response sample must be nonnegative integers")
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))

    @property
    def key(self):
        return self.prompt_id, self.visit, self.sample


@dataclass(frozen=True)
class Call:
    request_id: str
    key: tuple[str, int, int]
    prompt_ids: tuple[int, ...]
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class Completion:
    request_id: str
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    identities: tuple[tuple, ...]
    finish_reason: str
    stop_reason: int | str | None = None
    log_probs: tuple[float, ...] | None = None
    teacher_weight: float | None = None


@dataclass(frozen=True)
class Segment:
    identities: tuple[tuple, ...]
    start: int
    end: int
    finish_reason: str
    stop_reason: int | str | None
    teacher_weight: float | None = None


@dataclass(frozen=True)
class Trajectory:
    request: Request
    token_ids: tuple[int, ...]
    segments: tuple[Segment, ...]
    route: int
    termination: str
    behavior_log_probs: tuple[float, ...] | None = None

    @property
    def response_mask(self):
        # The valid length, not token != PAD, determines the mask when EOS=PAD.
        return (1,) * len(self.token_ids)


def _validate_batch(requests, plan, refs, context_limits, vocab_size, eos_token_id):
    if (
        type(vocab_size) is not int
        or type(eos_token_id) is not int
        or not 0 <= eos_token_id < vocab_size
    ):
        raise ValueError("EOS must belong to the valid generation vocabulary")
    if len(requests) != len(plan.requests) or len({r.key for r in requests}) != len(
        requests
    ):
        raise ValueError(
            "Global routing cardinality or prompt/visit/sample identity is invalid"
        )
    for role in plan.models:
        if (
            role not in refs
            or not isinstance(refs[role], ModelRef)
            or refs[role].role != role
        ):
            raise ValueError(f"Missing or mismatched model reference: {role}")
    for request, entry in zip(requests, plan.requests, strict=True):
        if not request.prompt_ids or any(
            type(t) is not int or not 0 <= t < vocab_size for t in request.prompt_ids
        ):
            raise ValueError("Prompt must contain valid token IDs")
        consumed = 0
        for stage in entry.stages:
            consumed += stage.max_tokens
            for role in stage.models:
                if (
                    role not in context_limits
                    or len(request.prompt_ids) + consumed > context_limits[role]
                ):
                    raise ValueError(
                        f"Cumulative response budget exceeds {role} context"
                    )
        if consumed != plan.cap:
            raise ValueError(
                "Every compiled request must have exactly one total response budget"
            )


async def generate(
    requests: list[Request],
    plan: BatchPlan,
    backend,
    *,
    refs: dict[str, ModelRef],
    context_limits: dict[str, int],
    vocab_size: int,
    eos_token_id: int,
    seed: int,
    concurrent_groups: bool = False,
):
    """Return original-order trajectories only after the complete batch succeeds.

    ``backend.generate(stage, calls, refs)`` preserves request order, raw token
    input and immutable identities. ``backend.abort_and_drain(ids)`` must return
    True after real engine requests stop; cancelling Python tasks is insufficient.
    """
    requests = list(requests)
    _validate_batch(requests, plan, refs, context_limits, vocab_size, eos_token_id)
    started = time.perf_counter()
    tokens, segments = [[] for _ in requests], [[] for _ in requests]
    scores = [[] for _ in requests]
    have_scores = [True] * len(requests)
    terminal = [False] * len(requests)
    timings, issued = [], []
    for phase in range(max((len(r.stages) for r in plan.requests), default=0)):
        groups = defaultdict(list)
        for i, (request, entry) in enumerate(zip(requests, plan.requests, strict=True)):
            if terminal[i] or phase >= len(entry.stages):
                continue
            stage = entry.stages[phase]
            identities = tuple(refs[m].identity for m in stage.models)
            request_seed = stable_seed(seed, "sample", request.key, stage.slot, phase)
            request_id = f"policy-{plan.batch_id}-{i}-{phase}-{request_seed}"
            call = Call(
                request_id,
                request.key,
                request.prompt_ids + tuple(tokens[i]),
                stage.max_tokens,
                request_seed,
            )
            issued.append(request_id)
            groups[stage].append((i, call, identities))

        async def run_group(stage: Stage, rows, phase=phase):
            tick = time.perf_counter()
            outputs = await backend.generate(
                stage, [r[1] for r in rows], tuple(refs[m] for m in stage.models)
            )
            elapsed = time.perf_counter() - tick
            if len(outputs) != len(rows):
                raise RuntimeError("Native backend changed request cardinality")
            for (i, call, identities), output in zip(rows, outputs, strict=True):
                if (output.request_id, output.prompt_ids, output.identities) != (
                    call.request_id,
                    call.prompt_ids,
                    identities,
                ):
                    raise RuntimeError(
                        "Native backend changed request order, raw prefix or model identity/version"
                    )
                if output.finish_reason not in {"length", "stop"}:
                    raise RuntimeError(
                        f"Native request did not complete: {output.finish_reason}"
                    )
                if not output.token_ids or len(output.token_ids) > call.max_tokens:
                    raise RuntimeError(
                        "Native output has an empty or excessive response length"
                    )
                if (
                    output.finish_reason == "length"
                    and len(output.token_ids) != call.max_tokens
                ):
                    raise RuntimeError(
                        "Native length termination did not exhaust its stage budget"
                    )
                if any(
                    type(t) is not int or not 0 <= t < vocab_size
                    for t in output.token_ids
                ):
                    raise RuntimeError("Native output contains invalid token IDs")
                if eos_token_id in output.token_ids and (
                    output.token_ids[-1] != eos_token_id
                    or output.token_ids.count(eos_token_id) != 1
                ):
                    raise RuntimeError("Native output continued after EOS")
                if output.log_probs is not None and len(output.log_probs) != len(
                    output.token_ids
                ):
                    raise RuntimeError(
                        "Behavior log-probabilities do not align with output tokens"
                    )
                if output.log_probs is not None and not all(
                    math.isfinite(x) for x in output.log_probs
                ):
                    raise RuntimeError("Behavior log-probabilities are not finite")
                if output.teacher_weight != stage.teacher_weight:
                    raise RuntimeError(
                        "Native backend used the wrong fusion coefficient"
                    )
                offset = len(tokens[i])
                tokens[i].extend(output.token_ids)
                segments[i].append(
                    Segment(
                        identities,
                        offset,
                        len(tokens[i]),
                        output.finish_reason,
                        output.stop_reason,
                        stage.teacher_weight,
                    )
                )
                if output.log_probs is None:
                    have_scores[i] = False
                else:
                    scores[i].extend(output.log_probs)
                # vLLM 0.10 checks the length limit before EOS. The raw reason
                # remains length, while the whole response still terminates.
                terminal[i] = (
                    output.finish_reason == "stop"
                    or output.token_ids[-1] == eos_token_id
                )
            return {
                "slot": stage.slot,
                "phase": phase,
                "requests": len(rows),
                "input_tokens": sum(len(call.prompt_ids) for _, call, _ in rows),
                "output_tokens": sum(len(out.token_ids) for out in outputs),
                "seconds": elapsed,
            }

        tasks = []
        try:
            if groups and hasattr(backend, "prepare_phase"):
                await backend.prepare_phase(
                    tuple((stage, len(rows)) for stage, rows in groups.items())
                )
            if concurrent_groups:
                tasks = [
                    asyncio.create_task(run_group(stage, rows))
                    for stage, rows in groups.items()
                ]
                timings.extend(await asyncio.gather(*tasks))
            else:
                for stage, rows in groups.items():
                    timings.append(await run_group(stage, rows))
        except BaseException:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            acknowledged = await asyncio.shield(backend.abort_and_drain(tuple(issued)))
            if acknowledged is not True:
                raise RuntimeError(
                    "Native abort/drain was not acknowledged; weight activation is unsafe"
                ) from None
            raise
    trajectories = [
        Trajectory(
            request,
            tuple(t),
            tuple(s),
            entry.route,
            "eos" if t and t[-1] == eos_token_id else "stop" if stopped else "length",
            tuple(logp) if known else None,
        )
        for request, t, s, entry, stopped, logp, known in zip(
            requests,
            tokens,
            segments,
            plan.requests,
            terminal,
            scores,
            have_scores,
            strict=True,
        )
    ]
    return trajectories, {"seconds": time.perf_counter() - started, "groups": timings}
