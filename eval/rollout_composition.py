"""Inference composition prototype; not wired into the production trainer.

The backend consumes raw token prefixes and preserves its native termination
reason. Routing happens globally, then requests are grouped by model and stage.
There are no model imports, text round trips, or per-token RPCs in this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    visit: int = 0
    sample: int = 0

    def __post_init__(self):
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("A stable nonempty prompt ID is required")
        if any(type(v) is not int or v < 0 for v in (self.visit, self.sample)):
            raise ValueError("Visit and sample must be nonnegative integers")
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))

    @property
    def key(self):
        return self.prompt_id, self.visit, self.sample


@dataclass(frozen=True)
class Single:
    model: str


@dataclass(frozen=True)
class Switch:
    first: Single
    second: Single
    boundary: int


@dataclass(frozen=True)
class Mixture:
    components: tuple[Single | Switch, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class Call:
    key: tuple[str, int, int]
    prompt_ids: tuple[int, ...]
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class Completion:
    key: tuple[str, int, int]
    model: str
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    finish_reason: str
    stop_reason: int | str | None = None


@dataclass(frozen=True)
class Segment:
    model: str
    start: int
    end: int
    finish_reason: str
    stop_reason: int | str | None


@dataclass(frozen=True)
class Trajectory:
    request: Request
    token_ids: tuple[int, ...]
    segments: tuple[Segment, ...]
    route: int
    termination: str

    @property
    def response_mask(self):
        # EOS can equal PAD. Every returned token, including EOS, is valid.
        return (1,) * len(self.token_ids)


def stable_seed(seed, *parts):
    value = repr((seed, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def exact_routes(size, weights, seed):
    """Largest-remainder quota, component-index ties, seeded global positions."""
    if type(size) is not int or size < 0 or not weights:
        raise ValueError("A nonnegative integer batch size and nonempty weights are required")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Mixture weights must be finite, nonnegative and have positive sum")
    # Scale before summing to avoid overflow with large finite weights.
    scaled = [w / max(weights) for w in weights]
    fractions = [size * w / sum(scaled) for w in scaled]
    counts = [math.floor(x) for x in fractions]
    order = sorted(range(len(weights)), key=lambda i: (-(fractions[i] - counts[i]), i))
    for i in order[: size - sum(counts)]:
        counts[i] += 1
    routes = [i for i, count in enumerate(counts) for _ in range(count)]
    random.Random(seed).shuffle(routes)
    return routes


def compile_stages(policy, cap):
    if type(cap) is not int or cap <= 0:
        raise ValueError("The total response budget must be a positive integer")
    if isinstance(policy, Single):
        if not isinstance(policy.model, str) or not policy.model:
            raise ValueError("A nonempty model reference is required")
        return [(policy.model, cap)]
    if not isinstance(policy, Switch):
        raise ValueError("Only single models and one switch are supported as mixture leaves")
    if not isinstance(policy.first, Single) or not isinstance(policy.second, Single):
        raise ValueError("Switch stages must be single-model policies")
    if type(policy.boundary) is not int or policy.boundary < 0:
        raise ValueError("The response-token boundary must be a nonnegative integer")
    # Validate both references, even when an endpoint will be removed.
    compile_stages(policy.first, cap)
    compile_stages(policy.second, cap)
    if policy.first.model == policy.second.model or policy.boundary >= cap:
        return [(policy.first.model, cap)]
    if policy.boundary == 0:
        return [(policy.second.model, cap)]
    return [(policy.first.model, policy.boundary), (policy.second.model, cap - policy.boundary)]


async def generate(
    requests,
    policy,
    backend,
    *,
    cap,
    seed,
    vocab_size,
    eos_token_id,
    context_limits,
    concurrent_groups=False,
):
    """Return original-order trajectories and disjoint group wall timings.

    Backend.generate(model, calls) must return completions in call order. A
    stopped first stage terminates the response. Only a length-completed stage
    with remaining total budget continues. Exceptions return no partial batch.
    """
    started = time.perf_counter()
    requests = list(requests)
    if type(vocab_size) is not int or type(eos_token_id) is not int or not 0 <= eos_token_id < vocab_size:
        raise ValueError("EOS must belong to the valid vocabulary")
    if len({r.key for r in requests}) != len(requests):
        raise ValueError("Duplicate prompt/visit/sample identity in the global batch")
    if isinstance(policy, Mixture):
        if len(policy.components) != len(policy.weights):
            raise ValueError("Every mixture component requires one weight")
        plans = [compile_stages(p, cap) for p in policy.components]
        routes = exact_routes(len(requests), policy.weights, stable_seed(seed, "routing"))
    else:
        plans = [compile_stages(policy, cap)]
        routes = [0] * len(requests)
    for request, route in zip(requests, routes, strict=True):
        if not request.prompt_ids or request.visit < 0 or request.sample < 0:
            raise ValueError("Nonempty prompts and nonnegative visit/sample values are required")
        if any(type(t) is not int or not 0 <= t < vocab_size for t in request.prompt_ids):
            raise ValueError("Prompt contains an invalid token identity")
        cumulative_budget = 0
        for model, budget in plans[route]:
            cumulative_budget += budget
            if model not in context_limits or len(request.prompt_ids) + cumulative_budget > context_limits[model]:
                raise ValueError(f"Total response budget does not fit model context: {model}")
    tokens = [[] for _ in requests]
    segments = [[] for _ in requests]
    terminal = [False] * len(requests)
    timings = []
    for stage_index in range(max((len(p) for p in plans), default=0)):
        groups = defaultdict(list)
        for i, request in enumerate(requests):
            plan = plans[routes[i]]
            if terminal[i] or stage_index >= len(plan):
                continue
            model, budget = plan[stage_index]
            call = Call(
                request.key,
                request.prompt_ids + tuple(tokens[i]),
                budget,
                stable_seed(seed, request.key, model, stage_index),
            )
            groups[model].append((i, call))

        async def run_group(model, rows, phase=stage_index):
            group_started = time.perf_counter()
            outputs = await backend.generate(model, [call for _, call in rows])
            elapsed = time.perf_counter() - group_started
            if len(outputs) != len(rows):
                raise RuntimeError("Backend changed the request count")
            for (i, call), output in zip(rows, outputs, strict=True):
                if output.model != model or output.key != call.key or output.prompt_ids != call.prompt_ids:
                    raise RuntimeError("Backend changed model, request identity, order or raw prefix")
                if output.finish_reason not in {"stop", "length"}:
                    raise RuntimeError(f"Unsuccessful native termination: {output.finish_reason}")
                if len(output.token_ids) > call.max_tokens:
                    raise RuntimeError("Backend exceeded the stage budget")
                if output.finish_reason == "length" and len(output.token_ids) != call.max_tokens:
                    raise RuntimeError("Length termination must exhaust the exact stage budget")
                if any(type(t) is not int or not 0 <= t < vocab_size for t in output.token_ids):
                    raise RuntimeError("Backend returned an invalid token identity")
                if eos_token_id in output.token_ids:
                    if output.token_ids[-1] != eos_token_id or output.token_ids.count(eos_token_id) != 1:
                        raise RuntimeError("Generation continued after EOS")
                offset = len(tokens[i])
                tokens[i].extend(output.token_ids)
                segments[i].append(Segment(model, offset, len(tokens[i]), output.finish_reason, output.stop_reason))
                # V1 checks max_tokens before EOS, so a final EOS may have a
                # native "length" reason. Preserve that reason but never resume.
                terminal[i] = output.finish_reason == "stop" or bool(
                    output.token_ids and output.token_ids[-1] == eos_token_id
                )
            return {
                "model": model,
                "stage": phase,
                "requests": len(rows),
                "input_tokens": sum(len(call.prompt_ids) for _, call in rows),
                "output_tokens": sum(len(out.token_ids) for out in outputs),
                "seconds": elapsed,
            }

        if concurrent_groups:
            tasks = [asyncio.create_task(run_group(model, rows)) for model, rows in groups.items()]
            try:
                timings.extend(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            for model, rows in groups.items():
                timings.append(await run_group(model, rows))
    trajectories = [
        Trajectory(r, tuple(t), tuple(s), route, "eos" if t and t[-1] == eos_token_id else "stop" if done else "length")
        for r, t, s, route, done in zip(requests, tokens, segments, routes, terminal, strict=True)
    ]
    return trajectories, {"seconds": time.perf_counter() - started, "groups": timings}
