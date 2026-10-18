"""Flat rollout policies, global routing and immutable model identities.

This module deliberately has no torch, Ray or model imports. A plan is compiled
for the global batch before DP slicing. It describes response-token budgets;
the executor must still stop the whole response on EOS or a semantic stop.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

MODEL_ROLES = frozenset({"current_student", "initial_student", "teacher"})


def _model(value):
    if not isinstance(value, str) or value not in MODEL_ROLES:
        raise ValueError(f"Unknown generation model role: {value!r}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def stable_seed(seed: int, *identity) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("Run seed must be a nonnegative integer")
    return int.from_bytes(hashlib.sha256(repr((seed, identity)).encode()).digest()[:4], "big")


@dataclass(frozen=True)
class ModelRef:
    role: str
    path: str
    version: int | None = None
    fingerprint: str | None = None

    def __post_init__(self):
        _model(self.role)
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Every model reference needs its canonical checkpoint path")
        if self.role == "current_student":
            if type(self.version) is not int or self.version < 0 or self.fingerprint is not None:
                raise ValueError("Current student requires an actor version, not a frozen fingerprint")
        elif self.version is not None or not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("Frozen models require an immutable fingerprint, not an actor version")

    @property
    def identity(self):
        return self.role, self.path, self.version if self.role == "current_student" else self.fingerprint


@dataclass(frozen=True)
class Single:
    model: str


@dataclass(frozen=True)
class Fusion:
    base: str
    teacher_weight: float


@dataclass(frozen=True)
class Switch:
    first: str
    second: str
    boundary: int


@dataclass(frozen=True)
class Mixture:
    models: tuple[str, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class Stage:
    models: tuple[str, ...]
    max_tokens: int
    teacher_weight: float | None = None

    @property
    def slot(self):
        return self.models[0] if len(self.models) == 1 else f"fusion:{self.models[0]}"


@dataclass(frozen=True)
class RequestPlan:
    route: int
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class BatchPlan:
    requests: tuple[RequestPlan, ...]
    cap: int
    batch_id: int

    @property
    def models(self):
        return frozenset(model for r in self.requests for s in r.stages for model in s.models)

    @property
    def uses_actor(self):
        return "current_student" in self.models


def exact_routes(size, weights, seed):
    """Global largest-remainder quota, declared-order ties, seeded positions."""
    if type(size) is not int or size < 0 or not weights:
        raise ValueError("A nonnegative global batch size and nonempty weights are required")
    for w in weights:
        _finite(w, "Mixture weight")
        if w < 0:
            raise ValueError("Mixture weights must be nonnegative")
    scale = max(weights)
    if scale <= 0:
        raise ValueError("At least one mixture weight must be positive")
    scaled = [w / scale for w in weights]
    desired = [size * w / sum(scaled) for w in scaled]
    counts = [math.floor(x) for x in desired]
    priority = sorted(range(len(weights)), key=lambda i: (-(desired[i] - counts[i]), i))
    for i in priority[: size - sum(counts)]:
        counts[i] += 1
    routes = [i for i, count in enumerate(counts) for _ in range(count)]
    random.Random(seed).shuffle(routes)
    return routes


def _fields(config, allowed):
    if not isinstance(config, Mapping):
        raise ValueError("A rollout policy must be a mapping")
    extra = set(config) - set(allowed) - {"kind"}
    if extra:
        raise ValueError(f"Unsupported policy fields: {sorted(extra)}")


def _single_ref(config):
    if isinstance(config, str):
        return _model(config)
    _fields(config, {"model"})
    if config.get("kind") != "single" or "model" not in config:
        raise ValueError("Switch stages and mixture components must be single models; nesting is disabled")
    return _model(config["model"])


def parse_policy(config, cap):
    """Validate the proposed config and reduce endpoints before allocating models."""
    if type(cap) is not int or cap < 1:
        raise ValueError("Total response-token cap must be a positive integer")
    if not isinstance(config, Mapping):
        raise ValueError("A rollout policy must be a mapping")
    kind = config.get("kind")
    if kind == "single":
        _fields(config, {"model"})
        return Single(_single_ref(config))
    if kind == "logit_fusion":
        _fields(config, {"base", "teacher_weight"})
        base = _model(config.get("base"))
        if base not in {"current_student", "initial_student"}:
            raise ValueError("Fusion base must be current or initial student")
        weight = _finite(config.get("teacher_weight"), "teacher_weight")
        if not 0 <= weight <= 1:
            raise ValueError("teacher_weight must be in [0,1]")
        return Single(base) if weight == 0 else Single("teacher") if weight == 1 else Fusion(base, weight)
    if kind == "sequence_switch":
        _fields(config, {"first", "second", "boundary", "keep_request_boundary"})
        first, second = _single_ref(config.get("first")), _single_ref(config.get("second"))
        keep = config.get("keep_request_boundary", False)
        if type(keep) is not bool or (keep and first != second):
            raise ValueError("keep_request_boundary must be boolean and is reserved for same-model execution controls")
        boundary = config.get("boundary")
        if not isinstance(boundary, Mapping) or set(boundary) != {"kind", "value"}:
            raise ValueError("Switch boundary needs exactly kind and value")
        value = _finite(boundary["value"], "Switch boundary")
        if boundary["kind"] == "response_tokens":
            if type(value) is not int or value < 0:
                raise ValueError("Response-token boundary must be a nonnegative integer")
            k = value
        elif boundary["kind"] == "budget_fraction":
            if not 0 <= value <= 1:
                raise ValueError("Budget fraction must be in [0,1]")
            k = math.floor(value * cap)
        else:
            raise ValueError("Switch boundary must use response_tokens or budget_fraction")
        return (
            Single(second)
            if k == 0
            else Single(first)
            if k >= cap or (first == second and not keep)
            else Switch(first, second, k)
        )
    if kind == "batch_mix":
        _fields(config, {"components", "assignment", "routing_unit"})
        if config.get("assignment", "exact_count") != "exact_count":
            raise ValueError("Only global exact-count mixture assignment is supported")
        if config.get("routing_unit", "trajectory") != "trajectory":
            raise ValueError("Mixture routing is per response trajectory")
        components = config.get("components")
        if not isinstance(components, Sequence) or isinstance(components, (str, bytes)) or len(components) != 2:
            raise ValueError("The first implementation supports exactly two single-model components")
        models, weights = [], []
        for component in components:
            if not isinstance(component, Mapping) or set(component) != {"policy", "weight"}:
                raise ValueError("Every mixture component needs exactly policy and weight")
            models.append(_single_ref(component["policy"]))
            weights.append(_finite(component["weight"], "Mixture weight"))
        exact_routes(0, weights, 0)
        if models[0] == models[1]:
            return Single(models[0])
        if weights[0] == 0:
            return Single(models[1])
        if weights[1] == 0:
            return Single(models[0])
        return Mixture(tuple(models), tuple(weights))
    raise ValueError(f"Unsupported rollout policy kind: {kind!r}")


def compile_policy(config, *, cap, batch_size, seed, batch_id):
    if type(batch_size) is not int or batch_size < 0 or type(batch_id) is not int or batch_id < 0:
        raise ValueError("Global batch size and batch identity must be nonnegative integers")
    policy = parse_policy(config, cap)
    routing_seed = stable_seed(seed, "global-routing", batch_id)
    if isinstance(policy, Mixture):
        routes = exact_routes(batch_size, policy.weights, routing_seed)
        plans = [(Stage((model,), cap),) for model in policy.models]
    else:
        routes = [0] * batch_size
        if isinstance(policy, Single):
            stages = (Stage((policy.model,), cap),)
        elif isinstance(policy, Switch):
            stages = (Stage((policy.first,), policy.boundary), Stage((policy.second,), cap - policy.boundary))
        else:
            stages = (Stage((policy.base, "teacher"), cap, policy.teacher_weight),)
        plans = [stages]
    return BatchPlan(tuple(RequestPlan(route, plans[route]) for route in routes), cap, batch_id)
