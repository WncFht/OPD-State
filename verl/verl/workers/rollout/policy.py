"""Compatibility exports during backend extraction; implementation lives in opd_state."""

from opd_state.core.policy import (
    MODEL_ROLES,
    stable_seed,
    ModelRef,
    Single,
    Fusion,
    Switch,
    Mixture,
    Stage,
    RequestPlan,
    BatchPlan,
    exact_routes,
    parse_policy,
    compile_policy,
)

__all__ = ['MODEL_ROLES', 'stable_seed', 'ModelRef', 'Single', 'Fusion', 'Switch', 'Mixture', 'Stage', 'RequestPlan', 'BatchPlan', 'exact_routes', 'parse_policy', 'compile_policy']
