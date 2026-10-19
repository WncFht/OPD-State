"""Compatibility exports during backend extraction; implementation lives in opd_state."""

from opd_state.rollout.executor import (
    Request,
    Call,
    Completion,
    Segment,
    Trajectory,
    generate,
)

__all__ = ['Request', 'Call', 'Completion', 'Segment', 'Trajectory', 'generate']
