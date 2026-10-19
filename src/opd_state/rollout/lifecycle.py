"""Admission and publication rules for one native generation engine.

The caller must actually drain native work and reset caches before publication.
These rules prevent a partially received actor stream, a frozen source, or a
sleeping engine from being used just because a Python RPC has returned.
"""

from __future__ import annotations

from opd_state.core.policy import ModelRef


class GenerationState:
    def __init__(self, role, path, fingerprint=None):
        self.role, self.path, self.fingerprint = role, path, fingerprint
        self.version = None
        self.pending_version = None
        self.active = set()
        self.drain_pending = False
        self.awake = True
        self.valid = role != "current_student"
        self.accepting = True
        if self.valid:
            ModelRef(role, path, fingerprint=fingerprint)
        elif role != "current_student" or not path:
            raise ValueError("Invalid current-student generation state")

    @property
    def identity(self):
        if not self.valid:
            raise RuntimeError("Generation weights have not been completely published")
        return ModelRef(self.role, self.path, self.version, self.fingerprint).identity

    def admit(self, request_id, expected):
        if (
            not self.accepting
            or not self.awake
            or not self.valid
            or self.pending_version is not None
        ):
            raise RuntimeError(
                "Generation engine is sleeping, updating or closed to new requests"
            )
        if request_id in self.active:
            raise RuntimeError("Native request identity is already active")
        if tuple(expected) != self.identity:
            raise RuntimeError(
                "Generation source identity or committed actor version does not match"
            )
        self.active.add(request_id)

    def complete(self, request_id):
        if request_id not in self.active:
            raise RuntimeError("Native completion has no matching admission")
        self.active.remove(request_id)

    def require_idle(self):
        if self.active or self.drain_pending:
            raise RuntimeError(
                "Native requests are active or drain is unacknowledged; weight/cache mutation is unsafe"
            )

    def begin_update(self, version):
        self.require_idle()
        if self.role != "current_student":
            raise RuntimeError("Frozen generation sources reject actor weight streams")
        if self.pending_version is not None:
            raise RuntimeError("Another actor weight stream is still incomplete")
        if (
            type(version) is not int
            or version < 0
            or (self.version is not None and version < self.version)
        ):
            raise ValueError("Actor versions must be nonnegative and cannot regress")
        self.pending_version = version
        self.valid = False
        self.accepting = False

    def commit_update(self, version, *, cache_reset):
        self.require_idle()
        if self.pending_version != version or type(version) is not int:
            raise RuntimeError(
                "Actor publication does not match its weight transaction"
            )
        if cache_reset is not True:
            raise RuntimeError(
                "Actor publication requires acknowledged KV/prefix cache invalidation"
            )
        self.version = version
        self.pending_version = None
        self.valid = True
        self.accepting = True

    def fail_update(self):
        self.pending_version = None
        self.valid = False
        self.accepting = False

    def close_admission(self):
        self.accepting = False

    def reopen(self):
        self.require_idle()
        if not self.valid or self.pending_version is not None or not self.awake:
            raise RuntimeError(
                "Cannot reopen an uninitialized, updating or sleeping engine"
            )
        self.accepting = True

    def slept(self, *, discarded_weights):
        self.require_idle()
        self.awake = False
        self.accepting = False
        if discarded_weights:
            self.valid = False

    def woke(self):
        self.require_idle()
        self.awake = True
        # Sleep level 2 discards weights; only a complete actor publication may
        # reopen that engine. Frozen sources must use weight-preserving sleep.
        self.accepting = self.valid and self.pending_version is None
