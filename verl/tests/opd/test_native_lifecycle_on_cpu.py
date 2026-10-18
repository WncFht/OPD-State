"""Admission, frozen-source and atomic actor-publication contract checks."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "verl/workers/rollout"
PACKAGE = "native_lifecycle_contract"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.lifecycle", ROOT / "lifecycle.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


class NativeLifecycleTests(unittest.TestCase):
    def actor(self, version=0):
        state = M.GenerationState("current_student", "/p")
        state.begin_update(version)
        state.commit_update(version, cache_reset=True)
        return state

    def test_initial_actor_cannot_generate_before_complete_weight_publication(self):
        state = M.GenerationState("current_student", "/p")
        with self.assertRaises(RuntimeError):
            state.admit("r", ("current_student", "/p", 0))
        state.begin_update(0)
        with self.assertRaises(RuntimeError):
            state.admit("r", ("current_student", "/p", 0))
        state.commit_update(0, cache_reset=True)
        state.admit("r", state.identity)
        state.complete("r")

    def test_frozen_initial_and_teacher_reject_every_actor_stream(self):
        for role in ["initial_student", "teacher"]:
            state = M.GenerationState(role, "/fixed", "sha")
            before = state.identity
            for version in [0, 1, 100]:
                with self.assertRaises(RuntimeError):
                    state.begin_update(version)
            self.assertEqual(state.identity, before)

    def test_inflight_request_blocks_weight_and_sleep_transitions(self):
        state = self.actor()
        state.admit("a", state.identity)
        with self.assertRaises(RuntimeError):
            state.begin_update(1)
        with self.assertRaises(RuntimeError):
            state.slept(discarded_weights=False)
        state.complete("a")
        state.begin_update(1)
        state.commit_update(1, cache_reset=True)
        self.assertEqual(state.identity, ("current_student", "/p", 1))

    def test_cache_invalidation_and_exact_transaction_are_required_to_publish(self):
        state = self.actor()
        state.begin_update(1)
        with self.assertRaises(RuntimeError):
            state.commit_update(2, cache_reset=True)
        with self.assertRaises(RuntimeError):
            state.commit_update(1, cache_reset=None)
        state.commit_update(1, cache_reset=True)

    def test_failed_actor_stream_stays_invalid_until_a_complete_retry(self):
        state = self.actor()
        state.begin_update(1)
        state.fail_update()
        with self.assertRaises(RuntimeError):
            state.reopen()
        with self.assertRaises(RuntimeError):
            state.admit("r", ("current_student", "/p", 0))
        state.begin_update(1)
        state.commit_update(1, cache_reset=True)
        state.admit("r", state.identity)

    def test_actor_version_regression_and_stale_requests_are_rejected(self):
        state = self.actor(3)
        with self.assertRaises(ValueError):
            state.begin_update(2)
        with self.assertRaises(RuntimeError):
            state.admit("r", ("current_student", "/p", 2))
        state.begin_update(3)  # Restoring the same actor after sleep is valid.
        state.commit_update(3, cache_reset=True)

    def test_sleep_level_one_preserves_frozen_identity_level_two_requires_actor_reload(self):
        frozen = M.GenerationState("teacher", "/q", "q-sha")
        frozen.slept(discarded_weights=False)
        with self.assertRaises(RuntimeError):
            frozen.admit("r", frozen.identity)
        frozen.woke()
        frozen.admit("r", frozen.identity)
        frozen.complete("r")
        actor = self.actor()
        actor.slept(discarded_weights=True)
        actor.woke()
        with self.assertRaises(RuntimeError):
            actor.reopen()
        actor.begin_update(0)
        actor.commit_update(0, cache_reset=True)

    def test_abort_closes_admission_until_native_completion_is_acknowledged(self):
        state = self.actor()
        state.admit("a", state.identity)
        state.close_admission()
        with self.assertRaises(RuntimeError):
            state.admit("b", state.identity)
        with self.assertRaises(RuntimeError):
            state.reopen()
        state.complete("a")
        state.reopen()
        state.admit("b", state.identity)


if __name__ == "__main__":
    unittest.main()
