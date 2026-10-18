"""Adversarial token/provenance and cancellation checks for production execution."""

import asyncio
import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "verl/workers/rollout"
PACKAGE = "native_executor_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P, E = load("policy"), load("native_executor")


class Backend:
    def __init__(self, *, stop_at=None, corrupt=None, drain=True):
        self.calls, self.aborted = [], []
        self.stop_at, self.corrupt, self.drain = stop_at, corrupt, drain

    async def generate(self, stage, calls, refs):
        self.calls.append((stage, calls, refs))
        result = []
        for call in calls:
            count = min(self.stop_at or call.max_tokens, call.max_tokens)
            ids = [2 if stage.models[0] == "teacher" else 3] * count
            if self.stop_at is not None and self.stop_at <= call.max_tokens:
                ids[-1] = 0
            # Preserve the real vLLM length-before-EOS boundary convention.
            reason = "length" if count == call.max_tokens else "stop"
            row = E.Completion(
                call.request_id,
                call.prompt_ids,
                tuple(ids),
                tuple(r.identity for r in refs),
                reason,
                None,
                (-0.25,) * count,
                stage.teacher_weight,
            )
            result.append(self.corrupt(row) if self.corrupt else row)
        return result

    async def abort_and_drain(self, ids):
        self.aborted.extend(ids)
        return self.drain


class NativeExecutorTests(unittest.TestCase):
    def run_case(self, config, backend=None, size=8, cap=8, **kwargs):
        backend = backend or Backend()
        requests = [E.Request(str(i), (1, 5)) for i in range(size)]
        plan = P.compile_policy(config, cap=cap, batch_size=size, seed=42, batch_id=1)
        refs = {
            "current_student": P.ModelRef("current_student", "/p", version=3),
            "initial_student": P.ModelRef("initial_student", "/p", fingerprint="p0"),
            "teacher": P.ModelRef("teacher", "/q", fingerprint="q"),
        }
        params = {
            "refs": refs,
            "context_limits": {m: 10 for m in refs},
            "vocab_size": 10,
            "eos_token_id": 0,
            "seed": 42,
            **kwargs,
        }
        return asyncio.run(E.generate(requests, plan, backend, **params)), backend

    def switch(self, first="teacher", second="current_student", k=3):
        return {
            "kind": "sequence_switch",
            "first": first,
            "second": second,
            "boundary": {"kind": "response_tokens", "value": k},
        }

    def test_all_directions_pass_exact_prefix_and_keep_original_training_prompt(self):
        for base in ["current_student", "initial_student"]:
            for first, second in [(base, "teacher"), ("teacher", base)]:
                (rows, _), backend = self.run_case(self.switch(first, second))
                first_tokens = (2 if first == "teacher" else 3,) * 3
                self.assertEqual(backend.calls[1][1][0].prompt_ids, (1, 5) + first_tokens)
                self.assertEqual(backend.calls[1][1][0].max_tokens, 5)
                for row in rows:
                    self.assertEqual(row.request.prompt_ids, (1, 5))
                    self.assertEqual(row.token_ids[:3], first_tokens)
                    self.assertEqual(row.response_mask, (1,) * 8)
                    self.assertEqual([(s.start, s.end) for s in row.segments], [(0, 3), (3, 8)])
                    self.assertEqual(row.behavior_log_probs, (-0.25,) * 8)

    def test_eos_at_length_boundary_prevents_second_model_and_remains_valid(self):
        (rows, _), backend = self.run_case(self.switch(k=1), Backend(stop_at=1))
        self.assertEqual(len(backend.calls), 1)
        for row in rows:
            self.assertEqual(row.token_ids, (0,))
            self.assertEqual(row.response_mask, (1,))
            self.assertEqual(row.termination, "eos")
            self.assertEqual(row.segments[0].finish_reason, "length")

    def test_early_eos_prevents_unused_second_stage(self):
        (rows, _), backend = self.run_case(self.switch(k=4), Backend(stop_at=2))
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(all(len(r.token_ids) == 2 for r in rows))

    def test_mixture_keeps_global_quota_and_original_request_order(self):
        config = {
            "kind": "batch_mix",
            "components": [
                {"policy": {"kind": "single", "model": "teacher"}, "weight": 0.25},
                {"policy": {"kind": "single", "model": "current_student"}, "weight": 0.75},
            ],
        }
        (rows, timing), backend = self.run_case(config, concurrent_groups=True)
        self.assertEqual([r.request.prompt_id for r in rows], [str(i) for i in range(8)])
        self.assertEqual(sum(r.token_ids[0] == 2 for r in rows), 2)
        self.assertEqual(sorted(t["requests"] for t in timing["groups"]), [2, 6])
        self.assertEqual(len(backend.calls), 2)

    def test_fusion_preserves_both_identities_and_coefficient(self):
        (rows, _), _ = self.run_case({"kind": "logit_fusion", "base": "initial_student", "teacher_weight": 0.4})
        segment = rows[0].segments[0]
        self.assertEqual([i[0] for i in segment.identities], ["initial_student", "teacher"])
        self.assertEqual(segment.teacher_weight, 0.4)

    def test_wrong_actor_version_source_prefix_and_order_are_rejected_and_drained(self):
        changes = [
            lambda r: replace(r, request_id="wrong"),
            lambda r: replace(r, prompt_ids=(1,)),
            lambda r: replace(r, identities=(("current_student", "/p", 4),)),
            lambda r: replace(r, finish_reason="abort"),
            lambda r: replace(r, token_ids=(2,) * 9),
            lambda r: replace(r, token_ids=()),
            lambda r: replace(r, token_ids=(True,) * 8),
            lambda r: replace(r, token_ids=(0, 2, 2, 2, 2, 2, 2, 2)),
            lambda r: replace(r, log_probs=(-0.2,)),
            lambda r: replace(r, log_probs=(float("nan"),) * 8),
        ]
        for corrupt in changes:
            backend = Backend(corrupt=corrupt)
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                self.run_case({"kind": "single", "model": "current_student"}, backend)
            self.assertEqual(len(backend.aborted), 8)

    def test_failed_drain_is_not_treated_as_completed_generation(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            self.run_case(self.switch(), Backend(corrupt=lambda r: replace(r, finish_reason="abort"), drain=False))

    def test_backend_must_acknowledge_the_actual_fusion_coefficient(self):
        config = {"kind": "logit_fusion", "base": "initial_student", "teacher_weight": 0.5}
        with self.assertRaisesRegex(RuntimeError, "coefficient"):
            self.run_case(config, Backend(corrupt=lambda row: replace(row, teacher_weight=0.6)))

    def test_context_limits_are_checked_per_source_before_any_native_call(self):
        backend = Backend()
        with self.assertRaises(ValueError):
            self.run_case(self.switch(), backend, context_limits={"teacher": 5, "current_student": 9})
        self.assertEqual(backend.calls, [])
        (rows, _), _ = self.run_case(self.switch(), context_limits={"teacher": 5, "current_student": 10})
        self.assertEqual(len(rows), 8)

    def test_empty_batch_has_no_native_requests_or_model_activation(self):
        (rows, metrics), backend = self.run_case(self.switch(), size=0)
        self.assertEqual(rows, [])
        self.assertEqual(metrics["groups"], [])
        self.assertEqual(backend.calls, [])

    def test_parallel_failure_cancels_other_groups_then_drains_native_work(self):
        class Failing(Backend):
            def __init__(self):
                super().__init__()
                self.cancelled = False

            async def generate(self, stage, calls, refs):
                if stage.models[0] == "teacher":
                    await asyncio.sleep(0.01)
                    raise RuntimeError("injected backend failure")
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        backend = Failing()
        config = {
            "kind": "batch_mix",
            "components": [
                {"policy": {"kind": "single", "model": "teacher"}, "weight": 1},
                {"policy": {"kind": "single", "model": "current_student"}, "weight": 1},
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_case(config, backend, concurrent_groups=True)
        self.assertTrue(backend.cancelled)
        self.assertEqual(len(backend.aborted), 8)


if __name__ == "__main__":
    unittest.main()
