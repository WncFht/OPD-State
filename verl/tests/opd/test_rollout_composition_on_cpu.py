"""Contract and adversarial checks for the standalone inference prototype."""

import asyncio
import random
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))

from probe_vllm_composition import reset_idle_engine
from rollout_composition import Completion, Mixture, Request, Single, Switch, compile_stages, exact_routes, generate


class Backend:
    def __init__(self, stop_at=None, corrupt=None):
        self.calls = []
        self.stop_at = stop_at or {}
        self.corrupt = corrupt

    async def generate(self, model, calls):
        self.calls.append((model, calls))
        results = []
        for call in calls:
            count = min(call.max_tokens, self.stop_at.get(call.key[0], call.max_tokens + 1))
            stop = call.key[0] in self.stop_at and self.stop_at[call.key[0]] <= call.max_tokens
            token = 2 if model == "student" else 3
            ids = tuple(([token] * count)[:-1] + [0]) if stop else (token,) * count
            result = Completion(
                call.key, model, call.prompt_ids, ids, "stop" if stop else "length", 0 if stop else None
            )
            results.append(self.corrupt(result) if self.corrupt else result)
        return results


class CompositionTests(unittest.TestCase):
    def test_idle_cache_reset_accepts_v1_none_and_checks_explicit_failure(self):
        for value in [None, True]:
            llm = SimpleNamespace(
                llm_engine=SimpleNamespace(has_unfinished_requests=lambda: False), reset_prefix_cache=lambda v=value: v
            )
            self.assertIs(reset_idle_engine(llm), value)
        llm.reset_prefix_cache = lambda: False
        with self.assertRaisesRegex(RuntimeError, "explicitly failed"):
            reset_idle_engine(llm)

    def test_weight_activation_rejects_active_requests_before_cache_reset(self):
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(has_unfinished_requests=lambda: True),
            reset_prefix_cache=lambda: self.fail("must not reset with active requests"),
        )
        with self.assertRaisesRegex(RuntimeError, "active"):
            reset_idle_engine(llm)

    def run_case(self, policy, backend=None, count=7, **kwargs):
        requests = kwargs.pop("requests", [Request(str(i), (1, 4 + i % 2)) for i in range(count)])
        backend = backend or Backend()
        result = asyncio.run(
            generate(
                requests,
                policy,
                backend,
                cap=kwargs.pop("cap", 8),
                seed=42,
                vocab_size=32,
                eos_token_id=0,
                context_limits=kwargs.pop("context_limits", {"student": 64, "teacher": 64}),
                **kwargs,
            )
        )
        return result, backend

    def test_zero_and_full_boundaries_prune_unused_model(self):
        for boundary, expected in [(0, "teacher"), (8, "student"), (100, "student")]:
            (_, _), backend = self.run_case(Switch(Single("student"), Single("teacher"), boundary))
            self.assertEqual([m for m, _ in backend.calls], [expected])

    def test_same_model_switch_is_merged(self):
        (_, _), backend = self.run_case(Switch(Single("student"), Single("student"), 3))
        self.assertEqual(len(backend.calls), 1)

    def test_second_prefix_and_remaining_budget(self):
        (rows, stats), backend = self.run_case(Switch(Single("teacher"), Single("student"), 3))
        for request, continuation in zip(rows, backend.calls[1][1], strict=True):
            self.assertEqual(continuation.prompt_ids, request.request.prompt_ids + request.token_ids[:3])
            self.assertEqual(continuation.max_tokens, 5)
            self.assertEqual([s.model for s in request.segments], ["teacher", "student"])
            self.assertEqual(request.response_mask, (1,) * 8)
        self.assertEqual(sum(g["input_tokens"] for g in stats["groups"]), 7 * (2 + 5))

    def test_eos_at_boundary_stops_whole_response_and_preserves_eos_equals_pad(self):
        (rows, _), backend = self.run_case(
            Switch(Single("teacher"), Single("student"), 3),
            Backend(stop_at={"0": 3}),
            count=2,
        )
        self.assertEqual(rows[0].token_ids, (3, 3, 0))
        self.assertEqual(rows[0].response_mask, (1, 1, 1))
        self.assertEqual(len(rows[0].segments), 1)
        self.assertEqual([c.key[0] for c in backend.calls[1][1]], ["1"])

    def test_each_stage_context_is_checked_at_its_own_cumulative_budget(self):
        self.run_case(Switch(Single("teacher"), Single("student"), 3), context_limits={"teacher": 5, "student": 10})
        with self.assertRaisesRegex(ValueError, "context"):
            self.run_case(Switch(Single("teacher"), Single("student"), 3), context_limits={"teacher": 5, "student": 9})

    def test_native_length_reason_with_eos_still_terminates(self):
        backend = Backend(stop_at={"0": 3}, corrupt=lambda r: replace(r, finish_reason="length", stop_reason=None))
        (rows, _), backend = self.run_case(Switch(Single("teacher"), Single("student"), 3), backend, count=1)
        self.assertEqual(rows[0].token_ids, (3, 3, 0))
        self.assertEqual(rows[0].termination, "eos")
        self.assertEqual(rows[0].segments[0].finish_reason, "length")
        self.assertEqual(len(backend.calls), 1)

    def test_exact_global_quota_and_original_order(self):
        (rows, _), backend = self.run_case(Mixture((Single("teacher"), Single("student")), (0.5, 0.5)))
        self.assertEqual(Counter(r.route for r in rows), {0: 4, 1: 3})
        self.assertEqual([r.request.prompt_id for r in rows], list(map(str, range(7))))
        self.assertEqual(len(backend.calls), 2)

    def test_mixture_can_contain_switch_and_no_stage_doubles_cap(self):
        policy = Mixture((Single("teacher"), Switch(Single("student"), Single("teacher"), 3)), (0.5, 0.5))
        (rows, _), _ = self.run_case(policy)
        self.assertTrue(all(len(r.token_ids) == 8 for r in rows))
        self.assertEqual(Counter(len(r.segments) for r in rows), {1: 4, 2: 3})

    def test_route_is_reproducible_and_zero_weight_makes_no_calls(self):
        policy = Mixture((Single("teacher"), Single("student")), (0, 1))
        (a, _), backend = self.run_case(policy)
        (b, _), _ = self.run_case(policy)
        self.assertEqual(a, b)
        self.assertEqual([model for model, _ in backend.calls], ["student"])

    def test_empty_batch_never_calls_backend(self):
        (rows, _), backend = self.run_case(Single("student"), count=0)
        self.assertEqual(rows, [])
        self.assertEqual(backend.calls, [])

    def test_prompt_response_identity_includes_visit_and_sample(self):
        requests = [Request("same", (1,), visit=i, sample=j) for i in range(2) for j in range(2)]
        (_, _), backend = self.run_case(Single("student"), requests=requests)
        self.assertEqual(len({call.seed for call in backend.calls[0][1]}), 4)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_case(Single("student"), requests=[requests[0], requests[0]])

    def test_source_order_prefix_length_and_token_corruption_rejected(self):
        corruptions = [
            lambda r: replace(r, model="wrong"),
            lambda r: replace(r, key=("wrong", 0, 0)),
            lambda r: replace(r, prompt_ids=r.prompt_ids + (2,)),
            lambda r: replace(r, token_ids=r.token_ids[:-1]),
            lambda r: replace(r, token_ids=r.token_ids + (2,)),
            lambda r: replace(r, token_ids=(99,) * len(r.token_ids)),
            lambda r: replace(r, token_ids=(0,) + r.token_ids[1:]),
            lambda r: replace(r, finish_reason="abort"),
            lambda r: replace(r, finish_reason="completed"),
        ]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                self.run_case(Single("student"), Backend(corrupt=corrupt))

    def test_invalid_policies_and_budgets_fail_before_inference(self):
        policies = [
            Switch(Single("student"), Single("teacher"), -1),
            Mixture((Single("student"),), (0,)),
            Mixture((Single("student"),), (float("nan"),)),
            Mixture((Single("student"),), (1, 2)),
            Single(""),
        ]
        for policy in policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.run_case(policy)
        for cap in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                compile_stages(Single("student"), cap)

    def test_five_hundred_random_global_quotas(self):
        rng = random.Random(713)
        for _ in range(500):
            n, k = rng.randrange(130), rng.randrange(1, 8)
            weights = [rng.random() for _ in range(k)]
            routes = exact_routes(n, weights, 42)
            self.assertEqual(len(routes), n)
            counts = Counter(routes)
            for i, weight in enumerate(weights):
                self.assertLess(abs(counts[i] - n * weight / sum(weights)), 1 + 1e-10)
        self.assertEqual(len(exact_routes(17, [1e308, 1e308], 42)), 17)

    def test_all_short_switch_boundaries_and_eos_positions(self):
        class PositionBackend(Backend):
            async def generate(self, model, calls):
                self.calls.append((model, calls))
                output = []
                for call in calls:
                    already = len(call.prompt_ids) - 2
                    remaining = self.stop_at[call.key[0]] - already
                    count = min(remaining, call.max_tokens)
                    stopped = remaining <= call.max_tokens
                    tokens = (2 if model == "student" else 3,) * count
                    if stopped:
                        tokens = tokens[:-1] + (0,)
                    output.append(
                        Completion(
                            call.key,
                            model,
                            call.prompt_ids,
                            tokens,
                            "stop" if stopped else "length",
                            0 if stopped else None,
                        )
                    )
                return output

        for cap in range(2, 10):
            for boundary in range(cap + 2):
                for eos_at in range(1, cap + 2):
                    for first, second in [("teacher", "student"), ("student", "teacher")]:
                        with self.subTest(cap=cap, boundary=boundary, eos=eos_at, first=first):
                            (rows, _), _ = self.run_case(
                                Switch(Single(first), Single(second), boundary),
                                PositionBackend(stop_at={"0": eos_at}),
                                count=1,
                                cap=cap,
                            )
                            row = rows[0]
                            length = min(cap, eos_at)
                            self.assertEqual(len(row.token_ids), length)
                            self.assertEqual(row.response_mask, (1,) * length)
                            if eos_at <= cap:
                                self.assertEqual(row.token_ids[-1], 0)
                                self.assertEqual(row.segments[-1].finish_reason, "stop")
                            self.assertEqual(sum(s.end - s.start for s in row.segments), length)
                            self.assertLessEqual(len(row.segments), 2)

    def test_concurrent_failure_cancels_other_groups(self):
        class FailingBackend:
            cancelled = False

            async def generate(self, model, calls):
                if model == "student":
                    await asyncio.sleep(0)
                    raise RuntimeError("injected failure")
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        backend = FailingBackend()
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.run_case(Mixture((Single("teacher"), Single("student")), (1, 1)), backend, concurrent_groups=True)
        self.assertTrue(backend.cancelled)


if __name__ == "__main__":
    unittest.main()
