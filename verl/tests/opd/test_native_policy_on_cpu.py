"""Production policy contracts without importing the GPU training stack."""

import importlib.util
import random
import sys
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "verl/workers/rollout/policy.py"
SPEC = importlib.util.spec_from_file_location("native_policy_contract", PATH)
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)


def single(model):
    return {"kind": "single", "model": model}


def switch(first="teacher", second="current_student", value=2, kind="response_tokens"):
    return {
        "kind": "sequence_switch",
        "first": single(first),
        "second": single(second),
        "boundary": {"kind": kind, "value": value},
    }


def mixture(weights=(0.25, 0.75), models=("teacher", "current_student")):
    return {
        "kind": "batch_mix",
        "components": [
            {"policy": single(model), "weight": weight} for model, weight in zip(models, weights, strict=True)
        ],
    }


class NativePolicyTests(unittest.TestCase):
    def compile(self, config, size=64, cap=8, batch_id=1):
        return P.compile_policy(config, cap=cap, batch_size=size, seed=42, batch_id=batch_id)

    def test_three_models_have_distinct_immutable_identities(self):
        refs = [
            P.ModelRef("current_student", "/same", version=0),
            P.ModelRef("initial_student", "/same", fingerprint="initial-sha"),
            P.ModelRef("teacher", "/teacher", fingerprint="teacher-sha"),
        ]
        self.assertEqual(len({r.identity for r in refs}), 3)
        with self.assertRaises(FrozenInstanceError):
            refs[1].version = 1
        for model in P.MODEL_ROLES:
            plan = self.compile(single(model))
            self.assertEqual(plan.models, {model})
            self.assertEqual(plan.uses_actor, model == "current_student")

    def test_model_version_and_frozen_identity_cannot_be_confused(self):
        invalid = [
            ("current_student", "/p", None, None),
            ("current_student", "/p", True, None),
            ("current_student", "/p", 0, "sha"),
            ("initial_student", "/p", 0, "sha"),
            ("teacher", "/q", None, ""),
            ("teacher", "", None, "sha"),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                P.ModelRef(*args)

    def test_fusion_endpoints_remove_unused_models_and_actor_sync(self):
        for base in ["current_student", "initial_student"]:
            config = {"kind": "logit_fusion", "base": base, "teacher_weight": 0.5}
            plan = self.compile(config)
            self.assertEqual(plan.models, {base, "teacher"})
            self.assertEqual(plan.requests[0].stages[0].teacher_weight, 0.5)
            self.assertEqual(plan.requests[0].stages[0].slot, f"fusion:{base}")
            self.assertEqual(self.compile({**config, "teacher_weight": 0}).models, {base})
            endpoint = self.compile({**config, "teacher_weight": 1})
            self.assertEqual(endpoint.models, {"teacher"})
            self.assertFalse(endpoint.uses_actor)

    def test_all_four_switch_directions_preserve_one_total_budget(self):
        for base in ["current_student", "initial_student"]:
            for first, second in [(base, "teacher"), ("teacher", base)]:
                for k in [1, 2, 7]:
                    plan = self.compile(switch(first, second, k))
                    stages = plan.requests[0].stages
                    self.assertEqual([s.models for s in stages], [(first,), (second,)])
                    self.assertEqual([s.max_tokens for s in stages], [k, 8 - k])
                    self.assertEqual(sum(s.max_tokens for s in stages), 8)

    def test_switch_endpoints_and_same_model_are_single_requests(self):
        for k, model in [(0, "current_student"), (8, "teacher"), (9, "teacher")]:
            plan = self.compile(switch(value=k))
            self.assertEqual(plan.models, {model})
            self.assertEqual(len(plan.requests[0].stages), 1)
            self.assertEqual(plan.requests[0].stages[0].max_tokens, 8)
        plan = self.compile(switch("teacher", "teacher", 3))
        self.assertEqual(plan.requests[0].stages, (P.Stage(("teacher",), 8),))

    def test_fraction_is_of_cap_not_unknown_final_length(self):
        for fraction, expected in [(0, None), (0.25, 1), (0.5, 3), (1, None)]:
            plan = self.compile(switch(value=fraction, kind="budget_fraction"), cap=7)
            stages = plan.requests[0].stages
            if expected is None:
                self.assertEqual(len(stages), 1)
            else:
                self.assertEqual(stages[0].max_tokens, expected)
                self.assertEqual(sum(s.max_tokens for s in stages), 7)

    def test_explicit_same_model_control_keeps_internal_boundary_and_prunes_endpoints(self):
        for role in P.MODEL_ROLES:
            config = {**switch(role, role, 3), "keep_request_boundary": True}
            plan = self.compile(config)
            self.assertEqual(plan.models, {role})
            self.assertEqual(plan.requests[0].stages, (P.Stage((role,), 3), P.Stage((role,), 5)))
            self.assertEqual(plan.uses_actor, role == "current_student")
            for endpoint in (0, 8, 9):
                endpoint_plan = self.compile({**config, "boundary": {"kind": "response_tokens", "value": endpoint}})
                self.assertEqual(endpoint_plan.requests[0].stages, (P.Stage((role,), 8),))

    def test_boundary_control_rejects_invalid_flags_and_cross_model_use(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "keep_request_boundary"):
                self.compile({**switch("teacher", "teacher", 3), "keep_request_boundary": value})
        with self.assertRaisesRegex(ValueError, "same-model"):
            self.compile({**switch(), "keep_request_boundary": True})

    def test_global_quota_is_assigned_before_dp_slicing(self):
        plan = self.compile(mixture())
        models = [r.stages[0].models[0] for r in plan.requests]
        self.assertEqual(Counter(models), {"teacher": 16, "current_student": 48})
        shards = [models[i::7] for i in range(7)]
        self.assertEqual(sum(x.count("teacher") for x in shards), 16)
        self.assertEqual(plan, self.compile(mixture()))
        self.assertNotEqual(models, [r.stages[0].models[0] for r in self.compile(mixture(), batch_id=2).requests])

    def test_small_odd_and_empty_batches_do_not_invent_a_request(self):
        self.assertEqual(P.exact_routes(1, [1, 1], 42), [0])
        self.assertEqual(Counter(P.exact_routes(3, [1, 1], 42)), {0: 2, 1: 1})
        plan = self.compile(mixture(), size=0)
        self.assertEqual(plan.requests, ())
        self.assertFalse(plan.models)
        self.assertFalse(plan.uses_actor)

    def test_zero_weight_and_duplicate_role_reduce_before_model_loading(self):
        for weights, expected in [((1, 0), "teacher"), ((0, 1), "current_student")]:
            self.assertEqual(self.compile(mixture(weights)).models, {expected})
        self.assertEqual(self.compile(mixture(models=("teacher", "teacher"))).models, {"teacher"})

    def test_500_weighted_global_quotas(self):
        rng = random.Random(123)
        for _ in range(500):
            size = rng.randrange(200)
            weights = [rng.random() for _ in range(rng.randrange(2, 10))]
            routes = P.exact_routes(size, weights, 42)
            self.assertEqual(len(routes), size)
            for i, w in enumerate(weights):
                self.assertLess(abs(routes.count(i) - size * w / sum(weights)), 1 + 1e-10)
        self.assertEqual(len(P.exact_routes(17, [1e308, 1e308], 42)), 17)

    def test_nested_modes_and_tcod_are_rejected_before_execution(self):
        nested = mixture()
        nested["components"][0]["policy"] = switch()
        invalid_switch = switch()
        invalid_switch["first"] = {"kind": "logit_fusion", "base": "initial_student", "teacher_weight": 0.5}
        for config in [nested, invalid_switch, {"kind": "TCOD"}, {**single("teacher"), "loss_scope": "suffix"}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.compile(config)

    def test_malformed_and_nonfinite_parameters_are_rejected(self):
        configs = [
            single([]),
            single("missing"),
            switch(value=True),
            switch(value=-1),
            switch(value=1.5),
            switch(value=float("nan"), kind="budget_fraction"),
            switch(value=1.1, kind="budget_fraction"),
            mixture((-1, 2)),
            mixture((0, 0)),
            mixture((float("inf"), 1)),
            {**mixture(), "routing_unit": "token"},
            {"kind": "logit_fusion", "base": "teacher", "teacher_weight": 0.5},
        ]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.compile(config)
        for kwargs in [{"cap": 0}, {"batch_size": -1}, {"batch_id": True}, {"seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                P.compile_policy(single("teacher"), **{"cap": 8, "batch_size": 3, "seed": 42, "batch_id": 1, **kwargs})

    def test_generation_identity_seed_includes_visit_sample_and_stage(self):
        values = {
            P.stable_seed(42, "prompt-id", visit, sample, stage)
            for visit in range(2)
            for sample in range(2)
            for stage in range(2)
        }
        self.assertEqual(len(values), 8)


if __name__ == "__main__":
    unittest.main()
