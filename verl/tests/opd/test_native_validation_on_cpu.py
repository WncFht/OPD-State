"""Validation length/termination aggregation and unbiased position selection."""

import importlib.util
import itertools
import math
import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "verl/trainer/ppo"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import opd_state.evaluation.rollout as rollout
from opd_state.integrations.verl.validation_data import dump_columns
import opd_state.evaluation.prefixes as kl


class ValidationDispatchTests(unittest.TestCase):
    def test_equal_collective_counts_and_inverse_order_for_four_and_eight_ranks(self):
        generator = random.Random(17)
        for world_size in (1, 2, 4, 8):
            for rounds in (1, 3, 29):
                records = [
                    {"prompt_ids": [1, 2], "response_ids": [5] * generator.randrange(1, 128)}
                    for _ in range(world_size * rounds)
                ]
                before = [len(r["response_ids"]) for r in records]
                order = kl.balanced_record_indices(records, world_size)
                self.assertEqual(sorted(order), list(range(len(records))))
                inverse = sorted(range(len(order)), key=order.__getitem__)
                self.assertEqual([order[i] for i in inverse], list(range(len(records))))
                self.assertEqual(before, [len(r["response_ids"]) for r in records])
                if world_size == 1:
                    self.assertEqual(order, tuple(range(len(records))))
                lengths = [len(r["prompt_ids"]) + len(r["response_ids"]) - 1 for r in records]
                old = sum(max(lengths[rank * rounds + step] for rank in range(world_size)) for step in range(rounds))
                new = sum(
                    max(lengths[order[rank * rounds + step]] for rank in range(world_size)) for step in range(rounds)
                )
                self.assertLessEqual(new, old)

    def test_bank_forward_length_uses_the_last_selected_prefix(self):
        records = [
            {"prompt_ids": [1], "response_ids": [5] * 100, "positions": [0, 1]},
            {"prompt_ids": [1], "response_ids": [5] * 8},
            {"prompt_ids": [1], "response_ids": [5] * 6},
            {"prompt_ids": [1], "response_ids": [5] * 3},
        ]
        self.assertEqual(kl.balanced_record_indices(records, 2), (1, 0, 2, 3))

    def test_invalid_shards_or_missing_padding_are_rejected(self):
        row = {"prompt_ids": [1], "response_ids": [2]}
        for records, world in (([], 4), ([row], 4), ([row], 0), ([row], True)):
            with self.subTest(world=world, count=len(records)), self.assertRaises(ValueError):
                kl.balanced_record_indices(records, world)


class ValidationAggregationTests(unittest.TestCase):
    def row(
        self,
        tokens,
        *,
        source="MATH-500",
        uid="a",
        cap=8,
        correct=True,
        pred="1",
        sample=0,
        finish=None,
        termination=None,
    ):
        eos = bool(tokens and tokens[-1] == 2)
        return rollout.rollout_record(
            source=source,
            uid=uid,
            prompt_length=3,
            response_ids=tokens,
            eos_token_id=2,
            info={
                "response_cap": cap,
                "termination": termination or ("eos" if eos else "length"),
                "finish_reason": finish or ("length" if len(tokens) == cap else "stop"),
                "prompt_id": uid,
                "sample": sample,
                "student_version": 50,
            },
            correct=correct,
            prediction=pred,
            prediction_available=True,
        )

    def test_eos_at_cap_is_not_truncation_and_raw_length_finish_survives_dump(self):
        row = self.row([7, 2], cap=2)
        self.assertTrue(row["eos"])
        self.assertFalse(row["truncated"])
        metrics = rollout.summarize_rollouts([row], samples_per_question=1)
        self.assertEqual(metrics["validation_rollout/MATH-500/eos_at_cap/count"], 1)
        self.assertEqual(metrics["validation_rollout/MATH-500/response_tokens"], 2)
        self.assertEqual(dump_columns([row])["rollout_finish_reason"], ["length"])

    def test_benchmark_macro_and_micro_use_rows_and_keep_missing_subgroups_finite(self):
        rows = [self.row([2], uid=str(i)) for i in range(4)]
        rows += [self.row([7] * 7 + [2], source="AIME25")]
        metrics = rollout.summarize_rollouts(rows, samples_per_question=1)
        self.assertEqual(metrics["validation_rollout/micro/response_length/mean"], 12 / 5)
        self.assertEqual(metrics["validation_rollout/macro/response_length/mean"], 4.5)
        self.assertAlmostEqual(metrics["validation_rollout/micro/response_length/p95"], 6.6)
        self.assertEqual(metrics["validation_rollout/micro/incorrect/responses"], 0)
        self.assertNotIn("validation_rollout/micro/incorrect/response_length_mean", metrics)
        self.assertNotIn("validation_rollout/macro/response_length/p95", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_exact_sample_cardinality_and_whole_epoch_not_batch_mean_reduction(self):
        rows = [self.row([2], sample=0), self.row([9, 9, 9, 2], sample=1, correct=False, pred=None)]
        metrics = rollout.summarize_rollouts(rows, samples_per_question=2)
        self.assertEqual(metrics["validation_rollout/MATH-500/questions"], 1)
        self.assertEqual(metrics["validation_rollout/MATH-500/answer_parse/fraction"], 0.5)
        self.assertEqual(metrics["validation_rollout/MATH-500/correct/response_length_mean"], 1)
        self.assertEqual(metrics["validation_rollout/MATH-500/incorrect/response_length_mean"], 4)
        with self.assertRaisesRegex(ValueError, "cardinality"):
            rollout.summarize_rollouts(rows[:1], samples_per_question=2)

    def test_fourgram_repetition_and_undefined_short_denominator(self):
        repeated = self.row([7] * 8, cap=8, correct=False, pred=None)
        short = self.row([2], uid="b")
        metrics = rollout.summarize_rollouts([repeated, short])
        self.assertEqual(metrics["validation_rollout/micro/repeat_4gram/eligible_responses"], 1)
        self.assertEqual(metrics["validation_rollout/micro/repeat_4gram/response_mean"], 4 / 5)
        self.assertEqual(metrics["validation_rollout/micro/truncated/fraction"], 0.5)
        self.assertNotIn("validation_rollout/micro/repeat_4gram/response_mean", rollout.summarize_rollouts([short]))

    def test_invalid_termination_and_tokens_fail_before_generating_successful_metrics(self):
        for tokens, kwargs in [
            ([2, 7], {}),
            ([2], {"termination": "length"}),
            ([7], {"cap": 8, "finish": "length"}),
            ([7], {"termination": "abort"}),
            ([2], {"correct": 0.3}),
        ]:
            with self.subTest(tokens=tokens, kwargs=kwargs), self.assertRaises(ValueError):
                self.row(tokens, **kwargs)

    def test_duplicate_samples_and_reserved_source_names_cannot_overwrite_results(self):
        row = self.row([2])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            rollout.summarize_rollouts([row, row])
        with self.assertRaisesRegex(ValueError, "reserved"):
            rollout.summarize_rollouts([self.row([2], source="micro")])


class PrefixSamplingTests(unittest.TestCase):
    def test_stratified_estimator_is_unbiased_by_exhaustive_position_enumeration(self):
        selected = kl.sample_prefix_positions(6, max_positions=3, seed=42)
        values = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        estimates = []
        for positions in itertools.product(*(range(s.begin, s.end) for s in selected)):
            sample = tuple(replace(s, position=i) for s, i in zip(selected, positions, strict=True))
            result = kl.estimate_trajectory_statistics(sample, {"kl": [values[i] for i in positions]}, length=6)
            estimates.append(result["kl/sum_est"])
            self.assertEqual(result["kl/token_mean_est"], result["kl/sum_est"] / 6)
        self.assertEqual(sum(estimates) / len(estimates), sum(values))

    def test_seeded_sampling_short_responses_and_full_coverage(self):
        for length in (1, 2, 7, 8, 9, 4096):
            samples = kl.sample_prefix_positions(length, seed=7)
            self.assertEqual(samples, kl.sample_prefix_positions(length, seed=7))
            self.assertEqual(samples[0].position, 0)
            self.assertEqual(sum(s.weight for s in samples), length)
            self.assertEqual(len(samples), min(8, length))
            out = kl.estimate_trajectory_statistics(samples, {"kl": [0.25] * len(samples)}, length=length)
            self.assertEqual(out["kl/sum_est"], length * 0.25)

    def test_invalid_budgets_positions_missing_mass_and_nonfinite_scores_are_rejected(self):
        for kwargs in ({"length": 0}, {"length": 10, "max_positions": 1}, {"length": 3, "seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kl.sample_prefix_positions(**kwargs)
        samples = kl.sample_prefix_positions(9, seed=2)
        for bad in (samples[:-1], (replace(samples[0], position=3), *samples[1:])):
            with self.assertRaises(ValueError):
                kl.estimate_trajectory_statistics(bad, {"kl": [0.1] * len(bad)}, length=9)
        with self.assertRaises(ValueError):
            kl.estimate_trajectory_statistics(samples, {"kl": [float("nan")] * len(samples)}, length=9)


if __name__ == "__main__":
    unittest.main()
