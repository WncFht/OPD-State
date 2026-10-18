"""Real Torch/DataProto/trainer validation checks without GPU generation."""

import itertools
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.fixed_rollout import pack_token_records
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.validation_kl import full_vocab_divergences
from verl.trainer.ppo.validation_rollout import rows_from_batch, summarize_rollouts


class FullVocabularyTests(unittest.TestCase):
    def test_qwen_sized_vocabulary_matches_double_precision_without_head_padding_mass(self):
        generator = torch.Generator().manual_seed(42)
        p = torch.randn((4, 151936), generator=generator).bfloat16()
        q = torch.randn((4, 151936), generator=generator).bfloat16()
        p[:, 151665:] = q[:, 151665:] = float("nan")
        logp = p[:, :151665].double().log_softmax(-1)
        logq = q[:, :151665].double().log_softmax(-1)
        expected = {
            "student_teacher_kl": (logp.exp() * (logp - logq)).sum(-1),
            "teacher_student_kl": (logq.exp() * (logq - logp)).sum(-1),
            "student_entropy": -(logp.exp() * logp).sum(-1),
            "teacher_entropy": -(logq.exp() * logq).sum(-1),
        }
        for chunk in (1024, 8192, 151665):
            actual = full_vocab_divergences(p, q, vocab_size=151665, chunk_size=chunk)
            for key, value in expected.items():
                torch.testing.assert_close(actual[key].double(), value, atol=1e-5, rtol=0)

    def test_chunked_both_directions_and_entropies_match_independent_double_reference(self):
        generator = torch.Generator().manual_seed(3)
        p = torch.randn((7, 29), generator=generator, requires_grad=True)
        q = torch.randn((7, 29), generator=generator, requires_grad=True)
        logp, logq = p.double().log_softmax(-1), q.double().log_softmax(-1)
        expected = {
            "student_teacher_kl": (logp.exp() * (logp - logq)).sum(-1),
            "teacher_student_kl": (logq.exp() * (logq - logp)).sum(-1),
            "student_entropy": -(logp.exp() * logp).sum(-1),
            "teacher_entropy": -(logq.exp() * logq).sum(-1),
        }
        before = p.detach().clone()
        for chunk in (1, 7, 29, 80):
            actual = full_vocab_divergences(p, q, chunk_size=chunk)
            for key in expected:
                torch.testing.assert_close(actual[key].double(), expected[key], atol=2e-6, rtol=2e-6)
                self.assertFalse(actual[key].requires_grad)
        self.assertTrue(torch.equal(p, before))
        self.assertIsNone(p.grad)

    def test_identical_models_extreme_logits_shift_and_head_padding(self):
        p = torch.tensor([[1000.0, -1000.0, 999.0, float("nan")], [-1000.0, 1000.0, 999.0, float("inf")]])
        actual = full_vocab_divergences(p, p, vocab_size=3, chunk_size=2)
        self.assertTrue(torch.equal(actual["student_teacher_kl"], torch.zeros(2)))
        self.assertTrue(torch.equal(actual["teacher_student_kl"], torch.zeros(2)))
        shifted = full_vocab_divergences(p[:, :3], p[:, :3] + 4, chunk_size=2)
        torch.testing.assert_close(shifted["student_teacher_kl"], torch.zeros(2), atol=1e-6, rtol=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            full_vocab_divergences(p, p)

    def test_bfloat16_matches_the_same_quantized_logits_and_invalid_inputs_fail(self):
        p = torch.tensor([[0.3, -0.9, 0.1]], dtype=torch.bfloat16)
        q = torch.tensor([[0.4, 0.8, -0.2]], dtype=torch.bfloat16)
        actual = full_vocab_divergences(p, q, chunk_size=2)
        reference = full_vocab_divergences(p.float(), q.float(), chunk_size=8)
        for key in actual:
            torch.testing.assert_close(actual[key], reference[key], atol=1e-6, rtol=1e-6)
        for a, b, kwargs in [
            (p, q[:, :2], {}),
            (p, q, {"vocab_size": 4}),
            (p.long(), q.long(), {}),
            (p, q, {"chunk_size": 0}),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                full_vocab_divergences(a, b, **kwargs)

    def test_sequence_chain_rule_matches_full_enumeration_with_eos_and_cap(self):
        # EOS=0; p/q differ at the prompt and at both surviving first actions.
        p = {(): [0.2, 0.5, 0.3], (1,): [0.7, 0.2, 0.1], (2,): [0.3, 0.3, 0.4]}
        q = {(): [0.4, 0.2, 0.4], (1,): [0.3, 0.4, 0.3], (2,): [0.6, 0.1, 0.3]}
        conditional = {
            prefix: float(
                full_vocab_divergences(torch.tensor([p[prefix]]).log(), torch.tensor([q[prefix]]).log())[
                    "student_teacher_kl"
                ][0]
            )
            for prefix in p
        }
        trajectories = [(0,), *itertools.product((1, 2), range(3))]
        exact = expected_sum = sampled_ratio_mean = 0.0
        negative_sample_seen = False
        for trajectory in trajectories:
            probp = probq = 1.0
            summed_kl = 0.0
            for i, action in enumerate(trajectory):
                prefix = trajectory[:i]
                probp *= p[prefix][action]
                probq *= q[prefix][action]
                summed_kl += conditional[prefix]
            ratio = math.log(probp / probq)
            negative_sample_seen |= ratio < 0
            exact += probp * ratio
            sampled_ratio_mean += probp * ratio
            expected_sum += probp * summed_kl
        self.assertTrue(negative_sample_seen)
        self.assertAlmostEqual(exact, expected_sum, places=6)
        self.assertAlmostEqual(sampled_ratio_mean, exact, places=6)
        self.assertGreater(exact, 0)


class ValidationTrainerTests(unittest.TestCase):
    def fixture(self):
        trainer = object.__new__(RayPPOTrainer)
        trainer.validation_enabled = True
        trainer.global_steps = trainer.total_training_steps = 50
        trainer.use_rm = False
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {"val_kwargs": {"n": 2, "do_sample": True}, "agent": {"num_workers": 4}}
                },
                "trainer": {"validation_data_dir": "unused-fixture-output"},
            }
        )
        trainer.tokenizer = NS(eos_token_id=2, pad_token_id=2, decode=lambda ids, **_: str(ids.tolist()))
        trainer.val_dataloader = [
            {
                "fixture_row_index": torch.arange(2),
                "data_source": np.array(["MATH-500", "AIME25"], dtype=object),
                "uid": np.array(["q-a", "q-b"], dtype=object),
                "reward_model": np.array([{"ground_truth": "1"}, {"ground_truth": "2"}], dtype=object),
            },
            {
                "fixture_row_index": torch.arange(1),
                "data_source": np.array(["MATH-500"], dtype=object),
                "uid": np.array(["q-c"], dtype=object),
                "reward_model": np.array([{"ground_truth": "3"}], dtype=object),
            },
        ]
        trainer._get_gen_batch = lambda data: data
        events, dumped = [], []
        trainer.rollout_coordinator = NS(
            begin_eval=lambda step: events.append(("begin", step)),
            end_eval=lambda step, **kwargs: events.append(("end", step)),
        )

        def generate(batch):
            records, infos = [], np.empty(len(batch), dtype=object)
            for index, uid in enumerate(batch.non_tensor_batch["uid"]):
                tokens = [6, 2] if uid != "q-b" else [7] * 6
                records.append(([1, 3], tokens, "fixture"))
                infos[index] = {
                    "termination": "eos" if tokens[-1] == 2 else "length",
                    "finish_reason": "stop" if tokens[-1] == 2 else "length",
                    "response_cap": 6,
                    "prompt_id": uid,
                    "sample": index % 2,
                    "student_version": 50,
                }
            result = pack_token_records(records, prompt_cap=3, response_cap=6, pad_token_id=2, source="fixture")
            result.non_tensor_batch["native_response_info"] = infos
            return result

        trainer.async_rollout_manager = NS(generate_sequences=generate)
        trainer._maybe_log_val_generations = lambda **kwargs: None
        trainer._dump_generations = lambda **kwargs: dumped.append(kwargs)
        trainer._val_metrics_update = lambda *args: {"original_accuracy_metric": 0.5}
        return trainer, events, dumped

    def reward(self, batch):
        n = len(batch)
        return batch.batch["rm_scores"], {"acc": [1.0] * n, "pred": ["1"] * n}

    def test_actual_validate_unpads_before_stats_and_preserves_accuracy_and_dump_columns(self):
        trainer, events, dumped = self.fixture()
        with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=self.reward):
            metrics = trainer._validate()
        self.assertEqual(events, [("begin", 50), ("end", 50)])
        self.assertEqual(metrics["original_accuracy_metric"], 0.5)
        self.assertEqual(metrics["validation_rollout/micro/responses"], 6)
        self.assertEqual(metrics["validation_rollout/micro/questions"], 3)
        self.assertAlmostEqual(metrics["validation_rollout/micro/response_length/mean"], 20 / 6)
        self.assertEqual(metrics["validation_rollout/macro/response_length/mean"], 4)
        columns = dumped[0]["reward_extra_infos_dict"]
        self.assertEqual(columns["rollout_response_length"], [2, 2, 6, 6, 2, 2])
        self.assertEqual(columns["rollout_student_version"], [50] * 6)
        self.assertEqual(columns["rollout_finish_reason"], ["stop", "stop", "length", "length", "stop", "stop"])

    def test_real_dataproto_reorder_preserves_raw_info_and_excludes_eos_pad_tail(self):
        trainer, _, _ = self.fixture()
        batch = DataProto.from_single_dict(trainer.val_dataloader[0])
        output = trainer.async_rollout_manager.generate_sequences(batch)
        batch = batch.union(output)
        batch.reorder(torch.tensor([1, 0]))
        rows = rows_from_batch(batch, {"acc": [0, 1], "pred": [None, "1"]}, 2)
        self.assertEqual([r["uid"] for r in rows], ["q-b", "q-a"])
        self.assertEqual([r["response_length"] for r in rows], [6, 2])
        self.assertEqual([r["student_version"] for r in rows], [50, 50])
        self.assertEqual(summarize_rollouts(rows)["validation_rollout/micro/response_tokens"], 8)
        batch.batch["response_mask"][0] = torch.tensor([1, 0, 1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rows_from_batch(batch, {}, 2)

    def test_merged_validation_keeps_raw_rows_for_a_single_pooled_aggregation(self):
        trainer, _, _ = self.fixture()
        with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=self.reward):
            raw = trainer._validate(merged=True)
        metrics = trainer._merge_validation_results(raw, None)
        self.assertEqual(metrics["validation_rollout/micro/responses"], 6)
        self.assertEqual(metrics["original_accuracy_metric"], 0.5)


if __name__ == "__main__":
    unittest.main()
