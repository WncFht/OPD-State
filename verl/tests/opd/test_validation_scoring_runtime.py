"""Actual tiny Qwen2 forward comparisons for bounded-memory KL scoring."""

import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.workers.validation_kl import record_fingerprint, score_causal_records, validate_record


class CausalScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = Qwen2Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.2,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        self.student = Qwen2ForCausalLM(config)
        self.teacher = Qwen2ForCausalLM(config)

    def record(self, response=(5, 6, 2), uid="q-a"):
        return {
            "state_source": "on_validation_states",
            "source": "MATH-500",
            "uid": uid,
            "sample": 0,
            "prompt_ids": [1, 4, 3],
            "response_ids": list(response),
        }

    def reference(self, record):
        self.student.eval()
        self.teacher.eval()
        prompt, response = record["prompt_ids"], record["response_ids"]
        ids = torch.tensor([prompt + response])
        with torch.no_grad():
            p = self.student(ids, use_cache=False).logits[0, len(prompt) - 1 : len(prompt) + len(response) - 1]
            q = self.teacher(ids, use_cache=False).logits[0, len(prompt) - 1 : len(prompt) + len(response) - 1]
        logp, logq = p.double().log_softmax(-1), q.double().log_softmax(-1)
        return {
            "student_teacher_kl": (logp.exp() * (logp - logq)).sum(-1),
            "teacher_student_kl": (logq.exp() * (logq - logp)).sum(-1),
            "student_entropy": -(logp.exp() * logp).sum(-1),
            "teacher_entropy": -(logq.exp() * logq).sum(-1),
        }

    def score(self, records, **kwargs):
        return score_causal_records(
            self.student,
            self.teacher,
            records,
            vocab_size=19,
            device="cpu",
            position_chunk_size=7,
            vocab_chunk_size=5,
            **kwargs,
        )

    def test_all_positions_shifted_by_one_match_dense_forward_and_double_reference(self):
        records = [self.record((2,), "short"), self.record(), self.record([7] * 31 + [2], "long")]
        references = [self.reference(record) for record in records]
        actual, timing = self.score(records)
        self.assertEqual(len(actual), 3)
        for record, ref, row in zip(records, references, actual, strict=True):
            self.assertEqual(row["record_sha256"], record_fingerprint(record))
            self.assertEqual(row["evaluated_positions"], len(record["response_ids"]))
            for key, values in ref.items():
                self.assertAlmostEqual(row["sums"][key], float(values.sum()), delta=2e-5)
                self.assertAlmostEqual(sum(d["sums"][key] for d in row["depths"].values()), row["sums"][key])
        self.assertEqual(timing["head_snapshot_bytes"], self.student.lm_head.weight.numel() * 4)
        self.assertGreater(timing["seconds"], 0)

    def test_readonly_scoring_restores_each_training_flag_hooks_rng_and_weights(self):
        self.student.train()
        self.student.model.layers[0].self_attn.eval()
        self.teacher.eval()
        flags = [(m, m.training) for root in (self.student, self.teacher) for m in root.modules()]
        p, q = copy.deepcopy(self.student.state_dict()), copy.deepcopy(self.teacher.state_dict())
        rng = torch.random.get_rng_state().clone()
        self.score([self.record()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for module, before in flags:
            self.assertEqual(module.training, before)
        for module, before in [(self.student, p), (self.teacher, q)]:
            self.assertFalse(module.lm_head._forward_pre_hooks)
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
            for name, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_sparse_fixed_bank_positions_use_full_raw_prefix_and_given_weights(self):
        record = self.record([7] * 15 + [2])
        record.update(state_source="on_initial_states", positions=[0, 5, 10, 15], weights=[1, 5, 5, 5])
        reference = self.reference(record)
        actual, _ = self.score([record])
        for key, values in reference.items():
            expected = float((values[record["positions"]] * torch.tensor(record["weights"])).sum())
            self.assertAlmostEqual(actual[0]["sums"][key], expected, delta=2e-5)
        self.assertEqual(actual[0]["evaluated_positions"], 4)
        self.assertEqual(actual[0]["depths"], {})

    def test_new_call_reads_updated_student_weights_and_keeps_teacher_unchanged(self):
        teacher_before = copy.deepcopy(self.teacher.state_dict())
        record = self.record()
        first, _ = self.score([record])
        with torch.no_grad():
            self.student.lm_head.weight.mul_(3)
        second, _ = self.score([record])
        self.assertNotAlmostEqual(first[0]["sums"]["student_teacher_kl"], second[0]["sums"]["student_teacher_kl"])
        for name, value in self.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[name]))

    def test_failure_removes_both_temporary_hooks_and_restores_training_flags(self):
        self.student.train()
        self.teacher.eval()
        with patch("verl.workers.validation_kl.full_vocab_divergences", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.score([self.record()])
        self.assertTrue(self.student.training)
        self.assertFalse(self.teacher.training)
        self.assertFalse(self.student.lm_head._forward_pre_hooks)
        self.assertFalse(self.teacher.lm_head._forward_pre_hooks)
        self.score([self.record()])

    def test_invalid_vocabulary_positions_and_partial_main_validation_fail(self):
        bad = [self.record() for _ in range(4)]
        bad[0]["response_ids"] = [19]
        bad[1].update(positions=[0, 2], weights=[1, 2])
        bad[2].update(state_source="on_teacher_states", positions=[0, 0], weights=[1, 2])
        bad[3].update(state_source="on_teacher_states", positions=[0], weights=[float("nan")])
        for record in bad:
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_record(record, 19)


if __name__ == "__main__":
    unittest.main()
