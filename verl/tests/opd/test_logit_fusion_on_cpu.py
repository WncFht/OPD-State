"""Fusion math, shared-prefix caching and weight ownership on tiny real models."""

import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.workers.rollout.fusion.decoder import LogitFusionDecoder, fuse_logits


def models():
    torch.manual_seed(9)
    config = Qwen2Config(
        vocab_size=19,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        max_position_embeddings=128,
        eos_token_id=18,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    student, teacher = Qwen2ForCausalLM(config), Qwen2ForCausalLM(copy.deepcopy(config))
    return student.eval(), teacher.eval()


def reference_greedy(student, teacher, prompts, weight, count):
    result = [[] for _ in prompts]
    for _ in range(count):
        seq = [p + r for p, r in zip(prompts, result, strict=True)]
        width = max(map(len, seq))
        ids, mask = torch.zeros(len(seq), width, dtype=torch.long), torch.zeros(len(seq), width, dtype=torch.long)
        for i, tokens in enumerate(seq):
            ids[i, -len(tokens) :] = torch.tensor(tokens)
            mask[i, -len(tokens) :] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0).masked_fill(mask == 0, 0)
        with torch.inference_mode():
            a = student(ids, attention_mask=mask, position_ids=positions, logits_to_keep=1).logits[:, -1]
            b = teacher(ids, attention_mask=mask, position_ids=positions, logits_to_keep=1).logits[:, -1]
            token = fuse_logits(a, b, weight).argmax(-1).tolist()
        for r, value in zip(result, token, strict=True):
            r.append(value)
    return result


class FusionMathTests(unittest.TestCase):
    def test_endpoints_and_geometric_distribution(self):
        a = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
        b = torch.tensor([[-0.5, 1.0, 2.0]], dtype=torch.float64)
        torch.testing.assert_close(fuse_logits(a, b, 0), a)
        torch.testing.assert_close(fuse_logits(a, b, 1), b)
        expected = a.softmax(-1).pow(0.5) * b.softmax(-1).pow(0.5)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(fuse_logits(a, b, 0.5).softmax(-1), expected)
        torch.testing.assert_close(fuse_logits(a + 700, b - 400, 0.5).softmax(-1), expected)

    def test_invalid_coefficients_and_empty_support_fail(self):
        a, b = torch.tensor([[0.0, -float("inf")]]), torch.tensor([[-float("inf"), 0.0]])
        for weight in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                fuse_logits(a, b, weight)
        with self.assertRaisesRegex(ValueError, "empty shared support"):
            fuse_logits(a, b, 0.5)
        self.assertTrue(torch.isfinite(fuse_logits(a, b, 0)[:, 0]).all())


class FusionDecodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.student, self.teacher = models()
        # Disable EOS for this deterministic full-prefix comparison by selecting
        # an unreachable token in a larger padded model vocabulary.
        self.prompts = [[1, 2, 3], [4]]

    def decoder(self, backend, weight=0.5):
        return LogitFusionDecoder(
            self.student,
            self.teacher,
            vocab_size=19,
            eos_token_id=18,
            pad_token_id=0,
            weight=weight,
            backend=backend,
            prompt_cap=8,
            response_cap=12,
            completion_check_interval=1,
        )

    def test_cached_decode_matches_full_prefix_and_static_reference(self):
        for weight in [0.0, 0.5, 1.0]:
            reference = reference_greedy(self.student, self.teacher, self.prompts, weight, 6)
            expected = [row[: row.index(18) + 1] if 18 in row else row for row in reference]
            for backend in ["dynamic", "static"]:
                actual = self.decoder(backend, weight).generate(self.prompts, max_new_tokens=6, temperature=0)
                self.assertEqual(actual["responses"], expected)

    def test_static_cache_is_reset_between_different_prompts(self):
        decoder = self.decoder("static")
        first = decoder.generate(self.prompts, max_new_tokens=8, temperature=0)
        decoder.generate([[5], [6, 7]], max_new_tokens=3, temperature=0)
        third = decoder.generate(self.prompts, max_new_tokens=8, temperature=0)
        self.assertEqual(first["responses"], third["responses"])

    def test_reproducible_sampling_and_finite_behavior_probabilities(self):
        decoder = self.decoder("dynamic")
        a = decoder.generate(self.prompts, max_new_tokens=7, seed=777)
        b = decoder.generate(self.prompts, max_new_tokens=7, seed=777)
        self.assertEqual(a["responses"], b["responses"])
        for row in a["behavior_log_probs"]:
            self.assertTrue(all(torch.isfinite(torch.tensor(row))))

    def test_weight_version_and_frozen_teacher(self):
        decoder = self.decoder("static")
        teacher_before = {k: v.detach().clone() for k, v in self.teacher.named_parameters()}
        weights = [(k, v.detach().clone() + 0.01) for k, v in self.student.named_parameters()]
        split = len(weights) // 2
        decoder.load_student_weights(weights[:split], version=0, final=False)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            decoder.generate(self.prompts)
        decoder.load_student_weights(weights[split:], version=0, final=True)
        decoder.generate(self.prompts, max_new_tokens=2, expected_version=0)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            decoder.generate(self.prompts, expected_version=1)
        for name, value in self.teacher.named_parameters():
            torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            decoder.load_student_weights(weights[:1], version=1, final=True)

    def test_eos_stops_each_row_and_prompt_bounds_fail(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = Qwen2Config(vocab_size=19, hidden_size=24, num_hidden_layers=1)

            def forward(self, input_ids, **kwargs):
                logits = torch.full((input_ids.shape[0], 1, 19), -100.0)
                logits[:, :, 18] = 100.0
                return type("Output", (), {"logits": logits})()

        decoder = LogitFusionDecoder(
            Constant(), Constant(), vocab_size=19, eos_token_id=18, pad_token_id=0, prompt_cap=8, response_cap=8
        )
        out = decoder.generate([[1], [2, 3]], temperature=0)
        self.assertEqual(out["responses"], [[18], [18]])
        self.assertEqual(out["finish_reasons"], ["eos", "eos"])
        for prompts in [[], [[]], [[22]], [[1] * 9]]:
            with self.assertRaises(ValueError):
                decoder.generate(prompts)


if __name__ == "__main__":
    unittest.main()
