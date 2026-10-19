"""Tensor-gradient contracts; these tests are not empirical LLM evidence."""

import unittest

import torch

from opd_state.kernels.head_tail import coefficients_and_metrics, score_loss, selected_and_tail, summarize_kl
from opd_state.integrations.verl.layouts import packed_response_heads


class TopKTailTests(unittest.TestCase):
    def test_enumerated_gradient_and_value_match_full_kl(self):
        torch.manual_seed(7)
        for size in (3, 9):
            for k in (1, size - 1, size):
                z = torch.randn(2, size, dtype=torch.float64, requires_grad=True)
                qz = torch.randn_like(z)
                p, q = z.log_softmax(-1), qz.log_softmax(-1)
                target = (p.exp() * (p - q)).sum()
                target_gradient = torch.autograd.grad(target, z, retain_graph=True)[0]
                head = selected_and_tail(z, k)
                teacher = selected_and_tail(qz, k, head["student_top_k_ids"])
                expected_gradient, expected_value = torch.zeros_like(z), torch.zeros(2, dtype=z.dtype)
                # Enumerate each state's next action, retaining original full-vocab probability.
                for action in range(size):
                    actions = torch.full((2,), action, dtype=torch.long)
                    mask = torch.ones(2, dtype=torch.bool)
                    hc, tc, metrics = coefficients_and_metrics(
                        head["student_top_k_log_probs"],
                        teacher["student_top_k_log_probs"],
                        p[:, action],
                        q[:, action],
                        head["student_tail_log_mass"],
                        teacher["student_tail_log_mass"],
                        head["student_top_k_ids"],
                        actions,
                        mask,
                    )
                    loss = score_loss(head["student_top_k_log_probs"], p[:, action], hc, tc, mask)
                    grad = torch.autograd.grad(loss.sum(), z, retain_graph=True)[0]
                    expected_value += p[:, action].detach().exp() * metrics["kl_unbiased"]
                    expected_gradient += p[:, action].detach().exp().unsqueeze(-1) * grad
                    self.assertTrue((metrics["kl_grouped"] <= (p.exp() * (p - q)).sum(-1) + 1e-12).all())
                torch.testing.assert_close(expected_gradient, target_gradient, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(expected_value, (p.exp() * (p - q)).sum(-1), atol=1e-12, rtol=1e-12)

    def test_tail_logsumexp_and_frozen_support(self):
        z = torch.tensor([[0.0, -40.0, -50.0]], dtype=torch.float64)
        head = selected_and_tail(z, 1)
        self.assertTrue(torch.isfinite(head["student_tail_log_mass"]).all())
        changed = torch.tensor([[-3.0, 4.0, 5.0]], dtype=torch.float64, requires_grad=True)
        frozen = selected_and_tail(changed, 1, head["student_top_k_ids"])
        self.assertEqual(frozen["student_top_k_ids"].item(), 0)
        torch.testing.assert_close(frozen["student_top_k_log_probs"], changed.log_softmax(-1)[:, :1])

    def test_behavior_importance_correction_recovers_value_and_gradient(self):
        torch.manual_seed(41)
        z = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
        qz, bz = torch.randn_like(z), torch.randn_like(z)
        p, q, b = z.log_softmax(-1), qz.log_softmax(-1), bz.log_softmax(-1)
        target = (p.exp() * (p - q)).sum(-1)
        target_gradient = torch.autograd.grad(target.sum(), z, retain_graph=True)[0]
        head = selected_and_tail(z, 2)
        teacher = selected_and_tail(qz, 2, head["student_top_k_ids"])
        expected_gradient, expected_value = torch.zeros_like(z), torch.zeros_like(target)
        for action in range(z.shape[-1]):
            actions = torch.full((2,), action, dtype=torch.long)
            mask = torch.ones(2, dtype=torch.bool)
            hc, tc, metrics = coefficients_and_metrics(
                head["student_top_k_log_probs"],
                teacher["student_top_k_log_probs"],
                p[:, action],
                q[:, action],
                head["student_tail_log_mass"],
                teacher["student_tail_log_mass"],
                head["student_top_k_ids"],
                actions,
                mask,
                behavior_action=b[:, action],
            )
            loss = score_loss(head["student_top_k_log_probs"], p[:, action], hc, tc, mask)
            gradient = torch.autograd.grad(loss.sum(), z, retain_graph=True)[0]
            expected_value += b[:, action].exp() * metrics["kl_unbiased"]
            expected_gradient += b[:, action].exp().unsqueeze(-1) * gradient
        torch.testing.assert_close(expected_value, target, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(expected_gradient, target_gradient, atol=1e-12, rtol=1e-12)

    def test_padding_nan_and_eos_position(self):
        head = torch.tensor([[[0.0]], [[float("nan")]]], requires_grad=True)
        action = torch.tensor([[0.2], [float("nan")]], requires_grad=True)
        coefficients = torch.tensor([[[2.0]], [[float("nan")]]])
        tail = torch.tensor([[3.0], [float("nan")]])
        mask = torch.tensor([[True], [False]])
        loss = score_loss(head, action, coefficients, tail, mask).sum()
        loss.backward()
        self.assertEqual(action.grad[1].item(), 0.0)
        self.assertEqual(head.grad[1].item(), 0.0)
        data = {
            "student_top_k_ids": torch.tensor([[[4, 5], [6, 7], [0, 0]], [[2, 3], [0, 0], [0, 0]]]),
            "prompts": torch.tensor([[0, 8, 9], [8, 9, 10]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
        }
        packed = packed_response_heads(data)
        self.assertEqual(packed[1:3].tolist(), [[4, 5], [6, 7]])
        self.assertEqual(packed[6].tolist(), [2, 3])
        self.assertEqual(packed.shape, (8, 2))

    def test_native_actor_loss_uses_predictor_positions_and_global_token_mean(self):
        from types import SimpleNamespace

        from tensordict import TensorDict

        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.losses import ppo_loss
        from verl.workers.utils.padding import left_right_2_no_padding

        data = TensorDict(
            {
                "input_ids": torch.tensor([[0, 8, 9, 4, 6, 0], [8, 9, 10, 2, 0, 0]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
                "prompts": torch.tensor([[0, 8, 9], [8, 9, 10]]),
                "responses": torch.tensor([[4, 6, 0], [2, 0, 0]]),
                "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0], [0, 1, 2, 3, 0, 0]]),
                "opd_head_coefficient": torch.ones(2, 3, 2),
                "opd_tail_coefficient": torch.full((2, 3), 2.0),
            },
            batch_size=[2],
        )
        data = left_right_2_no_padding(data)
        tu.assign_non_tensor(data, opd_topk_tail=True, dp_size=1, batch_num_tokens=3, global_batch_size=2)
        action = torch.arange(8, dtype=torch.float64, requires_grad=True)
        output = {"log_probs": action, "student_top_k_log_probs": torch.stack([action, 2 * action], dim=-1)}
        config = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None)
        loss, _ = ppo_loss(config, output, data)
        self.assertAlmostEqual(loss.item(), 15.0)
        loss.backward()
        expected = torch.zeros(8, dtype=torch.float64)
        expected[[1, 2, 6]] = 5 / 3
        torch.testing.assert_close(action.grad, expected)

    def test_native_engine_metadata_vocab_and_frozen_ids(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        from tensordict import TensorDict

        from verl.utils import tensordict_utils as tu
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        data = TensorDict(
            {
                "input_ids": torch.tensor([[0, 8, 9, 4, 6, 0], [8, 9, 10, 2, 0, 0]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
                "prompts": torch.tensor([[0, 8, 9], [8, 9, 10]]),
                "responses": torch.tensor([[4, 6, 0], [2, 0, 0]]),
                "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0], [0, 1, 2, 3, 0, 0]]),
            },
            batch_size=[2],
        )
        data = left_right_2_no_padding(data)
        tu.assign_non_tensor(data, opd_topk_tail=True, opd_effective_vocab_size=11, opd_student_top_k=2)
        engine = SimpleNamespace(
            pad_to_length=None,
            engine_config=SimpleNamespace(ulysses_sequence_parallel_size=1),
            _gather_and_unpad_packed=lambda value, padding: value,
        )
        z = torch.randn(1, 8, 13, dtype=torch.float64, requires_grad=True)
        args = {
            "pad_size": 0,
            "temperature_rmpad": torch.ones(8),
            "input_ids_rmpad_rolled": torch.roll(data["input_ids"].values(), -1),
        }
        with patch.dict(os.environ, {"VERL_DISABLE_FLASH_ATTN_CE": "1"}):
            out = FSDPEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=z), args, data, None)
            expected = z[0, :, :11].log_softmax(-1).gather(-1, args["input_ids_rmpad_rolled"][:, None]).squeeze(-1)
            torch.testing.assert_close(out["log_probs"].values(), expected)
            data["student_top_k_ids"] = no_padding_2_padding(out["student_top_k_ids"], data)
            tu.assign_non_tensor(data, opd_frozen_head=True)
            changed = torch.randn_like(z, requires_grad=True)
            out2 = FSDPEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=changed), args, data, None)
            restored = no_padding_2_padding(out2["student_top_k_ids"], data)
            torch.testing.assert_close(restored, data["student_top_k_ids"])
            (out2["student_top_k_log_probs"].values().sum() + out2["log_probs"].values().sum()).backward()
            self.assertTrue(torch.isfinite(changed.grad).all())
            self.assertEqual(changed.grad[..., 11:].abs().sum().item(), 0.0)

    def test_sequence_and_token_reduction(self):
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        metrics = summarize_kl({"kl_unbiased": torch.tensor([[1.0, 3.0, 0.0], [2.0, 0.0, 0.0]])}, mask)
        self.assertEqual(metrics["opd_alignment/kl_unbiased_sequence_mean"], 3.0)
        self.assertEqual(metrics["opd_alignment/kl_unbiased_token_mean"], 2.0)

    def test_training_guards_reject_changes_to_unbiased_contract(self):
        from omegaconf import OmegaConf

        from verl.trainer.ppo.opd_trainer_mixin import OPDTrainerMixin

        config = OmegaConf.create(
            {
                "distillation": {"fsdp_teacher": {"unbiased_topk_tail": True, "effective_vocab_size": 11}},
                "actor_rollout_ref": {
                    "actor": {
                        "ppo_epochs": 1,
                        "ppo_mini_batch_size": 8,
                        "policy_loss": {"loss_mode": "opd_topk_tail"},
                        "use_kl_loss": False,
                        "entropy_coeff": 0,
                        "loss_agg_mode": "token-mean",
                        "ulysses_sequence_parallel_size": 1,
                    },
                    "rollout": {"temperature": 1, "top_p": 1, "top_k": -1, "repetition_penalty": 1},
                    "model": {"use_fused_kernels": False},
                },
                "data": {"train_batch_size": 8},
                "algorithm": {"adv_estimator": "token_reward_direct", "rollout_correction": None},
            }
        )
        mixin = OPDTrainerMixin()
        mixin.use_opd_teacher, mixin.tokenizer = True, list(range(11))
        mixin.config = config
        mixin.validate_opd_top_k_config()
        invalid = {
            "actor_rollout_ref.actor.ppo_epochs": 2,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
            "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
            "actor_rollout_ref.actor.entropy_coeff": 0.1,
            "actor_rollout_ref.rollout.top_p": 0.9,
            "actor_rollout_ref.rollout.temperature": 0.7,
            "distillation.fsdp_teacher.effective_vocab_size": 12,
            "actor_rollout_ref.model.use_fused_kernels": True,
            "algorithm.rollout_correction": {"rollout_is": "token"},
        }
        for key, value in invalid.items():
            with self.subTest(key=key):
                mixin.config = OmegaConf.create(OmegaConf.to_container(config))
                OmegaConf.update(mixin.config, key, value)
                with self.assertRaises(ValueError):
                    mixin.validate_opd_top_k_config()


if __name__ == "__main__":
    unittest.main()
