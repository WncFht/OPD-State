"""Use real DataProto reordering and OPD scoring for source-stratified metrics."""

import math
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.native_state_metrics import summarize_native_states, trajectory_segments
from verl.trainer.ppo.opd_trainer_mixin import OPDTrainerMixin


class NativeStateMetricTests(unittest.TestCase):
    def fixture(self):
        provenance = np.empty(2, dtype=object)
        provenance[0] = (("teacher", 0, 0, 2), ("current_student", 1, 2, 3))
        provenance[1] = (("initial_student", 0, 0, 2),)
        return DataProto.from_dict(
            tensors={
                "response_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                "old_log_probs": torch.tensor(
                    [[-1.0, -2.0, -3.0, float("nan")], [-4.0, -5.0, float("nan"), float("nan")]]
                ),
                "fixture_teacher_logp": torch.tensor(
                    [[-2.0, -3.0, -4.0, float("nan")], [-2.0, -3.0, float("nan"), float("nan")]]
                ),
            },
            non_tensors={"native_segments": provenance},
        )

    def summarize(self, batch):
        return summarize_native_states(
            batch.batch["response_mask"],
            batch.batch["old_log_probs"],
            batch.batch["fixture_teacher_logp"],
            batch.non_tensor_batch["native_segments"],
        )

    def test_reordering_and_batch_slicing_preserve_action_sources_and_inherited_depth(self):
        batch = self.fixture()
        metrics = self.summarize(batch)
        self.assertEqual(metrics["native_state/all/tokens"], 5)
        self.assertEqual(metrics["native_state/segment_1/action_current_student/response_prefix_depth_mean"], 2)
        self.assertEqual(metrics["native_state/action_teacher/sampled_log_ratio_mean"], 1)
        self.assertEqual(metrics["native_state/action_initial_student/sampled_log_ratio_mean"], -2)
        self.assertAlmostEqual(metrics["native_state/all/log_ratio_rms"], math.sqrt(11 / 5), places=6)
        self.assertAlmostEqual(
            metrics["native_state/action_teacher/teacher_selected_action_probability_mean"],
            (math.exp(-2) + math.exp(-3)) / 2,
        )
        batch.reorder(torch.tensor([1, 0]))
        for key, value in self.summarize(batch).items():
            self.assertAlmostEqual(value, metrics[key], places=6, msg=key)
        self.assertEqual(self.summarize(batch[:1])["native_state/all/tokens"], 2)
        self.assertEqual(self.summarize(batch[1:])["native_state/all/tokens"], 3)
        tensor_dict = batch.to_tensordict()
        self.assertIn("native_segments", tensor_dict)
        round_trip = DataProto.from_tensordict(tensor_dict)
        for key, value in self.summarize(round_trip).items():
            self.assertAlmostEqual(value, metrics[key], places=6, msg=key)

    def test_zero_second_stage_keeps_zero_count_without_nan_and_padding_is_ignored(self):
        batch = self.fixture()[:1]
        batch.non_tensor_batch["native_segments"][0] = (("teacher", 0, 0, 3),)
        metrics = self.summarize(batch)
        self.assertEqual(metrics["native_state/segment_1/tokens"], 0)
        self.assertNotIn("native_state/segment_1/log_ratio_rms", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        batch.batch["old_log_probs"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.summarize(batch)

    def test_bad_provenance_and_holes_in_masks_fail_before_mislabeling_scores(self):
        for metadata in [
            (),
            (("teacher", 0, 0, 2),),
            (("teacher", 0, 1, 3),),
            (("teacher", 1, 0, 3),),
            (("other", 0, 0, 3),),
        ]:
            batch = self.fixture()
            batch.non_tensor_batch["native_segments"][0] = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.summarize(batch)
        batch = self.fixture()
        batch.batch["response_mask"][0] = torch.tensor([1, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.summarize(batch)

    def test_fusion_is_labeled_as_a_fusion_policy(self):
        trajectories = [
            NS(segments=[NS(identities=(("initial_student", "/p", "sha"), ("teacher", "/q", "sha")), start=0, end=3)])
        ]
        self.assertEqual(trajectory_segments(trajectories), [(("fusion_initial_student", 0, 0, 3),)])
        trajectories[0].segments[0].identities = (("teacher", "/q", "sha"), ("current_student", "/p", 0))
        with self.assertRaises(ValueError):
            trajectory_segments(trajectories)

    def test_actual_opd_hook_adds_diagnostics_without_changing_rewards(self):
        batch = self.fixture()
        # The production reward scorer uses zero padding; diagnostic-only tests
        # above separately check that NaN padding cannot contaminate summaries.
        for key in ["old_log_probs", "fixture_teacher_logp"]:
            batch.batch[key].nan_to_num_(nan=0)

        def score(data):
            q = data.batch["fixture_teacher_logp"].clone()
            return DataProto.from_dict(tensors={"rm_scores": q - data.batch["old_log_probs"], "teacher_log_probs": q})

        trainer = NS(
            opd_top_k_is_reward=lambda: False,
            opd_topk_tail_enabled=lambda: False,
            opd_teacher_wg=NS(compute_rm_score=score),
        )
        expected = batch.batch["fixture_teacher_logp"] - batch.batch["old_log_probs"]
        original_student = batch.batch["old_log_probs"].clone()
        metrics = {}
        result = OPDTrainerMixin.compute_opd_token_reward(trainer, batch, metrics)
        self.assertTrue(torch.equal(result.batch["token_level_scores"], expected))
        self.assertTrue(torch.equal(result.batch["old_log_probs"], original_student))
        self.assertEqual(metrics["native_state/all/tokens"], 5)
        self.assertAlmostEqual(
            metrics["native_state/all/sampled_log_ratio_mean"], -metrics["opd/token_reward_mean"], places=6
        )


if __name__ == "__main__":
    unittest.main()
