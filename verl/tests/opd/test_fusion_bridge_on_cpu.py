"""Configuration and worker-packing contracts for the fused training path."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import LogitFusionConfig
from verl.workers.engine_workers import ActorRolloutRefWorker


class FusionConfigurationTests(unittest.TestCase):
    def test_nested_hydra_configuration_has_typed_defaults(self):
        config = OmegaConf.create(
            {
                "_target_": "verl.workers.config.RolloutConfig",
                "name": "vllm",
                "logit_fusion": {
                    "_target_": "verl.workers.config.LogitFusionConfig",
                    "enabled": True,
                    "teacher_model_path": "/teacher",
                    "teacher_weight": 0.5,
                    "backend": "dynamic",
                },
            }
        )
        obj = omega_conf_to_dataclass(config)
        self.assertIsInstance(obj.logit_fusion, LogitFusionConfig)
        self.assertEqual(obj.logit_fusion.teacher_weight, 0.5)
        self.assertEqual(obj.logit_fusion.backend, "dynamic")
        with self.assertRaises(ValueError):
            LogitFusionConfig(enabled=True)
        with self.assertRaises(ValueError):
            LogitFusionConfig(teacher_weight=1.1)


class WorkerBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_uses_previous_actor_version_and_keeps_eos(self):
        captured = {}

        class Adapter:
            async def generate_logit_fusion(self, ids, **kwargs):
                captured.update(kwargs)
                self.ids = ids
                return {
                    "responses": [[3, 0]],
                    "finish_reasons": ["eos"],
                    "behavior_log_probs": [[-0.2, -0.4]],
                    "seconds": 0.1,
                    "lambda": 0.5,
                    "student_version": 4,
                }

        config = OmegaConf.create(
            {"rollout": {"seed": 42, "prompt_length": 4, "response_length": 4}, "model": {"pad_token_id": 0}}
        )
        worker = SimpleNamespace(config=config, rank=1, world_size=4, rollout=Adapter())
        ids = np.empty(1, dtype=object)
        ids[0] = [1, 2]
        batch = DataProto.from_dict(
            tensors={"dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8)},
            non_tensors={"fusion_prompt_ids": ids},
            meta_info={"global_steps": 5},
        )
        result = await ActorRolloutRefWorker.generate_logit_fusion(worker, batch)
        self.assertEqual(captured["expected_version"], 4)
        self.assertEqual(captured["seed"], 63)
        self.assertEqual(result.batch["responses"].tolist(), [[3, 0, 0, 0]])
        self.assertEqual(result.batch["response_mask"].tolist(), [[1, 1, 0, 0]])
        self.assertNotIn("old_log_probs", result.batch)
        self.assertNotIn("rollout_log_probs", result.batch)
        self.assertAlmostEqual(result.meta_info["metrics"]["fusion/logp_sum"], -0.6)


if __name__ == "__main__":
    unittest.main()
