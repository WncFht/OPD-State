"""Torch/Hydra bridge checks; run explicitly with the training environment."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from verl import DataProto
from verl.trainer.ppo.rollout_coordinator import RolloutCoordinator, epochs_for_budget, prepare_native_config

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    pad_token_id = eos_token_id = 2
    bos_token_id = None
    chat_template = "canonical"

    def __init__(self):
        self.rendered = []

    def __len__(self):
        return 10

    def get_vocab(self):
        return {str(i): i for i in range(10)}

    def apply_chat_template(self, messages, **kwargs):
        self.rendered.append((messages, kwargs))
        return [1, 3]


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tokenizer = Tokenizer()
        for name in ["student", "teacher"]:
            path = self.root / name
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"model_type": "qwen2", "vocab_size": 10}))
        (self.root / "train.parquet").write_bytes(b"fixture identity only")
        with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
            self.config = compose(config_name="ppo_trainer")
        c = self.config
        with open_dict(c):
            c.actor_rollout_ref.model.path = str(self.root / "student")
            c.actor_rollout_ref.rollout.name = "vllm"
            c.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
            c.actor_rollout_ref.rollout.policy = {"kind": "single", "model": "teacher"}
            c.actor_rollout_ref.rollout.gpu_memory_utilization = 0.2
            c.actor_rollout_ref.rollout.max_model_len = 16
            c.actor_rollout_ref.rollout.prompt_length = 4
            c.actor_rollout_ref.rollout.response_length = 4
            c.actor_rollout_ref.rollout.val_kwargs.max_tokens = 6
            c.actor_rollout_ref.rollout.val_kwargs.n = 16
            c.distillation.fsdp_teacher.enabled = True
            c.distillation.fsdp_teacher.model_path = str(self.root / "teacher")
            c.algorithm.adv_estimator = "token_reward_direct"
            c.algorithm.rollout_correction = None
            c.data.train_batch_size = 4
            c.data.train_files = [str(self.root / "train.parquet")]
            c.trainer.resume_mode = "disable"
            c.trainer.default_local_dir = str(self.root / "checkpoint")
        self.signature = patch("verl.trainer.ppo.rollout_coordinator.model_signature", return_value="sha")
        self.hf_tokenizer = patch("verl.utils.tokenizer.hf_tokenizer", return_value=self.tokenizer)
        self.signature.start()
        self.hf_tokenizer.start()

    def tearDown(self):
        self.signature.stop()
        self.hf_tokenizer.stop()
        self.tmp.cleanup()

    def prepare(self):
        return prepare_native_config(self.config, self.tokenizer, None)

    def batch(self, n=1, *, validate=False, step=1):
        raw = np.empty(n, dtype=object)
        for i in range(n):
            raw[i] = [{"role": "user", "content": "one question"}]
        return DataProto.from_dict(
            non_tensors={"raw_prompt": raw, "data_source": np.array(["math"] * n, dtype=object)},
            meta_info={"global_steps": step, "validate": validate},
        )

    def test_real_hydra_defaults_and_fixed_sources_do_not_change_actor_initialization(self):
        self.assertTrue(self.prepare())
        c = self.config
        self.assertEqual(c.actor_rollout_ref.model.path, str(self.root / "student"))
        self.assertEqual(c.trainer.native_required_models, ["teacher"])
        self.assertEqual(c.actor_rollout_ref.rollout.generation_models.teacher.fingerprint, "sha")
        first = c.actor_rollout_ref.rollout.generation_sync_namespace
        self.prepare()
        self.assertNotEqual(first, c.actor_rollout_ref.rollout.generation_sync_namespace)

    def test_switch_divides_one_memory_budget_and_tail_rejects_off_policy_actions(self):
        c = self.config
        c.actor_rollout_ref.rollout.policy = {
            "kind": "sequence_switch",
            "first": "current_student",
            "second": "teacher",
            "boundary": {"kind": "response_tokens", "value": 2},
        }
        self.prepare()
        self.assertEqual(c.actor_rollout_ref.rollout.gpu_memory_utilization, 0.1)
        self.assertEqual(c.actor_rollout_ref.rollout.generation_models.teacher.memory_fraction, 0.1)
        c.distillation.fsdp_teacher.unbiased_topk_tail = True
        with self.assertRaisesRegex(ValueError, "fresh current-student"):
            self.prepare()

    def test_prompt_visits_and_eval_samples_have_independent_counters(self):
        self.prepare()
        co = RolloutCoordinator(self.config, self.tokenizer, None, None, None)
        self.assertEqual([r.visit for r in co._requests(self.batch(2), False)], [0, 1])
        self.assertEqual([r.sample for r in co._requests(self.batch(16), True)], list(range(16)))
        self.assertEqual(co._requests(self.batch(), False)[0].visit, 2)
        self.assertEqual(len(self.tokenizer.rendered), 19)
        self.assertTrue(all(k["add_generation_prompt"] for _, k in self.tokenizer.rendered))

    def test_training_and_final_student_evaluation_recompute_real_scores_and_keep_eos(self):
        self.prepare()
        events = []

        class Pool:
            handles = {}
            awake = set()

            async def prepare(self, slots):
                events.append(("prepare", slots))

            async def sleep_all(self):
                events.append(("sleep",))

        class Server:
            async def generate(self, calls, identity, params):
                events.append(("generate", identity))
                return [
                    {
                        **call,
                        "token_ids": [4, 2],
                        "identity": identity,
                        "finish_reason": "stop",
                        "stop_reason": 2,
                        "log_probs": None,
                    }
                    for call in calls
                ]

        server = Server()
        handle = NS(generate_native_batch=NS(remote=server.generate))
        pool = Pool()
        pool.handles = {"teacher": [handle], "current_student": [handle]}

        class Reward:
            def compute_rm_score(self, batch):
                self.batch = batch
                scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                scores[:, 1] = 1
                return DataProto.from_dict(tensors={"rm_scores": scores})

        reward = Reward()
        checkpoint = NS(update_weights=lambda version: events.append(("sync", version)))
        co = RolloutCoordinator(self.config, self.tokenizer, pool, checkpoint, reward)
        self.assertEqual(co.update_weights(0)["source/rollout_sync"], 0)
        out = co.generate_sequences(self.batch())
        self.assertEqual(out.batch["response_mask"].tolist(), [[1, 1, 0, 0]])
        self.assertEqual(out.batch["rm_scores"].sum().item(), 1)
        self.assertIn("raw_prompt", reward.batch.non_tensor_batch)
        self.assertEqual(events[1][1][0], "teacher")
        trace = list((self.root / "native-trajectories").glob("step-*.jsonl"))
        self.assertEqual(len(trace), 1)
        self.assertEqual(json.loads(trace[0].read_text())["segments"][0]["identities"][0][0], "teacher")
        co.begin_eval(2)
        ev = co.generate_sequences(self.batch(16, validate=True, step=2))
        co.end_eval(2, resume_training=False)
        self.assertEqual(len(ev), 16)
        self.assertIn(("sync", 2), events)
        self.assertEqual([e[1][0] for e in events if e[0] == "generate"], ["teacher", "current_student"])

    def test_update_budget_covers_multiple_epochs_without_a_fifty_step_dataset_limit(self):
        self.assertEqual(epochs_for_budget(301, 50), 7)
        self.assertEqual(epochs_for_budget(25, 50), 1)
        with self.assertRaises(ValueError):
            epochs_for_budget(5, 0)


if __name__ == "__main__":
    unittest.main()
