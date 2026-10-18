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
from verl.trainer.ppo.fixed_rollout import digest_file, tokenizer_signature
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
        self.assertEqual(c.actor_rollout_ref.rollout.gpu_memory_utilization, 0.2)
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

    def test_same_question_with_different_row_ids_increments_the_same_visit_counter(self):
        self.prepare()
        co = RolloutCoordinator(self.config, self.tokenizer, None, None, None)
        batch = self.batch(2)
        batch.non_tensor_batch["opd_prompt_id"] = np.array(["row-a", "row-b"], dtype=object)
        requests = co._requests(batch, False)
        self.assertEqual([r.visit for r in requests], [0, 1])
        self.assertEqual(requests[0].prompt_id, requests[1].prompt_id)
        self.assertEqual([r.row_id for r in requests], ["row-a", "row-b"])

    def test_training_and_final_student_evaluation_recompute_real_scores_and_keep_eos(self):
        self.prepare()
        events = []

        class Pool:
            handles = {}
            awake = set()
            eval_slot = "eval:current_student"

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
        pool.handles = {"teacher": [handle], "eval:current_student": [handle]}

        class Reward:
            def compute_rm_score(self, batch):
                self.batch = batch
                scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                scores[:, 1] = 1
                return DataProto.from_dict(tensors={"rm_scores": scores})

        reward = Reward()
        checkpoint = NS(
            update_weights=lambda version: events.append(("sync", version)),
            actor_wg=NS(set_native_rollout_target=lambda *args: None),
        )
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
        with patch("ray.get"):
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

    def test_prepared_manifest_is_bound_to_data_tokenizer_template_and_prompt_cap(self):
        manifest = {
            "ordered_parquet_sha256": digest_file(self.root / "train.parquet"),
            "tokenizer_signature": tokenizer_signature(self.tokenizer),
            "prompt_cap": 4,
            "apply_chat_template_kwargs": {},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        with open_dict(self.config):
            self.config.trainer.native_data_manifest = str(path)
        self.prepare()
        (self.root / "train.parquet").write_bytes(b"changed data")
        with self.assertRaisesRegex(ValueError, "prepared prompt pool"):
            self.prepare()

    def test_unverified_native_unbiased_tail_cannot_silently_run_without_behavior_scores(self):
        self.config.actor_rollout_ref.rollout.policy = {"kind": "single", "model": "current_student"}
        self.config.distillation.fsdp_teacher.unbiased_topk_tail = True
        with self.assertRaisesRegex(ValueError, "behavior scores"):
            self.prepare()

    def test_fusion_allocates_one_paired_budget_and_leaves_plain_eval_engine_pure(self):
        for base in ["current_student", "initial_student"]:
            self.config.actor_rollout_ref.rollout.policy = {"kind": "logit_fusion", "base": base, "teacher_weight": 0.5}
            self.prepare()
            self.assertIsNone(self.config.actor_rollout_ref.rollout.generation_fusion)
            self.assertEqual(self.config.actor_rollout_ref.rollout.gpu_memory_utilization, 0.2)
            self.assertEqual(set(self.config.trainer.native_required_models), {base, "teacher"})

    def test_current_fusion_retargets_actor_stream_for_training_and_pure_student_eval(self):
        self.config.actor_rollout_ref.rollout.policy = {
            "kind": "logit_fusion",
            "base": "current_student",
            "teacher_weight": 0.5,
        }
        self.prepare()
        events = []
        fusion_config = {"teacher_path": "/q", "teacher_fingerprint": "sha", "teacher_weight": 0.5}
        version_state = {"version": None}

        async def status():
            return {"identity": ("current_student", str(self.root / "student"), version_state["version"])}

        main = NS(name="main", native_source_status=NS(remote=status))
        pair = NS(name="pair", native_source_status=NS(remote=status))

        class Pool:
            awake = set()
            eval_slot = "eval:current_student"

            async def sleep_all(self):
                self.awake.clear()

        pool = Pool()
        pool.handles = {"eval:current_student": [main], "fusion:current_student": [pair]}
        pool.fusion_config = fusion_config
        wg = NS(set_native_rollout_target=lambda handles, fusion, purpose: events.append((handles, fusion, purpose)))

        def sync(version):
            events.append(("sync", version))
            version_state["version"] = version

        cp = NS(actor_wg=wg, update_weights=sync)
        co = RolloutCoordinator(self.config, self.tokenizer, pool, cp, None)
        with patch("ray.get"):
            co.update_weights(0)
            co.begin_eval(2)
            co.end_eval(2, resume_training=True)
        self.assertEqual(
            [e for e in events if isinstance(e[0], list)],
            [([pair], fusion_config, "training"), ([main], None, "evaluation"), ([pair], fusion_config, "training")],
        )
        self.assertIn("fusion:current_student", pool.awake)


if __name__ == "__main__":
    unittest.main()
