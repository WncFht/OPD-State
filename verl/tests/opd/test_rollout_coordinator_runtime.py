"""Torch/Hydra bridge checks; run explicitly with the training environment."""

import asyncio
import copy
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
from verl.trainer.ppo.rollout_coordinator import (
    RolloutCoordinator,
    epochs_for_budget,
    native_evaluation_config,
    prepare_native_config,
)
from verl.workers.rollout.native_pool import NativeModelPool

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

    def test_validation_fast_hydra_profile_composes_without_changing_learning_batch(self):
        with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
            baseline = compose(config_name="ppo_trainer")
            profile = compose(config_name="ppo_trainer", overrides=["+experiment=validation_fast"])
        r = profile.actor_rollout_ref.rollout
        self.assertEqual(r.generation_eval_max_num_seqs, 128)
        self.assertEqual(r.generation_eval_memory_utilization, 0.55)
        self.assertFalse(r.generation_eval_enforce_eager)
        self.assertTrue(profile.trainer.validation_kl.enabled)
        self.assertEqual(profile.trainer.validation_metrics_mode, "math_exact")
        self.assertEqual(profile.data.train_batch_size, baseline.data.train_batch_size)
        self.assertEqual(profile.actor_rollout_ref.actor.optim.lr, baseline.actor_rollout_ref.actor.optim.lr)
        self.assertEqual(r.gpu_memory_utilization, baseline.actor_rollout_ref.rollout.gpu_memory_utilization)
        self.assertEqual(r.enforce_eager, baseline.actor_rollout_ref.rollout.enforce_eager)

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

    def test_layout_selection_does_not_change_actor_or_source_memory_budget(self):
        self.config.actor_rollout_ref.rollout.generation_layout = "split"
        self.prepare()
        self.assertEqual(self.config.actor_rollout_ref.rollout.generation_models.teacher.memory_fraction, 0.2)
        self.assertEqual(self.config.actor_rollout_ref.model.path, str(self.root / "student"))
        self.config.actor_rollout_ref.rollout.generation_layout = "automatic"
        with self.assertRaisesRegex(ValueError, "layout"):
            self.prepare()

    def test_eval_concurrency_is_optional_and_repeated_resolution_preserves_training_limit(self):
        r = self.config.actor_rollout_ref.rollout
        r.max_num_seqs = 8
        self.prepare()
        self.assertEqual(r.max_num_seqs, 8)
        self.assertIs(native_evaluation_config(self.config), self.config)
        r.generation_eval_max_num_seqs = 64
        self.prepare()
        evaluation = native_evaluation_config(self.config)
        self.assertEqual(evaluation.actor_rollout_ref.rollout.max_num_seqs, 64)
        self.assertEqual(r.max_num_seqs, 8)
        self.prepare()
        self.assertEqual(r.max_num_seqs, 8)
        self.assertEqual(self.config.data.train_batch_size, 4)
        self.assertEqual(r.val_kwargs.n, 16)
        r.max_num_seqs = 32
        self.prepare()
        self.assertEqual(r.max_num_seqs, 32)
        self.assertEqual(native_evaluation_config(self.config).actor_rollout_ref.rollout.max_num_seqs, 64)
        r.policy = None
        self.assertIs(native_evaluation_config(self.config), self.config)

    def test_invalid_eval_concurrency_fails_before_resource_or_config_mutation(self):
        r = self.config.actor_rollout_ref.rollout
        r.max_num_seqs = 8
        for value in (0, -1, True, 1.5, "64"):
            r.generation_eval_max_num_seqs = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                self.prepare()
            self.assertEqual(r.max_num_seqs, 8)
        r.generation_eval_max_num_seqs = int(r.max_num_batched_tokens) + 1
        with self.assertRaisesRegex(ValueError, "batched-token budget"):
            self.prepare()
        r.generation_eval_max_num_seqs = 64
        r.engine_kwargs = {"vllm": {"max_num_seqs": 128}}
        with self.assertRaisesRegex(ValueError, "engine_kwargs"):
            self.prepare()

    def test_eval_token_and_memory_overrides_leave_training_configuration_unchanged(self):
        r = self.config.actor_rollout_ref.rollout
        r.max_num_seqs, r.max_num_batched_tokens = 8, 512
        r.generation_eval_max_num_seqs = 64
        r.generation_eval_max_num_batched_tokens = 8192
        r.generation_eval_memory_utilization = 0.7
        r.generation_eval_enforce_eager = False
        r.generation_eval_engine_kwargs = {
            "compilation_config": {"custom_ops": ["all"], "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}
        }
        original_eager = r.enforce_eager
        self.prepare()
        evaluation = native_evaluation_config(self.config).actor_rollout_ref.rollout
        self.assertEqual((evaluation.max_num_seqs, evaluation.max_num_batched_tokens), (64, 8192))
        self.assertEqual(evaluation.gpu_memory_utilization, 0.7)
        self.assertFalse(evaluation.enforce_eager)
        self.assertEqual(list(evaluation.cudagraph_capture_sizes), [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(r.enforce_eager, original_eager)
        self.assertNotIn("compilation_config", r.engine_kwargs.get("vllm", {}))
        self.assertEqual((r.max_num_seqs, r.max_num_batched_tokens, r.gpu_memory_utilization), (8, 512, 0.2))
        self.assertEqual(self.config.trainer.native_generation_memory_budget, 0.2)
        self.prepare()
        self.assertEqual(r.gpu_memory_utilization, 0.2)

    def test_eval_token_memory_types_budgets_and_conflicts_fail_before_ray(self):
        r = self.config.actor_rollout_ref.rollout
        r.max_num_seqs = 8
        for key, values in (
            ("generation_eval_max_num_batched_tokens", (0, -1, True, 3.5, "8192", 4)),
            ("generation_eval_memory_utilization", (0, 1, -0.1, True, "0.7", float("nan"))),
            ("generation_eval_enforce_eager", (0, 1, "false")),
            ("generation_eval_engine_kwargs", (True, "bad", {"max_num_seqs": 64}, {"compilation_config": "bad"})),
        ):
            for value in values:
                r[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.prepare()
            r[key] = None
        for field, engine_key, value in (
            ("generation_eval_max_num_batched_tokens", "max_num_batched_tokens", 8192),
            ("generation_eval_memory_utilization", "gpu_memory_utilization", 0.7),
        ):
            r[field] = value
            r.engine_kwargs = {"vllm": {engine_key: value}}
            with self.assertRaisesRegex(ValueError, "engine_kwargs"):
                self.prepare()
            r[field], r.engine_kwargs = None, {}

    def test_actual_pool_creation_separates_eval_and_all_training_policy_limits(self):
        class Replica:
            def __init__(self, replica_rank, config, model_config, gpus_per_node=4, **kwargs):
                self.replica_rank, self.gpus_per_node = replica_rank, gpus_per_node
                self.config, self.model_config = copy.deepcopy(config), model_config
                self.workers, self.awake = [], False

                async def status():
                    return {"awake": self.awake}

                self._server_handle = NS(native_source_status=NS(remote=status))

            async def launch_servers(self):
                self.awake = True

            async def sleep(self):
                self.awake = False

        policies = [
            {"kind": "single", "model": "current_student"},
            {"kind": "single", "model": "teacher"},
            {"kind": "logit_fusion", "base": "initial_student", "teacher_weight": 0.5},
            {
                "kind": "batch_mix",
                "components": [
                    {"policy": {"kind": "single", "model": "teacher"}, "weight": 0.5},
                    {"policy": {"kind": "single", "model": "current_student"}, "weight": 0.5},
                ],
            },
        ]
        for policy in policies:
            r = self.config.actor_rollout_ref.rollout
            r.policy, r.max_num_seqs, r.generation_eval_max_num_seqs = policy, 8, 64
            r.generation_eval_max_num_batched_tokens, r.generation_eval_memory_utilization = 8192, 0.7
            self.prepare()
            evaluation = native_evaluation_config(self.config)
            mains = [
                Replica(i, evaluation.actor_rollout_ref.rollout, self.config.actor_rollout_ref.model) for i in range(4)
            ]
            manager = NS(get_replicas=lambda mains=mains: mains, server_handles=[m._server_handle for m in mains])
            pool = NativeModelPool(self.config, manager)
            with (
                self.subTest(policy=policy),
                patch("verl.workers.rollout.replica.get_rollout_replica_class", return_value=Replica),
            ):
                asyncio.run(pool.initialize())
                self.assertEqual(pool.describe()[pool.eval_slot]["max_num_seqs"], 64)
                self.assertEqual(pool.describe()[pool.eval_slot]["max_num_batched_tokens"], 8192)
                self.assertEqual(pool.describe()[pool.eval_slot]["gpu_memory_utilization"], 0.7)
                for slot, replicas in pool.replicas.items():
                    if slot == pool.eval_slot:
                        continue
                    self.assertEqual(len(replicas), 4)
                    for replica in replicas:
                        self.assertEqual(replica.config.max_num_seqs, 8)
                        self.assertEqual(replica.config.generation_purpose, "training")
                        self.assertEqual(replica.config.max_model_len, 8)
                        self.assertNotEqual(replica.config.gpu_memory_utilization, 0.7)
                self.assertEqual(r.max_num_seqs, 8)
                self.assertEqual(r.gpu_memory_utilization, 0.2)

    def test_named_memory_rpc_reads_worker_peaks_and_resets_only_when_requested(self):
        from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

        cuda = NS(
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 10,
            memory_reserved=lambda: 20,
            max_memory_allocated=lambda: 30,
            max_memory_reserved=lambda: 40,
        )
        with (
            patch("verl.workers.rollout.vllm_rollout.utils.torch.cuda", cuda),
            patch.object(cuda, "reset_peak_memory_stats") as reset,
        ):
            actual = vLLMColocateWorkerExtension.native_cuda_memory(NS())
            self.assertEqual(actual, {"allocated": 10, "reserved": 20, "peak_allocated": 30, "peak_reserved": 40})
            reset.assert_not_called()
            vLLMColocateWorkerExtension.native_cuda_memory(NS(), reset=True)
            reset.assert_called_once()
            with self.assertRaisesRegex(ValueError, "boolean"):
                vLLMColocateWorkerExtension.native_cuda_memory(NS(), reset=1)

    def test_worker_receipt_reads_the_actual_engine_scheduler_configuration(self):
        from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

        worker = NS(
            model_runner=NS(
                use_cuda_graph=True,
                vllm_config=NS(
                    scheduler_config=NS(max_num_seqs=64, max_num_batched_tokens=8192),
                    model_config=NS(max_model_len=32768),
                    cache_config=NS(gpu_memory_utilization=0.7),
                ),
            )
        )
        with (
            patch("verl.workers.rollout.vllm_rollout.native_weights.configure_source", return_value={}),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties", return_value=NS(uuid="GPU-CPU-fixture")),
        ):
            receipt = vLLMColocateWorkerExtension.configure_native_source(worker, "current_student", "/student")
        execution = receipt["worker_execution"]
        self.assertEqual(execution["max_num_seqs"], 64)
        self.assertEqual(execution["max_num_batched_tokens"], 8192)
        self.assertEqual(execution["max_model_len"], 32768)
        self.assertEqual(execution["gpu_memory_utilization"], 0.7)
        self.assertTrue(execution["cuda_graph_enabled"])
        self.assertFalse(torch.cuda.is_initialized())

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
                        "execution": {
                            "replica_rank": 0,
                            "device_uuid": "unit-gpu-0",
                            "pid": 100,
                            "local_device": 0,
                            "visible_devices": "0",
                        },
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
        self.assertEqual(out.non_tensor_batch["native_segments"].tolist(), [(("teacher", 0, 0, 2),)])
        self.assertEqual(out.batch["rm_scores"].sum().item(), 1)
        self.assertIn("raw_prompt", reward.batch.non_tensor_batch)
        self.assertEqual(events[1][1][0], "teacher")
        trace = list((self.root / "native-trajectories").glob("step-*.jsonl"))
        self.assertEqual(len(trace), 1)
        self.assertEqual(json.loads(trace[0].read_text())["segments"][0]["identities"][0][0], "teacher")
        audit = json.loads((trace[0].parent / "batches.jsonl").read_text().splitlines()[0])
        self.assertEqual(
            audit["generation_placement"],
            [{"layout": "overlap", "requests": {"teacher": 1}, "replicas": {"teacher": [0]}}],
        )
        with patch("ray.get"):
            co.begin_eval(2)
        ev = co.generate_sequences(self.batch(16, validate=True, step=2))
        self.assertNotIn("native_segments", ev.non_tensor_batch)
        self.assertEqual(len(ev.non_tensor_batch["native_response_info"]), 16)
        self.assertEqual(ev.non_tensor_batch["native_response_info"][0]["student_version"], 2)
        self.assertEqual(ev.non_tensor_batch["native_response_info"][0]["response_cap"], 6)
        self.assertEqual(ev.non_tensor_batch["native_response_info"][0]["termination"], "eos")
        co.end_eval(2, resume_training=False)
        self.assertEqual(len(ev), 16)
        self.assertIn(("sync", 2), events)
        self.assertEqual([e[1][0] for e in events if e[0] == "generate"], ["teacher", "current_student"])

    def test_update_budget_covers_multiple_epochs_without_a_fifty_step_dataset_limit(self):
        self.assertEqual(epochs_for_budget(301, 50), 7)
        self.assertEqual(epochs_for_budget(25, 50), 1)
        with self.assertRaises(ValueError):
            epochs_for_budget(5, 0)

    def test_same_model_two_requests_keep_source_counts_and_final_student_only_evaluation(self):
        from opd_state.rollout.executor import Completion

        with open_dict(self.config):
            self.config.actor_rollout_ref.rollout.policy = {
                "kind": "sequence_switch",
                "first": "teacher",
                "second": "teacher",
                "boundary": {"kind": "response_tokens", "value": 2},
                "keep_request_boundary": True,
            }
        self.prepare()
        self.assertEqual(self.config.trainer.native_required_models, ["teacher"])
        self.assertEqual(self.config.actor_rollout_ref.rollout.generation_models.teacher.memory_fraction, 0.2)
        calls = []

        class Backend:
            placement_audit, execution_audit, verifications = [], [], []

            def __init__(self, pool, sampling, *, validate):
                self.validate = validate

            async def generate(self, stage, batch, refs):
                calls.append((self.validate, stage.models, refs[0].identity))
                return [
                    Completion(
                        row.request_id,
                        row.prompt_ids,
                        (4,) * row.max_tokens,
                        tuple(ref.identity for ref in refs),
                        "length",
                    )
                    for row in batch
                ]

        class Pool:
            async def sleep_all(self):
                pass

        class Reward:
            def compute_rm_score(self, batch):
                return DataProto.from_dict(
                    tensors={"rm_scores": torch.zeros_like(batch.batch["responses"], dtype=torch.float32)}
                )

        coordinator = RolloutCoordinator(self.config, self.tokenizer, Pool(), None, Reward())
        with patch("verl.trainer.ppo.rollout_coordinator.NativeRayBackend", Backend):
            training = coordinator.generate_sequences(self.batch())
            coordinator.eval_version = 2
            evaluation = coordinator.generate_sequences(self.batch(16, validate=True, step=2))
        metrics = training.meta_info["native_metrics"]
        self.assertEqual(metrics["source/segmented_responses"], 1)
        self.assertEqual(metrics["source/switched_responses"], 0)
        self.assertEqual(metrics["source/teacher/responses"], 1)
        self.assertEqual(metrics["source/teacher/tokens"], 4)
        self.assertEqual(training.batch["response_mask"].tolist(), [[1] * 4])
        self.assertEqual([item[1] for item in calls], [("teacher",), ("teacher",), ("current_student",)])
        self.assertEqual(calls[-1][2][-1], 2)
        self.assertEqual(evaluation.meta_info["native_metrics"]["source/segmented_responses"], 0)

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
