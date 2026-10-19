"""Actual trainer/teacher scoring hooks, with CPU models and simulated DP RPCs."""

import copy
import hashlib
import json
import runpy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import ray
import test_validation_metrics_runtime as fixtures
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
from verl.trainer.ppo.validation_kl_runner import load_fixed_bank, run_validation_kl, summarize_kl, validate_settings
from verl.workers.fsdp_teacher_worker import TeacherWorker


class Tokenizer:
    eos_token_id = pad_token_id = 2

    def __len__(self):
        return 19

    def decode(self, ids, **kwargs):
        return str(ids.tolist())


class ScoringRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        config = Qwen2Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        self.student, self.teacher = Qwen2ForCausalLM(config), Qwen2ForCausalLM(config)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.worker = object.__new__(TeacherWorker)
        self.worker.teacher_modules = {"default": self.teacher}
        self.worker.teacher_config = NS(dtype="float32", ulysses_sequence_parallel_size=1)
        student = self.student

        class Engine:
            module = student
            engine_config = NS(strategy="fsdp", ulysses_sequence_parallel_size=1)
            model_config = NS(use_fused_kernels=False)

            @contextmanager
            def eval_mode(self):
                yield

        self.worker.fused_worker_dict = {"actor_rollout": NS(actor=NS(engine=Engine()))}

    def trainer(self):
        trainer, events, dumped = fixtures.ValidationTrainerTests().fixture()
        trainer.tokenizer = Tokenizer()
        trainer.config.trainer.validation_data_dir = str(self.directory)
        trainer.config.trainer.validation_kl = {
            "enabled": True,
            "position_chunk_size": 3,
            "vocab_chunk_size": 5,
            "fixed_prefix_bank": None,
        }
        trainer.rollout_coordinator.sleep_replicas = lambda: events.append(("sleep",))
        worker = self.worker

        class Group:
            world_size = 4

            def compute_validation_kl(self, data):
                events.append(("score", len(data)))
                return DataProto.concat([worker.compute_validation_kl(part) for part in data.chunk(4)])

        trainer.opd_teacher_wg = Group()
        return trainer, events, dumped

    def test_actual_validate_reuses_every_response_and_keeps_both_dp_paddings_out_of_results(self):
        trainer, events, dumped = self.trainer()
        original_generate = trainer.async_rollout_manager.generate_sequences
        generated = []

        def generate(batch):
            generated.append(len(batch))
            return original_generate(batch)

        trainer.async_rollout_manager.generate_sequences = generate
        reward = fixtures.ValidationTrainerTests().reward
        before = copy.deepcopy(self.teacher.state_dict())
        with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=reward):
            metrics = trainer._validate()
        self.assertEqual(generated, [4, 4])
        self.assertEqual(events, [("begin", 50), ("sleep",), ("score", 8), ("end", 50)])
        prefix = "validation_kl/on_validation_states/micro/"
        self.assertEqual(metrics[prefix + "responses"], 6)
        self.assertEqual(metrics[prefix + "response_tokens"], 20)
        self.assertEqual(metrics[prefix + "evaluated_positions"], 20)
        self.assertEqual(metrics["original_accuracy_metric"], 0.5)
        self.assertEqual(len(dumped[0]["outputs"]), 6)
        report = json.loads((self.directory / "kl/kl-step-50.json").read_text())
        tokens = self.directory / "kl/kl-tokens-step-50.jsonl"
        self.assertEqual(report["tokens_sha256"], hashlib.sha256(tokens.read_bytes()).hexdigest())
        self.assertEqual(len(tokens.read_text().splitlines()), 6)
        self.assertFalse(report["strict_raw_sequence_kl"])
        self.assertEqual(len(report["worker_timings"]["seconds"]), 4)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in self.teacher.state_dict().items()))

    def test_worker_requires_one_live_actor_and_one_teacher_and_returns_real_statistics(self):
        trainer, _, _ = self.trainer()
        rows = [
            {
                "state_source": "on_validation_states",
                "source": "MATH-500",
                "uid": "a",
                "sample": 0,
                "prompt_ids": [1, 3],
                "response_ids": [6, 2],
            }
        ]
        scored, _ = run_validation_kl(trainer, rows)
        self.assertEqual(scored[0]["evaluated_positions"], 2)
        self.worker.fused_worker_dict["actor_rollout_ref"] = self.worker.fused_worker_dict["actor_rollout"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            run_validation_kl(trainer, rows)

    def test_periodic_validation_uses_each_live_version_and_resumes_until_final_step(self):
        trainer, events, dumped = self.trainer()
        trainer.total_training_steps = 4
        resumptions = []
        original_end = trainer.rollout_coordinator.end_eval

        def end_eval(version, *, resume_training):
            resumptions.append((version, resume_training))
            return original_end(version, resume_training=resume_training)

        trainer.rollout_coordinator.end_eval = end_eval
        bank = [
            {
                "state_source": source,
                "source": "MATH-500",
                "uid": "fixed",
                "sample": 0,
                "prompt_ids": [1, 3],
                "response_ids": [6, 6, 2],
                "positions": [0, 2],
                "weights": [1, 2],
            }
            for source in ("on_initial_states", "on_teacher_states")
        ]
        trainer._validation_kl_bank = (bank, "same-bank-across-validation-steps")
        frozen_bank = copy.deepcopy(bank)
        frozen_teacher = copy.deepcopy(self.teacher.state_dict())
        optimizer = torch.optim.SGD(self.student.parameters(), lr=0.1)
        ids = torch.tensor([[1, 3, 6, 2]])
        reward = fixtures.ValidationTrainerTests().reward
        observed = []
        for step in (0, 2, 4):
            if step:
                # Real learner updates between validations catch cached heads or
                # a worker accidentally reading the initial student's weights.
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    self.student(input_ids=ids, labels=ids, use_cache=False).loss.backward()
                    optimizer.step()
            trainer.global_steps = step
            before = copy.deepcopy(self.student.state_dict())
            rng = torch.random.get_rng_state().clone()
            with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=reward):
                metrics = trainer._validate()
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.student.state_dict().items()))
            self.assertTrue(all(torch.equal(v, frozen_teacher[k]) for k, v in self.teacher.state_dict().items()))
            report = json.loads((self.directory / f"kl/kl-step-{step}.json").read_text())
            self.assertEqual(report["student_version"], step)
            self.assertEqual(report["fixed_bank_sha256"], "same-bank-across-validation-steps")
            tokens = self.directory / f"kl/kl-tokens-step-{step}.jsonl"
            self.assertEqual(report["tokens_sha256"], hashlib.sha256(tokens.read_bytes()).hexdigest())
            columns = dumped[-1]["reward_extra_infos_dict"]
            self.assertEqual(columns["rollout_student_version"], [step] * 6)
            self.assertEqual(metrics["validation_kl/on_validation_states/micro/evaluated_positions"], 20)
            for source in ("on_initial_states", "on_teacher_states"):
                self.assertEqual(metrics[f"validation_kl/{source}/micro/evaluated_positions"], 2)
            observed.append(metrics["validation_kl/on_validation_states/micro/student_teacher_kl/token_weighted"])
        self.assertEqual(resumptions, [(0, True), (2, True), (4, False)])
        self.assertEqual(
            events,
            [event for step in (0, 2, 4) for event in [("begin", step), ("sleep",), ("score", 8), ("end", step)]],
        )
        self.assertEqual(len(list((self.directory / "kl").glob("kl-step-*.json"))), 3)
        self.assertEqual(bank, frozen_bank)
        self.assertNotAlmostEqual(observed[0], observed[1], places=6)
        self.assertNotAlmostEqual(observed[1], observed[2], places=6)

    def test_shared_fixed_bank_runs_alongside_full_validation_without_changing_main_coverage(self):
        trainer, _, _ = self.trainer()
        bank = [
            {
                "state_source": source,
                "source": "MATH-500",
                "uid": "fixed",
                "sample": 0,
                "prompt_ids": [1, 3],
                "response_ids": [6, 6, 2],
                "positions": [0, 2],
                "weights": [1, 2],
            }
            for source in ("on_initial_states", "on_teacher_states")
        ]
        trainer._validation_kl_bank = (bank, "bound-bank-sha")
        reward = fixtures.ValidationTrainerTests().reward
        with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=reward):
            metrics = trainer._validate()
        self.assertEqual(metrics["validation_kl/on_validation_states/micro/evaluated_positions"], 20)
        for source in ("on_initial_states", "on_teacher_states"):
            self.assertEqual(metrics[f"validation_kl/{source}/micro/responses"], 1)
            self.assertEqual(metrics[f"validation_kl/{source}/micro/evaluated_positions"], 2)
        report = json.loads((self.directory / "kl/kl-step-50.json").read_text())
        self.assertEqual(report["fixed_bank_sha256"], "bound-bank-sha")
        self.assertLess(report["scoring_started_epoch_s"], report["scoring_finished_epoch_s"])

    def test_merged_validation_keeps_kl_statistics_and_performance(self):
        trainer, _, _ = self.trainer()
        reward = fixtures.ValidationTrainerTests().reward
        with patch("verl.trainer.ppo.ray_trainer.extract_reward", side_effect=reward):
            raw = trainer._validate(merged=True)
        metrics = trainer._merge_validation_results(raw, None)
        self.assertEqual(metrics["validation_kl/on_validation_states/micro/evaluated_positions"], 20)
        self.assertGreater(metrics["validation_kl/performance/scoring_seconds"], 0)
        self.assertIn("validation_kl/performance/rollout_sleep_seconds", metrics)

    def test_new_kl_artifacts_do_not_break_the_existing_accuracy_collector(self):
        root = Path(__file__).resolve().parents[3]
        support = runpy.run_path(str(root / "scripts/tests/test_math_snapshots.py"))
        run, log = support["fixture"](self.directory)
        with log.open("a") as stream:
            stream.write("Final validation metrics:\n")
        trainer, _, _ = self.trainer()
        trainer.config.trainer.validation_data_dir = str(run / "validation")
        records = [
            {
                "state_source": "on_validation_states",
                "source": "MATH-500",
                "uid": "a",
                "sample": 0,
                "prompt_ids": [1, 3],
                "response_ids": [6, 2],
            }
        ]
        run_validation_kl(trainer, records)
        summary = support["COLLECTOR"]["collect"](
            run, self.directory / "collected", steps=2, interval=2, require_full=False
        )
        self.assertEqual(summary["validation_steps"], [2])

    def test_changed_rpc_identity_or_skipped_tokens_cannot_be_reported_as_complete(self):
        trainer, _, _ = self.trainer()
        rows = [
            {
                "state_source": "on_validation_states",
                "source": "MATH-500",
                "uid": "a",
                "sample": 0,
                "prompt_ids": [1, 3],
                "response_ids": [6, 2],
            }
        ]
        original = trainer.opd_teacher_wg.compute_validation_kl
        for kind in ("hash", "positions", "version"):

            def tamper(data, kind=kind):
                output = original(data)
                if kind == "hash":
                    output.non_tensor_batch["validation_kl_stats"][0]["record_sha256"] = "wrong"
                elif kind == "positions":
                    output.non_tensor_batch["validation_kl_stats"][0]["evaluated_positions"] = 1
                else:
                    output.meta_info["student_version"] = 49
                return output

            trainer.opd_teacher_wg.compute_validation_kl = tamper
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run_validation_kl(trainer, rows)
        self.assertFalse(list(self.directory.iterdir()))

    def test_token_question_macro_and_cumulative_reductions_do_not_collapse_together(self):
        def row(source, uid, sample, length, total):
            return {
                "state_source": "on_validation_states",
                "source": source,
                "uid": uid,
                "sample": sample,
                "response_length": length,
                "evaluated_positions": length,
                "sums": {
                    key: total
                    for key in ("student_teacher_kl", "teacher_student_kl", "student_entropy", "teacher_entropy")
                },
            }

        rows = [
            row("MATH-500", "a", 0, 1, 1),
            row("MATH-500", "a", 1, 9, 9),
            row("MATH-500", "b", 0, 1, 3),
            row("AIME25", "c", 0, 2, 8),
        ]
        metrics = summarize_kl(rows)
        prefix = "validation_kl/on_validation_states/"
        self.assertAlmostEqual(metrics[prefix + "MATH-500/student_teacher_kl/token_weighted"], 13 / 11)
        self.assertEqual(metrics[prefix + "MATH-500/student_teacher_kl/question_mean"], 2)
        self.assertEqual(metrics[prefix + "MATH-500/student_teacher_kl/cumulative_on_states"], 4)
        self.assertEqual(metrics[prefix + "macro/student_teacher_kl/question_mean"], 3)
        self.assertAlmostEqual(metrics[prefix + "micro/student_teacher_kl/question_mean"], 8 / 3)
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_kl(rows + rows[:1])

    def test_fixed_bank_binds_models_tokenizer_and_both_state_sources(self):
        trainer, _, _ = self.trainer()
        trainer.config.actor_rollout_ref.model = {"path": "p"}
        trainer.config.distillation = {"fsdp_teacher": {"model_path": "q"}}
        record = {
            "source": "MATH-500",
            "uid": "a",
            "sample": 0,
            "prompt_ids": [1, 3],
            "response_ids": [6, 6, 2],
            "positions": [0, 2],
            "weights": [1, 2],
        }
        bank = {
            "schema": "fixed-prefix-bank-v1",
            "vocab_size": 19,
            "tokenizer_signature": "tok",
            "initial_student_fingerprint": "p0",
            "teacher_fingerprint": "teacher",
            "records": [{**record, "state_source": source} for source in ("on_initial_states", "on_teacher_states")],
        }
        path = self.directory / "bank.json"
        path.write_text(json.dumps(bank))
        with (
            patch("verl.trainer.ppo.fixed_rollout.tokenizer_signature", return_value="tok"),
            patch(
                "verl.trainer.ppo.fixed_rollout.model_signature", side_effect=lambda p: {"p": "p0", "q": "teacher"}[p]
            ),
        ):
            rows, digest = load_fixed_bank(path, trainer.config, trainer.tokenizer)
            self.assertEqual(len(rows), 2)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            bank["teacher_fingerprint"] = "wrong"
            path.write_text(json.dumps(bank))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_fixed_bank(path, trainer.config, trainer.tokenizer)

    def test_config_checks_reject_unsupported_or_unrecorded_scoring_before_workers_start(self):
        trainer, _, _ = self.trainer()
        trainer.config.distillation = {"fsdp_teacher": {"enabled": True, "teachers": []}}
        trainer.config.actor_rollout_ref.model = {"use_fused_kernels": False}
        validate_settings(trainer.config, native_enabled=True)
        with self.assertRaisesRegex(ValueError, "native"):
            validate_settings(trainer.config, native_enabled=False)
        trainer.config.trainer.validation_kl.position_chunk_size = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_settings(trainer.config, native_enabled=True)


class ColocatedPeerTests(unittest.TestCase):
    def test_original_colocation_factory_exposes_the_same_peer_objects_to_inner_roles(self):
        @ray.remote
        class Actor(Worker):
            def __init__(self):
                self.marker = "live actor"

        @ray.remote
        class Teacher(Worker):
            def __init__(self):
                self.marker = "frozen teacher"

        args = {"actor_rollout": RayClassWithInitArgs(cls=Actor), "opd_teacher": RayClassWithInitArgs(cls=Teacher)}
        with patch.object(Worker, "__init__", lambda self: None):
            remote = create_colocated_worker_cls(args)
            raw = remote.cls.__ray_actor_class__()
        actor, teacher = raw.worker_dict["actor_rollout"], raw.worker_dict["opd_teacher"]
        self.assertIs(teacher.get_fused_worker_by_name("actor_rollout"), actor)
        self.assertIs(actor.get_fused_worker_by_name("opd_teacher"), teacher)
        self.assertIsNone(teacher.get_fused_worker_by_name("missing"))


if __name__ == "__main__":
    unittest.main()
