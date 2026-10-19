"""Balanced input selection, exact native trajectory assembly and scorer loading."""

import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from verl.trainer.ppo.fixed_rollout import digest_file
from verl.trainer.ppo.prefix_bank import assemble
from opd_state.integrations.verl.validation import load_fixed_bank
from opd_state.rollout.executor import Request, Segment, Trajectory
from verl.workers.rollout.vllm_rollout.native_server import NativePolicyServerMixin

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prefix_bank_prepare", ROOT / "data/prep/prepare_prefix_bank.py")
PREPARE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PREPARE)
sys.path.insert(0, str(ROOT / "eval"))
from generate_prefix_bank import engine_configs  # noqa: E402


class Tokenizer:
    eos_token_id = pad_token_id = 2
    bos_token_id = 1
    chat_template = "fixed"

    def __len__(self):
        return 256

    def get_vocab(self):
        return {str(i): i for i in range(256)}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
        return [1] + [ord(value) for value in messages[0]["content"]]


class PrefixBankTests(unittest.TestCase):
    def test_both_engine_configs_activate_real_native_admission_with_the_bound_frozen_role(self):
        for role in ("initial_student", "teacher"):
            descriptor = {"path": str((ROOT / role).resolve()), "fingerprint": "frozen-test-identity"}
            cfg, model = engine_configs(
                OmegaConf.create({"policy": None, "enforce_eager": True}),
                OmegaConf.create({"path": "/unused"}),
                role,
                descriptor,
            )

            async def collective_rpc(method, kwargs, expected_role=role, expected_descriptor=descriptor):
                self.assertEqual(method, "configure_native_source")
                self.assertEqual(kwargs["role"], expected_role)
                return [
                    {
                        "path": expected_descriptor["path"],
                        "role": expected_role,
                        "fusion": None,
                        "worker_execution": {"device_uuid": "CPU-fixture-only"},
                    }
                ]

            server = NativePolicyServerMixin()
            server.config, server.model_config = cfg, model
            server.engine = NS(collective_rpc=collective_rpc)
            asyncio.run(server.initialize_native_source())
            self.assertEqual(server.native_state.identity, (role, descriptor["path"], descriptor["fingerprint"]))

    def test_vocabulary_size_is_not_requeried_for_each_input_token(self):
        class CountingTokenizer(Tokenizer):
            def __init__(self):
                self.length_queries = 0

            def __len__(self):
                self.length_queries += 1
                return super().__len__()

        tokenizer = CountingTokenizer()
        PREPARE.select_questions(self.rows(), tokenizer, per_benchmark=2, prompt_cap=64, seed=42, template_kwargs={})
        self.assertEqual(tokenizer.length_queries, 1)

    def rows(self):
        return [
            {
                "data_source": source,
                "prompt": [{"role": "user", "content": f"{source} Q{i}"}],
                "reward_model": {"ground_truth": str(i)},
            }
            for source in PREPARE.BENCHMARKS
            for i in range(5)
        ]

    def fixture(self, base):
        files = []
        for source in PREPARE.BENCHMARKS:
            path = base / f"{source}.parquet"
            pq.write_table(pa.Table.from_pylist([r for r in self.rows() if r["data_source"] == source]), path)
            files.append(path)
        for role in ("p", "q"):
            directory = base / role
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
            (directory / "model.safetensors").write_bytes(f"fixture identity only: {role}".encode())
        args = NS(
            input=files,
            student=base / "p",
            teacher=base / "q",
            output=base / "input.json",
            per_benchmark=2,
            prompt_cap=64,
            response_cap=16,
            positions=4,
            seed=42,
            template_kwargs="{}",
        )
        with patch("verl.utils.hf_tokenizer", return_value=Tokenizer()):
            prepared = PREPARE.prepare(args)
        return args, prepared

    def trajectories(self, prepared):
        result = {}
        for role, model in prepared["models"].items():
            identity = (role, model["path"], model["fingerprint"])
            tokens = (9, 2) if role == "initial_student" else (7, 7, 7, 7, 7, 7, 2)
            result[role] = [
                Trajectory(
                    Request(row["uid"], tuple(row["prompt_ids"])),
                    tokens,
                    (Segment((identity,), 0, len(tokens), "stop", 2),),
                    0,
                    "eos",
                )
                for row in prepared["questions"]
            ]
        return result

    def test_identity_based_balanced_selection_is_stable_under_input_order_changes(self):
        kwargs = {"per_benchmark": 2, "prompt_cap": 64, "seed": 42, "template_kwargs": {}}
        first, excluded = PREPARE.select_questions(self.rows(), Tokenizer(), **kwargs)
        second, _ = PREPARE.select_questions(list(reversed(self.rows())), Tokenizer(), **kwargs)
        self.assertEqual([r["uid"] for r in first], [r["uid"] for r in second])
        self.assertEqual(len(first), 6)
        self.assertEqual(set(excluded.values()), {0})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PREPARE.select_questions(self.rows() + self.rows()[:1], Tokenizer(), **kwargs)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            PREPARE.select_questions(self.rows()[:5], Tokenizer(), **kwargs)

    def test_real_parquet_preparation_and_both_frozen_sources_load_in_the_actual_scorer(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            args, prepared = self.fixture(base)
            trajectories = self.trajectories(prepared)
            for rows in trajectories.values():
                rows.reverse()
            bank = assemble(
                prepared,
                trajectories,
                input_sha256=digest_file(args.output),
                generation_provenance={"scope": "CPU fixture; no generation performed"},
            )
            self.assertEqual(len(bank["records"]), 12)
            self.assertEqual(sum(len(row["positions"]) for row in bank["records"]), 36)
            for record in bank["records"]:
                self.assertEqual(sum(record["weights"]), len(record["response_ids"]))
                self.assertEqual(record["positions"][0], 0)
                self.assertEqual(record["response_ids"][-1], Tokenizer.eos_token_id)
            path = base / "bank.json"
            path.write_text(json.dumps(bank))
            config = OmegaConf.create(
                {
                    "actor_rollout_ref": {"model": {"path": str(args.student)}},
                    "distillation": {"fsdp_teacher": {"model_path": str(args.teacher)}},
                }
            )
            records, sha = load_fixed_bank(path, config, Tokenizer())
            self.assertEqual(len(records), 12)
            self.assertEqual(sha, digest_file(path))

    def test_wrong_frozen_identity_raw_prefix_and_missing_or_duplicate_questions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, prepared = self.fixture(Path(temporary))
            original = self.trajectories(prepared)
            variants = []
            missing = copy.deepcopy(original)
            missing["teacher"].pop()
            variants.append(missing)
            duplicate = copy.deepcopy(original)
            duplicate["teacher"].append(duplicate["teacher"][0])
            variants.append(duplicate)
            wrong = copy.deepcopy(original)
            row = wrong["teacher"][0]
            segment = replace(row.segments[0], identities=(("teacher", "/wrong", "wrong"),))
            wrong["teacher"][0] = replace(row, segments=(segment,))
            variants.append(wrong)
            wrong = copy.deepcopy(original)
            row = wrong["teacher"][0]
            wrong["teacher"][0] = replace(row, request=replace(row.request, prompt_ids=(1, 3)))
            variants.append(wrong)
            for values in variants:
                with self.assertRaises(ValueError):
                    assemble(prepared, values, input_sha256=digest_file(args.output), generation_provenance={})

    def test_early_eos_and_invalid_raw_finish_reason_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, prepared = self.fixture(Path(temporary))
            for tokens, reason in [((2, 7, 2), "stop"), ((7, 2), "abort"), ((7, 2), "length")]:
                values = self.trajectories(prepared)
                row = values["teacher"][0]
                segment = replace(row.segments[0], end=len(tokens), finish_reason=reason)
                values["teacher"][0] = replace(row, token_ids=tokens, segments=(segment,))
                with self.subTest(tokens=tokens, reason=reason), self.assertRaises(ValueError):
                    assemble(prepared, values, input_sha256=digest_file(args.output), generation_provenance={})


if __name__ == "__main__":
    unittest.main()
