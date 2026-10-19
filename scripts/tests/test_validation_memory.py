"""Capacity arithmetic and header-only checkpoint inspection, without CUDA."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validation_memory", ROOT / "eval/estimate_validation_memory.py"
)
MEMORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY)


class ValidationMemoryTests(unittest.TestCase):
    def config(self):
        return {
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "use_sliding_window": False,
        }

    def test_qwen_kv_and_four_to_eight_replicas_do_not_shard_model_weights(self):
        four = MEMORY.estimate(self.config(), 1777088000, gpus=4)
        eight = MEMORY.estimate(self.config(), 1777088000, gpus=8)
        self.assertEqual(four["kv_bytes_per_sequence_position"], 28672)
        self.assertEqual(four["inference_weights_bytes_per_replica"], 3554176000)
        self.assertEqual(
            eight["inference_weights_bytes_per_replica"],
            four["inference_weights_bytes_per_replica"],
        )
        self.assertEqual(four["generation"]["max_queued_responses_per_gpu"], 128)
        self.assertEqual(eight["generation"]["max_queued_responses_per_gpu"], 64)
        self.assertEqual(four["generation"]["responses_per_submission"], 512)
        self.assertEqual(eight["generation"]["responses_per_submission"], 512)
        self.assertEqual(
            four["generation"]["kv_bytes_if_all_active_reach_cap"], 56 * MEMORY.GIB
        )
        self.assertFalse(four["generation"]["all_active_fit_at_cap_under_assumptions"])

    def test_kl_logits_use_padded_head_but_workspace_uses_effective_support(self):
        result = MEMORY.estimate(
            self.config(), 1777088000, position_chunk=128, vocabulary_chunk=8192
        )
        self.assertEqual(
            result["kl"]["two_bf16_or_fp32_logit_blocks_bytes"], 2 * 128 * 151936 * 2
        )
        self.assertEqual(
            result["kl"]["fp32_reduction_workspace_upper_bytes"], 32 * 128 * 8192
        )
        self.assertEqual(result["kl"]["student_head_snapshot_bytes"], 466747392)
        whole = MEMORY.estimate(
            self.config(), 1777088000, position_chunk=1024, vocabulary_chunk=200000
        )
        self.assertEqual(whole["kl"]["vocabulary_chunk"], 151665)
        batched = MEMORY.estimate(self.config(), 1777088000, forward_batch_size=4)
        self.assertEqual(
            batched["kl"]["two_selected_hidden_upper_bytes"],
            4 * whole["kl"]["two_selected_hidden_upper_bytes"],
        )
        self.assertGreater(
            whole["kl"]["listed_components_bytes"],
            result["kl"]["listed_components_bytes"],
        )

    def test_invalid_dimensions_and_unmodeled_attention_are_rejected(self):
        for kwargs in (
            {"gpus": 0},
            {"gpus": True},
            {"memory_budget": 1},
            {"memory_budget": float("nan")},
            {"non_kv_margin_gib": -1},
            {"mean_context": 40000},
            {"effective_vocabulary": 200000},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MEMORY.estimate(self.config(), 1777088000, **kwargs)
        config = self.config()
        config["use_sliding_window"] = True
        with self.assertRaises(ValueError):
            MEMORY.estimate(config, 1777088000)

    def test_checkpoint_headers_count_storage_without_loading_tensor_data(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            header = json.dumps(
                {"weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
            ).encode()
            data = struct.pack("<Q", len(header)) + header + bytes(12)
            (folder / "model.safetensors").write_bytes(data)
            result = MEMORY.checkpoint_size(folder)
            self.assertEqual(
                result,
                {"parameters": 6, "checkpoint_stored_bytes": 12, "tensor_count": 1},
            )
            (folder / "model-duplicate.safetensors").write_bytes(data)
            with self.assertRaisesRegex(ValueError, "Repeated"):
                MEMORY.checkpoint_size(folder)
            (folder / "model-duplicate.safetensors").unlink()
            (folder / "model.safetensors").write_bytes(struct.pack("<Q", 10000) + b"{}")
            with self.assertRaisesRegex(ValueError, "length"):
                MEMORY.checkpoint_size(folder)


if __name__ == "__main__":
    unittest.main()
