"""Check complete-pool deduplication and real Parquet/Datasets round trips."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fresh_data", ROOT / "data/prep/build_fresh_math_data.py")
DATA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DATA)


class Tokenizer:
    chat_template = "simple"
    eos_token_id = pad_token_id = 250
    bos_token_id = 251

    def apply_chat_template(self, messages, **kwargs):
        return [ord(x) for x in messages[0]["content"].strip()]

    def get_vocab(self):
        return {str(i): i for i in range(256)}


def row(text, answer):
    return {
        "prompt": [{"role": "user", "content": text}],
        "reward_model": {"ground_truth": answer},
        "data_source": "DAPO",
    }


class FreshDataTests(unittest.TestCase):
    def test_duplicate_tokens_conflicting_answers_and_length_filter_are_separate(self):
        rows = [row("a", "1"), row(" a ", "1"), row("b", "2"), row("b", "3"), row("x" * 20, "4")]
        indices, prompts, counts = DATA.select_prompts(rows, Tokenizer(), 10, {})
        self.assertEqual(indices, [0])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(counts["original_rows"], 5)
        self.assertEqual(counts["duplicates_removed"], 2)
        self.assertEqual(counts["conflicting_prompts_removed"], 1)
        self.assertEqual(counts["excluded_overlong"], 1)
        with self.assertRaisesRegex(ValueError, "Conflicting answers"):
            DATA.select_prompts(rows, Tokenizer(), 10, {}, "error")

    def test_prepared_pool_keeps_all_unique_rows_and_new_columns_survive_hf_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "raw.parquet"
            pq.write_table(pa.Table.from_pylist([row("a", "1"), row("a", "1"), row("b", "2")]), source)
            args = NS(
                input=[source],
                student=base / "tokenizer",
                prompt_cap=10,
                template_kwargs="{}",
                output=base / "prepared",
                conflict_policy="exclude",
            )
            with patch("verl.utils.hf_tokenizer", return_value=Tokenizer()):
                manifest = DATA.build(args)
            self.assertEqual(manifest["unique_prompts"], 2)
            result = load_dataset(
                "parquet", data_files=str(args.output / "ordered.parquet"), split="train", cache_dir=str(base / "cache")
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(len(set(result["opd_prompt_id"])), 2)
            self.assertEqual(result["opd_prompt_length"], [1, 1])
            with self.assertRaises(FileExistsError):
                DATA.build(args)


if __name__ == "__main__":
    unittest.main()
