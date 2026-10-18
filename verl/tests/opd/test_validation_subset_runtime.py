"""Real Parquet round trips for the short engineering validation protocol."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("validation_subset_data", ROOT / "data/prep/build_validation_subset.py")
DATA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DATA)
spec = importlib.util.spec_from_file_location("validation_subset_verify", ROOT / "eval/native_validation_subset.py")
VERIFY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VERIFY)


class ValidationSubsetTests(unittest.TestCase):
    def inputs(self, base):
        paths = []
        for name in DATA.BENCHMARKS:
            path = base / f"{name}.parquet"
            rows = [
                {
                    "data_source": name,
                    "prompt": [{"role": "user", "content": f"Question {i}"}],
                    "reward_model": {"ground_truth": str(i)},
                    "extra_info": {"original_index": i},
                }
                for i in range(5)
            ]
            pq.write_table(pa.Table.from_pylist(rows), path)
            paths.append(path)
        return paths

    def test_six_questions_preserve_nested_prompts_answers_schema_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inputs = self.inputs(base)
            before = {p: p.read_bytes() for p in inputs}
            report = DATA.build(inputs, base / "subset")
            self.assertEqual(report["questions"], 6)
            self.assertEqual(VERIFY.verify(base / "subset/manifest.json"), report)
            for original, selected in zip(inputs, VERIFY.paths(base / "subset"), strict=True):
                source, actual = pq.read_table(original), pq.read_table(selected)
                self.assertEqual(actual.to_pylist(), source.slice(0, 2).to_pylist())
                self.assertEqual(actual.schema, source.schema)
                self.assertEqual(original.read_bytes(), before[original])

    def test_missing_benchmark_duplicates_short_inputs_and_replacement_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inputs = self.inputs(base)
            for values in (inputs[:2], [inputs[0], inputs[0], inputs[2]]):
                with self.assertRaises(ValueError):
                    DATA.build(values, base / "invalid")
            self.assertFalse((base / "invalid").exists())
            with self.assertRaises(ValueError):
                DATA.build(inputs, base / "too-many", per_benchmark=6)
            DATA.build(inputs, base / "subset")
            with self.assertRaises(FileExistsError):
                DATA.build(inputs, base / "subset")

    def test_mutated_source_or_output_fails_before_gpu_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inputs = self.inputs(base)
            DATA.build(inputs, base / "subset")
            original = inputs[0].read_bytes()
            inputs[0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                VERIFY.verify(base / "subset/manifest.json")
            inputs[0].write_bytes(original)
            VERIFY.paths(base / "subset")[0].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                VERIFY.verify(base / "subset/manifest.json")


if __name__ == "__main__":
    unittest.main()
