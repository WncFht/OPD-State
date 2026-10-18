"""Check question normalization, consumed exposure and explicit macro reductions."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
spec = importlib.util.spec_from_file_location(
    "native_train_overlap", ROOT / "eval/audit_native_train_overlap.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class NativeTrainOverlapTests(unittest.TestCase):
    def test_whitespace_normalization_preserves_case_and_mathematical_content(self):
        self.assertEqual(
            audit.question_key([{"role": "user", "content": "Find  x\n + y"}]),
            audit.question_key([{"role": "user", "content": " Find x + y "}]),
        )
        self.assertNotEqual(
            audit.question_key([{"role": "user", "content": "x + y"}]),
            audit.question_key([{"role": "user", "content": "X + y"}]),
        )

    def test_macro_remains_equal_weight_after_excluding_a_question(self):
        rows = []
        for name in audit.BENCHMARKS:
            for i, value in enumerate([0.25, 0.75]):
                rows.append(
                    {
                        "data_source": name,
                        "question_sha256": f"{name}-{i}",
                        **{
                            field: value
                            for field in [
                                "mean@16",
                                "pass@1",
                                "pass@2",
                                "pass@4",
                                "pass@8",
                                "pass@16",
                            ]
                        },
                    }
                )
        result = audit.subset_metrics(rows, {"MATH-500-0"})
        self.assertAlmostEqual(result["macro"]["mean@16"], (0.75 + 0.5 + 0.5) / 3)
        with self.assertRaisesRegex(ValueError, "emptied"):
            audit.subset_metrics(rows, {"AMC23-0", "AMC23-1"})

    def test_definition_is_bound_to_dataset_and_tracks_all_matching_training_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [
                {
                    "data_source": name,
                    "question_sha256": f"{name}-{i}",
                    **{
                        field: 0.5
                        for field in [
                            "mean@16",
                            "pass@1",
                            "pass@2",
                            "pass@4",
                            "pass@8",
                            "pass@16",
                        ]
                    },
                }
                for name, n in zip(audit.BENCHMARKS, [500, 40, 30], strict=True)
                for i in range(n)
            ]
            definition = {
                "prepared_manifest_sha256": "fixture-sha",
                "evaluation_question_ids": [r["question_sha256"] for r in rows],
                "overlaps": [
                    {
                        "benchmark": "MATH-500",
                        "evaluation_row_index": 0,
                        "question_sha256": "MATH-500-0",
                        "training_row_ids": ["a", "b"],
                    },
                    {
                        "benchmark": "AMC23",
                        "evaluation_row_index": 0,
                        "question_sha256": "AMC23-0",
                        "training_row_ids": ["c"],
                    },
                ],
            }
            definition_path = directory / "definition.json"
            definition_path.write_text(json.dumps(definition))
            manifest = directory / "manifest.json"
            manifest.write_text(
                json.dumps({"prepared_data_manifest_sha256": "fixture-sha"})
            )
            questions = directory / "questions.csv"
            with questions.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = audit.apply_definition(
                definition_path, manifest, {"b"}, questions, directory / "result.json"
            )
            self.assertEqual(result["pool_overlap_questions"], 2)
            self.assertEqual(result["consumed_overlap_questions"], 1)
            self.assertEqual(
                result["views"]["exclude_consumed"]["MATH-500"]["questions"], 499
            )
            self.assertEqual(
                result["views"]["exclude_training_pool"]["AMC23"]["questions"], 39
            )
            manifest.write_text(
                json.dumps({"prepared_data_manifest_sha256": "different"})
            )
            with self.assertRaisesRegex(ValueError, "different prepared"):
                audit.apply_definition(
                    definition_path, manifest, {"b"}, questions, directory / "bad.json"
                )


if __name__ == "__main__":
    unittest.main()
