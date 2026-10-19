"""CPU checks for fixed-source integrity and finite-sample evaluation metrics."""

import importlib.util
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    path = ROOT / (
        "src/opd_state/evaluation/math.py"
        if name == "math_eval"
        else f"verl/verl/trainer/ppo/{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


fixed = module("fixed_rollout")
evaluation = module("math_eval")


class EvaluationTests(unittest.TestCase):
    def test_pass_estimator_matches_all_combinations(self):
        for n in [1, 2, 4, 16]:
            for c in range(n + 1):
                for k in range(1, n + 1):
                    expected = 1 - (
                        math.comb(n - c, k) if n - c >= k else 0
                    ) / math.comb(n, k)
                    self.assertAlmostEqual(
                        evaluation.estimate_pass_at_k(n, c, k), expected
                    )
        self.assertEqual(evaluation.estimate_pass_at_k(16, 1, 16), 1.0)
        for args in [(0, 0, 1), (16, 17, 1), (16, 1, 17), (16, -1, 1)]:
            with self.assertRaises(ValueError):
                evaluation.estimate_pass_at_k(*args)

    def test_equal_benchmark_macro_differs_from_question_weighted_micro(self):
        sources = ["math"] * 32 + ["aime"] * 16
        uids = ["q1"] * 16 + ["q2"] * 16 + ["q3"] * 16
        scores = [1] * 16 + [0] * 16 + [0] * 16
        metrics, records = evaluation.exact_math_metrics(
            sources, uids, scores, samples_per_question=16
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(metrics["eval/math/mean@16"], 0.5)
        self.assertEqual(metrics["eval/macro/mean@16"], 0.25)
        self.assertAlmostEqual(metrics["eval/micro/mean@16"], 1 / 3)

    def test_missing_replicates_and_benchmarks_fail(self):
        with self.assertRaisesRegex(ValueError, "Expected 16"):
            evaluation.exact_math_metrics(
                ["math"] * 15, ["q"] * 15, [1] * 15, samples_per_question=16
            )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluation.exact_math_metrics(
                ["math"] * 16,
                ["q"] * 16,
                [1] * 16,
                samples_per_question=16,
                required_sources=["math", "aime"],
            )
        with self.assertRaisesRegex(ValueError, "counts mismatch"):
            evaluation.exact_math_metrics(
                ["math"] * 16,
                ["q"] * 16,
                [1] * 16,
                samples_per_question=16,
                required_counts={"math": 2},
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            evaluation.exact_math_metrics(
                ["math"], ["q"], [0.5], samples_per_question=1
            )


class FixedCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompt = [{"role": "user", "content": "2+2"}]
        conn = sqlite3.connect(self.root / "trajectories.sqlite")
        conn.execute(
            "CREATE TABLE trajectories (prompt_id TEXT PRIMARY KEY, prompt_sha256 TEXT, "
            "prompt_ids TEXT,response_ids TEXT,finish_reason TEXT)"
        )
        conn.execute(
            "INSERT INTO trajectories VALUES (?,?,?,?,?)",
            ("q1", fixed.digest_json(self.prompt), "[1,2]", "[3,9]", "stop"),
        )
        conn.commit()
        conn.close()
        self.manifest = {
            "schema": 1,
            "source": "cached_teacher",
            "records": 1,
            "database_sha256": fixed.digest_file(self.root / "trajectories.sqlite"),
            "vocab_size": 10,
            "prompt_cap": 8,
            "response_cap": 4,
            "eos_token_id": 9,
            "pad_token_id": 0,
            "temperature": 1.0,
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_load_is_read_only_and_stale_prompt_cannot_be_reused(self):
        cache = fixed.FixedRolloutCache(
            self.root, expected={"source": "cached_teacher"}
        )
        self.addCleanup(cache.close)
        self.assertEqual(cache.lookup("q1", self.prompt), ([1, 2], [3, 9], "stop"))
        with self.assertRaises(KeyError):
            cache.lookup("another", self.prompt)
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            cache.lookup("q1", [{"role": "user", "content": "3+3"}])
        with self.assertRaises(sqlite3.OperationalError):
            cache.connection.execute("DELETE FROM trajectories")

    def test_protocol_and_database_tampering_fail(self):
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            fixed.FixedRolloutCache(self.root, expected={"temperature": 0.7})
        with (self.root / "trajectories.sqlite").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            fixed.FixedRolloutCache(self.root)

    def test_eos_and_token_domain_are_checked(self):
        conn = sqlite3.connect(self.root / "trajectories.sqlite")
        conn.execute("UPDATE trajectories SET response_ids='[9,3]'")
        conn.commit()
        conn.close()
        self.manifest["database_sha256"] = fixed.digest_file(
            self.root / "trajectories.sqlite"
        )
        self.write_manifest()
        cache = fixed.FixedRolloutCache(self.root)
        self.addCleanup(cache.close)
        with self.assertRaisesRegex(ValueError, "after EOS"):
            cache.lookup("q1", self.prompt)


if __name__ == "__main__":
    unittest.main()
