"""Native Torch tests for cached DataProto and bootstrap compatibility."""

import json
import sqlite3
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.fixed_rollout import FixedRolloutCache, digest_file, digest_json
from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val


def reference_bootstrap(data, size, functions, repeats=128, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(data), size=(repeats, size), replace=True)
    values = np.asarray(data, dtype=object)
    result = np.empty((len(functions), repeats))
    for i, function in enumerate(functions):
        for j in range(repeats):
            result[i, j] = function(values[indices[j]])
    return [(float(row.mean()), float(row.std())) for row in result]


class BootstrapCompatibilityTests(unittest.TestCase):
    def test_numeric_and_custom_reducers(self):
        for data in [[True, False, True], [0.1, -0.3, 2.4, 0.8], [0, 0, 1, 1]]:
            for size in [1, 2, 4, 8, 16]:
                functions = [np.max, np.min, np.mean, lambda x: float(np.median(x))]
                expected = reference_bootstrap(data, size, functions)
                expected_state = np.random.get_state()
                actual = bootstrap_metric(data, size, functions, n_bootstrap=128)
                actual_state = np.random.get_state()
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(actual_state[1], expected_state[1])
                self.assertEqual(actual_state[2:], expected_state[2:])

    def test_majority_ties_use_first_winning_occurrence(self):
        data = [
            {"pred": "A", "val": 0.0},
            {"pred": "B", "val": 1.0},
            {"pred": "A", "val": 1.0},
            {"pred": "C", "val": 0.0},
        ]
        function = partial(calc_maj_val, vote_key="pred", val_key="val")
        for seed in range(10):
            for size in [2, 4, 8, 16]:
                expected = reference_bootstrap(data, size, [function], seed=seed)
                actual = bootstrap_metric(data, size, [function], n_bootstrap=128, seed=seed)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


class CachedBatchTests(unittest.TestCase):
    def test_eos_is_valid_even_when_it_is_the_padding_token(self):
        raw = [{"role": "user", "content": "question"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            conn = sqlite3.connect(path / "trajectories.sqlite")
            conn.execute(
                "CREATE TABLE trajectories (prompt_id TEXT PRIMARY KEY,prompt_sha256 TEXT,"
                "prompt_ids TEXT,response_ids TEXT,finish_reason TEXT)"
            )
            conn.execute(
                "INSERT INTO trajectories VALUES (?,?,?,?,?)", ("sample", digest_json(raw), "[1,2]", "[3,0]", "stop")
            )
            conn.commit()
            conn.close()
            manifest = {
                "schema": 1,
                "source": "cached_teacher",
                "records": 1,
                "database_sha256": digest_file(path / "trajectories.sqlite"),
                "vocab_size": 10,
                "prompt_cap": 4,
                "response_cap": 4,
                "eos_token_id": 0,
                "pad_token_id": 0,
            }
            (path / "manifest.json").write_text(json.dumps(manifest))
            prompts = np.empty(1, dtype=object)
            prompts[0] = raw
            request = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8)},
                non_tensors={
                    "raw_prompt": prompts,
                    "opd_prompt_id": np.array(["sample"]),
                    "uid": np.array(["fresh-uid"]),
                },
            )
            cache = FixedRolloutCache(path)
            try:
                batch = cache.build_batch(request)
                self.assertEqual(batch.batch["input_ids"].tolist(), [[0, 0, 1, 2, 3, 0, 0, 0]])
                self.assertEqual(batch.batch["attention_mask"].tolist(), [[0, 0, 1, 1, 1, 1, 0, 0]])
                self.assertEqual(batch.batch["response_mask"].tolist(), [[1, 1, 0, 0]])
                self.assertEqual(batch.batch["position_ids"].tolist(), [[0, 0, 0, 1, 2, 3, 0, 0]])
                self.assertNotIn("old_log_probs", batch.batch)
                self.assertNotIn("rollout_log_probs", batch.batch)
                self.assertNotIn("uid", batch.non_tensor_batch)
                request.union(batch)
                self.assertEqual(request.non_tensor_batch["uid"].tolist(), ["fresh-uid"])
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
