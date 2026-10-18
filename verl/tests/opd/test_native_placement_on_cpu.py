"""Verify physical source allocation without changing trajectory quotas."""

import importlib.util
import random
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "verl/workers/rollout/placement.py"
spec = importlib.util.spec_from_file_location("native_placement_contract", SOURCE)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


class PlacementTests(unittest.TestCase):
    def test_equal_and_skewed_groups_have_disjoint_devices(self):
        self.assertEqual(
            placement.place_groups({"current": 32, "teacher": 32}, 4, "split"),
            {"current": (0, 1), "teacher": (2, 3)},
        )
        self.assertEqual(
            placement.place_groups({"current": 7, "teacher": 1}, 4, "split"),
            {"current": (0, 1, 2), "teacher": (3,)},
        )
        self.assertEqual(
            placement.place_groups({"teacher": 1, "current": 1}, 4, "split"),
            {"current": (0,), "teacher": (1,)},
        )

    def test_single_source_uses_all_available_devices_and_overlap_keeps_shared_ranks(self):
        for layout in ["overlap", "split"]:
            self.assertEqual(placement.place_groups({"teacher": 64}, 4, layout), {"teacher": (0, 1, 2, 3)})
            self.assertEqual(placement.place_groups({"teacher": 1}, 4, layout), {"teacher": (0,)})
        self.assertEqual(
            placement.place_groups({"current": 7, "teacher": 1}, 4, "overlap"),
            {"current": (0, 1, 2, 3), "teacher": (0,)},
        )

    def test_random_allocations_cover_devices_without_empty_shards_or_source_overlap(self):
        rng = random.Random(42)
        for _ in range(500):
            replicas = rng.randint(2, 8)
            counts = {"current": rng.randint(1, 128), "teacher": rng.randint(1, 128)}
            original = dict(counts)
            for layout in ["overlap", "split"]:
                allocated = placement.place_groups(counts, replicas, layout)
                self.assertEqual(counts, original)
                self.assertEqual(
                    allocated, placement.place_groups(dict(reversed(list(counts.items()))), replicas, layout)
                )
                for source, indices in allocated.items():
                    self.assertTrue(1 <= len(indices) <= min(replicas, counts[source]))
                    self.assertEqual(len(set(indices)), len(indices))
                    self.assertTrue(all(0 <= i < replicas for i in indices))
                if layout == "split":
                    flat = [i for indices in allocated.values() for i in indices]
                    self.assertEqual(sorted(flat), list(range(min(replicas, sum(counts.values())))))

    def test_invalid_capacities_and_layout_fail_before_activation(self):
        for counts, replicas, layout in [
            ({"a": 1, "b": 1}, 1, "split"),
            ({"a": 0}, 4, "overlap"),
            ({"a": -1}, 4, "overlap"),
            ({"a": True}, 4, "overlap"),
            ({"a": 1.5}, 4, "overlap"),
            ({"a": 1}, 0, "overlap"),
            ({"a": 1}, True, "overlap"),
            ({"a": 1}, 4, "automatic"),
        ]:
            with self.subTest(counts=counts, replicas=replicas, layout=layout), self.assertRaises(ValueError):
                placement.place_groups(counts, replicas, layout)
        self.assertEqual(placement.place_groups({}, 4, "split"), {})


if __name__ == "__main__":
    unittest.main()
