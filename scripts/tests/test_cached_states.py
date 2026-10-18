"""Check exact-prefix state counting against explicit state-set enumeration."""

import itertools
import runpy
import unittest
from pathlib import Path

COUNT = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "eval/analyze_cached_states.py")
)["paired_prefix_counts"]


class PrefixStateTests(unittest.TestCase):
    def test_prefix_counts_match_exhaustive_state_sets(self):
        sequences = [
            tokens
            for length in range(1, 5)
            for tokens in itertools.product((0, 1), repeat=length)
        ]
        for student, teacher in itertools.product(sequences, repeat=2):
            student_states = {student[:i] for i in range(len(student))}
            teacher_states = {teacher[:i] for i in range(len(teacher))}
            _, count = COUNT(student, teacher)
            self.assertEqual(count, len(student_states & teacher_states))

    def test_shared_suffix_does_not_restore_a_common_prefix(self):
        self.assertEqual(COUNT([0, 1, 2, 3], [9, 1, 2, 3]), (0, 1))
        self.assertEqual(COUNT([0, 1], [0, 1]), (2, 2))


if __name__ == "__main__":
    unittest.main()
