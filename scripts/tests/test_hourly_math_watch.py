"""Do not archive unlogged validation or a partially written training row."""

import runpy
import tempfile
import unittest
from pathlib import Path

WATCH = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "eval/watch_math_progress.py")
)


class HourlyWatchTests(unittest.TestCase):
    def test_raw_file_and_logged_validation_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "validation").mkdir()
            (run / "validation/100.jsonl").touch()
            rows = {
                50: {"eval/macro/mean@16": 0.5},
                100: {"actor/loss": 0.1},
            }
            self.assertEqual(WATCH["complete_steps"](run, rows, 50), [])
            (run / "validation/50.jsonl").touch()
            self.assertEqual(WATCH["complete_steps"](run, rows, 50), [50])
            rows[100]["eval/macro/mean@16"] = 0.6
            self.assertEqual(WATCH["complete_steps"](run, rows, 50), [50, 100])

    def test_partial_future_row_is_not_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "logs").mkdir()
            (run / "validation").mkdir()
            (run / "validation/50.jsonl").touch()
            (run / "validation/100.jsonl").touch()
            (run / "logs/train.log").write_text(
                "step:50 - actor/loss:0.1 - eval/macro/mean@16:0.5\n"
                "step:99 - actor/loss:0.05\n"
                "step:100 - eval/macro/mean@16:0."
            )
            report = WATCH["inspect"](run, 50)
            self.assertEqual(report["last_complete_training_log_step"], 99)
            self.assertEqual(report["complete_validation_steps"], [50])


if __name__ == "__main__":
    unittest.main()
