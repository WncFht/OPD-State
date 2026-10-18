"""Check scheduler preflight, committed snapshots and physical split bindings."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load("native_benchmark_controls", "benchmark_native_policies.py")
snapshot = load("native_benchmark_snapshot", "native_source_snapshot.py")
sweep = load("native_benchmark_sweep", "run_native_scheduler_sweep.py")


class NativeBenchmarkControlTests(unittest.TestCase):
    def test_sweep_controls_share_workload_and_plan_without_starting_gpu_jobs(self):
        conditions = sweep.cases()
        self.assertEqual(len(conditions), 15)
        for case in conditions:
            args = self.options(
                layout=case["layout"],
                replicas=case["replicas"],
                budget=case["budget"],
                max_num_seqs=case["sequences"],
                max_num_batched_tokens=case["batched_tokens"],
                chunked_prefill=case["chunked_prefill"],
                natural_repeats=3,
            )
            benchmark.validate_options(args)
            argv = sweep.command(case, Path("output"), "commit", Path("python"))
            self.assertIn("--natural-warmup", argv)
            self.assertEqual(argv[argv.index("--batches") + 1], "64")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "plan"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "eval/run_native_scheduler_sweep.py"),
                    "--plan-only",
                    "--source-commit",
                    "HEAD",
                    "--output",
                    str(directory),
                ],
                cwd=ROOT,
                check=True,
            )
            state = json.loads((directory / "state.json").read_text())
            self.assertEqual(state["state"], "planned")
            self.assertEqual(len(state["cases"]), 15)

    def options(self, **changes):
        return NS(
            **(
                {
                    "layout": "overlap",
                    "replicas": 4,
                    "batches": [64],
                    "repeats": 3,
                    "cap": 128,
                    "natural_cap": 4096,
                    "natural_repeats": 1,
                    "budget": 0.4,
                    "max_num_seqs": None,
                    "max_num_batched_tokens": 8192,
                    "chunked_prefill": True,
                }
                | changes
            )
        )

    def test_invalid_scheduler_settings_are_rejected_before_gpu_startup(self):
        for changes in [
            {"max_num_seqs": 0},
            {"max_num_batched_tokens": 0},
            {"repeats": 0},
            {"cap": 5120},
            {"batches": [0]},
            {"budget": 1},
            {"chunked_prefill": False, "max_num_batched_tokens": 2048},
            {"layout": "split", "replicas": 1},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                benchmark.validate_options(self.options(**changes))
        options = self.options(max_num_batched_tokens=512)
        benchmark.validate_options(options)
        self.assertEqual(options.max_num_seqs, 64)
        benchmark.validate_options(
            self.options(chunked_prefill=False, max_num_batched_tokens=5120)
        )

    def test_teacher_local_replica_zero_can_bind_physical_gpu_two(self):
        engines = {"teacher": {"gpus": [2, 3]}}
        uuids = {2: "GPU-card-2", 3: "GPU-card-3"}
        benchmark.check_execution(
            [{"slot": "teacher", "replica_rank": 0, "device_uuid": "card-2"}],
            engines,
            uuids,
        )
        for rank, uuid in [(2, "card-2"), (0, "card-0")]:
            with self.subTest(rank=rank, uuid=uuid), self.assertRaises(ValueError):
                benchmark.check_execution(
                    [{"slot": "teacher", "replica_rank": rank, "device_uuid": uuid}],
                    engines,
                    uuids,
                )
        with self.assertRaises(ValueError):
            benchmark.check_execution([], engines, uuids)

    def test_committed_snapshot_does_not_copy_later_worktree_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "verl").mkdir()
            module = root / "verl/model.py"
            module.write_text("VALUE = 1\n")
            driver = root / "probe.py"
            driver.write_text("# driver identity\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "verl/model.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "--no-gpg-sign",
                    "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            module.write_text("VALUE = 2\n")
            committed, working = root / "committed", root / "working"
            committed.mkdir()
            working.mkdir()
            frozen = snapshot.freeze(root, committed, driver, commit="HEAD")
            live = snapshot.freeze(root, working, driver)
            self.assertEqual((frozen / "model.py").read_text(), "VALUE = 1\n")
            self.assertEqual((live / "model.py").read_text(), "VALUE = 2\n")
            self.assertEqual(
                json.loads((committed / "source-manifest.json").read_text())[
                    "source_kind"
                ],
                "git_archive",
            )


if __name__ == "__main__":
    unittest.main()
