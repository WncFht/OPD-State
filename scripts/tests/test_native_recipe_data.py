"""Exercise the real native launcher with a non-training downstream capture."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class NativeRecipeDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        recipe = self.root / "recipe"
        recipe.mkdir()
        shutil.copyfile(ROOT / "recipe/math_native.sh", recipe / "math_native.sh")
        (recipe / "math_baseline.sh").write_text(
            "#!/usr/bin/env bash\nset -eu\npython3 - \"$@\" <<'PY'\n"
            "import json,os,sys\nfrom pathlib import Path\n"
            "Path(os.environ['CAPTURE_FILE']).write_text(json.dumps({"
            "'args':sys.argv[1:],'train':os.environ.get('TRAIN_FILES'),"
            "'manifest':os.environ.get('NATIVE_DATA_MANIFEST')}))\nPY\n"
        )
        self.capture = self.root / "capture.json"
        self.environment = dict(os.environ, CAPTURE_FILE=str(self.capture))
        for key in ("TRAIN_FILES", "NATIVE_DATA_MANIFEST"):
            self.environment.pop(key, None)

    def run_launcher(self, *overrides, **environment):
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "bash",
                str(self.root / "recipe/math_native.sh"),
                "student_fresh",
                "trainer.total_training_steps=2",
                *overrides,
            ],
            cwd=self.root,
            env={**self.environment, **environment},
            text=True,
            capture_output=True,
        )
        captured = (
            json.loads(self.capture.read_text()) if self.capture.exists() else None
        )
        return result, captured

    def manifest(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
        return path

    def test_default_uses_prepared_full_pool_and_forwards_overrides(self):
        path = self.manifest("runs/native-data/math-full/manifest.json")
        result, captured = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured["manifest"], str(path))
        self.assertEqual(captured["train"], f"[{path.parent / 'ordered.parquet'}]")
        self.assertIn(f"++trainer.native_data_manifest={path}", captured["args"])
        self.assertEqual(captured["args"][-1], "trainer.total_training_steps=2")

    def test_missing_default_does_not_start_the_downstream_launcher(self):
        result, captured = self.run_launcher()
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(captured)
        self.assertIn("prepared full-prompt manifest", result.stderr)
        self.assertIn("build_fresh_math_data.py", result.stderr)

    def test_explicit_engineering_train_files_keep_the_existing_contract(self):
        train = "[/fixture/repeated-question.parquet]"
        result, captured = self.run_launcher(TRAIN_FILES=train)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured["train"], train)
        self.assertIsNone(captured["manifest"])
        self.assertFalse(any("native_data_manifest" in arg for arg in captured["args"]))

    def test_cli_training_files_do_not_require_or_bind_the_default_manifest(self):
        override = "data.train_files=[/fixture/repeated-question.parquet]"
        for has_default in (False, True):
            if has_default:
                self.manifest("runs/native-data/math-full/manifest.json")
            result, captured = self.run_launcher(override)
            with self.subTest(has_default=has_default):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(captured["manifest"])
                self.assertIn(override, captured["args"])
                self.assertFalse(
                    any("native_data_manifest" in arg for arg in captured["args"])
                )

    def test_cli_manifest_and_data_pair_are_forwarded_without_a_second_binding(self):
        manifest = "++trainer.native_data_manifest=/fixture/prepared/manifest.json"
        train = "data.train_files=[/fixture/prepared/ordered.parquet]"
        result, captured = self.run_launcher(manifest, train)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(captured["manifest"])
        self.assertIn(manifest, captured["args"])
        self.assertIn(train, captured["args"])
        self.assertEqual(
            sum("native_data_manifest" in arg for arg in captured["args"]), 1
        )

    def test_explicit_manifest_with_spaces_and_matching_file_override_are_preserved(
        self,
    ):
        path = self.manifest("custom prepared pool/manifest.json")
        train = "[/fixture/copy-of-prepared.parquet]"
        result, captured = self.run_launcher(
            NATIVE_DATA_MANIFEST=str(path), TRAIN_FILES=train
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured["manifest"], str(path))
        self.assertEqual(captured["train"], train)
        self.assertIn(f"++trainer.native_data_manifest={path}", captured["args"])


if __name__ == "__main__":
    unittest.main()
