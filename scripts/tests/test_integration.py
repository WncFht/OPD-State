"""Check real Git index behavior, gate failures, and launcher path resolution."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_docs
import check_source
import check_spacing
import install_hooks


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-maintenance-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.root / ".gitignore").write_text("node_modules/\n")
        (self.root / "node_modules").symlink_to(ROOT / "node_modules")

    def git(self, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=check
        )

    def run_script(self, name, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *args],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ["PATH"],
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_scan_hidden_files_empty_scan_and_real_spacing_failure(self):
        self.assertNotEqual(self.run_script("check_spacing.py").returncode, 0)
        (self.root / ".hidden.py").write_text("# \u4e2d\u6587Python\n")
        (self.root / "README.md").write_text("# \u4e2d\u6587Markdown\n")
        self.assertEqual(
            set(check_spacing.selected_paths(self.root, [])),
            {".hidden.py", "README.md"},
        )
        self.assertNotEqual(self.run_script("check_spacing.py").returncode, 0)
        self.assertEqual(self.run_script("check_spacing.py", "--fix").returncode, 0)
        self.assertEqual(
            (self.root / ".hidden.py").read_text(), "# \u4e2d\u6587Python\n"
        )
        (self.root / ".hidden.py").write_text("# 中文 Python\n")
        self.assertEqual(self.run_script("check_spacing.py").returncode, 0)
        self.assertNotEqual(
            self.run_script("check_spacing.py", "--fix", ".hidden.py").returncode, 0
        )

    def test_real_partial_staging_and_hook_failure_preserve_worktree(self):
        path = self.root / "sample.py"
        path.write_text("value = 0\n")
        self.git("add", ".")
        self.git("commit", "-qm", "chore: Initialize fixture")
        path.write_text("value=1\n")
        self.git("add", "sample.py")
        path.write_text("value=1\n# unstaged user edit\n")
        config = {
            "repos": [
                {
                    "repo": "local",
                    "hooks": [
                        {
                            "id": "format",
                            "name": "format",
                            "language": "system",
                            "types": ["python"],
                            "stages": ["manual"],
                            "entry": f'{ROOT}/node_modules/.bin/git-format-staged --formatter "{Path(sys.executable).parent}/ruff format --stdin-filename {{}} -"',
                        },
                        {
                            "id": "lint",
                            "name": "lint",
                            "language": "system",
                            "types": ["python"],
                            "entry": f"{Path(sys.executable).parent}/ruff check --select F",
                        },
                    ],
                }
            ]
        }
        (self.root / ".pre-commit-config.yaml").write_text(yaml.safe_dump(config))
        install_hooks.install(self.root)
        success = self.git("commit", "-m", "feat: Check partial staging", check=False)
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
        self.assertEqual(self.git("show", "HEAD:sample.py").stdout, "value = 1\n")
        self.assertIn("# unstaged user edit", path.read_text())
        path.write_text("value = missing_name\n")
        self.git("add", "sample.py")
        # A valid working tree must not hide an invalid index snapshot.
        path.write_text("value = 2\n# user repair remains unstaged\n")
        rejected = self.git(
            "commit", "-m", "test: Reject invalid staged source", check=False
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("missing_name", self.git("show", ":sample.py").stdout)
        self.assertEqual(
            path.read_text(), "value = 2\n# user repair remains unstaged\n"
        )

    def test_document_gate_reads_index_and_rejects_broken_links(self):
        shutil.copytree(
            ROOT / ".agents",
            self.root / ".agents",
            ignore=shutil.ignore_patterns("implemented", "proposed"),
        )
        (self.root / "docs").mkdir()
        (self.root / "docs/development.md").write_text("# Fixture development\n")
        (self.root / "scripts").mkdir()
        for name in [
            "staged_checks.py",
            "check_docs.py",
            "notes_guard.py",
            "check_archive.py",
        ]:
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        doc = self.root / "README.md"
        doc.write_text("# Fixture\n")
        self.git("add", ".")
        self.assertEqual(self.run_script("staged_checks.py").returncode, 0)
        doc.write_text("# Fixture\n\n[broken](missing.md)\n")
        self.git("add", "README.md")
        doc.write_text("# Repaired only in worktree\n")
        rejected = self.run_script("staged_checks.py")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("missing.md", rejected.stdout)
        self.assertEqual(check_docs.check(self.root), [])

    def test_formatter_propagates_upstream_pipeline_failure(self):
        binary = self.root / "node_modules"
        binary.unlink()
        (binary / ".bin").mkdir(parents=True)
        for name, body in [("autocorrect", "exit 7"), ("prettier", "cat")]:
            path = binary / ".bin" / name
            path.write_text("#!/bin/sh\n" + body + "\n")
            path.chmod(0o755)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/fmt-markdown.sh"), "README.md"],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": str(binary / ".bin") + os.pathsep + os.environ["PATH"],
            },
            input="# Fixture\n",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 7)

    def test_secret_allowlist_only_accepts_the_upstream_constant(self):
        shutil.copy2(ROOT / ".gitleaks.toml", self.root / ".gitleaks.toml")
        source = self.root / "verl/verl/protocol.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "_padding_size_key = " + json.dumps("_padding_size_" + "key_x123d") + "\n"
        )
        self.git("add", ".")
        command = [
            str(Path(sys.executable).parent / "gitleaks"),
            "git",
            "--pre-commit",
            "--staged",
            "--redact",
            "--no-banner",
        ]
        accepted = subprocess.run(
            command, cwd=self.root, capture_output=True, check=False
        )
        self.assertEqual(accepted.returncode, 0)
        # The synthetic high-entropy value is generated only inside the fixture.
        token = hashlib.sha256(b"synthetic gitleaks rejection fixture").hexdigest()
        source.write_text(source.read_text() + "api_key = " + repr(token) + "\n")
        self.git("add", ".")
        rejected = subprocess.run(
            command, cwd=self.root, capture_output=True, check=False
        )
        self.assertNotEqual(rejected.returncode, 0)

    def test_manifest_hash_allowlist_preserves_path_key_and_format_boundaries(self):
        shutil.copy2(ROOT / ".gitleaks.toml", self.root / ".gitleaks.toml")
        digest = hashlib.sha256(b"synthetic tokenizer metadata fixture").hexdigest()
        allowed = "docs/experiment/results/fixture/cache-manifest.json"
        cases = [
            (allowed, "tokenizer_signature", digest, 0),
            (allowed, "api_key", digest, 1),
            (allowed, "tokenizer_signature", digest + "XYZ", 1),
            (allowed, "tokenizer_signature", digest.upper(), 1),
            ("docs/elsewhere/cache-manifest.json", "tokenizer_signature", digest, 1),
            (
                "docs/experiment/results/fixture/credentials.json",
                "tokenizer_signature",
                digest,
                1,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-summary.json",
                "tokens_sha256",
                digest,
                0,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-independent-audit.json",
                "tokens_sha256",
                digest,
                0,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-summary.json",
                "api_key",
                digest,
                1,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/other.json",
                "tokens_sha256",
                digest,
                1,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-summary.json",
                "tokens_sha256",
                digest.upper(),
                1,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-collection.json",
                "/fixture/runs/native-validation-kl-long-20261019-v2/validation/kl/kl-tokens-step-2.jsonl",
                digest,
                0,
            ),
            (
                "docs/experiment/results/native-rollout/validation-kl-long/kl-collection.json",
                "/fixture/runs/native-validation-kl-long-20261019-v2/validation/kl/kl-tokens-step-3.jsonl",
                digest,
                1,
            ),
        ]
        command = [
            str(Path(sys.executable).parent / "gitleaks"),
            "git",
            "--pre-commit",
            "--staged",
            "--redact",
            "--no-banner",
        ]
        previous = None
        for name, key, value, code in cases:
            with self.subTest(path=name, key=key, expected_exit=code):
                if previous is not None:
                    previous.unlink()
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({key: value}, indent=2) + "\n")
                self.git("add", "-A")
                result = subprocess.run(
                    command, cwd=self.root, capture_output=True, check=False
                )
                self.assertEqual(result.returncode, code)
                previous = path

    def test_training_source_syntax_and_editing(self):
        self.assertTrue(check_source.check(self.root))
        source = self.root / "verl"
        source.mkdir()
        path = source / "sample.py"
        path.write_text("value = 1\n")
        self.assertEqual(check_source.check(self.root), [])
        path.write_text("value = 2\n")
        self.assertEqual(check_source.check(self.root), [])
        path.write_text("value = (\n")
        self.assertTrue(check_source.check(self.root))
        path.unlink()
        shell = source / "launch.sh"
        shell.write_text("#!/bin/bash\necho valid\n")
        self.assertEqual(check_source.check(self.root), [])
        shell.write_text("#!/bin/bash\nif then\n")
        self.assertTrue(check_source.check(self.root))

    def test_launchers_forward_arguments_and_select_vendored_verl(self):
        binary = self.root / "training-env/bin"
        binary.mkdir(parents=True)
        fake = binary / "python3"
        fake.write_text(
            f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps({'args':sys.argv[1:],"
            "'verl':os.environ['VERL_ROOT'],'pythonpath':os.environ['PYTHONPATH']}))\n"
        )
        fake.chmod(0o755)
        capture = self.root / "capture.json"
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "CONDA_ENV_BIN": str(binary),
            "MODEL_BASE": str(self.root / "models"),
            "OPD_ROOT": str(self.root / "runtime"),
            "OPD_DRY_RUN": "1",
            "CAPTURE": str(capture),
            "N_GPUS_PER_NODE": "4",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
        upstream = ROOT
        for script, mode in [
            ("opd.sh", "full"),
            ("opd.sh", "oneshot"),
            ("opd.sh", "template"),
            ("grpo.sh", "full"),
            ("mopd.sh", "full"),
        ]:
            with self.subTest(script=script, mode=mode):
                result = subprocess.run(
                    [
                        "bash",
                        "-o",
                        "pipefail",
                        str(upstream / "recipe" / script),
                        "trainer.total_training_steps=10",
                    ],
                    cwd=self.root,
                    env={**env, "MODE": mode},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(capture.read_text())
                self.assertEqual(data["verl"], str(upstream / "verl"))
                self.assertTrue(
                    data["pythonpath"].startswith(str(upstream / "verl") + ":")
                )
                self.assertEqual(
                    data["args"][:5],
                    [
                        "-m",
                        "verl.trainer.main_ppo",
                        "--cfg",
                        "job",
                        "--config-name=ppo_trainer",
                    ],
                )
                self.assertIn("trainer.n_gpus_per_node=4", data["args"])
                self.assertIn("trainer.total_training_steps=10", data["args"])
        baseline = subprocess.run(
            ["bash", str(upstream / "recipe/math_baseline.sh")],
            cwd=self.root,
            env={
                **env,
                "OPD_TRAIN_PYTHON": str(fake),
                "EXPERIMENT_NAME": "fixture-baseline",
                "OPD_ROOT": str(self.root / "baseline"),
                "WANDB_DIR": str(self.root / "wandb"),
                "RAY_TMPDIR": str(self.root / "ray"),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        args = json.loads(capture.read_text())["args"]
        for required in [
            "trainer.total_training_steps=50",
            "trainer.test_freq=50",
            "trainer.save_freq=-1",
            "trainer.val_before_train=False",
            "trainer.project_name=OPD-State",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
            "actor_rollout_ref.rollout.val_kwargs.n=16",
            "+trainer.math_eval_samples=16",
        ]:
            self.assertIn(required, args)
        result = subprocess.run(
            ["bash", str(upstream / "recipe/opd.sh")],
            cwd=self.root,
            env={**env, "MODE": "invalid"},
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
