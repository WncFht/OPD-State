"""Exercise real pushes against committed snapshots and historical secrets."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import install_hooks
import pre_push_checks


class PrePushTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-push-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.remote = Path(self.temp.name) / "remote.git"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("init", "--bare", "-q", str(self.remote))
        (self.root / ".gitignore").write_text(".venv\nnode_modules\n")
        (self.root / ".venv").symlink_to(Path(sys.executable).parent.parent)
        (self.root / "node_modules").symlink_to(ROOT / "node_modules")
        (self.root / "scripts").mkdir()
        for name in ("run_checks.py", "pre_push_checks.py"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        config = {
            "version": 1,
            "checks": [
                {
                    "id": "syntax",
                    "command": [
                        "python3",
                        "-c",
                        "compile(open('sample.py').read(), 'sample.py', 'exec')",
                    ],
                    "needs": [],
                    "modes": ["ci"],
                    "timeout": 10,
                    "allow_failure": False,
                }
            ],
        }
        (self.root / "scripts/checks.json").write_text(json.dumps(config))
        (self.root / "sample.py").write_text("value = 1\n")
        self.commit("chore: Initialize fixture")
        install_hooks.install(self.root, "pre-push")

    def git(self, *args, check=True):
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
            check=check,
        )

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)

    def push(self, *refs):
        return self.git("push", str(self.remote), *refs, check=False)

    def assert_no_checkouts(self):
        worktrees = self.git("worktree", "list", "--porcelain").stdout
        self.assertEqual(worktrees.count("worktree "), 1, worktrees)

    def test_push_checks_commit_and_preserves_dirty_index_and_worktree(self):
        sample = self.root / "sample.py"
        sample.write_text("invalid syntax in index !\n")
        self.git("add", "sample.py")
        sample.write_text("invalid syntax in worktree !\n")
        index_before = self.git("write-tree").stdout
        accepted = self.push("HEAD:refs/heads/main")
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertEqual(self.git("write-tree").stdout, index_before)
        self.assertEqual(sample.read_text(), "invalid syntax in worktree !\n")
        self.assert_no_checkouts()
        self.commit("test: Commit invalid syntax")
        sample.write_text("value = 2\n")
        rejected = self.push("HEAD:refs/heads/main")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("SyntaxError", rejected.stdout + rejected.stderr)
        self.assertEqual(sample.read_text(), "value = 2\n")
        remote_sha = self.git(
            "--git-dir", str(self.remote), "rev-parse", "refs/heads/main"
        ).stdout
        self.assertEqual(remote_sha, self.git("rev-parse", "HEAD^").stdout)
        self.assert_no_checkouts()

    def test_push_rejects_secret_deleted_from_tip(self):
        token = hashlib.sha256(b"synthetic push rejection fixture").hexdigest()
        secret = self.root / "credentials.py"
        secret.write_text("api_key = " + repr(token) + "\n")
        self.commit("test: Add synthetic secret")
        secret.unlink()
        self.commit("test: Remove synthetic secret")
        rejected = self.push("HEAD:refs/heads/main")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("leaks found", rejected.stdout + rejected.stderr)
        self.assertNotIn(token, rejected.stdout + rejected.stderr)
        self.assert_no_checkouts()

    def test_protocol_checks_all_distinct_tips_and_ignores_deletions(self):
        zero, first, second = "0" * 40, "1" * 40, "2" * 40
        lines = [
            f"refs/heads/a {first} refs/heads/a {zero}",
            f"refs/heads/b {second} refs/heads/b {first}",
            f"refs/tags/v1 {first} refs/tags/v1 {zero}",
            f"(delete) {zero} refs/heads/old {second}",
        ]
        self.assertEqual(pre_push_checks.pushed_commits(lines), [first, second])
        self.assertEqual(pre_push_checks.pushed_commits([]), [])
        self.assertEqual(self.push(":refs/heads/absent").returncode, 0)

    def test_installer_preserves_foreign_pre_push(self):
        hook = self.root / ".git/hooks/pre-push"
        hook.write_text("#!/bin/sh\nexit 7\n")
        with self.assertRaises(SystemExit):
            install_hooks.install(self.root, "pre-push")
        self.assertEqual(hook.read_text(), "#!/bin/sh\nexit 7\n")


if __name__ == "__main__":
    unittest.main()
