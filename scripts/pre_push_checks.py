"""Validate every pushed commit in an isolated worktree, preserving local edits."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def pushed_commits(lines):
    """Read Git's pre-push protocol; deleted refs require no snapshot check."""
    commits = []
    for line in lines:
        local_ref, local_sha, remote_ref, remote_sha = line.split()
        if set(local_sha) == {"0"}:
            continue
        if local_sha not in commits:
            commits.append(local_sha)
    return commits


def check_commit(root, commit):
    # Git invokes hooks with repository-local variables. A linked checkout
    # must resolve its own index and HEAD rather than inheriting the caller's.
    local_vars = subprocess.check_output(
        ["git", "rev-parse", "--local-env-vars"], cwd=root, text=True
    ).splitlines()
    env = {key: value for key, value in os.environ.items() if key not in local_vars}
    with tempfile.TemporaryDirectory(prefix="opd-pre-push-") as directory:
        checkout = Path(directory) / "checkout"
        subprocess.run(
            ["git", "worktree", "add", "--quiet", "--detach", str(checkout), commit],
            cwd=root,
            env=env,
            check=True,
        )
        try:
            for name in (".venv", "node_modules"):
                source = root / name
                if not source.is_dir() or (checkout / name).exists():
                    raise RuntimeError(f"Missing or conflicting dependency: {name}")
                (checkout / name).symlink_to(source.resolve(), target_is_directory=True)
            print(f"Checking pushed commit {commit}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_checks.py",
                    "--config",
                    "scripts/checks.json",
                    "--mode",
                    "ci",
                ],
                cwd=checkout,
                env=env,
                check=True,
            )
            subprocess.run(
                ["gitleaks", "git", "--redact", "--no-banner", "--log-opts", commit],
                cwd=checkout,
                env=env,
                check=True,
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)],
                cwd=root,
                env=env,
                check=True,
            )


def main():
    root = Path.cwd().resolve()
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    try:
        for commit in pushed_commits(sys.stdin):
            check_commit(root, commit)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Pre-push check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
