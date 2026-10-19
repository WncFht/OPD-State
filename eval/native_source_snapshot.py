"""Freeze production Python/config files before native GPU probes start."""

import hashlib
import json
import os
import shutil
import subprocess
import sys


def freeze(root, output, driver, commit=None):
    root, output = root.resolve(), output.resolve()
    source = output / "source"
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", (commit or "HEAD") + "^{commit}"], cwd=root, text=True
    ).strip()
    if commit:
        source.mkdir()
        components = ["verl", "src/opd_state"]
        archive = subprocess.Popen(["git", "archive", resolved, *components], cwd=root, stdout=subprocess.PIPE)
        unpacked = subprocess.run(["tar", "-xf", "-", "-C", str(source)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() or unpacked.returncode:
            raise RuntimeError("Could not freeze the selected native source commit")
    else:
        shutil.copytree(root / "verl", source / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(root / "src/opd_state", source / "src/opd_state", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "base_commit": resolved,
        "source_kind": "git_archive" if commit else "worktree_copy",
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return source / "verl"


def activate(source):
    """Use the package paired with this snapshot, including in spawned workers."""
    paths = [str(source)]
    package = source.parent / "src"
    if (package / "opd_state").is_dir():
        paths.insert(0, str(package))
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
