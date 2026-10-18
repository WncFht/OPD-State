"""Freeze production Python/config files before native GPU probes start."""

import hashlib
import json
import shutil
import subprocess


def freeze(root, output, driver):
    source = output / "source"
    shutil.copytree(root / "verl", source / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return source / "verl"
