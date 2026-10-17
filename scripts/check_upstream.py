"""Validate the vendored training tree without importing GPU dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def check(root: Path, *, pristine: bool = False) -> list[str]:
    manifest = json.loads((root / "provenance/one-shot-opd.json").read_text())
    source = root / manifest["path"]
    errors, changed = [], []
    if not source.is_dir():
        return [f"Missing training source: {source}"]
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing or nonregular upstream file: {name}")
            continue
        executable = bool(path.stat().st_mode & 0o111)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["sha256"] or executable != (expected["mode"] == "100755"):
            changed.append(name)
    for path in source.rglob(".git"):
        errors.append(f"Nested Git entry: {path.relative_to(root)}")
    # Git enumerates ignored output correctly and includes new source files.
    names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                manifest["path"],
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    found = set()
    python_count = shell_count = 0
    for name in sorted(set(names) - {""}):
        path = root / name
        if not path.is_file():
            continue
        found.add(path.relative_to(source).as_posix())
        if path.suffix == ".py":
            try:
                compile(path.read_bytes(), str(path), "exec")
                python_count += 1
            except (SyntaxError, ValueError) as exc:
                errors.append(f"{name}: {exc}")
        elif path.suffix == ".sh":
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            shell_count += 1
            if result.returncode:
                errors.append(f"{name}: {result.stderr.strip()}")
    extra = sorted(found - manifest["files"].keys())
    if pristine:
        errors.extend(f"Changed upstream file: {name}" for name in changed)
        errors.extend(f"Extra upstream file: {name}" for name in extra)
    print(
        f"Upstream {manifest['commit']}: {len(manifest['files'])} baseline files, "
        f"{len(changed)} changed, {len(extra)} added; compiled {python_count} Python, checked {shell_count} shell"
    )
    if changed and not pristine:
        print("Modified training sources: " + ", ".join(changed))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--pristine", action="store_true")
    args = parser.parse_args()
    errors = check(args.root.resolve(), pristine=args.pristine)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
