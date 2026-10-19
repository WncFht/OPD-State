"""Check training source syntax without importing the GPU training stack."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SOURCE_DIRS = (
    "data",
    "eval",
    "recipe",
    "verl",
    "src",
    "recipes",
    "tools",
    "platforms",
    "benchmarks",
    "tests",
)


def check(root: Path) -> list[str]:
    errors = []
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
                *SOURCE_DIRS,
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    python_count = shell_count = 0
    for name in sorted(set(names) - {""}):
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
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
    if not python_count and not shell_count:
        errors.append("No training source files were selected")
    print(
        f"Training source: compiled {python_count} Python files, checked {shell_count} shell scripts"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    errors = check(args.root.resolve())
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
