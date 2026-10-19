"""Freeze every declared execution component and verify the complete closure."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

from opd_state.artifacts import atomic_json, content_id, file_digest, inside

SUFFIXES = {".py", ".json", ".yaml", ".yml", ".jinja", ".patch", ".toml", ".txt"}
IGNORED = {"__pycache__", ".git", "secrets", "node_modules", ".venv"}


def source_files(root):
    root = Path(root).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED or part.startswith(".env") for part in relative.parts):
            continue
        if not path.is_file() or not (
            path.suffix in SUFFIXES
            or path.name.startswith(("LICENSE", "NOTICE"))
            or path.name == "version"
        ):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(
                f"Executable source escapes the declared component: {path}"
            )
        files.append(relative.as_posix())
    return files


def snapshot_sources(repository, backend, destination, *, package=None):
    repository, backend, destination = map(
        lambda p: Path(p).resolve(), (repository, backend, destination)
    )
    if package is None:
        import opd_state

        package = Path(opd_state.__file__).parent
    components = {
        "opd_state": Path(package).resolve(),
        "backend/verl/verl": backend / "verl/verl",
    }
    for name in (
        "recipes",
        "configs",
        "tools",
        "platforms",
        "patches",
        "data/prep",
        "eval",
    ):
        if (repository / name).is_dir():
            components[name] = repository / name
    records = []
    original = {}
    for component, root in components.items():
        members = source_files(root)
        if not members:
            raise ValueError(f"Empty execution component: {root}")
        original[component] = (root, members)
        for member in members:
            source = inside(root, member)
            digest = file_digest(source)
            relative = f"{component}/{member}"
            target = inside(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_digest(target) != digest or file_digest(source) != digest:
                raise ValueError(f"Source changed while freezing: {source}")
            records.append(
                {"path": relative, "sha256": digest, "bytes": target.stat().st_size}
            )
    for component, (root, members) in original.items():
        if source_files(root) != members:
            raise ValueError(f"Execution component changed while freezing: {root}")
        for row in (r for r in records if r["path"].startswith(component + "/")):
            member = row["path"][len(component) + 1 :]
            if file_digest(inside(root, member)) != row["sha256"]:
                raise ValueError(f"Source changed during preparation: {root / member}")
    manifest = {"schema_version": 1, "files": records, "source_id": content_id(records)}
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def verify_snapshot(destination):
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or content_id(manifest["files"]) != manifest["source_id"]
    ):
        raise ValueError("Invalid source snapshot identity")
    expected = {row["path"] for row in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate frozen source member")
    actual = set(source_files(destination)) - {"manifest.json"}
    if actual != expected:
        raise ValueError("Frozen execution source membership changed")
    for row in manifest["files"]:
        if file_digest(inside(destination, row["path"])) != row["sha256"]:
            raise ValueError(f"Frozen execution source changed: {row['path']}")
    return manifest


def remap_source(path, repository, backend, destination, *, package=None):
    if package is None:
        import opd_state

        package = Path(opd_state.__file__).parent
    path, repository, backend, destination = map(
        lambda p: Path(p).resolve(), (path, repository, backend, destination)
    )
    mappings = [
        (backend / "verl/verl", destination / "backend/verl/verl"),
        (Path(package).resolve(), destination / "opd_state"),
    ]
    mappings += [
        (repository / name, destination / name)
        for name in ("recipes", "configs", "tools", "platforms", "data/prep", "eval")
    ]
    for original, target in mappings:
        if path.is_relative_to(original):
            result = target / path.relative_to(original)
            if not result.is_file():
                raise ValueError(
                    "Executed helper was not included in the source snapshot"
                )
            return str(result)
    raise ValueError("Executed helper is outside the declared frozen source roots")


def runtime_versions():
    result = {"python": platform.python_version()}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "ray",
        "tensordict",
        "numpy",
        "pyarrow",
        "datasets",
        "hydra-core",
        "omegaconf",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
