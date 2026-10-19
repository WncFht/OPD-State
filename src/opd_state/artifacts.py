"""Content identities and atomic, exclusively owned run artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def content_id(value):
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(data.encode()).hexdigest()


def file_digest(path):
    path = Path(path)
    previous = None
    for _ in range(2):
        before = path.stat()
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                result.update(block)
        after = path.stat()
        digest = result.hexdigest()
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or (previous is not None and digest != previous):
            break
        if (before.st_mtime_ns, before.st_ctime_ns) == (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            return digest
        # TidalFS can refresh a just-written file's timestamps on first read,
        # even after fsync. Accept only an identical second full read whose
        # metadata is stable; a new inode, size or different bytes still fail.
        previous = digest
    raise ValueError(f"Input changed while hashing: {path}")


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def create_run(path):
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("source", "data", "logs", "events", "checkpoints", "validation"):
        (root / name).mkdir()
    return root


@contextmanager
def execution_lock(root):
    """Never change a prepared/running directory without its nonblocking lock."""
    root = Path(root).resolve(strict=True)
    with (root / ".execution.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Run is already being prepared or executed: {root}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def inside(root, relative):
    """Resolve manifest members without permitting absolute paths or traversal."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact member must be relative to its declared root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact symlink escapes its declared root")
    return result
