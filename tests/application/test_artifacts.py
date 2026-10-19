"""Hash consistency across delayed metadata refresh and actual concurrent writes."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_state.artifacts import file_digest


def test_timestamp_refresh_requires_identical_stable_read(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"same bytes")
    original = Path.stat
    calls = 0

    def stat(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        if self == path:
            calls += 1
            if calls == 1:
                values = {
                    key: getattr(result, key)
                    for key in (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                values["st_mtime_ns"] -= 1
                return SimpleNamespace(**values)
        return result

    monkeypatch.setattr(Path, "stat", stat)
    assert file_digest(path) == hashlib.sha256(b"same bytes").hexdigest()
    assert calls == 4


def test_content_changes_during_timestamp_refresh_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"old bytes")
    original = Path.stat
    calls = 0

    def stat(self, *args, **kwargs):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 2:
                before = original(self, *args, **kwargs)
                path.write_bytes(b"new bytes")
                os.utime(
                    path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000)
                )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(ValueError, match="changed while hashing"):
        file_digest(path)
