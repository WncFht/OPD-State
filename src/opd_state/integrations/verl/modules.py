"""Load explicitly selected pure helpers while extracting the fixed backend."""

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from opd_state.artifacts import inside


@lru_cache(maxsize=32)
def _load(path, digest):
    name = "_opd_source_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def backend_module(backend, relative):
    path = inside(Path(backend) / "verl/verl", relative)
    return _load(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
