"""One native Hydra composer shared by the foreground app, studies and tests."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from opd_state.core.config import Objective
from opd_state.core.evaluation import Benchmark, Evaluation
from opd_state.integrations.verl.configuration import evaluation_field, validate_config


def _path(root, relative):
    if root is None:
        raise ValueError(
            "Set paths.model_root/paths.data_root or the native artifact path"
        )
    return str(Path(str(root)).expanduser() / str(relative))


def _manifest_data(manifest):
    if not manifest:
        raise ValueError("Select an explicit prepared dataset manifest")
    return str(Path(str(manifest)).expanduser().parent / "ordered.parquet")


def _context(prompt, response, evaluation):
    return max(int(prompt) + int(response), evaluation_field(evaluation, "context"))


def _evaluation(value, field, total_steps=None):
    result = evaluation_field(value, field, total_steps)
    return OmegaConf.create(result) if isinstance(result, (dict, list)) else result


def configure(backend, *, repository=None):
    """Register composition resources without reading model/data or allocating CUDA."""
    repository = Path(repository or Path(__file__).resolve().parents[1]).resolve()
    backend = Path(backend).expanduser().resolve()
    native = backend / "verl/verl/trainer/config"
    if not (native / "ppo_trainer.yaml").is_file():
        raise ValueError(
            "Backend must contain verl/verl/trainer/config/ppo_trainer.yaml"
        )
    resolvers = {
        "opd_backend_config": lambda: str(native),
        "opd_backend": lambda: str(backend),
        "opd_repository": lambda: str(repository),
        "opd_path": _path,
        "opd_manifest_data": _manifest_data,
        "opd_context": _context,
        "opd_eval": _evaluation,
        "opd_source_commit": lambda source: subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
    }
    for name, function in resolvers.items():
        OmegaConf.register_new_resolver(name, function, replace=True)
    for name, cls in (
        ("objective", Objective),
        ("evaluation", Evaluation),
        ("benchmark", Benchmark),
    ):
        # Mutable composition; the single decoder returns immutable typed objects.
        ConfigStore.instance().store(group="schema", name=name, node=asdict(cls()))
    return repository / "configs"


def compose_config(backend, overrides=(), *, repository=None, resolve=True):
    directory = configure(backend, repository=repository)
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        config = compose(config_name="train", overrides=list(overrides))
    bind_sources(config, backend=backend, repository=directory.parent)
    if resolve:
        OmegaConf.resolve(config)
        validate_config(config, backend=backend)
    return config


def bind_sources(config, *, backend, repository):
    # ConfigStore/resolver registration is process-global. Bind ownership now,
    # including for lazy configs, before another composition selects a backend.
    for key, path in (
        ("backend", Path(backend).resolve()),
        ("repository", Path(repository).resolve()),
    ):
        if Path(config.run[key]).resolve() != path:
            raise ValueError(
                f"run.{key} is derived from the selected configuration source"
            )
        config.run[key] = str(path)
    return config
