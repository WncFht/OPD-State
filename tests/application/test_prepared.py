"""Sealed source/data identity and the foreground state machine."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from opd_state.artifacts import atomic_json, content_id, execution_lock, file_digest
from recipes.application import prepare, run_prepared, validate_application
from tools.verify import configuration_id, verify_prepared

ROOT = Path(__file__).resolve().parents[2]


def test_configuration_identity_is_independent_of_source_identity(prepared):
    root = prepared["root"]
    config, manifest, inputs = verify_prepared(root)
    config.ray_kwargs.ray_init.runtime_env.env_vars.OPD_SOURCE_ID = "different-source"
    assert (
        configuration_id(config, root, inputs, manifest["scratch"])
        == manifest["config_id"]
    )
    config.actor_rollout_ref.actor.optim.lr *= 2
    assert (
        configuration_id(config, root, inputs, manifest["scratch"])
        != manifest["config_id"]
    )


def test_verify_cli_uses_frozen_decoder_before_current_schema(prepared):
    root = prepared["root"]
    # A historical decoder can support fields rejected by the current schema.
    # This stub proves handoff only; real completed runs have separate GPU evidence.
    config = OmegaConf.load(root / "resolved.yaml")
    config.distillation.opd.objective.kind = "historical-objective"
    OmegaConf.save(config, root / "resolved.yaml")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["configuration_sha256"] = file_digest(root / "resolved.yaml")
    manifest["manifest_id"] = content_id(
        {k: v for k, v in manifest.items() if k != "manifest_id"}
    )
    atomic_json(root / "manifest.json", manifest)
    result = subprocess.run(
        [sys.executable, "-m", "tools.verify", str(root), "--prepared"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "verification-dispatch-proof.txt").read_text() == str(
        root / "source/tools/verify.py"
    )


def test_preparation_owns_new_directory_and_preserves_native_inputs(prepared):
    root = prepared["root"]
    config, manifest, inputs = verify_prepared(root)
    assert manifest["run_id"] == prepared["manifest"]["run_id"]
    assert config.data.train_batch_size == 4
    assert config.trainer.total_training_steps == 2
    assert inputs["selected_counts"] == {"train": 4, "validation": 3}
    assert not Path(manifest["scratch"]).exists()
    scratch = Path(manifest["scratch"])
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.RAY_TMPDIR == str(scratch)
    assert (
        len(
            os.fsencode(
                scratch
                / "ray/session_2099-12-31_23-59-59_999999_4194304/sockets/plasma_store"
            )
        )
        <= 107
    )
    assert config.trainer.logger == ["console", "file"]
    assert Path(config.data.train_files[0]).is_relative_to(root / "data")
    assert Path(config.reward.custom_reward_function.path).is_relative_to(
        root / "source"
    )
    assert json.loads((root / "status.json").read_text())["state"] == "prepared"
    with pytest.raises(FileExistsError):
        prepare(
            prepared["config"],
            backend=prepared["backend"],
            repository=prepared["repository"],
            input_preparer=prepared["inputs"],
        )


def test_relative_run_outputs_survive_frozen_working_directory(prepared, monkeypatch):
    root = prepared["root"].parent / "relative-run"
    config = OmegaConf.create(OmegaConf.to_container(prepared["config"], resolve=True))
    relative = os.path.relpath(root, Path.cwd())
    config.run.dir = relative
    config.run.source_dir = relative + "/source"
    config.trainer.default_local_dir = relative + "/checkpoints"
    config.trainer.validation_data_dir = relative + "/validation"
    prepare(
        config,
        backend=prepared["backend"],
        repository=prepared["repository"],
        input_preparer=prepared["inputs"],
    )
    monkeypatch.chdir(root / "source")
    frozen, _, _ = verify_prepared(root)
    assert validate_application(frozen) == root
    assert frozen.trainer.default_local_dir == str(root / "checkpoints")
    assert frozen.trainer.validation_data_dir == str(root / "validation")


@pytest.mark.parametrize(
    "member",
    [
        "resolved.yaml",
        "inputs.json",
        "source/recipes/train.py",
        "data/train/ordered.parquet",
        "data/selection.json",
    ],
)
def test_prepared_mutations_are_rejected(prepared, member):
    path = prepared["root"] / member
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify_prepared(prepared["root"])


def test_external_tokenizer_change_is_rejected(prepared):
    path = prepared["models"] / "JustRL-DeepSeek-1.5B/tokenizer.json"
    path.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="bytes changed"):
        verify_prepared(prepared["root"])


@pytest.mark.parametrize("member", ["source/recipes/injected.py", "data/injected.json"])
def test_added_members_are_not_silently_ignored(prepared, member):
    path = prepared["root"] / member
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="membership"):
        verify_prepared(prepared["root"])


def test_execution_lock_is_exclusive(prepared):
    with execution_lock(prepared["root"]):
        with pytest.raises(RuntimeError, match="already"):
            with execution_lock(prepared["root"]):
                pass


def test_config_and_backend_ownership_cannot_be_rebound(prepared):
    root = prepared["root"]
    config = OmegaConf.load(root / "resolved.yaml")
    config.run.backend = str(prepared["backend"])
    OmegaConf.save(config, root / "resolved.yaml")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["configuration_sha256"] = file_digest(root / "resolved.yaml")
    manifest["manifest_id"] = content_id(
        {k: v for k, v in manifest.items() if k != "manifest_id"}
    )
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="frozen backend"):
        verify_prepared(root)


def test_zero_exit_without_training_evidence_cannot_complete(prepared):
    def runner(*args, **kwargs):
        return 0

    with pytest.raises(FileNotFoundError):
        run_prepared(
            prepared["root"],
            resource_check=lambda cfg: {"gpu_count": 4, "fixture": True},
            command_runner=runner,
        )
    status = json.loads((prepared["root"] / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["cleanup_complete"]
    assert not Path(prepared["manifest"]["scratch"]).exists()
    with pytest.raises(ValueError, match="unexecuted"):
        run_prepared(prepared["root"], resource_check=lambda cfg: {})


def test_prepared_cli_rejects_overrides_before_launch(prepared):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recipes.train",
            "--prepared",
            str(prepared["root"]),
            "trainer.total_training_steps=1",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "immutable" in result.stderr
    assert not (prepared["root"] / ".executed").exists()


def test_prepared_cli_dispatches_frozen_source_after_worktree_changes(prepared):
    # This fixture entry only records dispatch. It does not simulate GPU success.
    (prepared["repository"] / "recipes/train.py").write_text(
        "raise RuntimeError('unfrozen code executed')\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "recipes.train", "--prepared", str(prepared["root"])],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (prepared["root"] / "dispatch-proof.txt").read_text() == str(
        prepared["root"] / "source/recipes/train.py"
    )
    assert (
        json.loads((prepared["root"] / "status.json").read_text())["state"]
        == "prepared"
    )
