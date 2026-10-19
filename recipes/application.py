"""Prepare immutable runs, supervise their frozen backend, and verify completion."""

from __future__ import annotations

import codecs
import json
import os
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from opd_state.artifacts import (
    atomic_json,
    content_id,
    create_run,
    execution_lock,
    file_digest,
)
from opd_state.events import EventWriter
from opd_state.integrations.verl.configuration import validate_config
from opd_state.processes import (
    OwnedProcess,
    ProcessCleanupError,
    finish_cleanup,
    process_record,
)
from recipes.inputs import prepare_inputs
from recipes.snapshots import remap_source, runtime_versions, snapshot_sources
from tools.verify import configuration_id, verify_prepared, verify_run


def validate_application(config):
    if not isinstance(config.run.dir, str) or not config.run.dir.strip():
        raise ValueError("Set a new persistent run.dir")
    if config.run.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("Unsupported W&B mode")
    names = (
        [config.trainer.logger]
        if isinstance(config.trainer.logger, str)
        else list(config.trainer.logger)
    )
    if not names or set(names) - {"console", "file", "wandb"}:
        raise ValueError("Select the admitted console/file/W&B logging backends")
    if type(config.run.prepare_only) is not bool:
        raise TypeError("run.prepare_only must be boolean")
    if (
        type(config.run.cleanup_timeout_seconds) is not int
        or config.run.cleanup_timeout_seconds <= 0
    ):
        raise ValueError("Cleanup timeout must be a positive integer")
    root = Path(config.run.dir).expanduser().resolve()
    for configured, expected in (
        (config.run.source_dir, root / "source"),
        (config.trainer.default_local_dir, root / "checkpoints"),
        (config.trainer.validation_data_dir, root / "validation"),
    ):
        if Path(configured).expanduser().resolve() != expected:
            raise ValueError(
                "Runtime artifacts must belong to the exclusively owned run.dir"
            )
    if config.trainer.val_only:
        raise ValueError(
            "Use a declared evaluation suite; this entry owns a training run"
        )
    if config.trainer.save_freq > 0:
        raise ValueError(
            "Checkpoint publication must be integrated before enabling saving"
        )
    if config.actor_rollout_ref.rollout.policy is None:
        raise ValueError(
            "Stock head-tail execution still requires its migration admission"
        )
    return root


def public_environment(config, root, source_id, run_id, scratch):
    """Serialize only public controls; credentials stay in the inherited environment."""
    source = root / "source"
    return {
        "OPD_RUN_ID": run_id,
        "OPD_SOURCE_ID": source_id,
        "OPD_MANIFEST_PATH": str(root / "manifest.json"),
        "PYTHONPATH": os.pathsep.join([str(source), str(source / "backend/verl")]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(scratch / "pycache"),
        "TOKENIZERS_PARALLELISM": "false",
        "VERL_USE_EXTERNAL_PLUGINS": "none",
        "VLLM_USE_V1": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "RAY_ADDRESS": "local",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        # Ray appends its own /ray directory to RAY_TMPDIR.
        "RAY_TMPDIR": str(scratch),
        "TMPDIR": str(scratch / "tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor"),
        "VERL_FILE_LOGGER_PATH": str(root / "logs/metrics.jsonl"),
        "WANDB_MODE": config.run.wandb_mode,
        "WANDB_RUN_ID": run_id,
        "WANDB_PROJECT": config.trainer.project_name,
        "WANDB_NAME": config.run.name,
        "WANDB_RESUME": "never",
        "WANDB_DIR": str(root / "wandb"),
        "WANDB_CACHE_DIR": str(root / "cache/wandb"),
        "WANDB_DATA_DIR": str(root / "cache/wandb-data"),
    }


def _state(root, run_id, state, **details):
    atomic_json(
        root / "status.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "state": state,
            "updated_epoch_s": time.time(),
            **details,
        },
    )


def prepare(config, *, backend, repository=None, input_preparer=prepare_inputs):
    """Preparation does not enumerate CUDA devices or create a Ray cluster."""
    repository = Path(repository or Path(__file__).resolve().parents[1]).resolve()
    backend = Path(backend).expanduser().resolve()
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if (
        Path(config.run.backend).resolve() != backend
        or Path(config.run.repository).resolve() != repository
    ):
        raise ValueError(
            "Configuration ownership differs from the selected source roots"
        )
    validate_config(config, backend=backend)
    root = validate_application(config)
    root = create_run(root)
    run_id = uuid.uuid4().hex
    with execution_lock(root):
        _state(root, run_id, "preparing")
        events = EventWriter(root / "events", run_id, "prepare")
        events.emit("run.preparing")
        try:
            config.run.dir = str(root)
            config.run.source_dir = str(root / "source")
            # Interpolations were resolved before admission. Rebind every owned
            # output before dispatch changes cwd to the frozen source tree.
            config.trainer.default_local_dir = str(root / "checkpoints")
            config.trainer.validation_data_dir = str(root / "validation")
            scratch_base = (
                Path(config.run.scratch_root).expanduser().resolve()
                if config.run.scratch_root
                else Path("/tmp") / f"opd-state-{os.getuid()}"
            )
            scratch = scratch_base / ("j" + run_id[:12])
            socket = (
                scratch
                / "ray"
                / "session_2099-12-31_23-59-59_999999_4194304"
                / "sockets"
                / "plasma_store"
            )
            if len(os.fsencode(socket)) > 107:
                raise ValueError(
                    "Choose a short scratch_root for Ray's Unix socket paths"
                )
            source = snapshot_sources(repository, backend, root / "source")
            events.context["source_id"] = source["source_id"]
            events.emit("source.frozen", files=len(source["files"]))
            print(
                "Prepared source snapshot; checking model, tokenizer and data inputs",
                flush=True,
            )
            inputs = input_preparer(config, root, backend=backend)
            events.emit(
                "inputs.prepared",
                inputs_id=inputs["inputs_id"],
                populations=inputs["selected_counts"],
            )
            config.reward.custom_reward_function.path = remap_source(
                config.reward.custom_reward_function.path,
                repository,
                backend,
                root / "source",
            )
            config.run.backend = str(root / "source/backend")
            config.run.repository = str(root / "source")
            loggers = (
                [config.trainer.logger]
                if isinstance(config.trainer.logger, str)
                else list(config.trainer.logger)
            )
            if "file" not in loggers:
                loggers.append("file")
            if config.run.wandb_mode == "disabled":
                loggers = [name for name in loggers if name != "wandb"]
            config.trainer.logger = loggers
            environment = public_environment(
                config, root, source["source_id"], run_id, scratch
            )
            with open_dict(config.ray_kwargs.ray_init):
                config.ray_kwargs.ray_init.runtime_env = {"env_vars": environment}
            config_id = configuration_id(config, root, inputs, scratch)
            with open_dict(config.ray_kwargs.ray_init.runtime_env.env_vars):
                config.ray_kwargs.ray_init.runtime_env.env_vars["OPD_CONFIG_ID"] = (
                    config_id
                )
            events.context["config_id"] = config_id
            validate_config(config, backend=config.run.backend)
            atomic_json(root / "inputs.json", inputs)
            from opd_state.artifacts import atomic_text

            atomic_text(root / "resolved.yaml", OmegaConf.to_yaml(config, resolve=True))
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "config_id": config_id,
                "source_id": source["source_id"],
                "inputs_id": inputs["inputs_id"],
                "configuration_sha256": file_digest(root / "resolved.yaml"),
                "inputs_sha256": file_digest(root / "inputs.json"),
                "scratch": str(scratch),
                "runtime": runtime_versions(),
                "source_repository": str(repository),
                "source_backend": str(backend),
            }
            manifest["manifest_id"] = content_id(manifest)
            atomic_json(root / "manifest.json", manifest)
            verify_prepared(root, verify_assets=False)
            _state(root, run_id, "prepared", manifest_id=manifest["manifest_id"])
            events.emit("run.prepared", manifest_id=manifest["manifest_id"])
            return manifest
        except BaseException as error:
            events.emit("run.failed", error_type=type(error).__name__, error=str(error))
            _state(
                root,
                run_id,
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed",
                phase="preparing",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise


def allocated_resources(config):
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise ValueError("Set the allocated CUDA_VISIBLE_DEVICES explicitly")
    devices = visible.split(",")
    if (
        len(set(devices)) != len(devices)
        or len(devices) != config.trainer.n_gpus_per_node
    ):
        raise ValueError("Visible allocation and configured GPU count differ")
    if torch.cuda.device_count() != len(devices) or not torch.cuda.is_available():
        raise ValueError("Allocated CUDA devices cannot initialize this runtime")
    return {
        "visible_devices": visible,
        "gpu_count": len(devices),
        "cuda_runtime": torch.version.cuda,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def run_command(
    command, *, cwd, environment, log_path, cleanup_timeout=20, subreaper=False
):
    child, offset = None, 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def display():
        nonlocal offset
        with log_path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
            offset = stream.tell()
        sys.stdout.write(decoder.decode(data))
        sys.stdout.flush()

    try:
        with log_path.open("xb") as stream:
            child = OwnedProcess.launch(
                command,
                cwd=cwd,
                environment=environment,
                log=stream,
                subreaper=subreaper,
            )
            while child.process.poll() is None:
                child.observe()
                display()
                time.sleep(0.2)
            code = child.process.returncode
        display()
        return code
    finally:
        if child is not None:
            with finish_cleanup():
                child.stop(timeout=cleanup_timeout)


def run_prepared(
    directory,
    *,
    resource_check=allocated_resources,
    command_runner=run_command,
    verifier=verify_run,
    subreaper=False,
):
    """Execute once; terminal failures require a new prepared run directory."""
    root = Path(directory).expanduser().resolve(strict=True)
    with execution_lock(root):
        config, manifest, _inputs = verify_prepared(root)
        run_id = manifest["run_id"]
        status = json.loads((root / "status.json").read_text())
        if status.get("run_id") != run_id or status.get("state") != "prepared":
            raise ValueError("Only an unexecuted prepared run may start")
        if (root / ".executed").exists():
            raise ValueError("This run has already acquired an execution identity")
        validate_application(config)
        if runtime_versions() != manifest["runtime"]:
            raise ValueError(
                "Prepared dependency versions changed; prepare a new run in the selected environment"
            )
        scratch = Path(manifest["scratch"])
        if scratch.name != "j" + run_id[:12]:
            raise ValueError("Scratch ownership does not match this run")
        started_scratch, safe, handlers = False, True, {}
        events = EventWriter(
            root / "events",
            run_id,
            "execution",
            source_id=manifest["source_id"],
            config_id=manifest["config_id"],
        )

        def interrupted(signum, _frame):
            raise InterruptedError(f"Foreground run cancelled by signal {signum}")

        try:
            with (root / ".executed").open("x") as stream:
                json.dump(
                    {
                        "run_id": run_id,
                        "pid": os.getpid(),
                        "process": process_record(os.getpid()),
                    },
                    stream,
                )
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, interrupted)
            scratch.mkdir(parents=True, exist_ok=False)
            started_scratch = True
            for name in ("ray", "tmp", "inductor", "pycache"):
                (scratch / name).mkdir()
            for name in ("wandb", "cache/wandb", "cache/wandb-data"):
                (root / name).mkdir(parents=True, exist_ok=True)
            resources = resource_check(config)
            atomic_json(root / "resources.json", {"run_id": run_id, **resources})
            events.emit("resources.bound", resources=resources)
            environment = {
                **os.environ,
                **dict(config.ray_kwargs.ray_init.runtime_env.env_vars),
            }
            command = [
                sys.executable,
                "-m",
                "verl.trainer.main_ppo",
                "--config-path",
                str(root),
                "--config-name",
                "resolved",
                "hydra.run.dir=" + str(root / "logs"),
                "hydra.output_subdir=null",
                "hydra.job.chdir=false",
            ]
            atomic_json(
                root / "command.json",
                {"run_id": run_id, "command": command, "cwd": str(root / "source")},
            )
            _state(
                root,
                run_id,
                "running",
                pid=os.getpid(),
                process=process_record(os.getpid()),
            )
            events.emit(
                "training.started", command_sha256=file_digest(root / "command.json")
            )
            code = command_runner(
                command,
                cwd=root / "source",
                environment=environment,
                log_path=root / "logs/train.log",
                cleanup_timeout=config.run.cleanup_timeout_seconds,
                subreaper=subreaper,
            )
            atomic_json(
                root / "train-exit.json", {"run_id": run_id, "return_code": code}
            )
            events.emit("training.finished", return_code=code)
            if code:
                raise RuntimeError(
                    f"Backend exited with {code}; inspect logs/train.log"
                )
            _state(root, run_id, "verifying")
            verification = verifier(root)
            atomic_json(root / "verification.json", verification)
            if started_scratch:
                shutil.rmtree(scratch)
                started_scratch = False
            _state(
                root,
                run_id,
                "completed",
                verification_sha256=file_digest(root / "verification.json"),
                cleanup_complete=True,
            )
            events.emit(
                "run.completed",
                verification_sha256=file_digest(root / "verification.json"),
            )
            return verification
        except BaseException as error:
            safe = not isinstance(error, ProcessCleanupError)
            events.emit("run.failed", error_type=type(error).__name__, error=str(error))
            _state(
                root,
                run_id,
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed",
                error_type=type(error).__name__,
                error=str(error),
                cleanup_complete=False,
            )
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if started_scratch and safe:
                try:
                    with finish_cleanup():
                        shutil.rmtree(scratch)
                    value = json.loads((root / "status.json").read_text())
                    value["cleanup_complete"] = True
                    atomic_json(root / "status.json", value)
                    events.emit("cleanup.finished")
                except BaseException as error:
                    _state(
                        root,
                        run_id,
                        "failed_cleanup",
                        cleanup_complete=False,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    raise
