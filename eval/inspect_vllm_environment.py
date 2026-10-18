"""Inspect an isolated vLLM/plugin installation with every GPU hidden.

This validates imports, registration and import-time feature configuration.
It does not construct a model, initialize CUDA, or prove GPU/kernel support.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch


def package_snapshot():
    return {
        distribution.metadata["Name"]: {
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256((distribution.read_text("METADATA") or "").encode()).hexdigest(),
        }
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }


def inspect(expected_version, *, plugin=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES='' so this import inspection cannot use a GPU")
    import torch
    import vllm

    if torch.cuda.is_initialized() or vllm.__version__ != expected_version:
        raise ValueError("Inspection requires an uninitialized CUDA runtime and the exact requested vLLM version")
    report = {
        "scope": "CPU import/registration/configuration only; no CUDA context or model inference",
        "python": sys.version,
        "prefix": sys.prefix,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "module_paths": {"vllm": vllm.__file__, "torch": torch.__file__},
        "packages": package_snapshot(),
        "batch_invariant_requested": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
    }
    try:
        invariant = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    except ModuleNotFoundError as error:
        if error.name != "vllm.model_executor.layers.batch_invariant":
            raise
        report["batch_invariance"] = {"recognized_implementation": False}
    else:
        requested = invariant.vllm_is_batch_invariant()
        previous = os.environ.get("VLLM_BATCH_INVARIANT")
        try:
            os.environ["VLLM_BATCH_INVARIANT"] = "0" if requested else "1"
            after_environment_change = invariant.vllm_is_batch_invariant()
        finally:
            if previous is None:
                os.environ.pop("VLLM_BATCH_INVARIANT", None)
            else:
                os.environ["VLLM_BATCH_INVARIANT"] = previous
        report["batch_invariance"] = {
            "recognized_implementation": True,
            "import_time_flag": invariant.vllm_is_batch_invariant(),
            "flag_after_late_environment_change": after_environment_change,
            "import_time_configuration_cached": requested == after_environment_change,
            "kernel_overrides_initialized": bool(invariant._batch_invariant_MODE),
            "source_path": invariant.__file__,
            "source_sha256": hashlib.sha256(Path(invariant.__file__).read_bytes()).hexdigest(),
        }
    if plugin:
        registration = importlib.import_module("vllm_logits.register")
        registration.register_dual_qwen()
        compatibility = importlib.import_module("vllm_logits._compat")
        backbone = importlib.import_module("vllm_logits.backbones")
        registered = "DualQwen2ForCausalLM" in compatibility.ModelRegistry.get_supported_archs()
        if not registered or not issubclass(backbone.DualQwen2ForCausalLM, torch.nn.Module):
            raise ValueError("Dual Qwen plugin did not register its actual native model class")
        report["plugin"] = {
            "version": metadata.version("vllm-logits"),
            "registered": registered,
            "entry_points": {entry.name: entry.value for entry in metadata.entry_points(group="vllm.general_plugins")},
            "files_sha256": {
                module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (registration, compatibility, backbone)
            },
        }
        # Execute the actual loader on a one-parameter CPU module while its
        # checkpoint iterators are empty. This isolates its incoming-stream
        # contract; it neither constructs native backbones nor loads GPU weights.
        model = backbone.DualQwen2ForCausalLM.__new__(backbone.DualQwen2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.register_parameter("sentinel", torch.nn.Parameter(torch.zeros(1)))
        model.layers_a = 28
        observed = {"incoming_iterated": False, "parameter_writes": 0}

        class Incoming:
            def __iter__(self):
                observed["incoming_iterated"] = True
                yield "sentinel", torch.ones(1)

        def loader(parameter, value):
            observed["parameter_writes"] += 1
            parameter.data.copy_(value)

        def empty_iterator(*args):
            return iter(())

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"VLLM_LOGIT_MIX_ANCESTOR": directory, "VLLM_LOGIT_MIX_SPECIALIST": directory}),
            patch.object(compatibility, "get_weight_iterators", return_value=(empty_iterator, empty_iterator)),
            patch.object(compatibility, "get_default_weight_loader", return_value=loader),
        ):
            claimed = model.load_weights(Incoming())
        report["plugin"]["weight_stream_probe"] = {
            "scope": "Actual loader method with stubbed empty CPU checkpoint iterators; no native GPU update",
            **observed,
            "claimed_loaded_names": sorted(claimed),
            "sentinel_value_after": model.sentinel.item(),
            "incoming_actor_stream_consumed": observed["incoming_iterated"],
        }
    driver = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    code = driver.cuDriverGetVersion(ctypes.byref(version))
    report["driver_api"] = {
        "query_exit_code": code,
        "version": version.value,
        "mapped_libraries": sorted(
            {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcuda.so" in line}
        ),
        "scope": "cuDriverGetVersion without cuInit; does not establish runtime/kernel compatibility",
    }
    report["cuda_initialized_after"] = torch.cuda.is_initialized()
    if report["cuda_initialized_after"]:
        raise RuntimeError("CPU environment inspection initialized CUDA")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--plugin", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = {"status": "cpu_import_passed", **inspect(args.expected_version, plugin=args.plugin)}
    except Exception as error:
        report = {"status": "cpu_import_failed", "error": str(error), "traceback": traceback.format_exc()}
    report["inspector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "vllm", "torch", "error") if key in report}))
    return int(report["status"] != "cpu_import_passed")


if __name__ == "__main__":
    raise SystemExit(main())
