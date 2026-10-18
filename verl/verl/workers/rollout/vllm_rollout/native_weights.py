"""Named-tensor stream ownership checks in the real vLLM inference worker."""

from pathlib import Path


def configure_source(worker, role, path, fingerprint, sync_namespace=""):
    from safetensors import safe_open

    from ..policy import ModelRef

    path = str(Path(path).resolve())
    actual = str(Path(worker.model_runner.vllm_config.model_config.model).resolve())
    if actual != path:
        raise RuntimeError("Native model path does not match the declared source")
    if role == "current_student":
        ModelRef(role, path, version=0)
    else:
        ModelRef(role, path, fingerprint=fingerprint)
    files = sorted(Path(path).glob("*.safetensors"))
    names = set()
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in names:
                    raise RuntimeError("Checkpoint repeats a parameter across safetensor shards")
                names.add(name)
    if not names:
        raise ValueError("Native source requires a local dense safetensors checkpoint")
    worker._native_generation_role = role
    worker._native_sync_namespace = sync_namespace
    worker._native_generation_path = path
    worker._native_generation_fingerprint = fingerprint
    worker._native_expected_weight_names = names
    worker._native_actor_version = None
    worker._native_weight_write_serial = 0
    return {"role": role, "path": path, "parameter_count": len(names)}


def check_actor_stream(worker, names, received, *, final):
    role = getattr(worker, "_native_generation_role", None)
    if role is None:
        return
    if role != "current_student":
        raise RuntimeError("Frozen native generation workers reject actor weight writes")
    expected = worker._native_expected_weight_names
    for name in names:
        if name not in expected or name in received:
            raise RuntimeError(f"Actor stream contains unexpected or duplicate parameter: {name}")
        received.add(name)
    if final and received != expected:
        raise RuntimeError(f"Actor stream is incomplete: {len(expected - received)} missing parameters")
