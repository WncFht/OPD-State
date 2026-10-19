"""Named-tensor stream ownership checks in the real vLLM inference worker."""

from pathlib import Path


def parameter_addresses(module):
    import hashlib

    fields = [
        (name, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        for name, tensor in sorted(module.named_parameters())
    ]
    return hashlib.sha256(repr(fields).encode()).hexdigest()


def parameter_digest(module):
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen_digests(worker):
    return {role: parameter_digest(model) for role, model in getattr(worker, "_native_frozen_modules", {}).items()}


def configure_source(
    worker, role, path, fingerprint, sync_namespace="", fusion=None, verify_frozen=False, verify_addresses=False
):
    from safetensors import safe_open

    from opd_state.core.policy import ModelRef

    path = str(Path(path).resolve())
    actual = str(Path(worker.model_runner.vllm_config.model_config.model).resolve())
    if actual != path:
        raise RuntimeError("Native model path does not match the declared source")
    if role == "current_student":
        ModelRef(role, path, version=0)
    else:
        ModelRef(role, path, fingerprint=fingerprint)
    model = worker.model_runner.model
    load_format = worker.model_runner.vllm_config.load_config.load_format
    if (role != "current_student" or fusion is not None) and str(getattr(load_format, "value", load_format)) == "dummy":
        raise RuntimeError("Frozen native weights require checkpoint loading, not dummy initialization")
    companion = None
    if fusion is not None:
        companion = ModelRef(
            "teacher", str(Path(fusion["teacher_path"]).resolve()), fingerprint=fusion["teacher_fingerprint"]
        )
        if (
            type(model).__name__ != "NativeQwen2Fusion"
            or str(Path(model.config.opd_teacher_path).resolve()) != companion.path
            or model.config.opd_teacher_fingerprint != companion.fingerprint
            or model.teacher_weight != fusion["teacher_weight"]
        ):
            raise RuntimeError("Native paired worker does not match its declared teacher and coefficient")
    elif type(model).__name__ == "NativeQwen2Fusion":
        raise RuntimeError("Native paired worker requires a declared frozen companion")
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
    worker._native_frozen_modules = {}
    worker._native_verify_addresses = verify_addresses
    if verify_frozen:
        if fusion is not None:
            worker._native_frozen_modules["teacher"] = model.teacher
        if role != "current_student":
            worker._native_frozen_modules[role] = model.base if fusion is not None else model
    return {
        "role": role,
        "path": path,
        "parameter_count": len(names),
        "fusion": fusion,
        "companion": companion.identity if companion is not None else None,
        "frozen_digests": frozen_digests(worker),
        "parameter_addresses": parameter_addresses(model) if verify_addresses else None,
    }


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
