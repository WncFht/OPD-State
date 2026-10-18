"""Shared-prefix student/teacher logit fusion for training rollouts."""


def register_native_fusion():
    from vllm import ModelRegistry

    if "OPDNativeQwen2Fusion" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("OPDNativeQwen2Fusion", "verl.workers.rollout.fusion.vllm_model:NativeQwen2Fusion")
