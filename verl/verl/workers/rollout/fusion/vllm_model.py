"""Two native Qwen2 backbones, independent KV and one vLLM sampler.

The outer model returns concatenated hidden states so vLLM can select the same
scheduled positions from both models before computing and mixing their logits.
Teacher attention names use disjoint global layer indices for vLLM 0.10's KV
binder; checkpoint and PyTorch parameter names retain the original local indices.
"""

from __future__ import annotations

import copy
import re
from itertools import groupby

import torch
from torch import nn
from transformers import AutoConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class NativeQwen2Fusion(nn.Module):
    packed_modules_mapping = Qwen2ForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Native exact fusion currently requires TP=PP=1")
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError("Native exact fusion requires dense full weights")
        config = vllm_config.model_config.hf_config
        self.config = config
        self.teacher_weight = float(config.opd_teacher_weight)
        if not 0 < self.teacher_weight < 1:
            raise ValueError("Compile fusion endpoints to a single model before loading")
        self.hidden_size = int(config.hidden_size)
        self.base = Qwen2ForCausalLM(vllm_config=vllm_config, prefix="base")

        teacher_config = copy.copy(vllm_config)
        teacher_config.model_config = copy.copy(vllm_config.model_config)
        teacher_hf = AutoConfig.from_pretrained(config.opd_teacher_path, local_files_only=True)
        if teacher_hf.model_type != "qwen2":
            raise ValueError("Native paired teacher must be a Qwen2 checkpoint")
        teacher_config.model_config.hf_config = teacher_hf
        self.teacher = Qwen2ForCausalLM(vllm_config=teacher_config, prefix="teacher")

        context = vllm_config.compilation_config.static_forward_context
        offset = int(config.num_hidden_layers)
        for old in list(context):
            if not old.startswith("teacher.model.layers."):
                continue
            match = re.fullmatch(r"teacher\.model\.layers\.(\d+)(\..+)", old)
            if match is None:
                raise RuntimeError("Unexpected teacher attention layer name")
            new = f"teacher.model.layers.{offset + int(match[1])}{match[2]}"
            if new in context:
                raise RuntimeError("Paired attention layer indices overlap")
            layer = context.pop(old)
            layer.layer_name = new
            context[new] = layer
        self.secondary_weights = [
            DefaultModelLoader.Source(str(config.opd_teacher_path), revision=None, prefix="teacher.")
        ]

    @property
    def model(self):
        # Legacy loader inspection expects a decoder here. A property avoids
        # registering aliases that duplicate checkpoint parameter names.
        return self.base.model

    def get_input_embeddings(self, input_ids):
        return self.base.get_input_embeddings(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("Native paired MVP accepts token IDs and a single pipeline stage")
        base = self.base(input_ids, positions)
        teacher = self.teacher(input_ids, positions)
        return torch.cat((base, teacher), dim=-1)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        base, teacher = hidden_states.split(self.hidden_size, dim=-1)
        base_logits = self.base.compute_logits(base, sampling_metadata).float()
        teacher_logits = self.teacher.compute_logits(teacher, sampling_metadata).float()
        return base_logits.mul_(1 - self.teacher_weight).add_(teacher_logits, alpha=self.teacher_weight)

    def load_weights(self, weights):
        # The native default loader streams the primary checkpoint followed by
        # secondary teacher weights. Actor updates stream only primary names.
        loaded = set()
        for is_teacher, group in groupby(weights, key=lambda item: item[0].startswith("teacher.")):
            if is_teacher:
                names = self.teacher.load_weights((name.removeprefix("teacher."), tensor) for name, tensor in group)
                loaded.update("teacher." + name for name in names)
            else:
                names = self.base.load_weights(group)
                loaded.update("base." + name for name in names)
        return loaded
