"""Estimate validation tensor capacity without importing Torch or reserving GPUs.

KV estimates assume TP=1, full-attention replicas and no shared prefixes. The
caller supplies non-KV/other-process margins; neither CUDA Graph workspace nor
FSDP's transient gathers can be inferred exactly from a model config.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20


def checkpoint_size(path):
    """Read safetensors headers only, retaining distinct parameter identities."""
    parameters, stored_bytes, names = 0, 0, set()
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise ValueError("Memory estimation requires local safetensors weights")
    for file in files:
        with file.open("rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Invalid safetensors header: " + str(file))
            size = struct.unpack("<Q", header)[0]
            if not 2 <= size <= min(file.stat().st_size - 8, 64 * MIB):
                raise ValueError("Invalid safetensors metadata length")
            tensors = json.loads(stream.read(size))
        for name, tensor in tensors.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError("Repeated checkpoint parameter: " + name)
            names.add(name)
            shape, offsets = tensor["shape"], tensor["data_offsets"]
            if any(type(value) is not int or value < 0 for value in shape):
                raise ValueError("Invalid checkpoint parameter shape")
            if len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= file.stat().st_size - size - 8:
                raise ValueError("Invalid checkpoint tensor offsets")
            parameters += math.prod(shape)
            stored_bytes += offsets[1] - offsets[0]
    return {"parameters": parameters, "checkpoint_stored_bytes": stored_bytes, "tensor_count": len(names)}


def estimate(
    config,
    parameter_count,
    *,
    gpus=4,
    memory_mib=81559,
    prompt_batch=32,
    samples=16,
    max_sequences=64,
    memory_budget=0.55,
    context_cap=32768,
    mean_context=6301,
    effective_vocabulary=151665,
    position_chunk=1024,
    forward_batch_size=1,
    vocabulary_chunk=151665,
    weight_bytes=2,
    kv_bytes=2,
    non_kv_margin_gib=4,
    other_process_gib=0,
):
    """Return component sizes and capacity checks, not a predicted whole-card peak."""
    values = (
        parameter_count,
        gpus,
        memory_mib,
        prompt_batch,
        samples,
        max_sequences,
        context_cap,
        mean_context,
        effective_vocabulary,
        position_chunk,
        forward_batch_size,
        vocabulary_chunk,
        weight_bytes,
        kv_bytes,
    )
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Validation memory dimensions must be positive integers")
    if (
        type(memory_budget) not in (int, float)
        or not 0 < memory_budget < 1
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in (non_kv_margin_gib, other_process_gib)
        )
        or mean_context > context_cap
    ):
        raise ValueError("Invalid validation memory budget, margin or mean context")
    if config.get("use_sliding_window", False):
        raise ValueError("This capacity estimate requires full attention")
    for name in ("hidden_size", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError("Model dimensions must be positive integers: " + name)
    if weight_bytes not in (2, 4) or kv_bytes not in (1, 2, 4):
        raise ValueError("Unsupported inference weight/KV element size")
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    dimension = config.get("head_dim") or hidden // heads
    if type(dimension) is not int or dimension < 1 or hidden != heads * dimension:
        raise ValueError("Attention dimensions do not match")
    layers, kv_heads, padded_vocab = config["num_hidden_layers"], config["num_key_value_heads"], config["vocab_size"]
    if effective_vocabulary > padded_vocab:
        raise ValueError("Effective vocabulary exceeds the output head")
    capacity = memory_mib * MIB
    weights = parameter_count * weight_bytes
    kv_per_position = 2 * layers * kv_heads * dimension * kv_bytes
    queued = math.ceil(prompt_batch * samples / gpus)
    active = min(queued, max_sequences)
    reserve = int(memory_budget * capacity)
    available_kv = max(0, reserve - weights - int(non_kv_margin_gib * GIB))
    padded_logits = 2 * position_chunk * padded_vocab * weight_bytes
    # Upper bookkeeping bound for FP32 log-probabilities/probabilities,
    # differences, cast/reduction temporaries; not allocator/kernel profiling.
    reduction = 32 * position_chunk * min(vocabulary_chunk, effective_vocabulary)
    head_snapshot = padded_vocab * hidden * weight_bytes
    hidden_states = 2 * forward_batch_size * context_cap * hidden * weight_bytes
    return {
        "scope": "Tensor/capacity estimates, TP=1 replicas; not measured allocator or whole-card peaks",
        "gpus": gpus,
        "parameters": parameter_count,
        "inference_weights_bytes_per_replica": weights,
        "kv_bytes_per_sequence_position": kv_per_position,
        "generation": {
            "responses_per_submission": prompt_batch * samples,
            "max_queued_responses_per_gpu": queued,
            "max_active_responses_per_gpu": active,
            "instance_memory_budget_bytes": reserve,
            "assumed_non_kv_margin_bytes": int(non_kv_margin_gib * GIB),
            "kv_capacity_bytes_after_assumed_margin": available_kv,
            "kv_capacity_positions_after_assumed_margin": available_kv // kv_per_position,
            "kv_bytes_at_assumed_mean_context": active * mean_context * kv_per_position,
            "kv_bytes_if_all_active_reach_cap": active * context_cap * kv_per_position,
            "max_simultaneous_at_context_cap": available_kv // (context_cap * kv_per_position),
            "all_active_fit_at_cap_under_assumptions": active * context_cap * kv_per_position <= available_kv,
            "reservation_plus_other_process_bytes": reserve + int(other_process_gib * GIB),
            "reservation_and_other_processes_fit": reserve + int(other_process_gib * GIB) < capacity,
        },
        "kl": {
            "position_chunk": position_chunk,
            "forward_batch_size": forward_batch_size,
            "vocabulary_chunk": min(vocabulary_chunk, effective_vocabulary),
            "two_bf16_or_fp32_logit_blocks_bytes": padded_logits,
            "fp32_reduction_workspace_upper_bytes": reduction,
            "student_head_snapshot_bytes": head_snapshot,
            "two_selected_hidden_upper_bytes": hidden_states,
            "two_full_inference_weight_copies_bytes": 2 * weights,
            "listed_components_bytes": 2 * weights + padded_logits + reduction + head_snapshot + hidden_states,
        },
        "excluded_or_assumed": [
            "Actual vLLM non-KV profiling, CUDA Graph pools and other-process allocations",
            "KV page rounding, shared-prefix savings and preemption/recompute",
            "FSDP sharding, parameter precision, transient gathers and offload transfers",
            "Backbone activation/workspace and allocator fragmentation",
            "No throughput or eight-GPU measurement is implied",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--memory-mib", type=int, default=81559)
    parser.add_argument("--prompt-batch", type=int, default=32)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--memory-budget", type=float, default=0.55)
    parser.add_argument("--context-cap", type=int, default=32768)
    parser.add_argument("--mean-context", type=int, default=6301)
    parser.add_argument("--effective-vocabulary", type=int, default=151665)
    parser.add_argument("--position-chunk", type=int, default=1024)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--vocabulary-chunk", type=int, default=151665)
    parser.add_argument("--non-kv-margin-gib", type=float, default=4)
    parser.add_argument("--other-process-gib", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    size = checkpoint_size(args.model)
    result = estimate(
        json.loads((args.model / "config.json").read_text()),
        size["parameters"],
        **{name: value for name, value in vars(args).items() if name not in {"model", "output"}},
    )
    result["checkpoint"] = size
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
