"""Measure native batch/order/re-prefill effects in isolated vLLM stacks.

Only native vLLM generates tokens. A fixed dual-plugin probe is optional and
does not support actor weight updates. Batch invariance and same-model splitting
are separate observations; this is not a math accuracy or training experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

from native_devices import matches_device, query_devices
from run_native_acceptance import gpu_rows


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_responses(reference, observed):
    if set(reference) != set(observed) or not reference:
        raise ValueError("Batch comparison requires the same nonempty prompt identities")
    divergences = {}
    for uid, expected in reference.items():
        actual = observed[uid]
        if not expected or not actual:
            raise ValueError("Batch comparison cannot hide empty native responses")
        if expected != actual:
            position = next(
                (i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
                min(len(expected), len(actual)),
            )
            divergences[uid] = position
    return {
        "responses": len(reference),
        "exact_responses": len(reference) - len(divergences),
        "first_divergence": divergences,
    }


def worker_info(worker):
    import torch
    from vllm.compilation.counter import compilation_counter

    model = worker.model_runner.model
    context = worker.model_runner.vllm_config.compilation_config.static_forward_context
    regions, backends = [], set()
    for name, layer in context.items():
        backends.add(type(getattr(layer, "impl", layer)).__name__)
        cache = getattr(layer, "kv_cache", ())
        if isinstance(cache, torch.Tensor):
            cache = (cache,)
        for tensor in cache:
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                regions.append(
                    {"layer": name, "pointer": tensor.data_ptr(), "bytes": tensor.numel() * tensor.element_size()}
                )
    try:
        invariant = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    except ModuleNotFoundError as error:
        if error.name != "vllm.model_executor.layers.batch_invariant":
            raise
        flag = initialized = False
    else:
        flag, initialized = invariant.vllm_is_batch_invariant(), bool(invariant._batch_invariant_MODE)
    return {
        "device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        "pid": os.getpid(),
        "model_class": type(model).__name__,
        "batch_flag": flag,
        "kernel_overrides_initialized": initialized,
        "attention_implementations": sorted(backends),
        "kv_regions": regions,
        "cuda_graphs_captured": compilation_counter.num_cudagraph_captured,
        "cuda_graph_enabled": compilation_counter.num_cudagraph_captured > 0,
    }


def full_vocab_distance(expected, observed):
    import numpy as np

    if (
        expected.shape != observed.shape
        or expected.ndim != 2
        or not np.isfinite(expected).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("Full-vocabulary comparison needs matching finite state/vocabulary arrays")

    def log_probability(values):
        values = values.astype(np.float64)
        shifted = values - values.max(-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    logp, logq = log_probability(expected), log_probability(observed)
    p, q = np.exp(logp), np.exp(logq)
    return {
        "max_logit_error": float(np.abs(expected - observed).max()),
        "max_tv": float(np.abs(p - q).sum(-1).max() / 2),
        "max_reference_observed_kl": float((p * (logp - logq)).sum(-1).max()),
        "max_observed_reference_kl": float((q * (logq - logp)).sum(-1).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-vllm", required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--boundary", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.boundary < args.cap <= 4096 or args.repeats < 2 or not 0 <= args.teacher_weight <= 1:
        parser.error("Use an internal boundary, cap <=4096, at least two replays and a valid fusion weight")
    if metadata.version("vllm") != args.expected_vllm:
        raise ValueError("Probe must execute inside the exact requested vLLM environment")
    if any(used > 128 for _, used, _ in gpu_rows()):
        raise RuntimeError("Wait until every owned physical GPU 0–3 is idle")
    devices = query_devices()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(devices[args.gpu]["cuda_ordinal"]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_BATCH_INVARIANT=str(int(args.batch_invariant)),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
    )
    cache = args.output / "compiler-cache"
    os.environ.update(
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
    )
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    raw = args.prompts.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()[:16]]
    if len(records) != 16 or len({row["prompt_id"] for row in records}) != 16:
        raise ValueError("Version probes require sixteen distinct canonical training prompts")
    for row in records:
        if (
            not row["prompt_ids"]
            or len(row["prompt_ids"]) + args.cap > 5120
            or any(type(token) is not int or not 0 <= token < len(tokenizer) for token in row["prompt_ids"])
        ):
            raise ValueError("Probe prompt violates the common token/context budget")
    (args.output / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    layers = config.num_hidden_layers
    paths = [model_path]
    if args.teacher:
        teacher = args.teacher.resolve()
        other = AutoConfig.from_pretrained(teacher, local_files_only=True)
        other_tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
        if tokenizer.get_vocab() != other_tokenizer.get_vocab() or any(
            getattr(config, name) != getattr(other, name)
            for name in (
                "hidden_size",
                "vocab_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "rope_theta",
            )
        ):
            raise ValueError("Dual probe requires the accepted Qwen2 architecture and shared token mapping")
        import vllm_logits.register

        vllm_logits.register.register_dual_qwen()
        os.environ.update(
            VLLM_PLUGINS="vllm_logits_register",
            VLLM_LOGIT_MIX_ANCESTOR=str(model_path),
            VLLM_LOGIT_MIX_SPECIALIST=str(teacher),
            VLLM_LOGIT_MIX_ALPHA=str(args.teacher_weight),
            VLLM_LOGIT_MIX_A_LAYERS=str(layers),
            VLLM_LOGIT_MIX_MODE="logit",
            VLLM_LOGIT_MIX_PROTECT_IDS="",
        )
        dual = args.output / "dual-config"
        config.architectures, config.num_hidden_layers = ["DualQwen2ForCausalLM"], 2 * layers
        config.save_pretrained(dual)
        for file in model_path.glob("*.safetensors*"):
            (dual / file.name).symlink_to(file)
        paths.append(teacher)
        model_path = dual
    fingerprints = {
        str(path): {
            file.name: digest(file)
            for file in sorted(path.iterdir())
            if file.is_file() and (file.suffix == ".safetensors" or file.name == "config.json")
        }
        for path in paths
    }
    if args.batch_invariant:
        module = importlib.import_module("vllm.model_executor.layers.batch_invariant")
        if not module.vllm_is_batch_invariant():
            raise RuntimeError("The vLLM batch-invariance module did not read the pre-import flag")
    kwargs = {"attention_config": {"backend": "FLASH_ATTN"}} if args.expected_vllm.startswith("0.15.") else {}
    if not kwargs:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    options = dict(
        model=str(model_path),
        tokenizer=str(args.model.resolve()),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=not args.graph,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
        max_model_len=5120,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_logprobs=len(tokenizer),
        logprobs_mode="raw_logits",
        **kwargs,
    )
    if args.graph:
        options["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16], "custom_ops": ["all"]}
    (args.output / "command.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "devices": devices,
                "checkpoint_files_sha256": fingerprints,
                "prompts_sha256": hashlib.sha256(raw).hexdigest(),
                "driver_sha256": digest(Path(__file__)),
                "device_helper_sha256": digest(Path(__file__).with_name("native_devices.py")),
                "owned_probe_tag": os.environ.get("OPD_VERSION_PROBE_ID"),
                "process_group": os.getpgrp(),
            },
            default=str,
            indent=2,
        )
        + "\n"
    )
    llm = None
    observations = []
    started = time.perf_counter()
    try:
        llm = LLM(**options)
        cold = time.perf_counter() - started
        info = llm.collective_rpc(worker_info)[0]
        (args.output / "worker.json").write_text(json.dumps(info, indent=2) + "\n")
        if (
            not matches_device(info["device_uuid"], devices[args.gpu])
            or info["batch_flag"] != args.batch_invariant
            or (args.batch_invariant and not info["kernel_overrides_initialized"])
        ):
            raise RuntimeError("Actual worker GPU/feature initialization disagrees with the requested probe")
        regions = sorted(info["kv_regions"], key=lambda row: row["pointer"])
        if len(regions) != layers * (2 if args.teacher else 1) or any(
            a["pointer"] + a["bytes"] > b["pointer"] for a, b in zip(regions, regions[1:], strict=False)
        ):
            raise RuntimeError("Probe model lacks the complete independent KV regions")
        if args.graph and not info["cuda_graph_enabled"]:
            raise RuntimeError("Requested CUDA Graph probe did not capture native graphs")
        allowed = list(range(len(tokenizer)))

        def generate(rows, cap, *, prefixes=None, scores=False):
            prompts = [list(row["prompt_ids"] if prefixes is None else prefixes[row["prompt_id"]]) for row in rows]
            output = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompts],
                SamplingParams(
                    max_tokens=cap,
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    seed=42,
                    allowed_token_ids=allowed,
                    logprobs=len(tokenizer) if scores else None,
                ),
                use_tqdm=False,
            )
            if len(output) != len(rows) or any(
                list(value.prompt_token_ids) != ids for value, ids in zip(output, prompts, strict=True)
            ):
                raise ValueError("Native version probe changed request cardinality or original token prefixes")
            result = {}
            for row, value in zip(rows, output, strict=True):
                ids = list(value.outputs[0].token_ids)
                if not ids or len(ids) > cap or any(not 0 <= token < len(tokenizer) for token in ids):
                    raise ValueError("Native version probe produced invalid effective tokens")
                if tokenizer.eos_token_id in ids[:-1]:
                    raise ValueError("Native version probe continued after EOS")
                result[row["prompt_id"]] = ids
            return result, output

        generate(records[:2], 8)
        reference = None
        for size, reverse in ((1, False), (4, False), (16, False), (16, True)):
            ordered = list(reversed(records)) if reverse else records
            for repeat in range(args.repeats):
                tick = time.perf_counter()
                result = {}
                for offset in range(0, len(ordered), size):
                    result.update(generate(ordered[offset : offset + size], args.cap)[0])
                elapsed = time.perf_counter() - tick
                if reference is None:
                    reference = result
                row = {
                    "batch": size,
                    "reverse": reverse,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "tokens": sum(map(len, result.values())),
                    "responses": result,
                    "comparison": compare_responses(reference, result),
                }
                observations.append(row)
                with (args.output / "observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
        first, _ = generate(records, args.boundary)
        remaining = [
            row
            for row in records
            if first[row["prompt_id"]][-1] != tokenizer.eos_token_id and len(first[row["prompt_id"]]) == args.boundary
        ]
        prefixes = {row["prompt_id"]: row["prompt_ids"] + first[row["prompt_id"]] for row in remaining}
        second = generate(remaining, args.cap - args.boundary, prefixes=prefixes)[0] if remaining else {}
        split = {row["prompt_id"]: first[row["prompt_id"]] + second.get(row["prompt_id"], []) for row in records}
        split_path = args.output / "split.json"
        split_path.write_text(
            json.dumps({"first": first, "second": second, "second_prefixes": prefixes, "responses": split}, indent=2)
            + "\n"
        )
        logit_checks = []
        logit_reference = None
        for name, order, batch in (
            ("single", records[:2], 1),
            ("paired", records[:2], 2),
            ("reversed", list(reversed(records[:2])), 2),
        ):
            arrays = {}
            for offset in range(0, len(order), batch):
                selected = order[offset : offset + batch]
                _, values = generate(selected, 1, scores=True)
                for record, value in zip(selected, values, strict=True):
                    logits = value.outputs[0].logprobs[0]
                    arrays[record["prompt_id"]] = np.array(
                        [logits[token].logprob for token in range(len(tokenizer))], dtype=np.float32
                    )
            values = np.stack([arrays[row["prompt_id"]] for row in records[:2]])
            path = args.output / f"logits-{name}.npy"
            np.save(path, values)
            if logit_reference is None:
                logit_reference = values
            logit_checks.append(
                {
                    "name": name,
                    "file": path.name,
                    "sha256": digest(path),
                    **full_vocab_distance(logit_reference, values),
                }
            )
        report = {
            "scope": "Native greedy inference diagnostics; not training quality or actor hot-update acceptance",
            "vllm": metadata.version("vllm"),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_vocab_sha256": hashlib.sha256(
                json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "eos_token_id": tokenizer.eos_token_id,
            "expected_kv_layers": layers * (2 if args.teacher else 1),
            "prompt_ids": [row["prompt_id"] for row in records],
            "probe_prefixes": [row["prompt_ids"] for row in records[:2]],
            "cold_seconds": cold,
            "worker": info,
            "batch_invariance_requested": args.batch_invariant,
            "all_batch_order_replays_equal": all(row["comparison"]["exact_responses"] == 16 for row in observations),
            "same_model_split": compare_responses(reference, split),
            "full_vocab_checks": logit_checks,
            "observations_sha256": digest(args.output / "observations.jsonl"),
            "inputs_sha256": digest(args.output / "inputs.json"),
            "split_sha256": digest(split_path),
            "worker_sha256": digest(args.output / "worker.json"),
            "checkpoint_files_sha256": fingerprints,
            "plugin_teacher_weight": args.teacher_weight if args.teacher else None,
            "limitations": [
                "Two fixed states do not prove equivalence on every prefix",
                "Stack comparison also changes Torch/CUDA",
                "Plugin incoming actor-stream support is not implemented",
                "Same-model splitting and batch/order invariance are separate checks",
            ],
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "vllm": report["vllm"],
                    "batch_order_equal": report["all_batch_order_replays_equal"],
                    "split_equal": report["same_model_split"]["exact_responses"],
                }
            )
        )
        return 2 if args.batch_invariant and not report["all_batch_order_replays_equal"] else 0
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
