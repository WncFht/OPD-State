"""Measure shared-prefix dual-model decoding on one explicitly assigned GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def compare_prefix_logits(student, teacher, prompts, *, tokenizer, prompt_cap, response_cap, flash_kv=False, steps=8):
    import torch
    from transformers import DynamicCache

    from verl.workers.rollout.fusion.decoder import LogitFusionDecoder, StaticDecodeContext, fuse_logits

    reference = LogitFusionDecoder(
        student,
        teacher,
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        prompt_cap=prompt_cap,
        response_cap=response_cap,
    )
    ids, mask, positions = reference._inputs(prompts)
    device = ids.device
    bsz, width = ids.shape
    capacity = prompt_cap + response_cap
    valid = torch.zeros((bsz, capacity), dtype=torch.bool, device=device)
    valid[:, :width] = mask.bool()
    caches = [DynamicCache(config=model.config) for model in (student, teacher)]
    static = StaticDecodeContext(student, teacher, bsz, capacity, capture=False, flash_kv=flash_kv)
    graph = StaticDecodeContext(student, teacher, bsz, capacity, capture=True, flash_kv=flash_kv)
    with torch.inference_mode():
        dynamic_logits = [
            model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                cache_position=torch.arange(width, device=device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1]
            for model, cache in zip((student, teacher), caches, strict=True)
        ]
        static_logits = static.prefill(ids, mask, positions)
        graph_logits = graph.prefill(ids, mask, positions)
        differences, variation, graph_static = [], [], []
        for step in range(steps):
            for ref, a, b in zip(dynamic_logits, static_logits, graph_logits, strict=True):
                ref, a, b = [value[:, : len(tokenizer)] for value in (ref, a, b)]
                differences.append(float((ref.float() - b.float()).abs().max()))
                variation.append(float((ref.float().softmax(-1) - b.float().softmax(-1)).abs().sum(-1).max() / 2))
                graph_static.append(float((a.float() - b.float()).abs().max()))
            chosen = fuse_logits(
                dynamic_logits[0][:, : len(tokenizer)], dynamic_logits[1][:, : len(tokenizer)], 0.5
            ).argmax(-1)[:, None]
            valid[:, width + step] = True
            logical_positions = (mask.sum(-1) + step)[:, None]
            static_logits = static.decode(chosen, valid, logical_positions, width + step)
            graph_logits = graph.decode(chosen, valid, logical_positions, width + step)
            dynamic_logits = [
                model(
                    input_ids=chosen,
                    attention_mask=valid[:, : width + step + 1].long(),
                    position_ids=logical_positions,
                    cache_position=torch.tensor([width + step], device=device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits[:, -1]
                for model, cache in zip((student, teacher), caches, strict=True)
            ]
    result = {
        "max_logit_error_graph_vs_dynamic": max(differences),
        "max_distribution_tv_graph_vs_dynamic": max(variation),
        "max_logit_error_graph_vs_static": max(graph_static),
        "tested_prefix_steps": steps,
        "batch": len(prompts),
        "logit_errors_by_step_and_model": differences,
        "distribution_tv_by_step_and_model": variation,
    }
    print(json.dumps({"parity": result}), flush=True)
    fp32 = next(student.parameters()).dtype == torch.float32
    # Independent FP32 and raw-attention checks distinguish masking mistakes
    # from accumulated BF16 rounding across different attention kernels.
    tolerance = 1e-5 if fp32 else 0.05
    if result["max_distribution_tv_graph_vs_dynamic"] > tolerance:
        raise AssertionError(f"Graph probability drift exceeds the BF16 parity gate: {result}")
    if result["max_logit_error_graph_vs_dynamic"] > (1e-4 if fp32 else 0.5):
        raise AssertionError(f"Graph logit drift exceeds the precision gate: {result}")
    if result["max_logit_error_graph_vs_static"] > 1e-5:
        raise AssertionError(f"Captured and eager versions of the same kernel disagree: {result}")
    del reference, static, graph, caches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify_flash_kv_attention():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from verl.workers.rollout.fusion.decoder import flash_kv_attention

    generator = torch.Generator(device="cuda").manual_seed(173)
    errors = []
    for batch, width, queries in ((1, 9, 9), (3, 11, 11), (3, 13, 1)):
        q = torch.randn((batch, 12, queries, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, 2, 128, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(k.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        lengths = torch.full((batch,), width, device="cuda", dtype=torch.int32)
        leftpad = torch.arange(batch, device="cuda", dtype=torch.int32) * 2
        output, _ = flash_kv_attention(
            None,
            q,
            k,
            v,
            None,
            scaling=128**-0.5,
            fusion_kv_lengths=lengths,
            fusion_kv_leftpad=leftpad,
        )
        keys = torch.arange(128, device="cuda")
        allowed = (keys[None, :] >= leftpad[:, None]) & (keys[None, :] < lengths[:, None])
        allowed = allowed[:, None, None, :]
        if queries > 1:
            physical_query = torch.arange(queries, device="cuda") + width - queries
            allowed = allowed & (keys[None, :] <= physical_query[:, None])[None, None]
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=allowed, enable_gqa=True
            ).transpose(1, 2)
        torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.008)
        difference = output.float() - reference
        relative_l2 = float(difference.norm() / reference.norm())
        if relative_l2 > 0.005:
            raise AssertionError(f"Flash KV relative L2 error exceeds BF16 tolerance: {relative_l2}")
        errors.append(
            {
                "batch": batch,
                "width": width,
                "queries": queries,
                "max_abs_error_vs_fp32": float(difference.abs().max()),
                "relative_l2": relative_l2,
            }
        )
    return errors


def verify_graph_weight_updates(student, teacher, prompt, *, tokenizer, prompt_cap, response_cap, flash_kv=False):
    import torch

    from verl.workers.rollout.fusion.decoder import LogitFusionDecoder

    decoder = LogitFusionDecoder(
        student,
        teacher,
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        weight=0,
        backend="cuda_graph_flash" if flash_kv else "cuda_graph",
        prompt_cap=prompt_cap,
        response_cap=response_cap,
    )
    decoder.load_student_weights(list(student.named_parameters()), version=0, final=True)
    original = decoder.generate([prompt], max_new_tokens=4, temperature=0, expected_version=0)
    graph = decoder.contexts[1].graph
    head = student.get_output_embeddings().weight
    backup = head.detach().clone()
    teacher_probes = [parameter.reshape(-1)[:32].detach().clone() for parameter in teacher.parameters()]
    try:
        with torch.no_grad():
            head.zero_()
        decoder.load_student_weights(list(student.named_parameters()), version=1, final=True)
        changed = decoder.generate([prompt], max_new_tokens=4, temperature=0, expected_version=1)
        if changed["responses"] != [[0, 0, 0, 0]]:
            raise AssertionError("Captured decoder ignored in-place actor weight updates")
        if decoder.contexts[1].graph is not graph:
            raise AssertionError("Graph was unexpectedly recaptured after copying weights")
        with torch.no_grad():
            head.copy_(backup)
        decoder.load_student_weights(list(student.named_parameters()), version=2, final=True)
        restored = decoder.generate([prompt], max_new_tokens=4, temperature=0, expected_version=2)
        if restored["responses"] != original["responses"]:
            raise AssertionError("Restored actor weights did not restore the original token stream")
        if any(
            not torch.equal(probe, parameter.reshape(-1)[:32])
            for probe, parameter in zip(teacher_probes, teacher.parameters(), strict=True)
        ):
            raise AssertionError("Teacher parameter probes changed with student updates")
    finally:
        with torch.no_grad():
            head.copy_(backup)
    return {"versions_verified": [0, 1, 2], "graph_reused": True, "teacher_parameter_probes_unchanged": True}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verl.workers.rollout.fusion.decoder import LogitFusionDecoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", default="1,8,16")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompt-cap", type=int, default=1024)
    parser.add_argument("--response-cap", type=int, default=4096)
    parser.add_argument("--backends", default="dynamic,static,cuda_graph")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("auto", "efficient", "math"), default="auto")
    parser.add_argument("--graph-attention", choices=("sdpa", "flash-kv"), default="sdpa")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise ValueError("Benchmark expects exactly one assigned CUDA device")
    torch.backends.cuda.enable_flash_sdp(args.attention_backend == "auto")
    torch.backends.cuda.enable_mem_efficient_sdp(args.attention_backend != "math")
    torch.backends.cuda.enable_math_sdp(args.attention_backend != "efficient")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    models = [
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
        .to("cuda")
        .eval()
        for path in (args.student, args.teacher)
    ]
    prompts = [json.loads(line)["prompt_ids"] for line in Path(args.prompts).read_text().splitlines()]
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "decoder_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[2] / "verl/verl/workers/rollout/fusion/decoder.py").read_bytes()
        ).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tokens": args.tokens,
        "prompt_cap": args.prompt_cap,
        "response_cap": args.response_cap,
        "lambda": 0.5,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "graph_attention": args.graph_attention,
        "measurements": [],
        "parity": [],
    }
    query = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Driver PIDs can use a different namespace from os.getpid(). Our CUDA
    # context is already live, so exactly one compute process means isolation.
    process_count = len([line for line in query.stdout.splitlines() if line.strip()])
    report["other_gpu_process_count"] = max(0, process_count - 1)
    report["timing_isolated"] = process_count == 1
    if args.graph_attention == "flash-kv":
        report["flash_attention_vs_fp32"] = verify_flash_kv_attention()
    report["weight_updates"] = verify_graph_weight_updates(
        *models,
        prompts[0],
        tokenizer=tokenizer,
        prompt_cap=args.prompt_cap,
        response_cap=args.response_cap,
        flash_kv=args.graph_attention == "flash-kv",
    )
    for batch in map(int, args.batch_sizes.split(",")):
        selected = prompts[:batch]
        parity = compare_prefix_logits(
            *models,
            selected,
            tokenizer=tokenizer,
            prompt_cap=args.prompt_cap,
            response_cap=args.response_cap,
            flash_kv=args.graph_attention == "flash-kv",
            steps=min(8, args.tokens - 1),
        )
        report["parity"].append(parity)
        for backend in args.backends.split(","):
            decoder = LogitFusionDecoder(
                *models,
                vocab_size=len(tokenizer),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                weight=0.5,
                backend=backend,
                prompt_cap=args.prompt_cap,
                response_cap=args.response_cap,
            )
            cold = decoder.generate(selected, max_new_tokens=4, temperature=0)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = decoder.generate(selected, max_new_tokens=args.tokens, temperature=0)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            count = sum(map(len, result["responses"]))
            row = {
                "backend": backend,
                "batch": batch,
                "seconds": elapsed,
                "tokens": count,
                "tokens_per_second": count / elapsed,
                "cold_seconds": cold["seconds"],
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "sample": tokenizer.decode(result["responses"][0][:48]),
            }
            report["measurements"].append(row)
            print(json.dumps(row), flush=True)
            del decoder, result, cold
            gc.collect()
            torch.cuda.empty_cache()
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("FUSION_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
