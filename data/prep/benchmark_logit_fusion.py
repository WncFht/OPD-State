"""Measure shared-prefix dual-model decoding on one explicitly assigned GPU."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def compare_prefix_logits(student, teacher, prompts, *, tokenizer, prompt_cap, response_cap, steps=8):
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
    static = StaticDecodeContext(student, teacher, bsz, capacity, capture=False)
    graph = StaticDecodeContext(student, teacher, bsz, capacity, capture=True)
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
    }
    if result["max_distribution_tv_graph_vs_dynamic"] > 0.01:
        raise AssertionError(f"Graph probability drift exceeds the BF16 parity gate: {result}")
    if result["max_logit_error_graph_vs_static"] > 0.125:
        raise AssertionError(f"Graph/static drift exceeds one high-logit BF16 increment: {result}")
    del reference, static, graph, caches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify_graph_weight_updates(student, teacher, prompt, *, tokenizer, prompt_cap, response_cap):
    import torch

    from verl.workers.rollout.fusion.decoder import LogitFusionDecoder

    decoder = LogitFusionDecoder(
        student,
        teacher,
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        weight=0,
        backend="cuda_graph",
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
    args = parser.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise ValueError("Benchmark expects exactly one assigned CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    models = [
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
        for path in (args.student, args.teacher)
    ]
    prompts = [json.loads(line)["prompt_ids"] for line in Path(args.prompts).read_text().splitlines()]
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tokens": args.tokens,
        "prompt_cap": args.prompt_cap,
        "response_cap": args.response_cap,
        "lambda": 0.5,
        "measurements": [],
        "parity": [],
    }
    report["weight_updates"] = verify_graph_weight_updates(
        *models, prompts[0], tokenizer=tokenizer, prompt_cap=args.prompt_cap, response_cap=args.response_cap
    )
    for batch in map(int, args.batch_sizes.split(",")):
        selected = prompts[:batch]
        parity = compare_prefix_logits(
            *models,
            selected,
            tokenizer=tokenizer,
            prompt_cap=args.prompt_cap,
            response_cap=args.response_cap,
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
