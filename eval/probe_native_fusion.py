"""Compare native paired Qwen2 full logits against HF on identical prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(weight, gpu, output, tokens, graph):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    sys.path.insert(0, str(ROOT / "verl"))
    llm = None
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from vllm import LLM, SamplingParams

        from verl.trainer.ppo.fixed_rollout import model_signature
        from verl.utils import hf_tokenizer
        from verl.workers.rollout.fusion import register_native_fusion

        register_native_fusion()
        paths = [
            str(ROOT.parent / "models" / name) for name in ["DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B"]
        ]
        tokenizer = hf_tokenizer(paths[0])
        vocab = len(tokenizer)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
            )
            for text in ["What is 7 times 8? Show your reasoning.", "Solve x^2 - 5x + 6 = 0."]
        ]
        paired = 0 < weight < 1
        fingerprints = [model_signature(path) for path in paths]
        overrides = (
            {
                "architectures": ["OPDNativeQwen2Fusion"],
                "opd_teacher_path": paths[1],
                "opd_teacher_fingerprint": fingerprints[1],
                "opd_teacher_weight": weight,
            }
            if paired
            else {}
        )
        llm = LLM(
            model=paths[1] if weight == 1 else paths[0],
            dtype="bfloat16",
            enforce_eager=not graph,
            compilation_config={"cudagraph_capture_sizes": [1, 2], "custom_ops": ["all"]} if graph else {},
            hf_overrides=overrides,
            enable_prefix_caching=True,
            enable_sleep_mode=True,
            max_model_len=256,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            gpu_memory_utilization=0.15,
            tensor_parallel_size=1,
            max_logprobs=vocab,
            logprobs_mode="raw_logits",
            worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        )
        llm.collective_rpc("monkey_patch_model", kwargs={"vocab_size": vocab})
        source_info = llm.collective_rpc(
            "configure_native_source",
            kwargs={
                "role": "teacher" if weight == 1 else "initial_student",
                "path": paths[1] if weight == 1 else paths[0],
                "fingerprint": fingerprints[1] if weight == 1 else fingerprints[0],
                "fusion": {"teacher_path": paths[1], "teacher_fingerprint": fingerprints[1], "teacher_weight": weight}
                if paired
                else None,
                "verify_frozen": True,
                "verify_addresses": graph,
            },
        )[0]
        layout = llm.collective_rpc("native_model_layout")[0]
        if graph and (not layout["cuda_graph_enabled"] or layout["cuda_graphs_captured"] < 1):
            raise RuntimeError("Requested native CUDA graphs were not actually captured")
        regions = sorted(layout["kv_regions"], key=lambda r: r["pointer"])
        if len(regions) != (56 if paired else 28) or any(
            a["pointer"] + a["bytes"] > b["pointer"] for a, b in zip(regions, regions[1:], strict=False)
        ):
            raise RuntimeError("Native backbones do not have complete disjoint KV regions")

        def native():
            results = llm.generate(
                [{"prompt_token_ids": p} for p in prompts],
                SamplingParams(max_tokens=tokens, temperature=0, logprobs=vocab),
                use_tqdm=False,
            )
            arrays = []
            responses = []
            for result in results:
                response = result.outputs[0]
                responses.append(list(response.token_ids))
                for scores in response.logprobs:
                    if set(range(vocab)) - scores.keys():
                        raise RuntimeError("Native full-vocabulary logits are incomplete")
                    arrays.append(np.array([scores[i].logprob for i in range(vocab)], dtype=np.float32))
            return np.stack(arrays), responses

        logits, responses = native()
        llm.sleep(level=1)
        llm.wake_up()
        llm.reset_prefix_cache()
        restored, restored_responses = native()
        if not np.array_equal(logits, restored) or responses != restored_responses:
            raise RuntimeError("Native fusion logits changed after weight-preserving sleep")
        restored_source = llm.collective_rpc("native_source_status")[0]
        if restored_source["frozen_digests"] != source_info["frozen_digests"] or (
            graph and restored_source["parameter_addresses"] != source_info["parameter_addresses"]
        ):
            raise RuntimeError("Frozen parameters or CUDA graph addresses changed during native generation/sleep")
        llm.llm_engine.engine_core.shutdown()
        llm = None

        components = []
        for path in paths:
            model = (
                AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                .to("cuda:0")
                .eval()
            )
            with torch.inference_mode():
                rows = [
                    model(input_ids=torch.tensor([p + r[:-1]], device="cuda:0"), use_cache=False)
                    .logits[0, len(p) - 1 :, :vocab]
                    .float()
                    .cpu()
                    for p, r in zip(prompts, responses, strict=True)
                ]
            components.append(torch.cat(rows))
            del model
            torch.cuda.empty_cache()
        expected = (1 - weight) * components[0] + weight * components[1]
        observed = torch.from_numpy(logits)
        error = (observed - expected).abs().amax(dim=-1)
        tv = (observed.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
        if not torch.isfinite(observed).all() or error.max() > 0.5 or tv.max() > 0.05:
            raise RuntimeError(f"Native/HF BF16 bound failed: max logits={error.tolist()}, TV={tv.tolist()}")
        np.savez_compressed(
            str(Path(output).with_suffix(".npz")),
            native=logits,
            hf=expected.numpy(),
            student=components[0].numpy(),
            teacher=components[1].numpy(),
        )
        result = {
            "teacher_weight": weight,
            "native_paired_backbones": paired,
            "vocab_size": vocab,
            "prompt_ids": prompts,
            "response_ids": responses,
            "tokens_per_request_cap": tokens,
            "cuda_graph_requested": graph,
            "max_abs_logit_error": error.tolist(),
            "total_variation": tv.tolist(),
            "sleep_wake_logits_bitwise_equal": True,
            "native_sampler": True,
            "native_layout": layout,
            "native_source": source_info,
            "reference": "HF BF16 FlashAttention2, same token prefix and valid vocabulary",
            "source_sha256": hashlib.sha256(
                (ROOT / "verl/verl/workers/rollout/fusion/vllm_model.py").read_bytes()
            ).hexdigest(),
        }
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "teacher_weight": weight,
                    "positions": len(logits),
                    "max_logit_error": float(error.max()),
                    "max_tv": float(tv.max()),
                    "graph": graph,
                }
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        for process in mp.active_children():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    args = parser.parse_args()
    if args.tokens < 1 or any(not math.isfinite(w) or not 0 <= w <= 1 for w in args.weights):
        parser.error("tokens must be positive and weights must be finite values in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=False)
    for weight in args.weights:
        process = mp.get_context("spawn").Process(
            target=child, args=(weight, args.gpu, str(args.output / f"lambda-{weight}.json"), args.tokens, args.graph)
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Native fusion probe failed at lambda={weight}, exit={process.exitcode}")


if __name__ == "__main__":
    main()
