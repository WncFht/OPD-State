"""Compare native paired Qwen2 full logits against HF on identical prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(weight, gpu, output):
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
        overrides = (
            {"architectures": ["OPDNativeQwen2Fusion"], "opd_teacher_path": paths[1], "opd_teacher_weight": weight}
            if paired
            else {}
        )
        llm = LLM(
            model=paths[1] if weight == 1 else paths[0],
            dtype="bfloat16",
            enforce_eager=True,
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
        layout = llm.collective_rpc("native_model_layout")[0]
        regions = sorted(layout["kv_regions"], key=lambda r: r["pointer"])
        if len(regions) != (56 if paired else 28) or any(
            a["pointer"] + a["bytes"] > b["pointer"] for a, b in zip(regions, regions[1:], strict=False)
        ):
            raise RuntimeError("Native backbones do not have complete disjoint KV regions")

        def native():
            results = llm.generate(
                [{"prompt_token_ids": p} for p in prompts],
                SamplingParams(max_tokens=1, temperature=0, logprobs=vocab),
                use_tqdm=False,
            )
            arrays = []
            for result in results:
                scores = result.outputs[0].logprobs[0]
                if set(range(vocab)) - scores.keys():
                    raise RuntimeError("Native full-vocabulary logits are incomplete")
                arrays.append(np.array([scores[i].logprob for i in range(vocab)], dtype=np.float32))
            return np.stack(arrays)

        logits = native()
        llm.sleep(level=1)
        llm.wake_up()
        llm.reset_prefix_cache()
        restored = native()
        if not np.array_equal(logits, restored):
            raise RuntimeError("Native fusion logits changed after weight-preserving sleep")
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
                    model(input_ids=torch.tensor([p], device="cuda:0"), use_cache=False)
                    .logits[0, -1, :vocab]
                    .float()
                    .cpu()
                    for p in prompts
                ]
            components.append(torch.stack(rows))
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
            "max_abs_logit_error": error.tolist(),
            "total_variation": tv.tolist(),
            "sleep_wake_logits_bitwise_equal": True,
            "native_sampler": True,
            "native_layout": layout,
            "reference": "HF BF16 FlashAttention2, same token prefix and valid vocabulary",
            "source_sha256": hashlib.sha256(
                (ROOT / "verl/verl/workers/rollout/fusion/vllm_model.py").read_bytes()
            ).hexdigest(),
        }
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for weight in [0.0, 0.5, 1.0]:
        process = mp.get_context("spawn").Process(
            target=child, args=(weight, args.gpu, str(args.output / f"lambda-{weight}.json"))
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Native fusion probe failed at lambda={weight}, exit={process.exitcode}")


if __name__ == "__main__":
    main()
