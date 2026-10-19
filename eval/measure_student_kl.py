#!/usr/bin/env python3
# Copyright 2026 OPD-State contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate fresh student rollouts and calibrate full-vocabulary conditional KL.

Both models must be immutable HF checkpoint directories. No parameter updates.
Generation and scoring use the same effective vocabulary, temperature 1 and
untruncated sampling. Forward KL below is on student states, not sequence FKL.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL or One-Shot-OPD parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0, help="Physical GPU; only 0-3")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.limit, args.samples, args.max_prompt_tokens, args.max_new_tokens, args.top_k) < 1:
        parser.error("Positive sample, token, and top-k budgets required")
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else str(args.gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "verl")]
    from opd_state.kernels.head_tail import coefficients_and_metrics, selected_and_tail

    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu else "cuda:0"
    dtype = torch.float32 if args.cpu else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("This calibration requires identical tokenizer token-ID mappings")
    size = len(tokenizer)
    if set(tokenizer.get_vocab().values()) != set(range(size)) or args.top_k > size:
        raise ValueError("Expected contiguous vocabulary IDs and top-k <= vocabulary size")

    class EffectiveVocabulary(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[..., size:] = -torch.inf
            return scores

    student = (
        AutoModelForCausalLM.from_pretrained(
            args.student, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            args.teacher, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    if args.prompts.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(args.prompts).to_pylist()
    else:
        rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    records = []
    accepted = 0
    with torch.inference_mode():
        for row in rows:
            messages = row.get("messages", row.get("prompt"))
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if not isinstance(messages, list):
                raise ValueError("Each prompt requires a messages/prompt list or prompt string")
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            ).to(device)
            if prompt.shape[1] > args.max_prompt_tokens:
                continue
            prompt_id = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            for sample in range(args.samples):
                generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                generated = student.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    generation_config=generation_config,
                    logits_processor=[EffectiveVocabulary()],
                    use_model_defaults=False,
                )
                sequence = generated.sequences[0]
                actions = sequence[prompt.shape[1] :]
                # Retain the actual generation distributions, avoiding a second
                # student's scoring pass with numerically different KV/batching.
                logits_p = torch.cat(generated.scores, dim=0)[:, :size]
                del generated
                if not torch.isfinite(logits_p).all():
                    raise FloatingPointError("Expected finite, untruncated generation scores in effective vocabulary")
                logits_q = teacher(sequence[None], use_cache=False).logits[0, prompt.shape[1] - 1 : -1, :size]
                sums = dict(
                    reverse_full=0.0, forward_on_student_states=0.0, grouped=0.0, unbiased=0.0, sampled_logratio=0.0
                )
                for start in range(0, len(actions), 32):
                    lp, lq = (
                        logits_p[start : start + 32].float().log_softmax(-1),
                        logits_q[start : start + 32].float().log_softmax(-1),
                    )
                    a = actions[start : start + 32]
                    head = selected_and_tail(logits_p[start : start + 32], args.top_k)
                    qhead = selected_and_tail(logits_q[start : start + 32], args.top_k, head["student_top_k_ids"])
                    _, _, stats = coefficients_and_metrics(
                        head["student_top_k_log_probs"],
                        qhead["student_top_k_log_probs"],
                        lp.gather(-1, a[:, None]).squeeze(-1),
                        lq.gather(-1, a[:, None]).squeeze(-1),
                        head["student_tail_log_mass"],
                        qhead["student_tail_log_mass"],
                        head["student_top_k_ids"],
                        a,
                        torch.ones_like(a, dtype=torch.bool),
                    )
                    sums["reverse_full"] += float((lp.exp() * (lp - lq)).sum(-1).double().sum())
                    sums["forward_on_student_states"] += float((lq.exp() * (lq - lp)).sum(-1).double().sum())
                    sums["grouped"] += float(stats["kl_grouped"].double().sum())
                    sums["unbiased"] += float(stats["kl_unbiased"].double().sum())
                    sums["sampled_logratio"] += float(stats["behavior_logratio"].double().sum())
                records.append(
                    dict(
                        prompt_id=prompt_id,
                        sample=sample,
                        response_tokens=len(actions),
                        stopped_on_eos=int(actions[-1]) == tokenizer.eos_token_id,
                        **sums,
                    )
                )
                del logits_p, logits_q
            accepted += 1
            if accepted == args.limit:
                break
    if not records:
        raise ValueError("No prompt passed the token-length filter")
    tokens = sum(r["response_tokens"] for r in records)
    keys = ["reverse_full", "forward_on_student_states", "grouped", "unbiased", "sampled_logratio"]
    summary = {key + "_sequence_mean": sum(r[key] for r in records) / len(records) for key in keys}
    summary.update({key + "_token_mean": sum(r[key] for r in records) / tokens for key in keys})
    summary["mean_response_tokens"] = tokens / len(records)
    output = {
        "student": str(args.student.resolve()),
        "teacher": str(args.teacher.resolve()),
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampling": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "student_probability_source": "actual Transformers generate output_scores",
        "effective_vocab_size": size,
        "top_k": args.top_k,
        "prompts": accepted,
        "trajectories": len(records),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
