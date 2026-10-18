"""Two-rank CPU FSDP check of head capture, optimizer updates and teacher freezing."""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, output_dir, rendezvous):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from verl.workers.validation_kl import score_causal_records

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method="file://" + rendezvous, timeout=datetime.timedelta(seconds=90)
    )
    try:
        torch.manual_seed(17)
        config = Qwen2Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        p, q = Qwen2ForCausalLM(config).eval(), Qwen2ForCausalLM(config).eval()
        for parameter in q.parameters():
            parameter.requires_grad_(False)
        record = {
            "state_source": "on_validation_states",
            "source": "MATH-500",
            "uid": f"rank-{rank}",
            "sample": 0,
            "prompt_ids": [1, 4, 3],
            "response_ids": [7] * (3 + 8 * rank) + [2],
        }
        ids = torch.tensor([record["prompt_ids"] + record["response_ids"]])
        with torch.no_grad():
            logp = p(ids, use_cache=False).logits[0, 2:-1].double().log_softmax(-1)
            logq = q(ids, use_cache=False).logits[0, 2:-1].double().log_softmax(-1)
        reference = {
            "student_teacher_kl": float((logp.exp() * (logp - logq)).sum()),
            "teacher_student_kl": float((logq.exp() * (logq - logp)).sum()),
            "student_entropy": float(-(logp.exp() * logp).sum()),
            "teacher_entropy": float(-(logq.exp() * logq).sum()),
        }
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})
        student = FSDP(p, auto_wrap_policy=policy, device_id=torch.device("cpu"), use_orig_params=False)
        teacher = FSDP(q, auto_wrap_policy=policy, device_id=torch.device("cpu"), use_orig_params=False)
        teacher_before = [parameter.detach().clone() for parameter in teacher.parameters()]
        first, timings = score_causal_records(
            student, teacher, [record], vocab_size=19, device="cpu", position_chunk_size=3, vocab_chunk_size=5
        )
        errors = {key: abs(first[0]["sums"][key] - value) for key, value in reference.items()}
        if max(errors.values()) > 2e-5:
            raise RuntimeError(f"Rank {rank} FSDP output differs from dense reference: {errors}")
        student.train()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss = student(ids, labels=ids, use_cache=False).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second, _ = score_causal_records(
            student, teacher, [record], vocab_size=19, device="cpu", position_chunk_size=3, vocab_chunk_size=5
        )
        updated = abs(second[0]["sums"]["student_teacher_kl"] - first[0]["sums"]["student_teacher_kl"]) > 1e-8
        frozen = all(
            torch.equal(before, after) for before, after in zip(teacher_before, teacher.parameters(), strict=True)
        )
        if not updated or not frozen:
            raise RuntimeError("FSDP scoring did not read the updated student or changed the teacher")
        report = {
            "rank": rank,
            "response_tokens": len(record["response_ids"]),
            "max_absolute_errors": errors,
            "student_optimizer_update_observed": updated,
            "teacher_unchanged": frozen,
            "timings": timings,
            "initial_sums": first[0]["sums"],
            "updated_sums": second[0]["sums"],
        }
        (Path(output_dir) / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path.insert(0, str(ROOT / "verl"))
    import torch.multiprocessing as mp

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(args.output), str(args.output / "rendezvous")), nprocs=2, join=True)
    ranks = [json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    sources = [
        "eval/probe_validation_fsdp_cpu.py",
        "verl/verl/workers/validation_kl.py",
        "verl/verl/trainer/ppo/validation_kl.py",
    ]
    report = {
        "schema": "validation-fsdp-cpu-v1",
        "world_size": 2,
        "backend": "gloo",
        "device": "cpu",
        "actual_qwen_checkpoints": False,
        "real_gpu_verified": False,
        "ranks": ranks,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(args.output / "receipt.json"),
                "ranks": len(ranks),
                "optimizer_updates_observed": all(r["student_optimizer_update_observed"] for r in ranks),
            }
        )
    )


if __name__ == "__main__":
    main()
