"""Isolated exact reverse-KL experiments; does not modify the training pipeline.

FP32 student parameters/Adam states, BF16 projection and FlashAttention2,
FP32 probabilities. Chunked first-order autograd computes each head gradient
before releasing its vocabulary intermediates. No second-order support.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def dense_kl(h, w, th, tw):
    logits = F.linear(h, w)
    lp = (
        logits.float().log_softmax(-1)
        if logits.dtype != torch.float64
        else logits.log_softmax(-1)
    )
    with torch.no_grad():
        q_logits = F.linear(th, tw)
        lq = (
            q_logits.float().log_softmax(-1)
            if q_logits.dtype != torch.float64
            else q_logits.log_softmax(-1)
        )
    return (lp.exp() * (lp - lq)).sum() / h.shape[0]


class ChunkedKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, th, tw, chunk):
        dh = torch.zeros_like(h)
        dw = torch.zeros_like(w)
        loss = torch.zeros(
            (),
            device=h.device,
            dtype=torch.float64 if h.dtype == torch.float64 else torch.float32,
        )
        for start in range(0, len(h), chunk):
            end = min(start + chunk, len(h))
            # Detached weight leaves differ each chunk. Autocast's weight cache
            # would retain one low-precision head copy per leaf until scope exit.
            with (
                torch.enable_grad(),
                torch.autocast(
                    h.device.type,
                    enabled=torch.is_autocast_enabled(h.device.type),
                    dtype=torch.get_autocast_dtype(h.device.type),
                    cache_enabled=False,
                ),
            ):
                hc = h[start:end].detach().requires_grad_(True)
                wc = w.detach().requires_grad_(True)
                lc = dense_kl(hc, wc, th[start:end], tw) * ((end - start) / len(h))
                gh, gw = torch.autograd.grad(lc, (hc, wc))
            dh[start:end] = gh
            dw.add_(gw)
            loss.add_(lc.detach())
            del hc, wc, lc, gh, gw
        ctx.save_for_backward(dh, dw)
        return loss

    @staticmethod
    def backward(ctx, scale):
        dh, dw = ctx.saved_tensors
        return dh * scale, dw * scale, None, None, None


def verify():
    torch.manual_seed(19)
    output = []
    for dtype, device in [(torch.float64, "cpu"), (torch.float32, "cuda")]:
        h = torch.randn(17, 11, device=device, dtype=dtype, requires_grad=True)
        w = (torch.randn(53, 11, device=device, dtype=dtype) / 3).requires_grad_()
        th = torch.randn(17, 7, device=device, dtype=dtype, requires_grad=True)
        tw = (torch.randn(53, 7, device=device, dtype=dtype) / 3).requires_grad_()
        reference = dense_kl(h, w, th, tw)
        expected = torch.autograd.grad(reference, (h, w))
        for chunk in [1, 8, 17]:
            loss = ChunkedKL.apply(h, w, th, tw, chunk)
            actual = torch.autograd.grad(loss * 1.7, (h, w, th, tw), allow_unused=True)
            errors = [
                float((a / 1.7 - b).abs().max()) for a, b in zip(actual[:2], expected)
            ]
            assert actual[2:] == (None, None)
            tolerance = 1e-12 if dtype == torch.float64 else 2e-6
            assert max(errors) < tolerance and abs(float(loss - reference)) < tolerance
            output.append(
                dict(
                    dtype=str(dtype),
                    chunk=chunk,
                    loss_error=abs(float(loss - reference)),
                    gradient_errors=errors,
                )
            )
    return output


def load_records(path, length, count):
    records = []
    for line in Path(path).open():
        row = json.loads(line)
        prompt, response = row["prompt_ids"], row["response_ids"]
        if len(prompt) + len(response) >= length and len(prompt) < length:
            ids = (prompt + response)[:length]
            records.append((ids, len(prompt)))
        if len(records) >= count:
            break
    if len(records) < count:
        raise ValueError(
            f"Only {len(records)} real records meet length {length}; need {count}"
        )
    return records


def sync():
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "head", "train"], default="verify")
    parser.add_argument("--student", default="../models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--teacher", default="../models/JustRL-DeepSeek-1.5B")
    parser.add_argument(
        "--records",
        default="runs/validation-full-fast-20261019-v1/validation/kl/kl-tokens-step-2.jsonl",
    )
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=151665)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--loss", choices=["dense", "chunked"], default="chunked")
    parser.add_argument("--strict-bf16", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank, world = int(os.getenv("LOCAL_RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    if args.strict_bf16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(19)
    if world > 1:
        dist.init_process_group("nccl")
    result = {
        "args": vars(args),
        "rank": rank,
        "world": world,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.mode == "verify":
        result["checks"] = verify()
    else:
        records = load_records(args.records, args.length, world * args.accumulation)
        result["input_sha256"] = hashlib.sha256(
            json.dumps(records).encode()
        ).hexdigest()
        result["response_tokens_global"] = sum(len(ids) - plen for ids, plen in records)
        student = (
            AutoModelForCausalLM.from_pretrained(
                args.student,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .cuda()
            .float()
        )
        teacher = (
            AutoModelForCausalLM.from_pretrained(
                args.teacher,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        result["parameters"] = sum(p.numel() for p in student.parameters())
        student.config.use_cache = False
        teacher.config.use_cache = False
        local_records = records[rank::world]
        if args.mode == "head":
            ids, plen = local_records[0]
            inputs = torch.tensor([ids], device="cuda")
            student.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = (
                    student.model(inputs, use_cache=False)
                    .last_hidden_state[0, plen - 1 : -1]
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
                th = (
                    teacher.model(inputs, use_cache=False)
                    .last_hidden_state[0, plen - 1 : -1]
                    .detach()
                    .clone()
                )
            w = (
                student.lm_head.weight[: args.vocab]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            tw = teacher.lm_head.weight[: args.vocab].detach().clone()
            del student, teacher
            gc.collect()
            torch.cuda.empty_cache()
            cases = [
                ("dense", 0),
                ("chunk128", 128),
                ("chunk512", 512),
                ("chunk1024", 1024),
            ]
            results, reference = [], None
            for name, chunk in cases:
                times = []
                for repeat in range(4):
                    h.grad = None
                    w.grad = None
                    sync()
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = (
                            ChunkedKL.apply(h, w, th, tw, chunk)
                            if chunk
                            else dense_kl(h, w, th, tw)
                        )
                    loss.backward()
                    sync()
                    elapsed = time.perf_counter() - tick
                    if repeat:
                        times.append(elapsed)
                gradients = [h.grad.detach().cpu(), w.grad.detach().cpu()]
                if reference is None:
                    reference = (float(loss), gradients)
                errors = [
                    float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))
                    for a, b in zip(gradients, reference[1])
                ]
                row = dict(
                    case=name,
                    seconds=times,
                    median_seconds=statistics.median(times),
                    loss=float(loss),
                    loss_error=abs(float(loss) - reference[0]),
                    gradient_relative_l2=errors,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                del gradients, loss
                h.grad = None
                w.grad = None
                gc.collect()
                torch.cuda.empty_cache()
            result["measurements"] = results
        else:
            student.train()
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            class Learner(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, ids, prompt_len, teacher_hidden, teacher_weight):
                    hidden = self.model.model(ids, use_cache=False).last_hidden_state[
                        0, prompt_len - 1 : -1
                    ]
                    if args.loss == "dense":
                        return dense_kl(
                            hidden,
                            self.model.lm_head.weight[: args.vocab],
                            teacher_hidden,
                            teacher_weight[: args.vocab],
                        )
                    return ChunkedKL.apply(
                        hidden,
                        self.model.lm_head.weight[: args.vocab],
                        teacher_hidden,
                        teacher_weight[: args.vocab],
                        args.chunk,
                    )

            learner = Learner(student)
            if world > 1:
                learner = torch.nn.parallel.DistributedDataParallel(
                    learner,
                    device_ids=[rank],
                    gradient_as_bucket_view=True,
                    bucket_cap_mb=100,
                )
            optimizer = torch.optim.AdamW(
                student.parameters(), lr=1e-6, weight_decay=0, fused=True
            )
            updates = []
            total_response = result["response_tokens_global"]
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                if world > 1:
                    dist.barrier()
                sync()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                teacher_seconds, learner_seconds, losses = 0.0, 0.0, []
                for index, (ids, plen) in enumerate(local_records):
                    inputs = torch.tensor([ids], device="cuda")
                    sync()
                    part = time.perf_counter()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        th = (
                            teacher.model(inputs, use_cache=False)
                            .last_hidden_state[0, plen - 1 : -1]
                            .contiguous()
                        )
                    sync()
                    teacher_seconds += time.perf_counter() - part
                    part = time.perf_counter()
                    context = (
                        learner.no_sync()
                        if world > 1 and index + 1 < len(local_records)
                        else contextlib.nullcontext()
                    )
                    with context, torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = learner(inputs, plen, th, teacher.lm_head.weight)
                        # DDP averages ranks; this gives a global valid-token mean.
                        scaled = loss * (world * (len(ids) - plen) / total_response)
                        scaled.backward()
                    sync()
                    learner_seconds += time.perf_counter() - part
                    losses.append(float(loss.detach()))
                    del th, loss, scaled
                sync()
                part = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                assert torch.isfinite(grad_norm)
                optimizer.step()
                sync()
                optimizer_seconds = time.perf_counter() - part
                row = dict(
                    step=step,
                    seconds=time.perf_counter() - tick,
                    teacher_seconds=teacher_seconds,
                    learner_seconds=learner_seconds,
                    optimizer_seconds=optimizer_seconds,
                    losses=losses,
                    grad_norm=float(grad_norm),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                )
                updates.append(row)
                print(json.dumps({"rank": rank, **row}), flush=True)
            result["updates"] = updates
            result["optimizer_state_dtypes"] = sorted(
                {
                    str(v.dtype)
                    for state in optimizer.state.values()
                    for v in state.values()
                    if torch.is_tensor(v)
                }
            )
            result["teacher_has_gradient"] = any(
                p.grad is not None for p in teacher.parameters()
            )
            if world > 1:
                check = torch.stack(
                    [p.detach().flatten()[0] for p in student.parameters()]
                )
                base = check.clone()
                dist.broadcast(base, src=0)
                result["rank_parameter_sample_max_difference"] = float(
                    (check - base).abs().max()
                )
    output = Path(args.output)
    if world > 1:
        output = output.with_name(output.stem + f"-rank{rank}" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
