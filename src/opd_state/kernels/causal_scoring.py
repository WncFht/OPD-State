"""Read-only causal scoring using bounded output-head projection memory."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from opd_state.core.records import STATISTICS, record_fingerprint, validate_record
from opd_state.kernels.divergence import full_vocab_divergences


def _output_head(module):
    head = module.get_output_embeddings()
    # A size-based wrapping policy can wrap the head itself. Install the hook
    # inside that wrapper, after FSDP has gathered its parameter views.
    while hasattr(head, "_fsdp_wrapped_module"):
        head = head._fsdp_wrapped_module
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("Validation KL requires an ordinary linear output head")
    return head


@contextmanager
def _head_capture(module, callback, *, batch_size=1, keep_batch=False):
    head = _output_head(module)
    calls = []

    def before_head(layer, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 3 or inputs[0].shape[0] != batch_size:
            raise ValueError(
                "Validation KL output-head batch differs from its causal inputs"
            )
        calls.append(1)
        callback(layer, (inputs[0] if keep_batch else inputs[0][0]).detach())
        # The backbone has already run. A zero-length head input suppresses the
        # ordinary [full sequence, vocabulary] projection without replacing any
        # parameters or model-forward implementation.
        return (inputs[0][:, :0, :],)

    handle = head.register_forward_pre_hook(before_head)
    try:
        yield calls
    finally:
        handle.remove()


def score_causal_records(
    student,
    teacher,
    records,
    *,
    vocab_size,
    device,
    position_chunk_size=128,
    vocab_chunk_size=8192,
    dtype=torch.bfloat16,
    forward_batch_size=1,
):
    """Score identical raw prefixes, including EOS actions, without backward.

    The student head is snapshotted once while its FSDP parameters are gathered.
    Per response we retain its selected hidden states until the teacher head is
    reached, then project both heads in bounded position blocks. The FSDP model
    is still called normally, so all ranks must process equal record counts.
    """
    if type(vocab_size) is not int or vocab_size < 1 or not records:
        raise ValueError(
            "Validation KL requires a positive vocabulary and nonempty local records"
        )
    if any(
        type(size) is not int or size < 1
        for size in (position_chunk_size, vocab_chunk_size)
    ):
        raise ValueError("Validation KL block sizes must be positive integers")
    if type(forward_batch_size) is not int or forward_batch_size < 1:
        raise ValueError("Validation KL forward batch size must be a positive integer")
    for module in (student, teacher):
        if module.config.model_type != "qwen2":
            raise ValueError(
                "Validation KL head capture currently supports the native Qwen2 pair"
            )
    if forward_batch_size > 1:
        return _score_causal_batches(
            student,
            teacher,
            records,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            position_chunk_size=position_chunk_size,
            vocab_chunk_size=vocab_chunk_size,
            forward_batch_size=forward_batch_size,
        )
    specifications = [validate_record(record, vocab_size) for record in records]
    device = torch.device(device)
    flags = [
        (module, module.training)
        for root in (student, teacher)
        for module in root.modules()
    ]
    output, snapshot = [], {}
    started = time.perf_counter()
    timings = {
        "student_forward_seconds": 0.0,
        "teacher_forward_seconds": 0.0,
        "head_kl_seconds": 0.0,
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        student.eval()
        teacher.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ),
        ):
            for record, (positions, weights) in zip(
                records, specifications, strict=True
            ):
                prompt = record["prompt_ids"]
                # Action t is predicted from prompt[-1] for t=0, or y[t-1].
                # No token after the last requested prefix is needed.
                tokens = prompt + record["response_ids"][: positions[-1]]
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                head_positions = (
                    torch.tensor(positions, dtype=torch.long, device=device)
                    + len(prompt)
                    - 1
                )
                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "position_ids": torch.arange(len(tokens), device=device).unsqueeze(
                        0
                    ),
                    "use_cache": False,
                    "logits_to_keep": head_positions,
                }
                captured = {}
                position_count = len(positions)

                def capture_student(
                    head, hidden, count=position_count, capture=captured
                ):
                    if hidden.shape[0] != count or head.weight.shape[0] < vocab_size:
                        raise ValueError(
                            "Student output head does not match selected response positions/vocabulary"
                        )
                    if not snapshot:
                        projection_dtype = (
                            dtype if device.type == "cuda" else head.weight.dtype
                        )
                        snapshot["weight"] = head.weight.detach().to(
                            dtype=projection_dtype, copy=True
                        )
                        snapshot["bias"] = (
                            None
                            if head.bias is None
                            else head.bias.detach().to(
                                dtype=projection_dtype, copy=True
                            )
                        )
                    capture["student_hidden"] = hidden

                parts = {name: [] for name in STATISTICS}

                def score_teacher(
                    head, hidden, count=position_count, capture=captured, results=parts
                ):
                    if hidden.shape[0] != count or head.weight.shape[0] < vocab_size:
                        raise ValueError(
                            "Teacher output head does not match selected response positions/vocabulary"
                        )
                    sync()
                    tick = time.perf_counter()
                    # A master FP32 head must not be re-cast for every position
                    # block by autocast's non-trainable-weight path.
                    projection_dtype = (
                        dtype if device.type == "cuda" else head.weight.dtype
                    )
                    teacher_weight = head.weight.to(dtype=projection_dtype)
                    teacher_bias = (
                        None
                        if head.bias is None
                        else head.bias.to(dtype=projection_dtype)
                    )
                    for begin in range(0, count, position_chunk_size):
                        end = min(begin + position_chunk_size, count)
                        p = F.linear(
                            capture["student_hidden"][begin:end],
                            snapshot["weight"],
                            snapshot["bias"],
                        )
                        q = F.linear(hidden[begin:end], teacher_weight, teacher_bias)
                        values = full_vocab_divergences(
                            p, q, vocab_size=vocab_size, chunk_size=vocab_chunk_size
                        )
                        for name in STATISTICS:
                            results[name].append(values[name])
                    sync()
                    timings["head_kl_seconds"] += time.perf_counter() - tick

                sync()
                tick = time.perf_counter()
                with _head_capture(student, capture_student) as calls:
                    student_output = student(**kwargs)
                    if len(calls) != 1 or student_output.logits.numel():
                        raise RuntimeError(
                            "Student capture did not suppress the full-sequence vocabulary projection"
                        )
                sync()
                timings["student_forward_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                kl_before = timings["head_kl_seconds"]
                with _head_capture(teacher, score_teacher) as calls:
                    teacher_output = teacher(**kwargs)
                    if len(calls) != 1 or teacher_output.logits.numel():
                        raise RuntimeError(
                            "Teacher capture did not suppress the full-sequence vocabulary projection"
                        )
                sync()
                # Teacher forward contains the head callback; subtract that
                # measured interval instead of double-counting it.
                timings["teacher_forward_seconds"] += (
                    time.perf_counter()
                    - tick
                    - (timings["head_kl_seconds"] - kl_before)
                )
                values = {
                    name: torch.cat(chunks).double().cpu()
                    for name, chunks in parts.items()
                }
                weight_tensor = torch.tensor(weights, dtype=torch.float64)
                sums = {
                    name: float((value * weight_tensor).sum())
                    for name, value in values.items()
                }
                row = {
                    key: record[key]
                    for key in ("state_source", "source", "uid", "sample")
                }
                row.update(
                    {
                        "record_sha256": record_fingerprint(record),
                        "response_length": len(record["response_ids"]),
                        "evaluated_positions": len(positions),
                        "sums": sums,
                        "depths": {},
                    }
                )
                if record["state_source"] == "on_validation_states":
                    for name, lower, upper in (
                        ("0_255", 0, 256),
                        ("256_1023", 256, 1024),
                        ("1024_2047", 1024, 2048),
                        ("2048_plus", 2048, len(positions)),
                    ):
                        upper = min(upper, len(positions))
                        if lower < upper:
                            row["depths"][name] = {
                                "tokens": upper - lower,
                                "sums": {
                                    name: float(value[lower:upper].sum())
                                    for name, value in values.items()
                                },
                            }
                output.append(row)
    finally:
        for module, training in flags:
            module.training = training
    timings["seconds"] = time.perf_counter() - started
    timings["head_snapshot_bytes"] = sum(
        v.numel() * v.element_size() for v in snapshot.values() if v is not None
    )
    return output, timings


def _score_causal_batches(
    student,
    teacher,
    records,
    *,
    vocab_size,
    device,
    dtype,
    position_chunk_size,
    vocab_chunk_size,
    forward_batch_size,
):
    """Amortize FSDP gathers over equal-count groups; project valid states only."""
    device = torch.device(device)
    specifications = [validate_record(record, vocab_size) for record in records]
    flags = [
        (module, module.training)
        for root in (student, teacher)
        for module in root.modules()
    ]
    if device.type == "cuda" and any(
        module.config._attn_implementation != "flash_attention_2"
        for module in (student, teacher)
    ):
        raise ValueError(
            "Batched CUDA validation KL currently requires FlashAttention2"
        )
    snapshot, output = {}, []
    timings = {
        "student_forward_seconds": 0.0,
        "teacher_forward_seconds": 0.0,
        "head_kl_seconds": 0.0,
        "forward_batches": 0,
    }
    started = time.perf_counter()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        student.eval()
        teacher.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ),
        ):
            for begin in range(0, len(records), forward_batch_size):
                group = records[begin : begin + forward_batch_size]
                selected = specifications[begin : begin + forward_batch_size]
                sequences = [
                    row["prompt_ids"] + row["response_ids"][: positions[-1]]
                    for row, (positions, _) in zip(group, selected, strict=True)
                ]
                width = max(map(len, sequences))
                input_ids = torch.zeros(
                    (len(group), width), dtype=torch.long, device=device
                )
                mask = torch.zeros_like(input_ids)
                for index, sequence in enumerate(sequences):
                    input_ids[index, : len(sequence)] = torch.tensor(
                        sequence, device=device
                    )
                    mask[index, : len(sequence)] = 1
                head_indices = [
                    torch.tensor(positions, device=device) + len(row["prompt_ids"]) - 1
                    for row, (positions, _) in zip(group, selected, strict=True)
                ]
                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": mask,
                    "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
                    "use_cache": False,
                    "logits_to_keep": 0,
                }
                capture, parts = {}, {name: [] for name in STATISTICS}

                def pick(hidden, indices_by_row=head_indices):
                    return torch.cat(
                        [hidden[i, indices] for i, indices in enumerate(indices_by_row)]
                    )

                def capture_student(head, hidden, captured=capture, select=pick):
                    if head.weight.shape[0] < vocab_size:
                        raise ValueError(
                            "Batched student head has the wrong effective vocabulary"
                        )
                    if not snapshot:
                        projection_dtype = (
                            dtype if device.type == "cuda" else head.weight.dtype
                        )
                        snapshot["weight"] = head.weight.detach().to(
                            dtype=projection_dtype, copy=True
                        )
                        snapshot["bias"] = (
                            None
                            if head.bias is None
                            else head.bias.detach().to(
                                dtype=projection_dtype, copy=True
                            )
                        )
                    captured["student"] = select(hidden)

                def score_teacher(
                    head, hidden, captured=capture, selected_parts=parts, select=pick
                ):
                    if head.weight.shape[0] < vocab_size:
                        raise ValueError(
                            "Batched teacher head has the wrong effective vocabulary"
                        )
                    sync()
                    tick = time.perf_counter()
                    q_hidden = select(hidden)
                    projection_dtype = (
                        dtype if device.type == "cuda" else head.weight.dtype
                    )
                    weight = head.weight.to(dtype=projection_dtype)
                    bias = (
                        None
                        if head.bias is None
                        else head.bias.to(dtype=projection_dtype)
                    )
                    for offset in range(0, len(q_hidden), position_chunk_size):
                        end = offset + position_chunk_size
                        p = F.linear(
                            captured["student"][offset:end],
                            snapshot["weight"],
                            snapshot["bias"],
                        )
                        q = F.linear(q_hidden[offset:end], weight, bias)
                        values = full_vocab_divergences(
                            p, q, vocab_size=vocab_size, chunk_size=vocab_chunk_size
                        )
                        for name in STATISTICS:
                            selected_parts[name].append(values[name])
                    sync()
                    timings["head_kl_seconds"] += time.perf_counter() - tick

                sync()
                tick = time.perf_counter()
                with _head_capture(
                    student, capture_student, batch_size=len(group), keep_batch=True
                ) as calls:
                    result = student(**kwargs)
                    if len(calls) != 1 or result.logits.numel():
                        raise RuntimeError(
                            "Batched student head capture did not suppress the full projection"
                        )
                sync()
                timings["student_forward_seconds"] += time.perf_counter() - tick
                tick, before = time.perf_counter(), timings["head_kl_seconds"]
                with _head_capture(
                    teacher, score_teacher, batch_size=len(group), keep_batch=True
                ) as calls:
                    result = teacher(**kwargs)
                    if len(calls) != 1 or result.logits.numel():
                        raise RuntimeError(
                            "Batched teacher head capture did not suppress the full projection"
                        )
                sync()
                timings["teacher_forward_seconds"] += (
                    time.perf_counter() - tick - (timings["head_kl_seconds"] - before)
                )
                timings["forward_batches"] += 1
                values = {
                    name: torch.cat(chunks).double().cpu()
                    for name, chunks in parts.items()
                }
                cursor = 0
                for record, (positions, weights) in zip(group, selected, strict=True):
                    count = len(positions)
                    local = {
                        name: value[cursor : cursor + count]
                        for name, value in values.items()
                    }
                    row = {
                        key: record[key]
                        for key in ("state_source", "source", "uid", "sample")
                    }
                    weight_tensor = torch.tensor(weights, dtype=torch.float64)
                    row.update(
                        record_sha256=record_fingerprint(record),
                        response_length=len(record["response_ids"]),
                        evaluated_positions=count,
                        sums={
                            key: float((value * weight_tensor).sum())
                            for key, value in local.items()
                        },
                        depths={},
                    )
                    if record["state_source"] == "on_validation_states":
                        for name, lower, upper in (
                            ("0_255", 0, 256),
                            ("256_1023", 256, 1024),
                            ("1024_2047", 1024, 2048),
                            ("2048_plus", 2048, count),
                        ):
                            upper = min(upper, count)
                            if lower < upper:
                                row["depths"][name] = {
                                    "tokens": upper - lower,
                                    "sums": {
                                        key: float(value[lower:upper].sum())
                                        for key, value in local.items()
                                    },
                                }
                    output.append(row)
                    cursor += count
    finally:
        for module, training in flags:
            module.training = training
    timings["seconds"] = time.perf_counter() - started
    timings["head_snapshot_bytes"] = sum(
        value.numel() * value.element_size()
        for value in snapshot.values()
        if value is not None
    )
    return output, timings
