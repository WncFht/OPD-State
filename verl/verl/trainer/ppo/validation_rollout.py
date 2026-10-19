"""Validation rollout statistics, kept separate from answer-score bootstraps."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def decode_effective_tokens(tokenizer, ids, mask):
    """Decode actual prompt/response positions instead of full padded widths."""
    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError("Validation decoding requires matching token and mask matrices")
    ids, mask = ids.detach().cpu(), mask.detach().cpu().bool()
    rows = [row[selected] for row, selected in zip(ids, mask, strict=True)]
    if callable(getattr(tokenizer, "batch_decode", None)):
        return tokenizer.batch_decode(rows, skip_special_tokens=True)
    return [tokenizer.decode(row, skip_special_tokens=True) for row in rows]


def rollout_record(
    *,
    source,
    uid,
    prompt_length,
    response_ids,
    eos_token_id,
    info,
    correct=None,
    prediction=None,
    prediction_available=False,
):
    """Use effective tokens, including EOS even when its ID is also PAD.

    Native completion failures raise before this point. A successful summary
    therefore describes completed responses; it cannot establish a failure rate
    for a run which aborted before validation completed.
    """
    tokens = list(response_ids)
    cap = info["response_cap"]
    if type(cap) is not int or cap < 1 or len(tokens) > cap or prompt_length < 1:
        raise ValueError("Invalid validation prompt/response length or response cap")
    eos = bool(tokens and tokens[-1] == eos_token_id)
    if eos_token_id in tokens[:-1]:
        raise ValueError("Validation response continued after EOS")
    if info["termination"] not in {"eos", "length", "stop"}:
        raise ValueError("Validation native completion did not finish successfully")
    if (info["termination"] == "eos") != eos:
        raise ValueError("Validation native EOS metadata disagrees with effective tokens")
    if info["finish_reason"] not in {"stop", "length"}:
        raise ValueError("Validation native finish reason is invalid")
    if info["finish_reason"] == "length" and len(tokens) != cap:
        raise ValueError("Validation native length finish did not exhaust its response cap")
    if info["termination"] == "length" and (eos or len(tokens) != cap):
        raise ValueError("Validation truncation metadata disagrees with effective length")
    if correct is not None and float(correct) not in (0.0, 1.0):
        raise ValueError("Validation accuracy must be binary")
    grams = Counter(tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3)))
    gram_count = sum(grams.values())
    return {
        "source": str(source),
        "uid": str(uid),
        "prompt_length": int(prompt_length),
        "response_length": len(tokens),
        "response_cap": cap,
        "termination": info["termination"],
        "finish_reason": info["finish_reason"],
        "stop_reason": info.get("stop_reason"),
        "eos": eos,
        "truncated": info["termination"] == "length",
        "empty": not tokens,
        "fourgram_count": gram_count,
        "repeated_fourgram_count": gram_count - len(grams),
        "correct": None if correct is None else bool(correct),
        "prediction_available": bool(prediction_available),
        "prediction": None if prediction is None or not str(prediction).strip() else str(prediction),
        "generation_prompt_id": info["prompt_id"],
        "generation_sample": info["sample"],
        "student_version": info["student_version"],
    }


def rows_from_batch(batch, reward_extra_info, eos_token_id):
    """Collect after DP unpadding, preserving row identity and raw stop reasons."""
    import torch

    mask = batch.batch["response_mask"].cpu()
    valid = mask.bool()
    lengths = valid.sum(-1)
    depth = torch.arange(mask.shape[1]).unsqueeze(0)
    if not torch.equal(mask, valid.to(mask.dtype)) or not torch.equal(valid, depth < lengths.unsqueeze(-1)):
        raise ValueError("Validation responses require a binary contiguous effective mask")
    responses = batch.batch["responses"].cpu()
    prompt_width = batch.batch["prompts"].shape[-1]
    prompt_lengths = batch.batch["attention_mask"][:, :prompt_width].sum(-1).cpu().tolist()
    info = batch.non_tensor_batch["native_response_info"]
    sources = batch.non_tensor_batch["data_source"]
    uids = batch.non_tensor_batch["uid"]
    accuracy = reward_extra_info.get("acc", [None] * len(batch))
    predictions = reward_extra_info.get("pred", [None] * len(batch))
    if not all(len(column) == len(batch) for column in (info, sources, uids, accuracy, predictions)):
        raise ValueError("Validation diagnostic columns do not match the unpadded response count")
    return [
        rollout_record(
            source=sources[i],
            uid=uids[i],
            prompt_length=prompt_lengths[i],
            response_ids=responses[i, : int(lengths[i])].tolist(),
            eos_token_id=eos_token_id,
            info=info[i],
            correct=accuracy[i],
            prediction=predictions[i],
            prediction_available="pred" in reward_extra_info,
        )
        for i in range(len(batch))
    ]


def _quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize_rollouts(rows, *, samples_per_question=None):
    """Return per-benchmark and pooled statistics; never average batch means."""
    if not rows:
        return {}
    grouped = defaultdict(list)
    by_question = defaultdict(list)
    sample_keys = set()
    for row in rows:
        if row["source"] in {"macro", "micro"}:
            raise ValueError("Validation benchmark names cannot use reserved aggregation names")
        key = row["source"], row["uid"], row["generation_sample"]
        if key in sample_keys:
            raise ValueError("Validation diagnostics contain a duplicated response sample")
        sample_keys.add(key)
        grouped[row["source"]].append(row)
        by_question[row["source"], row["uid"]].append(row)
    if samples_per_question is not None:
        if type(samples_per_question) is not int or samples_per_question < 1:
            raise ValueError("Validation samples per question must be a positive integer")
        if any(len(values) != samples_per_question for values in by_question.values()):
            raise ValueError("Validation diagnostic sample cardinality differs from the accuracy protocol")
    metrics = {}

    def record(source, values):
        prefix = f"validation_rollout/{source}/"
        n = len(values)
        questions = {(v["source"], v["uid"]) for v in values}
        out = {"responses": n, "questions": len(questions)}
        lengths = [r["response_length"] for r in values]
        out["response_tokens"] = sum(lengths)
        out["response_length/mean"] = sum(lengths) / n
        for name, probability in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("max", 1.0)]:
            out[f"response_length/{name}"] = _quantile(lengths, probability)
        out["prompt_length/mean"] = sum(r["prompt_length"] for r in values) / n
        for flag in ("eos", "truncated", "empty"):
            out[f"{flag}/count"] = sum(r[flag] for r in values)
            out[f"{flag}/fraction"] = out[f"{flag}/count"] / n
        out["other_stop/count"] = sum(r["termination"] == "stop" for r in values)
        out["eos_at_cap/count"] = sum(r["eos"] and r["response_length"] == r["response_cap"] for r in values)
        eligible = [r for r in values if r["fourgram_count"]]
        out["repeat_4gram/eligible_responses"] = len(eligible)
        if eligible:
            out["repeat_4gram/response_mean"] = sum(
                r["repeated_fourgram_count"] / r["fourgram_count"] for r in eligible
            ) / len(eligible)
            out["repeat_4gram/token_weighted"] = sum(r["repeated_fourgram_count"] for r in eligible) / sum(
                r["fourgram_count"] for r in eligible
            )
        available = [r for r in values if r["prediction_available"]]
        out["answer_parse/available_responses"] = len(available)
        if available:
            out["answer_parse/fraction"] = sum(r["prediction"] is not None for r in available) / len(available)
            diverse = [
                len({r["prediction"] for r in by_question[key] if r["prediction"] is not None}) for key in questions
            ]
            out["answers/unique_per_question_mean"] = sum(diverse) / len(diverse)
        for name, subset in [
            ("correct", [r for r in values if r["correct"] is True]),
            ("incorrect", [r for r in values if r["correct"] is False]),
            ("truncated", [r for r in values if r["truncated"]]),
        ]:
            out[f"{name}/responses"] = len(subset)
            if subset:
                out[f"{name}/response_length_mean"] = sum(r["response_length"] for r in subset) / len(subset)
                out[f"{name}/response_length_p95"] = _quantile([r["response_length"] for r in subset], 0.95)
        metrics.update({prefix + key: value for key, value in out.items()})

    for source, values in sorted(grouped.items()):
        record(source, values)
    record("micro", rows)
    # Macro percentiles would be means of percentiles, not a pooled quantile.
    for field in (
        "response_length/mean",
        "prompt_length/mean",
        "eos/fraction",
        "truncated/fraction",
        "empty/fraction",
        "answer_parse/fraction",
        "repeat_4gram/response_mean",
    ):
        keys = [f"validation_rollout/{source}/{field}" for source in grouped]
        if all(key in metrics for key in keys):
            metrics[f"validation_rollout/macro/{field}"] = sum(metrics[key] for key in keys) / len(keys)
    return metrics


def dump_columns(rows):
    """Append raw per-response evidence without entering reward metric reducers."""
    if not rows:
        return {}
    fields = (
        "prompt_length",
        "response_length",
        "response_cap",
        "termination",
        "finish_reason",
        "stop_reason",
        "fourgram_count",
        "repeated_fourgram_count",
        "generation_prompt_id",
        "generation_sample",
        "student_version",
    )
    return {f"rollout_{key}": [row[key] for row in rows] for key in fields}
