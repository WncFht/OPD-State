"""Convert verl validation batches into preserved tokens and diagnostic rows."""

from opd_state.evaluation.rollout import rollout_record


def decode_effective_tokens(tokenizer, ids, mask):
    """Decode actual prompt/response positions instead of full padded widths."""
    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError(
            "Validation decoding requires matching token and mask matrices"
        )
    ids, mask = ids.detach().cpu(), mask.detach().cpu().bool()
    rows = [row[selected] for row, selected in zip(ids, mask, strict=True)]
    if callable(getattr(tokenizer, "batch_decode", None)):
        return tokenizer.batch_decode(rows, skip_special_tokens=True)
    return [tokenizer.decode(row, skip_special_tokens=True) for row in rows]


def rows_from_batch(batch, reward_extra_info, eos_token_id):
    """Collect after DP unpadding, preserving row identity and raw stop reasons."""
    import torch

    mask = batch.batch["response_mask"].cpu()
    valid = mask.bool()
    lengths = valid.sum(-1)
    depth = torch.arange(mask.shape[1]).unsqueeze(0)
    if not torch.equal(mask, valid.to(mask.dtype)) or not torch.equal(
        valid, depth < lengths.unsqueeze(-1)
    ):
        raise ValueError(
            "Validation responses require a binary contiguous effective mask"
        )
    responses = batch.batch["responses"].cpu()
    prompt_width = batch.batch["prompts"].shape[-1]
    prompt_lengths = (
        batch.batch["attention_mask"][:, :prompt_width].sum(-1).cpu().tolist()
    )
    info = batch.non_tensor_batch["native_response_info"]
    sources = batch.non_tensor_batch["data_source"]
    uids = batch.non_tensor_batch["uid"]
    accuracy = reward_extra_info.get("acc", [None] * len(batch))
    predictions = reward_extra_info.get("pred", [None] * len(batch))
    if not all(
        len(column) == len(batch)
        for column in (info, sources, uids, accuracy, predictions)
    ):
        raise ValueError(
            "Validation diagnostic columns do not match the unpadded response count"
        )
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
