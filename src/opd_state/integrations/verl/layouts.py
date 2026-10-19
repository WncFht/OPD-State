# Copyright 2026 OPD-State contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert verl response candidate IDs into packed predictor positions."""

import torch


def packed_response_heads(data):
    """Place cached response head IDs at the preceding predictor positions."""
    ids = data["student_top_k_ids"]
    prompt_width = data["prompts"].shape[1]
    attention = data["attention_mask"]
    lengths = attention.sum(-1).long()
    prompts = attention[:, :prompt_width].sum(-1).long()
    pieces = []
    for row, (length, prompt) in enumerate(
        zip(lengths.tolist(), prompts.tolist(), strict=True)
    ):
        count = length - prompt
        piece = (
            torch.arange(ids.shape[-1], device=ids.device).expand(length, -1).clone()
        )
        piece[prompt - 1 : prompt + count - 1] = ids[row, :count]
        pieces.append(piece)
    return torch.cat(pieces)
