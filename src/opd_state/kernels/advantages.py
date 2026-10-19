# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Detached token rewards shared by sampled and head-candidate OPD."""

from __future__ import annotations

import torch


def compute_token_reward_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the token-level teacher reward directly as the advantage.

    Args:
        token_level_rewards: ``(bs, response_length)`` for scalar-per-token OPD,
            or ``(bs, response_length, K)`` when top-K distillation is enabled.
        response_mask: ``(bs, response_length)``.
        config: unused; accepted to match the estimator calling convention.

    Returns:
        ``(advantages, returns)``, both the same shape as ``token_level_rewards``.
    """
    with torch.no_grad():
        mask = response_mask
        if token_level_rewards.dim() == 3:
            # top-K rewards are (bs, T, K); broadcast the (bs, T) mask over K.
            mask = response_mask.unsqueeze(-1)
        advantages = token_level_rewards * mask
        returns = advantages.clone()
    return advantages, returns
