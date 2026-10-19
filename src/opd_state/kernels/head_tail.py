# Copyright 2026 OPD-State contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-normalization head and importance-corrected on-policy tail."""

import torch


def selected_and_tail(logits, k, ids=None, *, include_tail=True, chunk_size=64):
    """Use full normalization and complement logsumexp (no 1-head cancellation)."""
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError("top-k must lie inside the effective vocabulary")
    if ids is None:
        ids = logits.topk(k, dim=-1).indices
    if ids.shape != (*logits.shape[:-1], k):
        raise ValueError("Frozen head shape does not match prefix logits")
    normalizer = logits.to(dtype).logsumexp(-1)
    head = logits.gather(-1, ids).to(dtype) - normalizer.unsqueeze(-1)
    result = {"student_top_k_ids": ids.long(), "student_top_k_log_probs": head}
    if include_tail:
        with torch.no_grad():
            flat = logits.detach().reshape(-1, logits.shape[-1])
            flat_ids = ids.reshape(-1, k)
            pieces = []
            for start in range(0, len(flat), chunk_size):
                block = flat[start : start + chunk_size].to(dtype).clone()
                block.scatter_(-1, flat_ids[start : start + chunk_size], -torch.inf)
                pieces.append(block.logsumexp(-1))
            tail = torch.cat(pieces).reshape(normalizer.shape) - normalizer.detach()
            result["student_tail_log_mass"] = tail
    return result


def coefficients_and_metrics(
    student_head,
    teacher_head,
    student_action,
    teacher_action,
    student_tail,
    teacher_tail,
    head_ids,
    actions,
    mask,
    behavior_action=None,
):
    """Compute frozen coefficients only on valid response positions."""
    valid = mask.bool()
    head_coeff = torch.zeros_like(student_head)
    tail_coeff = torch.zeros_like(student_action)
    names = (
        "kl_head",
        "kl_grouped",
        "kl_unbiased",
        "tail_residual",
        "student_tail_mass",
        "teacher_tail_mass",
        "sample_in_tail",
        "action_importance_ratio",
        "behavior_logratio",
    )
    stats = {name: torch.zeros_like(student_action) for name in names}
    with torch.no_grad():
        ph, qh = student_head[valid], teacher_head[valid]
        pa, qa = student_action[valid], teacher_action[valid]
        pt, qt = student_tail[valid], teacher_tail[valid]
        behavior = pa if behavior_action is None else behavior_action[valid]
        weight = (pa - behavior).exp()
        event = ~head_ids[valid].eq(actions[valid].unsqueeze(-1)).any(-1)
        coeff = ph.exp() * (ph - qh)
        head = coeff.sum(-1)
        nonempty = torch.isfinite(pt)
        beta = torch.zeros_like(pt)
        beta[nonempty] = pt[nonempty] - qt[nonempty]
        grouped = head + pt.exp() * beta
        residual = torch.where(event, weight * (pa - qa - beta), torch.zeros_like(pa))
        head_coeff[valid] = coeff
        tail_coeff[valid] = torch.where(event, weight * (pa - qa), torch.zeros_like(pa))
        values = (
            head,
            grouped,
            grouped + residual,
            residual,
            pt.exp(),
            qt.exp(),
            event.to(pa.dtype),
            weight,
            behavior - qa,
        )
        for name, value in zip(names, values, strict=True):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Nonfinite {name}; check teacher support/normalization"
                )
            stats[name][valid] = value
    return head_coeff.detach(), tail_coeff.detach(), stats


def score_loss(head_logp, action_logp, head_coefficient, tail_coefficient, mask):
    """One loss per prefix; invalid positions never enter arithmetic."""
    valid = mask.bool()
    out = torch.zeros_like(action_logp)
    out[valid] = (head_logp[valid] * head_coefficient[valid].detach()).sum(-1)
    out[valid] += action_logp[valid] * tail_coefficient[valid].detach()
    return out


def summarize_kl(stats, mask):
    """Global batch readouts; sequence sums and token means have separate names."""
    valid = mask.bool()
    n = int(valid.sum())
    if not n:
        raise ValueError("KL diagnostics require at least one valid response token")
    result = {
        f"opd_alignment/{key}_token_mean": float(value[valid].double().mean())
        for key, value in stats.items()
    }
    seq = stats["kl_unbiased"].double().sum(-1)
    result["opd_alignment/kl_unbiased_sequence_mean"] = float(seq.mean())
    result["opd_alignment/kl_unbiased_sequence_se"] = (
        float(seq.std(unbiased=True) / len(seq) ** 0.5) if len(seq) > 1 else 0.0
    )
    if "behavior_logratio" in stats:
        behavior_seq = stats["behavior_logratio"].double().sum(-1)
        result["opd_alignment/behavior_kl_sequence_mean"] = float(behavior_seq.mean())
    result["opd_alignment/mean_response_tokens"] = n / len(seq)
    result["opd_alignment/sequences"] = len(seq)
    result["opd_alignment/valid_tokens"] = n
    return result
