"""Full-vocabulary conditional divergences and auditable prefix sampling.

These functions do not infer a trajectory's sampling distribution. A weighted
sum is a sequence-KL estimator only for trajectories actually sampled from the
corresponding student distribution with the same EOS and length-cap protocol.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixSample:
    position: int
    begin: int
    end: int

    @property
    def weight(self):
        return self.end - self.begin


def sample_prefix_positions(length, *, max_positions=8, seed=0):
    """Always include action zero, then sample uniformly within disjoint strata.

    Positions index response actions, not input-token positions. Weighting by
    stratum size makes the estimated sum unbiased conditional on the trajectory.
    """
    if type(length) is not int or length < 1:
        raise ValueError("KL prefix sampling requires a nonempty effective response")
    if type(max_positions) is not int or max_positions < 2:
        raise ValueError("KL prefix sampling needs at least two available positions")
    if type(seed) is not int or seed < 0:
        raise ValueError("KL prefix sampling requires a nonnegative integer seed")
    if length <= max_positions:
        return tuple(PrefixSample(i, i, i + 1) for i in range(length))
    generator = random.Random(seed)
    strata = max_positions - 1
    result = [PrefixSample(0, 0, 1)]
    for i in range(strata):
        begin = 1 + (length - 1) * i // strata
        end = 1 + (length - 1) * (i + 1) // strata
        result.append(PrefixSample(generator.randrange(begin, end), begin, end))
    return tuple(result)


def estimate_trajectory_statistics(samples, values, *, length):
    """Estimate per-response sums and means; reject incomplete stratum coverage."""
    if type(length) is not int or length < 1 or not samples or not values:
        raise ValueError("KL estimation requires positions, statistics and a valid effective length")
    cursor = 0
    for sample in samples:
        if (
            any(type(v) is not int for v in (sample.position, sample.begin, sample.end))
            or sample.begin != cursor
            or not sample.begin <= sample.position < sample.end <= length
        ):
            raise ValueError("KL sample strata must exactly partition the effective response")
        cursor = sample.end
    if cursor != length:
        raise ValueError("KL sample strata do not cover the entire effective response")
    result = {"response_length": length, "evaluated_positions": len(samples)}
    for name, entries in values.items():
        if len(entries) != len(samples) or not all(math.isfinite(float(v)) for v in entries):
            raise ValueError("KL statistics must be finite and align with the sampled positions")
        total = math.fsum(sample.weight * float(value) for sample, value in zip(samples, entries, strict=True))
        result[name + "/sum_est"] = total
        result[name + "/token_mean_est"] = total / length
    return result


def full_vocab_divergences(student_logits, teacher_logits, *, vocab_size=None, chunk_size=8192):
    """Compute both KL directions and entropies for selected prefixes in FP32.

    Inputs are [states, vocabulary] logits at T=1, before top-p/top-k. A supplied
    vocabulary size excludes head padding; the caller must establish that the
    rollout sampler uses this same support before claiming sequence KL. Chunked
    two-pass normalization avoids materializing full FP32 probability tensors.
    No tensors are modified and no result participates in autograd.
    """
    import torch

    if student_logits.ndim != 2 or teacher_logits.shape != student_logits.shape or not student_logits.shape[0]:
        raise ValueError("KL logits require matching nonempty [states, vocabulary] shapes")
    if student_logits.device != teacher_logits.device:
        raise ValueError("KL student and teacher logits must be on the same device")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("KL logits must have floating-point types")
    size = student_logits.shape[-1] if vocab_size is None else vocab_size
    if type(size) is not int or not 1 <= size <= student_logits.shape[-1]:
        raise ValueError("KL vocabulary size must select a nonempty common support")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("KL vocabulary chunk size must be positive")
    with torch.no_grad():

        def chunks():
            for begin in range(0, size, chunk_size):
                end = min(begin + chunk_size, size)
                yield student_logits[:, begin:end].float(), teacher_logits[:, begin:end].float()

        normalizer_parts = [[], []]
        for p, q in chunks():
            if not torch.isfinite(p).all() or not torch.isfinite(q).all():
                raise ValueError("KL logits must be finite on the effective vocabulary")
            for index, value in enumerate((p, q)):
                normalizer_parts[index].append(torch.logsumexp(value, dim=-1))
        # Reduce the small block summaries together. Sequential logaddexp/add_
        # accumulated visible FP32 error with the 151,665-token vocabulary.
        normalizers = [torch.logsumexp(torch.stack(parts), dim=0) for parts in normalizer_parts]
        parts = {key: [] for key in ("student_teacher_kl", "teacher_student_kl", "student_entropy", "teacher_entropy")}
        for p, q in chunks():
            logp, logq = p - normalizers[0].unsqueeze(-1), q - normalizers[1].unsqueeze(-1)
            probp, probq = logp.exp(), logq.exp()
            parts["student_teacher_kl"].append((probp * (logp - logq)).sum(-1))
            parts["teacher_student_kl"].append((probq * (logq - logp)).sum(-1))
            parts["student_entropy"].append(-(probp * logp).sum(-1))
            parts["teacher_entropy"].append(-(probq * logq).sum(-1))
        result = {key: torch.stack(values).sum(0) for key, values in parts.items()}
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("KL reduction produced non-finite statistics")
        return result
