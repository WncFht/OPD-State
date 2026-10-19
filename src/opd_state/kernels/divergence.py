"""Full-vocabulary conditional KL and entropy with bounded FP32 reductions."""

from __future__ import annotations


def full_vocab_divergences(
    student_logits, teacher_logits, *, vocab_size=None, chunk_size=8192
):
    """Compute both KL directions and entropies for selected prefixes in FP32.

    Inputs are [states, vocabulary] logits at T=1, before top-p/top-k. A supplied
    vocabulary size excludes head padding; the caller must establish that the
    rollout sampler uses this same support before claiming sequence KL. Chunked
    two-pass normalization avoids materializing full FP32 probability tensors.
    No tensors are modified and no result participates in autograd.
    """
    import torch

    if (
        student_logits.ndim != 2
        or teacher_logits.shape != student_logits.shape
        or not student_logits.shape[0]
    ):
        raise ValueError(
            "KL logits require matching nonempty [states, vocabulary] shapes"
        )
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
        if chunk_size >= size:
            # Native full-block reductions avoid Python vocabulary launches.
            # logsumexp retains the existing FP32 normalization accuracy on
            # the large effective vocabulary (including the CPU reference).
            p_logits, q_logits = student_logits[:, :size], teacher_logits[:, :size]
            finite = torch.isfinite(p_logits).all() & torch.isfinite(q_logits).all()
            p, q = p_logits.float(), q_logits.float()
            logp = p - torch.logsumexp(p, dim=-1, keepdim=True)
            logq = q - torch.logsumexp(q, dim=-1, keepdim=True)
            probp, probq = logp.exp(), logq.exp()
            difference = logp - logq
            result = {
                "student_teacher_kl": (probp * difference).sum(-1),
                "teacher_student_kl": -(probq * difference).sum(-1),
                "student_entropy": -(probp * logp).sum(-1),
                "teacher_entropy": -(probq * logq).sum(-1),
            }
            finite.logical_and_(
                torch.stack(
                    [torch.isfinite(value).all() for value in result.values()]
                ).all()
            )
            if not finite:
                raise ValueError(
                    "KL logits and reduction must be finite on the effective vocabulary"
                )
            return result

        def chunks():
            for begin in range(0, size, chunk_size):
                end = min(begin + chunk_size, size)
                yield (
                    student_logits[:, begin:end].float(),
                    teacher_logits[:, begin:end].float(),
                )

        normalizer_parts = [[], []]
        finite = torch.ones((), dtype=torch.bool, device=student_logits.device)
        for p, q in chunks():
            finite.logical_and_(torch.isfinite(p).all()).logical_and_(
                torch.isfinite(q).all()
            )
            for index, value in enumerate((p, q)):
                normalizer_parts[index].append(torch.logsumexp(value, dim=-1))
        if not finite:
            raise ValueError("KL logits must be finite on the effective vocabulary")
        # Reduce the small block summaries together. Sequential logaddexp/add_
        # accumulated visible FP32 error with the 151,665-token vocabulary.
        normalizers = [
            torch.logsumexp(torch.stack(parts), dim=0) for parts in normalizer_parts
        ]
        parts = {
            key: []
            for key in (
                "student_teacher_kl",
                "teacher_student_kl",
                "student_entropy",
                "teacher_entropy",
            )
        }
        for p, q in chunks():
            logp, logq = (
                p - normalizers[0].unsqueeze(-1),
                q - normalizers[1].unsqueeze(-1),
            )
            probp, probq = logp.exp(), logq.exp()
            parts["student_teacher_kl"].append((probp * (logp - logq)).sum(-1))
            parts["teacher_student_kl"].append((probq * (logq - logp)).sum(-1))
            parts["student_entropy"].append(-(probp * logp).sum(-1))
            parts["teacher_entropy"].append(-(probq * logq).sum(-1))
        result = {key: torch.stack(values).sum(0) for key, values in parts.items()}
        if not torch.stack(
            [torch.isfinite(value).all() for value in result.values()]
        ).all():
            raise ValueError("KL reduction produced non-finite statistics")
        return result
