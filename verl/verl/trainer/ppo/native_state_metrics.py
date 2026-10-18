"""Sampled-action diagnostics on native prefixes, without extra model forwards."""

import torch

SOURCES = {"current_student", "initial_student", "teacher", "fusion_current_student", "fusion_initial_student"}


def trajectory_segments(trajectories):
    """Keep compact row metadata so DataProto reordering also reorders provenance."""
    rows = []
    for trajectory in trajectories:
        segments = []
        for index, segment in enumerate(trajectory.segments):
            roles = tuple(identity[0] for identity in segment.identities)
            if len(roles) == 1:
                source = roles[0]
            elif roles in [("current_student", "teacher"), ("initial_student", "teacher")]:
                source = "fusion_" + roles[0]
            else:
                raise ValueError("Unsupported action-policy identity in native diagnostic metadata")
            if source not in SOURCES:
                raise ValueError("Unknown native action-policy source")
            segments.append((source, index, segment.start, segment.end))
        rows.append(tuple(segments))
    return rows


@torch.no_grad()
def summarize_native_states(response_mask, student_logp, teacher_logp, segments):
    """Describe visited prefix/action pairs; signed log-ratio is not a full KL.

    A stage label describes the next action's policy. The state at that position
    contains every earlier token, including tokens from a different first stage.
    Absolute response depth excludes the original prompt's token count.
    """
    if (
        response_mask.ndim != 2
        or student_logp.shape != response_mask.shape
        or teacher_logp.shape != response_mask.shape
    ):
        raise ValueError("Native diagnostic scores and response mask need identical batch/token shapes")
    valid = response_mask.bool()
    lengths = valid.sum(-1)
    depth = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    if len(segments) != valid.shape[0] or not torch.equal(valid, depth < lengths.unsqueeze(-1)):
        raise ValueError("Native diagnostic metadata must match contiguous valid responses")
    if (lengths < 1).any() or not torch.equal(response_mask, valid.to(response_mask.dtype)):
        raise ValueError("Native diagnostics require nonempty responses and a binary mask")
    if not torch.isfinite(student_logp[valid]).all() or not torch.isfinite(teacher_logp[valid]).all():
        raise ValueError("Valid native diagnostic action log probabilities must be finite")
    names = sorted(SOURCES)
    source_codes = torch.zeros_like(response_mask, dtype=torch.int8)
    stages = torch.full_like(source_codes, -1)
    for i, row in enumerate(segments):
        end = 0
        for expected_stage, (source, stage, start, stop) in enumerate(row):
            if (
                source not in SOURCES
                or stage != expected_stage
                or stage > 1
                or start != end
                or not start < stop <= lengths[i]
            ):
                raise ValueError("Native diagnostic stages have invalid provenance or response coverage")
            source_codes[i, start:stop] = names.index(source) + 1
            stages[i, start:stop] = stage
            end = stop
        if end != int(lengths[i]):
            raise ValueError("Native diagnostic stages do not cover the complete valid response")

    metrics = {}

    def record(name, selected):
        count = int(selected.sum())
        prefix = f"native_state/{name}/"
        metrics[prefix + "tokens"] = count
        if not count:
            return
        p, q = student_logp[selected].float(), teacher_logp[selected].float()
        delta = p - q
        metrics.update(
            {
                prefix + "response_prefix_depth_mean": float(depth[selected].float().mean()),
                prefix + "student_logp_mean": float(p.mean()),
                prefix + "teacher_logp_mean": float(q.mean()),
                prefix + "teacher_selected_action_probability_mean": float(q.exp().mean()),
                prefix + "sampled_log_ratio_mean": float(delta.mean()),
                prefix + "log_ratio_rms": float(delta.square().mean().sqrt()),
            }
        )

    record("all", valid)
    for stage in [0, 1]:
        record(f"segment_{stage}", valid & (stages == stage))
    for code, name in enumerate(names, 1):
        selected = valid & (source_codes == code)
        if selected.any():
            record(f"action_{name}", selected)
            for stage in [0, 1]:
                if (selected & (stages == stage)).any():
                    record(f"segment_{stage}/action_{name}", selected & (stages == stage))
    for lower, upper in [(0, 256), (256, 1024), (1024, 2048), (2048, valid.shape[1])]:
        if lower < upper:
            record(f"depth_{lower}_{upper - 1}", valid & (depth >= lower) & (depth < upper))
    return metrics
