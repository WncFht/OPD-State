"""Token, question and cumulative KL reductions on preserved response records."""

import math
from collections import defaultdict

from opd_state.core.records import STATISTICS


def summarize_kl(rows):
    """Keep token weighting, per-question weighting and cumulative mismatch separate."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["state_source"], row["source"], row["uid"], row["sample"]
        if key in seen or row["source"] in {"micro", "macro"}:
            raise ValueError(
                "KL response identities must be unique and benchmark names unreserved"
            )
        seen.add(key)
        if row["response_length"] < 1 or any(
            not math.isfinite(value) for value in row["sums"].values()
        ):
            raise ValueError(
                "KL response lengths and statistics must be valid and finite"
            )
        groups[row["state_source"], row["source"]].append(row)
    metrics = {}

    def record(state_source, source, values):
        prefix = f"validation_kl/{state_source}/{source}/"
        total_tokens = sum(row["response_length"] for row in values)
        questions = defaultdict(list)
        for row in values:
            questions[row["source"], row["uid"]].append(row)
        out = {
            "responses": len(values),
            "questions": len(questions),
            "response_tokens": total_tokens,
            "evaluated_positions": sum(row["evaluated_positions"] for row in values),
        }
        for name in STATISTICS:
            out[name + "/token_weighted"] = (
                math.fsum(row["sums"][name] for row in values) / total_tokens
            )
            out[name + "/question_mean"] = math.fsum(
                math.fsum(row["sums"][name] / row["response_length"] for row in items)
                / len(items)
                for items in questions.values()
            ) / len(questions)
            out[name + "/cumulative_on_states"] = math.fsum(
                math.fsum(row["sums"][name] for row in items) / len(items)
                for items in questions.values()
            ) / len(questions)
        for depth in ("0_255", "256_1023", "1024_2047", "2048_plus"):
            parts = [row.get("depths", {}).get(depth) for row in values]
            parts = [part for part in parts if part is not None]
            count = sum(part["tokens"] for part in parts)
            out[f"depth/{depth}/tokens"] = count
            if count:
                for name in STATISTICS:
                    out[f"depth/{depth}/{name}/token_weighted"] = (
                        math.fsum(part["sums"][name] for part in parts) / count
                    )
        metrics.update({prefix + key: value for key, value in out.items()})

    for (state_source, source), values in sorted(groups.items()):
        record(state_source, source, values)
    for state_source in sorted({state for state, _ in groups}):
        sources = [source for state, source in groups if state == state_source]
        record(
            state_source,
            "micro",
            [row for row in rows if row["state_source"] == state_source],
        )
        for name in STATISTICS:
            for reduction in (
                "question_mean",
                "token_weighted",
                "cumulative_on_states",
            ):
                key = name + "/" + reduction
                metrics[f"validation_kl/{state_source}/macro/{key}"] = math.fsum(
                    metrics[f"validation_kl/{state_source}/{source}/{key}"]
                    for source in sources
                ) / len(sources)
    return metrics
