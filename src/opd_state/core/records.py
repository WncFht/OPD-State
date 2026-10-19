"""Raw-token validation record identity and complete position coverage."""

import hashlib
import json
import math

STATISTICS = (
    "student_teacher_kl",
    "teacher_student_kl",
    "student_entropy",
    "teacher_entropy",
)
STATE_SOURCES = {"on_validation_states", "on_initial_states", "on_teacher_states"}


def record_fingerprint(record):
    fields = (
        "state_source",
        "source",
        "uid",
        "sample",
        "prompt_ids",
        "response_ids",
        "positions",
        "weights",
    )
    payload = {key: record.get(key) for key in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_record(record, vocab_size):
    prompt, response = record["prompt_ids"], record["response_ids"]
    if (
        not prompt
        or not response
        or any(
            type(i) is not int or not 0 <= i < vocab_size for i in (*prompt, *response)
        )
    ):
        raise ValueError(
            "KL scoring needs nonempty raw prompt/response IDs within the effective vocabulary"
        )
    if record["state_source"] not in STATE_SOURCES:
        raise ValueError("Unknown validation KL state source")
    length = len(response)
    positions = record.get("positions", list(range(length)))
    weights = record.get("weights", [1.0] * len(positions))
    if not positions or any(
        type(i) is not int or not 0 <= i < length for i in positions
    ):
        raise ValueError("KL scoring positions must index valid response actions")
    if sorted(set(positions)) != positions:
        raise ValueError("KL scoring positions must be sorted and unique")
    if len(weights) != len(positions) or any(
        not math.isfinite(float(w)) or w <= 0 for w in weights
    ):
        raise ValueError(
            "KL scoring weights must be finite, positive and match positions"
        )
    if not math.isclose(math.fsum(weights), length, rel_tol=0, abs_tol=1e-8):
        raise ValueError("KL scoring position weights must cover the response length")
    if record["state_source"] == "on_validation_states" and (
        positions != list(range(length)) or any(w != 1 for w in weights)
    ):
        raise ValueError(
            "Main validation KL must score every effective response action"
        )
    return positions, weights
