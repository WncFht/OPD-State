"""Assemble audited native trajectories into an immutable common-prefix bank."""

from __future__ import annotations

from dataclasses import asdict

from verl.trainer.ppo.fixed_rollout import digest_json
from opd_state.evaluation.prefixes import sample_prefix_positions
from opd_state.core.policy import stable_seed

ROLES = {"initial_student": "on_initial_states", "teacher": "on_teacher_states"}


def assemble(prepared, trajectories, *, input_sha256, generation_provenance):
    """Require both frozen sources on every exact prompt before publishing a bank."""
    if prepared.get("schema") != "fixed-prefix-input-v1" or set(trajectories) != set(ROLES):
        raise ValueError("Prefix-bank generation must cover both prepared frozen sources")
    questions = prepared["questions"]
    if not questions or len({row["uid"] for row in questions}) != len(questions):
        raise ValueError("Prefix-bank prepared questions must be nonempty and unique")
    records = []
    for role, state_source in ROLES.items():
        rows = trajectories[role]
        indexed = {row.request.prompt_id: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != {row["uid"] for row in questions}:
            raise ValueError("Prefix-bank generated questions are missing or duplicated")
        model = prepared["models"][role]
        expected_identity = (role, model["path"], model["fingerprint"])
        for question in questions:
            row = indexed[question["uid"]]
            if (
                tuple(row.request.prompt_ids) != tuple(question["prompt_ids"])
                or row.request.visit != 0
                or row.request.sample != 0
                or len(row.segments) != 1
            ):
                raise ValueError("Prefix-bank source changed the original prefix or request identity")
            segment = row.segments[0]
            if (
                tuple(map(tuple, segment.identities)) != (expected_identity,)
                or segment.start != 0
                or segment.end != len(row.token_ids)
                or segment.teacher_weight is not None
            ):
                raise ValueError("Prefix-bank trajectory used the wrong frozen model or segment coverage")
            length = len(row.token_ids)
            if not 1 <= length <= prepared["response_cap"]:
                raise ValueError("Prefix-bank generated an empty or excessive response")
            if segment.finish_reason not in {"stop", "length"} or (
                segment.finish_reason == "length" and length != prepared["response_cap"]
            ):
                raise ValueError("Prefix-bank raw finish reason disagrees with its response budget")
            eos = prepared["eos_token_id"]
            if eos in row.token_ids[:-1] or (row.termination == "eos") != (row.token_ids[-1] == eos):
                raise ValueError("Prefix-bank EOS metadata disagrees with actual tokens")
            if row.termination not in {"eos", "length", "stop"}:
                raise ValueError("Prefix-bank native request did not complete")
            if row.termination == "length" and length != prepared["response_cap"]:
                raise ValueError("Prefix-bank truncation did not exhaust the declared response budget")
            if any(type(token) is not int or not 0 <= token < prepared["vocab_size"] for token in row.token_ids):
                raise ValueError("Prefix-bank response contains invalid token IDs")
            position_seed = stable_seed(prepared["selection_seed"], "bank-positions", question["uid"], role)
            samples = sample_prefix_positions(
                length, max_positions=prepared["positions_per_response"], seed=position_seed
            )
            records.append(
                {
                    "state_source": state_source,
                    "source": question["source"],
                    "uid": question["uid"],
                    "sample": 0,
                    "prompt_ids": list(question["prompt_ids"]),
                    "response_ids": list(row.token_ids),
                    "positions": [sample.position for sample in samples],
                    "weights": [sample.weight for sample in samples],
                    "strata": [asdict(sample) for sample in samples],
                    "position_seed": position_seed,
                    "termination": row.termination,
                    "native_segment": asdict(segment),
                }
            )
    return {
        "schema": "fixed-prefix-bank-v1",
        "tokenizer_signature": prepared["tokenizer_signature"],
        "vocab_size": prepared["vocab_size"],
        "initial_student_fingerprint": prepared["models"]["initial_student"]["fingerprint"],
        "teacher_fingerprint": prepared["models"]["teacher"]["fingerprint"],
        "input_sha256": input_sha256,
        "question_ids_sha256": digest_json([row["uid"] for row in questions]),
        "sampling": prepared["sampling"],
        "response_cap": prepared["response_cap"],
        "per_benchmark": prepared["per_benchmark"],
        "generation_provenance": generation_provenance,
        "records": records,
    }
