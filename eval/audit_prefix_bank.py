"""Audit a published prefix bank against prepared prompts and native receipts.

This is a CPU artifact audit. It does not rerun a neural model or establish
CUDA Graph execution from a requested configuration flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from native_devices import matches_device

ROLES = {"initial_student": "on_initial_states", "teacher": "on_teacher_states"}
BENCHMARKS = {"MATH-500", "AMC23", "AIME25"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, original_input, *, per_benchmark=30):
    directory, original_input = Path(directory), Path(original_input)
    hashes = {}

    def read(name):
        path = directory / name
        hashes[name] = digest(path)
        return json.loads(path.read_text())

    prepared = read("input.json")
    bank = read("prefix-bank.json")
    provenance = bank["generation_provenance"]
    if hashes["input.json"] != digest(original_input) or bank["input_sha256"] != hashes["input.json"]:
        raise ValueError("Bank does not match the unchanged original prepared input")
    if bank["schema"] != "fixed-prefix-bank-v1" or prepared["schema"] != "fixed-prefix-input-v1":
        raise ValueError("Unknown fixed-prefix input/bank schema")
    for key in ("tokenizer_signature", "vocab_size", "sampling", "response_cap", "per_benchmark"):
        if bank[key] != prepared[key]:
            raise ValueError("Bank changed its prepared sampling/token protocol: " + key)
    for role, key in (("initial_student", "initial_student_fingerprint"), ("teacher", "teacher_fingerprint")):
        if bank[key] != prepared["models"][role]["fingerprint"]:
            raise ValueError("Bank changed its frozen checkpoint identity")
    questions = {row["uid"]: row for row in prepared["questions"]}
    if (
        len(questions) != len(prepared["questions"])
        or prepared["per_benchmark"] != per_benchmark
        or Counter(row["source"] for row in questions.values()) != Counter({b: per_benchmark for b in BENCHMARKS})
    ):
        raise ValueError("Prepared bank question identities/counts differ from the required benchmarks")
    question_hash = hashlib.sha256(
        json.dumps(list(questions), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if bank["question_ids_sha256"] != question_hash:
        raise ValueError("Bank changed its selected question order")
    source = read("source-manifest.json")
    if source["source_kind"] != "git_archive" or not source["files"]:
        raise ValueError("Bank requires a nonempty immutable neural source snapshot")
    for name, expected in source["files"].items():
        if digest(directory / "source" / name) != expected:
            raise ValueError("Bank neural source snapshot changed: " + name)
    devices = read("device-identities.json")
    if (
        provenance["source_manifest_sha256"] != hashes["source-manifest.json"]
        or provenance["physical_gpu_identities"] != devices
    ):
        raise ValueError("Bank provenance differs from its saved source/device manifests")
    if set(devices) != {"0", "1", "2", "3"}:
        raise ValueError("Bank requires exactly physical GPUs 0–3")
    initialized = [json.loads(line) for line in (directory / "engine-initialization.jsonl").read_text().splitlines()]
    hashes["engine-initialization.jsonl"] = digest(directory / "engine-initialization.jsonl")
    expected_workers = {(role, gpu) for role in ROLES for gpu in range(4)}
    if len(initialized) != 8 or {(r["slot"], r["physical_gpu"]) for r in initialized} != expected_workers:
        raise ValueError("Bank does not retain all eight distinct role/device initialization receipts")
    for row in initialized:
        expected = devices[str(row["physical_gpu"])]
        if row["expected"] != expected or not matches_device(row["observed"]["device_uuid"], expected):
            raise ValueError("Bank initialized on an unexpected physical GPU")
    indexed = {(row["state_source"], row["uid"]): row for row in bank["records"]}
    if len(indexed) != len(bank["records"]) or set(indexed) != {
        (state, uid) for state in ROLES.values() for uid in questions
    }:
        raise ValueError("Bank lost or duplicated one of the paired source/question records")
    summaries, unique_prefixes = {}, set()
    for role, state in ROLES.items():
        path = directory / f"{role}-trajectories.jsonl"
        hashes[path.name] = digest(path)
        trajectories = [json.loads(line) for line in path.read_text().splitlines()]
        if len(trajectories) != len(questions) or len({row["request"]["prompt_id"] for row in trajectories}) != len(
            questions
        ):
            raise ValueError("Raw native bank trajectories are missing or duplicated")
        execution = read(f"{role}-execution.json")
        if execution != provenance["generation"][role]:
            raise ValueError("Bank changed its original native execution receipts")
        calls = execution["execution"]
        ids = [uid for row in calls for uid in row["request_ids"]]
        if (
            len(ids) != len(set(ids))
            or len(ids) != len(questions)
            or {r["replica_rank"] for r in calls} != set(range(4))
        ):
            raise ValueError("Bank native RPC request/rank coverage is incomplete")
        for row in calls:
            if (
                row["slot"] != role
                or row["requests"] != len(row["request_ids"])
                or not matches_device(row["device_uuid"], devices[str(row["replica_rank"])])
            ):
                raise ValueError("Bank native execution changed its source, work or physical device")
        verified = execution["frozen_verifications"]
        if len(verified) != 4 or any(row["slot"] != role or set(row["frozen_digests"]) != {role} for row in verified):
            raise ValueError("Bank lacks four source-specific frozen-parameter verification receipts")
        digests = {row["frozen_digests"][role] for row in verified}
        if len(digests) != 1 or len(next(iter(digests))) != 64:
            raise ValueError("Bank frozen-parameter receipts disagree across replicas")
        lengths, terminations, prefix_count = [], Counter(), 0
        for raw in trajectories:
            request = raw["request"]
            question = questions[request["prompt_id"]]
            row = indexed[state, question["uid"]]
            tokens, prompt = raw["token_ids"], question["prompt_ids"]
            length = len(tokens)
            if (
                request["prompt_ids"] != prompt
                or request["visit"] != 0
                or request["sample"] != 0
                or row["prompt_ids"] != prompt
                or row["response_ids"] != tokens
                or row["sample"] != 0
                or row["source"] != question["source"]
                or row["termination"] != raw["termination"]
                or not 1 <= length <= prepared["response_cap"]
                or any(type(token) is not int or not 0 <= token < prepared["vocab_size"] for token in prompt + tokens)
            ):
                raise ValueError("Bank changed original native prompts, response tokens or sample identities")
            eos = prepared["eos_token_id"]
            if (
                eos in tokens[:-1]
                or (tokens[-1] == eos) != (raw["termination"] == "eos")
                or raw["termination"] not in {"eos", "length", "stop"}
            ):
                raise ValueError("Bank continued after EOS or changed native termination")
            segment = row["native_segment"]
            model = prepared["models"][role]
            if (
                raw["segments"] != [segment]
                or segment["identities"] != [[role, model["path"], model["fingerprint"]]]
                or segment["start"] != 0
                or segment["end"] != length
                or segment["teacher_weight"] is not None
                or segment["finish_reason"] not in {"stop", "length"}
                or (segment["finish_reason"] == "length" or raw["termination"] == "length")
                and length != prepared["response_cap"]
            ):
                raise ValueError("Bank source/segment/budget differs from its frozen native trajectory")
            seed = int.from_bytes(
                hashlib.sha256(
                    repr((prepared["selection_seed"], ("bank-positions", question["uid"], role))).encode()
                ).digest()[:4],
                "big",
            )
            rng = random.Random(seed)
            maximum = prepared["positions_per_response"]
            if length <= maximum:
                strata = [{"position": p, "begin": p, "end": p + 1} for p in range(length)]
            else:
                strata = [{"position": 0, "begin": 0, "end": 1}]
                for part in range(maximum - 1):
                    begin = 1 + (length - 1) * part // (maximum - 1)
                    end = 1 + (length - 1) * (part + 1) // (maximum - 1)
                    strata.append({"position": rng.randrange(begin, end), "begin": begin, "end": end})
            if (
                row["position_seed"] != seed
                or row["strata"] != strata
                or row["positions"] != [part["position"] for part in strata]
                or row["weights"] != [part["end"] - part["begin"] for part in strata]
                or sum(row["weights"]) != length
            ):
                raise ValueError("Bank sampled positions/strata/weights do not reproduce from its fixed seed")
            hasher = hashlib.sha256(b"opd-prefix-tokens-v1")
            for token in prompt:
                hasher.update(token.to_bytes(4, "little"))
            selected = set(row["positions"])
            for position, token in enumerate(tokens):
                if position in selected:
                    unique_prefixes.add(hasher.copy().hexdigest())
                hasher.update(token.to_bytes(4, "little"))
            lengths.append(length)
            terminations[raw["termination"]] += 1
            prefix_count += len(row["positions"])
        summaries[state] = {
            "responses": len(lengths),
            "positions": prefix_count,
            "response_tokens": sum(lengths),
            "length": {"min": min(lengths), "mean": sum(lengths) / len(lengths), "max": max(lengths)},
            "termination": dict(terminations),
            "frozen_worker_digest": next(iter(digests)),
            "native_generation_seconds": execution["timing"]["seconds"],
        }
    return {
        "scope": __doc__,
        "passed": True,
        "audit_source_sha256": digest(Path(__file__)),
        "input_files_sha256": hashes,
        "original_prepared_input": str(original_input.resolve()),
        "neural_source_commit": source["base_commit"],
        "neural_source_files_verified": len(source["files"]),
        "driver_dependencies_sha256": provenance["driver_dependencies_sha256"],
        "graph_requested": provenance["graph"],
        "initialization_receipts": len(initialized),
        "question_count": len(questions),
        "responses": len(indexed),
        "states": summaries,
        "selected_positions": sum(r["positions"] for r in summaries.values()),
        "distinct_hashed_token_prefixes": len(unique_prefixes),
        "cold_engine_seconds": provenance["cold_seconds"],
        "limitations": [
            "Does not rerun the neural models",
            "Positions are not distinct states; both sources share each prompt-only prefix",
            "Fixed bank stops at its own response cap; full validation KL has a separate longer cap",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.prepared_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("responses", "selected_positions", "distinct_hashed_token_prefixes", "passed")
            }
        )
    )


if __name__ == "__main__":
    main()
