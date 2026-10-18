"""Independently audit native version-probe work, tokens and full-vocab arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from native_devices import matches_device


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(reference, observed):
    if set(reference) != set(observed):
        raise ValueError("Probe compared different prompt identities")
    result = {}
    for uid in reference:
        a, b = reference[uid], observed[uid]
        if a != b:
            index = 0
            while index < min(len(a), len(b)) and a[index] == b[index]:
                index += 1
            result[uid] = index
    return {"responses": len(reference), "exact_responses": len(reference) - len(result), "first_divergence": result}


def collect(directory):
    import numpy as np

    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    command = json.loads((directory / "command.json").read_text())
    args = command["arguments"]
    for file, key in (
        ("inputs.json", "inputs_sha256"),
        ("worker.json", "worker_sha256"),
        ("split.json", "split_sha256"),
        ("observations.jsonl", "observations_sha256"),
    ):
        if digest(directory / file) != summary[key]:
            raise ValueError("Native version-probe evidence changed: " + file)
    inputs = json.loads((directory / "inputs.json").read_text())
    identities = [row["prompt_id"] for row in inputs]
    if len(identities) != len(set(identities)) or len(identities) != 16 or summary["prompt_ids"] != identities:
        raise ValueError("Version probe requires the same sixteen distinct original prompts")
    worker = json.loads((directory / "worker.json").read_text())
    if worker != summary["worker"] or not matches_device(worker["device_uuid"], command["devices"][str(args["gpu"])]):
        raise ValueError("Version probe changed its worker or physical GPU identity")
    if (
        worker["batch_flag"] != args["batch_invariant"]
        or args["batch_invariant"]
        and not worker["kernel_overrides_initialized"]
    ):
        raise ValueError("Batch-invariance flag did not initialize in the actual GPU worker")
    regions = sorted(worker["kv_regions"], key=lambda row: row["pointer"])
    if len(regions) != summary["expected_kv_layers"] or any(
        a["pointer"] + a["bytes"] > b["pointer"] for a, b in zip(regions, regions[1:], strict=False)
    ):
        raise ValueError("Native probe KV regions are missing or overlap")
    if args["graph"] and not worker["cuda_graph_enabled"]:
        raise ValueError("Native graph probe did not capture graphs")
    rows = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
    expected_conditions = {
        (size, order, repeat)
        for size, order in ((1, False), (4, False), (16, False), (16, True))
        for repeat in range(args["repeats"])
    }
    if (
        len(rows) != len(expected_conditions)
        or {(row["batch"], row["reverse"], row["repeat"]) for row in rows} != expected_conditions
    ):
        raise ValueError("Missing or duplicated native batch/order replay conditions")
    reference = rows[0]["responses"]
    if (rows[0]["batch"], rows[0]["reverse"], rows[0]["repeat"]) != (1, False, 0):
        raise ValueError("Native replay reference must be the first single-request observation")
    all_equal = True
    for row in rows:
        responses = row["responses"]
        if set(responses) != set(identities) or any(
            not 1 <= len(ids) <= args["cap"]
            or any(type(token) is not int or not 0 <= token < summary["tokenizer_vocab_size"] for token in ids)
            for ids in responses.values()
        ):
            raise ValueError("Native replay response cardinality/token support is invalid")
        comparison = differences(reference, responses)
        if (
            comparison != row["comparison"]
            or row["tokens"] != sum(map(len, responses.values()))
            or not math.isfinite(row["seconds"])
            or row["seconds"] <= 0
        ):
            raise ValueError("Native replay comparison or work/time disagrees with its raw outputs")
        all_equal &= comparison["exact_responses"] == 16
    if summary["all_batch_order_replays_equal"] != all_equal:
        raise ValueError("Advertised batch-invariance result disagrees with raw responses")
    split = json.loads((directory / "split.json").read_text())
    first, second = split["first"], split["second"]
    if set(first) != set(identities) or not set(second) <= set(first) or set(split["second_prefixes"]) != set(second):
        raise ValueError("Same-model split lost or added original prompt identities")
    original = {row["prompt_id"]: row["prompt_ids"] for row in inputs}
    eos = summary["eos_token_id"]
    for uid, tokens in first.items():
        if not 1 <= len(tokens) <= args["boundary"] or eos in tokens[:-1]:
            raise ValueError("Same-model first request has invalid length/EOS")
        expected_second = len(tokens) == args["boundary"] and tokens[-1] != eos
        if expected_second != (uid in second):
            raise ValueError("Same-model split skipped a live continuation or continued after EOS")
        if uid in second and (not 1 <= len(second[uid]) <= args["cap"] - args["boundary"] or eos in second[uid][:-1]):
            raise ValueError("Same-model second request exceeds the remaining budget or continues after EOS")
    for uid, prefix in split["second_prefixes"].items():
        if prefix != original[uid] + first[uid] or len(first[uid]) != args["boundary"]:
            raise ValueError("Same-model second request changed the exact generated prefix")
    combined = {uid: first[uid] + second.get(uid, []) for uid in identities}
    if combined != split["responses"] or differences(reference, combined) != summary["same_model_split"]:
        raise ValueError("Same-model comparison does not reconstruct its two raw requests")
    arrays = {}
    for check in summary["full_vocab_checks"]:
        path = directory / check["file"]
        if digest(path) != check["sha256"] or check["name"] in arrays:
            raise ValueError("Full-vocabulary probe array changed or was duplicated")
        values = np.load(path, allow_pickle=False)
        if values.shape != (2, summary["tokenizer_vocab_size"]) or not np.isfinite(values).all():
            raise ValueError("Full-vocabulary probe does not cover both finite states and all valid tokens")
        arrays[check["name"]] = values
    if set(arrays) != {"single", "paired", "reversed"}:
        raise ValueError("Missing a full-vocabulary batch-shape comparison")
    if summary["probe_prefixes"] != [row["prompt_ids"] for row in inputs[:2]]:
        raise ValueError("Full-vocabulary probes changed their fixed raw states")

    def distribution(logits):
        values = logits.astype(np.float64)
        normalizer = np.logaddexp.reduce(values, axis=-1, keepdims=True)
        logs = values - normalizer
        return logs, np.exp(logs)

    logp, p = distribution(arrays["single"])
    for check in summary["full_vocab_checks"]:
        actual = arrays[check["name"]]
        logq, q = distribution(actual)
        metrics = {
            "max_logit_error": float(np.abs(arrays["single"] - actual).max()),
            "max_tv": float(np.abs(p - q).sum(-1).max() / 2),
            "max_reference_observed_kl": float(np.einsum("ij,ij->i", p, logp - logq).max()),
            "max_observed_reference_kl": float(np.einsum("ij,ij->i", q, logq - logp).max()),
        }
        if any(not math.isclose(value, check[key], abs_tol=1e-10, rel_tol=1e-8) for key, value in metrics.items()):
            raise ValueError("Full-vocabulary diagnostics disagree with independent recomputation")
    return {
        "scope": "Raw-token, resource identity and numerical artifact audit; not a neural-model recomputation",
        "batch_order_replays_equal": all_equal,
        "same_model_split": summary["same_model_split"],
        "conditions": len(rows),
        "states_per_full_vocab_shape": 2,
        "effective_vocab_size": summary["tokenizer_vocab_size"],
        "summary_sha256": digest(directory / "summary.json"),
        "auditor_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
