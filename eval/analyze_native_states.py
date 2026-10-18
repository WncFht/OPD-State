"""Compare exact sampled prefix states on paired native training trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from array import array
from collections import Counter
from pathlib import Path

from analyze_cached_states import paired_prefix_counts, sha256


def key(row):
    request = row["request"]
    return request["prompt_id"], request["visit"], request["sample"]


def shared_prefix_states(left, right):
    """Count s_t=(prompt, response[:t]), t=0..L-1; no terminal extra state."""
    if not left or not right:
        raise ValueError("Native trajectories need a nonempty response")
    common, shared = paired_prefix_counts(left, right)
    union = len(left) + len(right) - shared
    return common, shared, shared / union


def trace_rows(run, steps):
    run = Path(run)
    manifests = list(run.glob("checkpoints/**/native-trajectories/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one native trajectory manifest")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    command = json.loads((run / "command.json").read_text())
    batch_size = int(command["environment"]["MINI_BATCH_SIZE"])
    hashes = {"manifest.json": sha256(manifest_path), "command.json": sha256(run / "command.json")}
    if (run / "source-manifest.json").exists():
        hashes["source-manifest.json"] = sha256(run / "source-manifest.json")
    seen = set()
    rows = []
    for step in range(1, steps + 1):
        path = manifest_path.parent / f"step-{step}.jsonl"
        raw = path.read_bytes()
        batch = [json.loads(line) for line in raw.splitlines()]
        if len(batch) != batch_size:
            raise ValueError(f"Step {step} has an incomplete global batch")
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for row in batch:
            identity = key(row)
            if identity in seen or not row["token_ids"]:
                raise ValueError("Duplicate prompt/visit/sample or empty training response")
            seen.add(identity)
            tokens = array("I", row["token_ids"])
            if len(tokens) > manifest["response_cap"]:
                raise ValueError("Response exceeds the declared training budget")
            eos = manifest["eos_token_id"]
            if eos in tokens and (tokens[-1] != eos or tokens.count(eos) != 1):
                raise ValueError("Training response continued after EOS")
            if row["termination"] not in {"eos", "length", "stop"} or (row["termination"] == "eos") != (
                tokens[-1] == eos
            ):
                raise ValueError("Trajectory termination disagrees with its valid tokens")
            if row["termination"] == "length" and len(tokens) != manifest["response_cap"]:
                raise ValueError("Length termination did not exhaust the shared response cap")
            end = 0
            for segment in row["segments"]:
                if segment["start"] != end or not end < segment["end"] <= len(tokens):
                    raise ValueError("Segment coverage has a gap or invalid range")
                end = segment["end"]
            if end != len(tokens):
                raise ValueError("Segments do not cover the complete response")
            rows.append({**row, "token_ids": tokens, "step": step})
    return rows, manifest, hashes


def summarize(rows, cap):
    lengths = [len(row["token_ids"]) for row in rows]
    if not lengths:
        raise ValueError("State analysis requires training trajectories")
    segments = {}
    for row in rows:
        for segment in row["segments"]:
            roles = "+".join(identity[0] for identity in segment["identities"])
            label = roles if segment["teacher_weight"] is None else f"fusion:{roles}:lambda={segment['teacher_weight']}"
            stats = segments.setdefault(label, {"training_positions": 0, "response_prefix_depth_sum": 0})
            start, end = segment["start"], segment["end"]
            stats["training_positions"] += end - start
            stats["response_prefix_depth_sum"] += (start + end - 1) * (end - start) // 2
    for stats in segments.values():
        stats["mean_response_prefix_depth"] = stats["response_prefix_depth_sum"] / stats["training_positions"]
    return {
        "responses": len(rows),
        "unique_prompts": len({row["request"]["prompt_id"] for row in rows}),
        "revisited_responses": sum(row["request"]["visit"] > 0 for row in rows),
        "sampled_training_positions": sum(lengths),
        "response_length": {"mean": sum(lengths) / len(lengths), "min": min(lengths), "max": max(lengths)},
        "termination": dict(Counter(row["termination"] for row in rows)),
        "response_survival_at_depth": {
            str(depth): sum(length > depth for length in lengths) / len(lengths)
            for depth in [0, 1, 16, 64, 256, 1024, 2048, 4096]
            if depth <= cap
        },
        "action_policy_segments": segments,
    }


def compare(reference, candidate):
    indexed = {key(row): row for row in reference}
    if len(indexed) != len(reference) or len({key(row) for row in candidate}) != len(candidate):
        raise ValueError("Comparison inputs contain duplicate trajectory identities")
    if set(indexed) != {key(row) for row in candidate}:
        raise ValueError("Paired runs must cover exactly the same prompt/visit/sample identities")
    pairs = []
    for row in candidate:
        identity = key(row)
        other = indexed[identity]
        if other["request"]["prompt_ids"] != row["request"]["prompt_ids"] or other["step"] != row["step"]:
            raise ValueError("Paired question tokens or training order differ")
        common, shared, jaccard = shared_prefix_states(other["token_ids"], row["token_ids"])
        pairs.append(
            {
                "step": row["step"],
                "prompt_id": identity[0],
                "visit": identity[1],
                "sample": identity[2],
                "reference_length": len(other["token_ids"]),
                "candidate_length": len(row["token_ids"]),
                "common_response_prefix_tokens": common,
                "shared_sampled_states": shared,
                "sampled_state_jaccard": jaccard,
                "identical_response": other["token_ids"] == row["token_ids"],
            }
        )
    return pairs


def summarize_pairs(pairs):
    shared = sum(row["shared_sampled_states"] for row in pairs)
    total_left = sum(row["reference_length"] for row in pairs)
    total_right = sum(row["candidate_length"] for row in pairs)
    return {
        "responses": len(pairs),
        "identical_responses": sum(row["identical_response"] for row in pairs),
        "first_action_agreement": sum(row["common_response_prefix_tokens"] > 0 for row in pairs) / len(pairs),
        "shared_sampled_states": shared,
        "mean_common_response_prefix_tokens": sum(row["common_response_prefix_tokens"] for row in pairs) / len(pairs),
        "mean_per_response_state_jaccard": sum(row["sampled_state_jaccard"] for row in pairs) / len(pairs),
        "pooled_state_jaccard_on_matched_trajectories": shared / (total_left + total_right - shared),
    }


def analyze(reference, candidate, output, steps):
    if steps < 1:
        raise ValueError("State analysis needs a positive step budget")
    left, left_manifest, left_hashes = trace_rows(reference, steps)
    right, right_manifest, right_hashes = trace_rows(candidate, steps)
    for field in ["train_files", "prepared_data_manifest_sha256", "prompt_cap", "response_cap", "eos_token_id", "seed"]:
        if left_manifest.get(field) != right_manifest.get(field):
            raise ValueError(f"Paired run protocol differs: {field}")
    pairs = compare(left, right)
    result = {
        "scope": "Exact sampled prefix overlap; not a conditional KL or population state-distribution distance",
        "steps": steps,
        "reference": {
            "run": str(Path(reference).resolve()),
            "files_sha256": left_hashes,
            "statistics": summarize(left, left_manifest["response_cap"]),
        },
        "candidate": {
            "run": str(Path(candidate).resolve()),
            "files_sha256": right_hashes,
            "statistics": summarize(right, right_manifest["response_cap"]),
        },
        "paired": summarize_pairs(pairs),
        "by_step": {
            str(step): summarize_pairs([r for r in pairs if r["step"] == step]) for step in range(1, steps + 1)
        },
        "interpretation": [
            "One training state precedes each valid action, including EOS; "
            "the terminal post-response prefix is excluded.",
            "A shared sampled prefix is exact token equality conditional on a matched question and visit.",
            "Pooled overlap counts matched visits separately; "
            "it is not a deduplicated state-support Jaccard across visits.",
            "Action-policy labels describe the model selecting the next action; "
            "the state retains the entire prior mixed prefix.",
            "Stochastic branching lowers exact overlap even for the same policy; "
            "overlap alone cannot rank distribution shift or quality.",
        ],
        "analysis_source_sha256": sha256(Path(__file__)),
        "prefix_counter_source_sha256": sha256(Path(__file__).with_name("analyze_cached_states.py")),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired-prefixes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    result["paired_csv_sha256"] = sha256(output / "paired-prefixes.csv")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    result = analyze(args.reference, args.candidate, args.output, args.steps)
    print(json.dumps(result["paired"]))


if __name__ == "__main__":
    main()
