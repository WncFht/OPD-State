"""Validate and project native composition probes into compact research tables."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_run(path, allow_numerical_observation=False):
    summary = json.loads((path / "summary.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    samples = json.loads((path / "gpu.json").read_text())
    failed = [check for check in summary["checks"] if not check["passed"]]
    allowed = {"native_same_model_split_vs_uninterrupted"} if allow_numerical_observation else set()
    assert all(check["check"] in allowed for check in failed), (path, failed)
    assert summary["all_checks_passed"] or failed, path
    rows = []
    for case in summary["cases"]:
        trajectories = case["trajectories"]
        assert len(trajectories) == case["batch"]
        assert len({tuple(t["request"][k] for k in ("prompt_id", "visit", "sample")) for t in trajectories}) == len(
            trajectories
        )
        token_counts = defaultdict(int)
        expected_inputs = 0
        for row in trajectories:
            position = 0
            assert 0 < len(row["token_ids"]) <= case["cap"]
            for segment in row["segments"]:
                assert segment["start"] == position
                assert segment["start"] <= segment["end"] <= len(row["token_ids"])
                expected_inputs += len(row["request"]["prompt_ids"]) + position
                token_counts[segment["model"]] += segment["end"] - position
                position = segment["end"]
            assert position == len(row["token_ids"])
            if row["termination"] == "eos":
                assert row["token_ids"][-1] == metadata["eos_token_id"]
        assert expected_inputs == case["input_tokens"]
        assert sum(token_counts.values()) == case["tokens"]
        assert set(token_counts) <= {"student", "teacher"}
        rows.append(
            {
                "run": path.name,
                "mode": metadata["mode"],
                "name": case["name"],
                "batch": case["batch"],
                "response_cap": case["cap"],
                "output_tokens": case["tokens"],
                "submitted_input_tokens": case["input_tokens"],
                "student_action_tokens": token_counts["student"],
                "teacher_action_tokens": token_counts["teacher"],
                "seconds": case["seconds"],
                "output_tokens_per_second": case["output_tokens_per_second"],
                "native_group_count": len(case["groups"]),
                "concurrent_groups": case["concurrent_groups"],
            }
        )
    paired = defaultdict(list)
    for row in rows:
        if row["name"].startswith("benchmark/"):
            _, batch, _, kind = row["name"].split("/")
            paired[batch, kind].append(row)
    benchmarks = []
    for (batch, kind), values in sorted(paired.items()):
        benchmarks.append(
            {
                "batch": int(batch[1:]),
                "policy": kind,
                "repeats": len(values),
                "mean_seconds": statistics.mean(v["seconds"] for v in values),
                "mean_output_tokens_per_second": statistics.mean(v["output_tokens_per_second"] for v in values),
                "min_seconds": min(v["seconds"] for v in values),
                "max_seconds": max(v["seconds"] for v in values),
            }
        )
    proof = {
        "directory": str(path),
        "metadata": metadata,
        "case_count": len(rows),
        "case_responses": sum(c["batch"] for c in rows),
        "case_output_tokens": sum(c["output_tokens"] for c in rows),
        "all_checks_passed": summary["all_checks_passed"],
        "contract_checks_passed": True,
        "check_count": len(summary["checks"]),
        "passed_checks": sum(c["passed"] for c in summary["checks"]),
        "failed_numerical_observations": failed,
        "checks": summary["checks"],
        "benchmarks": benchmarks,
        "lifecycle": summary["lifecycle"],
        "weight_switches": summary.get("weight_switches", []),
        "whole_gpu_max_observed_mib": max(s["used_mib"] for s in samples),
        "whole_gpu_min_observed_mib": min(s["used_mib"] for s in samples),
        "gpu_samples": len(samples),
        "evidence_sha256": {
            name: digest(path / name) for name in ["summary.json", "metadata.json", "gpu.json", "raw-calls.jsonl"]
        },
        "projection_source_sha256": digest(Path(__file__)),
    }
    return proof, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-numerical-observation", action="store_true")
    args = parser.parse_args()
    proofs, rows = [], []
    for path in args.run:
        proof, cases = summarize_run(path, args.allow_numerical_observation)
        proofs.append(proof)
        rows.extend(cases)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for proof in proofs:
        label = proof["metadata"]["mode"]
        (args.output / f"{label}.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "mode": p["metadata"]["mode"],
                    "cases": p["case_count"],
                    "responses": p["case_responses"],
                    "tokens": p["case_output_tokens"],
                    "checks": p["passed_checks"],
                }
                for p in proofs
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
