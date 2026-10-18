"""Audit fixed-response lengths and paired exact-prefix overlap without GPUs.

The prefix statistic matches records with the same prompt. It is a lexical
prefix-position overlap, not semantic/representation-space state coverage.
Benchmark metadata is checked against the actual evaluation parquet contents.
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    ordered = sorted(values)
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "max": max(values),
    }


def paired_prefix_counts(student, teacher):
    """Count common response tokens and shared states BEFORE each action."""
    common = 0
    for a, b in zip(student, teacher, strict=False):
        if a != b:
            break
        common += 1
    return common, min(common + 1, len(student), len(teacher))


def benchmark_profile(data):
    import pandas as pd

    cache = data / ".cache"
    amc = pd.read_parquet(cache / "amc23.parquet").to_dict("records")
    maths = [json.loads(line) for line in (cache / "upstream/math500_test.jsonl").read_text().splitlines()]
    aime = [
        json.loads(line) for part in ("I", "II") for line in (cache / f"aime2025-{part}.jsonl").read_text().splitlines()
    ]
    raw = {
        "MATH-500": [(r["problem"], r["answer"]) for r in maths],
        "AMC23": [(r["problem"], r["answer"]) for r in amc],
        "AIME25": [(r["question"], r["answer"]) for r in aime],
    }
    for name, expected in raw.items():
        actual = pd.read_parquet(data / "math" / name / "test.parquet").to_dict("records")
        assert len(actual) == len(expected)
        for i, (row, (question, answer)) in enumerate(zip(actual, expected, strict=True)):
            assert row["prompt"][0]["content"] == question.strip(), (name, i)
            assert row["reward_model"]["ground_truth"].strip() == str(answer).strip(), (name, i)
    amc_numbers = defaultdict(list)
    for row in amc:
        match = re.search(r"2023_AMC_(12[AB])_Problems/Problem_(\d+)", row["url"])
        if match is None:
            raise ValueError(f"Unrecognized AMC provenance: {row['url']}")
        amc_numbers[match[1]].append(int(match[2]))
    paths = [
        cache / "amc23.parquet",
        cache / "upstream/math500_test.jsonl",
        *(cache / f"aime2025-{p}.jsonl" for p in ("I", "II")),
        *(data / "math" / name / "test.parquet" for name in raw),
    ]
    return {
        "question_counts": {k: len(v) for k, v in raw.items()},
        "raw_and_evaluation_content_match": True,
        "math_levels": dict(sorted(Counter(str(r["level"]) for r in maths).items())),
        "math_subjects": dict(Counter(r["subject"] for r in maths)),
        "amc_source_counts": {k: len(v) for k, v in amc_numbers.items()},
        "amc_problem_numbers": {k: sorted(v) for k, v in amc_numbers.items()},
        "aime_parts": {p: len((cache / f"aime2025-{p}.jsonl").read_text().splitlines()) for p in ("I", "II")},
        "input_sha256": {str(p): sha256(p) for p in paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "runs/rollout-sources")
    parser.add_argument("--data", type=Path, default=ROOT.parent / "MD-OPD/data")
    parser.add_argument("--long-run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment/results/state-diagnostics")
    args = parser.parse_args()
    schedule = args.cache_root / "paired-schedule/prompts.jsonl"
    records = [json.loads(line) for line in schedule.read_text().splitlines()]
    assert len(records) == 3200 and len({r["prompt_id"] for r in records}) == 3200
    connections, manifests, hashes = {}, {}, {}
    for name, source in (("student", "cached_initial_student"), ("teacher", "cached_teacher")):
        directory = args.cache_root / f"{name}-cache"
        manifest = json.loads((directory / "manifest.json").read_text())
        database = directory / "trajectories.sqlite"
        assert manifest["source"] == source and manifest["records"] == len(records)
        assert sha256(database) == manifest["database_sha256"]
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] == len(records)
        connections[name], manifests[name], hashes[name] = con, manifest, manifest["database_sha256"]
    for key in ("tokenizer_signature", "schedule_parquet_sha256", "student_signature", "teacher_signature"):
        assert manifests["student"][key] == manifests["teacher"][key]
    lengths, common, shared, finishes = {"student": [], "teacher": []}, [], [], Counter()
    identical = 0
    for record in records:
        responses = {}
        for name, con in connections.items():
            row = con.execute(
                "SELECT prompt_sha256,prompt_ids,response_ids,finish_reason FROM trajectories WHERE prompt_id=?",
                (record["prompt_id"],),
            ).fetchone()
            assert row is not None and row[0] == record["prompt_sha256"]
            assert json.loads(row[1]) == record["prompt_ids"]
            responses[name] = json.loads(row[2])
            lengths[name].append(len(responses[name]))
            finishes[f"{name}/{row[3]}"] += 1
        student, teacher = responses["student"], responses["teacher"]
        lcp, shared_states = paired_prefix_counts(student, teacher)
        common.append(lcp)
        # States are prefixes BEFORE the action: the first divergent action
        # is still taken from a common state. Never count a terminal state.
        shared.append(shared_states)
        identical += student == teacher
    for con in connections.values():
        con.close()
    semi = ROOT / "docs/experiment/results/2026-10-18-math-semi-opd-50steps-eval16-20261018-025524/training-1-50.csv"
    with semi.open() as stream:
        logged = list(csv.DictReader(stream))
    batches = []
    for i in range(50):
        batch = {"batch": i + 1}
        for name in lengths:
            part = lengths[name][i * 64 : (i + 1) * 64]
            batch[f"{name}_mean_tokens"] = statistics.mean(part)
            batch[f"{name}_cap_fraction"] = sum(x == 4096 for x in part) / 64
        batch["semi_logged_mean_tokens"] = float(logged[i]["response_length/mean"])
        assert batch["semi_logged_mean_tokens"] == batch["student_mean_tokens"], i + 1
        batch["paired_lcp_mean_tokens"] = statistics.mean(common[i * 64 : (i + 1) * 64])
        batches.append(batch)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cache-batch-diagnostics.csv", batches)
    long_rows = []
    if args.long_run:
        logs = list((args.long_run / "logs").glob("*.log"))
        assert len(logs) == 1
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", logs[0].read_text())
        for match in re.finditer(r"step:(\d+) - ([^\n]*)\n", text):
            step = int(match[1])
            length = re.search(r"(?:^| - )response_length/mean:([0-9.eE+\-]+)", match[2])
            assert length is not None
            batch_index = (step - 1) % 50
            value, expected = float(length[1]), batches[batch_index]["student_mean_tokens"]
            assert value == expected, (step, value, expected)
            long_rows.append(
                {"step": step, "batch": batch_index + 1, "mean_tokens": value, "cached_expected": expected}
            )
        if long_rows:
            assert [r["step"] for r in long_rows] == list(range(1, len(long_rows) + 1))
            write_csv(args.output / "long-length-observations.csv", long_rows)
    student_n, teacher_n, intersection = sum(lengths["student"]), sum(lengths["teacher"]), sum(shared)
    summary = {
        "analyzer_sha256": sha256(Path(__file__)),
        "schedule_sha256": sha256(schedule),
        "semi_training_csv_sha256": sha256(semi),
        "database_sha256": hashes,
        "records": len(records),
        "cached_response_lengths": {k: summarize(v) for k, v in lengths.items()},
        "batch_mean_lengths": {k: summarize([r[f"{k}_mean_tokens"] for r in batches]) for k in lengths},
        "semi_all_50_batch_means_match_cache": True,
        "long_observed_steps": len(long_rows),
        "long_completed_cycles": len(long_rows) // 50,
        "long_all_observed_lengths_match_periodic_cache": bool(long_rows),
        "expected_period_steps": 50,
        "teacher_student_common_prefix_tokens": summarize(common),
        "identical_response_pairs": identical,
        "finish_reasons": dict(finishes),
        "paired_shared_prefix_positions": intersection,
        "paired_prefix_position_jaccard": intersection / (student_n + teacher_n - intersection),
        "student_positions_also_in_paired_teacher": intersection / student_n,
        "teacher_positions_also_in_paired_student": intersection / teacher_n,
        "state_metric_scope": (
            "Exact lexical prefix positions within paired prompt records; not semantic state coverage"
        ),
        "benchmark_profile": benchmark_profile(args.data),
        "outputs_sha256": {
            name: sha256(args.output / name)
            for name in ["cache-batch-diagnostics.csv"] + (["long-length-observations.csv"] if long_rows else [])
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"benchmark_profile", "database_sha256"}}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
