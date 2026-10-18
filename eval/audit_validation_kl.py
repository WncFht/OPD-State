"""Independently bind KL artifacts to effective tokens and question-cluster intervals.

This audit checks evidence and reductions, not numerical model correctness.
Question bootstrap conditions on the fitted models and the chosen fixed bank;
it does not capture training-seed or fixed-bank position-sampling uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

STATISTICS = ("student_teacher_kl", "teacher_student_kl", "student_entropy", "teacher_entropy")
STATES = {"on_validation_states", "on_initial_states", "on_teacher_states"}
DEPTHS = (("0_255", 0, 256), ("256_1023", 256, 1024), ("1024_2047", 1024, 2048), ("2048_plus", 2048, None))
BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_fingerprint(record):
    keys = ("state_source", "source", "uid", "sample", "prompt_ids", "response_ids", "positions", "weights")
    return hashlib.sha256(
        json.dumps({key: record.get(key) for key in keys}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_pairs(path, rows):
    """Stream raw trajectories; full long-context validation need not live twice in RAM."""
    with Path(path).open() as stream:
        for raw, row in zip_longest((json.loads(line) for line in stream), rows):
            if raw is None or row is None:
                raise ValueError("KL report and token evidence contain different response counts")
            yield raw, row


def key(record):
    return record["state_source"], record["source"], record["uid"], record["sample"]


def reductions(rows):
    """Reconstruct producer metrics with no import from the training package."""
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_source"]].append(row)
    result = {}

    def group(state, source, items):
        questions = defaultdict(list)
        for row in items:
            questions[row["source"], row["uid"]].append(row)
        tokens = sum(row["response_length"] for row in items)
        values = {
            "responses": len(items),
            "questions": len(questions),
            "response_tokens": tokens,
            "evaluated_positions": sum(row["evaluated_positions"] for row in items),
        }
        for name in STATISTICS:
            per_question = [
                (
                    math.fsum(row["sums"][name] / row["response_length"] for row in samples) / len(samples),
                    math.fsum(row["sums"][name] for row in samples) / len(samples),
                )
                for samples in questions.values()
            ]
            values[name + "/question_mean"] = math.fsum(pair[0] for pair in per_question) / len(questions)
            values[name + "/cumulative_on_states"] = math.fsum(pair[1] for pair in per_question) / len(questions)
            values[name + "/token_weighted"] = math.fsum(row["sums"][name] for row in items) / tokens
        for depth, _, _ in DEPTHS:
            parts = [row.get("depths", {}).get(depth) for row in items]
            parts = [value for value in parts if value is not None]
            n = sum(part["tokens"] for part in parts)
            values[f"depth/{depth}/tokens"] = n
            if n:
                for name in STATISTICS:
                    values[f"depth/{depth}/{name}/token_weighted"] = math.fsum(part["sums"][name] for part in parts) / n
        result.update({f"validation_kl/{state}/{source}/{name}": value for name, value in values.items()})

    for state, items in by_state.items():
        sources = sorted({row["source"] for row in items})
        for source in sources:
            group(state, source, [row for row in items if row["source"] == source])
        group(state, "micro", items)
        for name in STATISTICS:
            for reduction in ("question_mean", "cumulative_on_states", "token_weighted"):
                result[f"validation_kl/{state}/macro/{name}/{reduction}"] = math.fsum(
                    result[f"validation_kl/{state}/{source}/{name}/{reduction}"] for source in sources
                ) / len(sources)
    return result


def question_bootstrap(rows, *, replicates=5000, seed=42):
    import numpy as np

    if type(replicates) is not int or replicates < 100 or type(seed) is not int or seed < 0:
        raise ValueError("KL uncertainty requires >=100 question bootstrap draws and a nonnegative integer seed")
    rng = np.random.default_rng(seed)
    intervals = {}
    for state in sorted({row["state_source"] for row in rows}):
        sources = sorted({row["source"] for row in rows if row["state_source"] == state})
        draws, question_counts = {}, {}
        for source in sources:
            questions = defaultdict(list)
            for row in rows:
                if row["state_source"] == state and row["source"] == source:
                    questions[row["uid"]].append(row)
            clusters = [questions[uid] for uid in sorted(questions)]
            n = len(clusters)
            question_counts[source] = n
            indices = rng.integers(0, n, size=(replicates, n))
            total_lengths = np.array([sum(row["response_length"] for row in cluster) for cluster in clusters])
            draws[source] = {}
            for name in STATISTICS:
                means = np.array(
                    [
                        sum(row["sums"][name] / row["response_length"] for row in cluster) / len(cluster)
                        for cluster in clusters
                    ]
                )
                cumulative = np.array(
                    [sum(row["sums"][name] for row in cluster) / len(cluster) for cluster in clusters]
                )
                total = np.array([sum(row["sums"][name] for row in cluster) for cluster in clusters])
                draws[source][name + "/question_mean"] = means[indices].mean(-1)
                draws[source][name + "/cumulative_on_states"] = cumulative[indices].mean(-1)
                draws[source][name + "/token_weighted"] = total[indices].sum(-1) / total_lengths[indices].sum(-1)
                # Retain sums for the pooled ratio; averaging benchmark ratios
                # would incorrectly make a token-weighted micro statistic macro.
                draws[source][name + "/_total"] = total[indices].sum(-1)
            draws[source]["_length"] = total_lengths[indices].sum(-1)
        for source in [*sources, "macro", "micro"]:
            for name in STATISTICS:
                for reduction in ("question_mean", "cumulative_on_states", "token_weighted"):
                    field = name + "/" + reduction
                    if source in sources:
                        values = draws[source][field]
                    elif source == "macro":
                        values = sum(draws[s][field] for s in sources) / len(sources)
                    elif reduction == "token_weighted":
                        values = sum(draws[s][name + "/_total"] for s in sources) / sum(
                            draws[s]["_length"] for s in sources
                        )
                    else:
                        values = sum(question_counts[s] * draws[s][field] for s in sources) / sum(
                            question_counts.values()
                        )
                    lo, hi = np.quantile(values, [0.025, 0.975])
                    intervals[f"validation_kl/{state}/{source}/{field}"] = {"low": float(lo), "high": float(hi)}
    return intervals


def audit(
    report_path,
    token_path,
    *,
    accuracy_path=None,
    bank_path=None,
    samples=16,
    workers=4,
    required_counts=None,
    replicates=5000,
    seed=42,
):
    report_path, token_path = Path(report_path), Path(token_path)
    report = json.loads(report_path.read_text())
    rows = report.get("rows", [])
    if report.get("schema") != "validation-kl-v1" or not rows:
        raise ValueError("KL report must contain nonzero response counts")
    if digest(token_path) != report.get("tokens_sha256"):
        raise ValueError("KL raw token evidence changed after scoring")
    if report.get("scoring_temperature") != 1 or report.get("strict_raw_sequence_kl") is not False:
        raise ValueError("KL report uses an unsupported probability or sequence-divergence convention")
    epochs = [report.get("scoring_started_epoch_s"), report.get("scoring_finished_epoch_s")]
    if (
        any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in epochs)
        or epochs[0] > epochs[1]
    ):
        raise ValueError("KL scoring timestamps are invalid")
    vocab = report.get("effective_vocab_size")
    if type(vocab) is not int or vocab < 1 or type(samples) is not int or samples < 1:
        raise ValueError("KL vocabulary and expected per-question sampling count must be positive integers")
    if (
        type(workers) is not int
        or workers < 1
        or type(report.get("student_version")) is not int
        or report["student_version"] < 0
    ):
        raise ValueError("KL worker count and student version are invalid")
    seen, main, fixed = set(), {}, []
    groups = defaultdict(list)
    for record, row in token_pairs(token_path, rows):
        identity = key(record)
        if identity in seen or record["state_source"] not in STATES or record["source"] not in BENCHMARKS:
            raise ValueError("KL evidence has duplicate responses or unsupported source identities")
        seen.add(identity)
        if key(row) != identity or raw_fingerprint(record) != row.get("record_sha256"):
            raise ValueError("KL score is bound to different raw tokens or response identities")
        length = len(record["response_ids"])
        if (
            not record["prompt_ids"]
            or not length
            or any(type(t) is not int or not 0 <= t < vocab for t in record["prompt_ids"] + record["response_ids"])
        ):
            raise ValueError("KL evidence contains empty or invalid effective tokens")
        positions = record.get("positions", list(range(length)))
        weights = record.get("weights", [1] * len(positions))
        if (
            not positions
            or sorted(set(positions)) != positions
            or any(type(p) is not int or not 0 <= p < length for p in positions)
            or len(weights) != len(positions)
            or any(not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(math.fsum(weights), length, rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("KL positions or weights do not cover the effective response")
        if row.get("response_length") != length or row.get("evaluated_positions") != len(positions):
            raise ValueError("KL score dropped valid response positions")
        if set(row.get("sums", {})) != set(STATISTICS) or any(
            not math.isfinite(v) or v < -1e-4 * length for v in row["sums"].values()
        ):
            raise ValueError("KL statistics are missing, nonfinite or numerically invalid")
        if record["state_source"] == "on_validation_states":
            if positions != list(range(length)) or any(w != 1 for w in weights):
                raise ValueError("Main validation KL must cover all effective positions")
            main[(record["source"], record["uid"], record["sample"])] = length
            groups[record["source"], record["uid"]].append(record)
            expected_depths = {
                name: min(length, hi if hi is not None else length) - lo for name, lo, hi in DEPTHS if lo < length
            }
            if {name: value["tokens"] for name, value in row.get("depths", {}).items()} != expected_depths:
                raise ValueError("KL depth bins do not cover the complete validation response")
            for name in STATISTICS:
                total = math.fsum(value["sums"][name] for value in row["depths"].values())
                if not math.isclose(total, row["sums"][name], rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("KL depth statistics disagree with the complete response")
        else:
            fixed.append(record)
    if not main or any(len(values) != samples for values in groups.values()):
        raise ValueError("KL validation has missing or extra samples per question")
    counts = Counter(source for source, _ in groups)
    if set(counts) != set(BENCHMARKS) or (required_counts is not None and dict(counts) != required_counts):
        raise ValueError("KL validation benchmark question coverage is incomplete")
    if accuracy_path is not None:
        accuracy = {}
        with Path(accuracy_path).open() as stream:
            for line in stream:
                item = json.loads(line)
                identity = item["data_source"], item["validation_uid"], item["rollout_generation_sample"]
                if identity in accuracy or item["rollout_student_version"] != report["student_version"]:
                    raise ValueError("Accuracy evidence has duplicate samples or a different student version")
                accuracy[identity] = item["rollout_response_length"]
        if accuracy != main:
            raise ValueError("KL token coverage differs from the complete accuracy response set")
    if required_counts is not None and accuracy_path is None:
        raise ValueError("Claiming complete benchmark coverage requires binding the raw accuracy evidence")
    if bank_path is not None:
        bank = json.loads(Path(bank_path).read_text())
        if (
            bank.get("schema") != "fixed-prefix-bank-v1"
            or digest(bank_path) != report.get("fixed_bank_sha256")
            or [raw_fingerprint(r) for r in fixed] != [raw_fingerprint(r) for r in bank["records"]]
        ):
            raise ValueError("KL fixed states differ from the declared common prefix bank")
    elif fixed:
        raise ValueError("Fixed-state KL evidence requires the original immutable bank")
    timing = report.get("worker_timings", {})
    for name in (
        "seconds",
        "student_forward_seconds",
        "teacher_forward_seconds",
        "head_kl_seconds",
        "head_snapshot_bytes",
    ):
        values = timing.get(name, [])
        if not isinstance(values, list) or len(values) != workers or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("KL worker timing evidence is incomplete or invalid")
    expected = reductions(rows)
    actual = {k: v for k, v in report.get("metrics", {}).items() if not k.startswith("validation_kl/performance/")}
    if set(actual) != set(expected) or any(
        not math.isfinite(actual[k]) or not math.isclose(actual[k], value, rel_tol=1e-8, abs_tol=1e-8)
        for k, value in expected.items()
    ):
        raise ValueError("KL aggregate metrics disagree with independent response reductions")
    return {
        "schema": "validation-kl-audit-v1",
        "scope": "artifact integrity, coverage and independent reductions; no model recomputation",
        "report_sha256": digest(report_path),
        "tokens_sha256": digest(token_path),
        "accuracy_sha256": digest(accuracy_path) if accuracy_path else None,
        "bank_sha256": digest(bank_path) if bank_path else None,
        "student_version": report["student_version"],
        "coverage": "bound_to_accuracy" if accuracy_path else "internal_only",
        "questions": dict(counts),
        "responses": len(main),
        "effective_positions": sum(main.values()),
        "worker_count": workers,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "interval_semantics": (
            "benchmark-stratified question-cluster bootstrap; conditional on fitted models and fixed bank"
        ),
        "excluded_uncertainty": ["training seeds", "fixed-bank construction and position sampling"],
        "intervals": question_bootstrap(rows, replicates=replicates, seed=seed),
        "audit_source_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--questions-per-benchmark", type=int)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = (
        BENCHMARKS
        if args.full
        else (
            {name: args.questions_per_benchmark for name in BENCHMARKS}
            if args.questions_per_benchmark is not None
            else None
        )
    )
    result = audit(
        args.report,
        args.tokens,
        accuracy_path=args.accuracy,
        bank_path=args.bank,
        samples=args.samples,
        workers=args.workers,
        required_counts=counts,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("responses", "effective_positions", "coverage")}))


if __name__ == "__main__":
    main()
