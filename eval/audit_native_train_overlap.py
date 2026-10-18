"""Audit exact normalized question overlap and retain full-benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from collect_math_experiment import question_identity, sha256

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("MATH-500", "AMC23", "AIME25")


def question_key(messages):
    text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def subset_metrics(questions, excluded):
    fields = ["mean@16", "pass@1", "pass@2", "pass@4", "pass@8", "pass@16"]
    result = {}
    for name in BENCHMARKS:
        rows = [row for row in questions if row["data_source"] == name and row["question_sha256"] not in excluded]
        if not rows:
            raise ValueError("An overlap exclusion emptied a benchmark; macro averages would change their meaning")
        result[name] = {
            "questions": len(rows),
            **{field: sum(float(row[field]) for row in rows) / len(rows) for field in fields},
        }
    result["macro"] = {field: sum(result[name][field] for name in BENCHMARKS) / len(BENCHMARKS) for field in fields}
    return result


def build_definition(manifest_path, tokenizer_path, output):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    train_path = manifest_path.parent / "ordered.parquet"
    if sha256(train_path) != manifest["ordered_parquet_sha256"]:
        raise ValueError("Prepared training parquet differs from its manifest")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    train_rows = pq.read_table(train_path).to_pylist()
    train = defaultdict(list)
    for row in train_rows:
        train[question_key(row["prompt"])].append(row["opd_prompt_id"])
    questions, overlaps, sources = [], [], {}
    for name in BENCHMARKS:
        path = ROOT.parent / "MD-OPD/data/math" / name / "test.parquet"
        sources[name] = sha256(path)
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            ids = tokenizer.apply_chat_template(
                row["prompt"], add_generation_prompt=True, tokenize=True, **manifest["apply_chat_template_kwargs"]
            )
            identity = question_identity(
                {
                    "data_source": name,
                    "input": tokenizer.decode(ids, skip_special_tokens=True),
                    "gts": row["reward_model"]["ground_truth"],
                }
            )
            questions.append(identity)
            matched = train.get(question_key(row["prompt"]), [])
            if matched:
                overlaps.append(
                    {
                        "benchmark": name,
                        "evaluation_row_index": index,
                        "question_sha256": identity,
                        "training_row_ids": matched,
                    }
                )
    if len(questions) != len(set(questions)) or len(questions) != 570:
        raise ValueError("Expected 570 distinct canonical evaluation questions")
    record = {
        "scope": "Exact user-question overlap, with whitespace collapsed; not semantic or pretraining exposure",
        "prepared_manifest_sha256": sha256(manifest_path),
        "ordered_parquet_sha256": sha256(train_path),
        "validation_parquet_sha256": sources,
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "train_pool_rows": len(train_rows),
        "evaluation_question_ids": questions,
        "overlaps": overlaps,
        "analysis_source_sha256": sha256(Path(__file__)),
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return record


def apply_definition(definition_path, manifest_path, consumed, question_csv, output):
    definition = json.loads(Path(definition_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("prepared_data_manifest_sha256") != definition["prepared_manifest_sha256"]:
        raise ValueError("Overlap definition is bound to a different prepared training manifest")
    with Path(question_csv).open() as stream:
        questions = list(csv.DictReader(stream))
    if not consumed or any(not isinstance(value, str) or not value for value in consumed):
        raise ValueError("Actual training exposure requires nonempty stable row identities")
    if Counter(row["data_source"] for row in questions) != Counter({"MATH-500": 500, "AMC23": 40, "AIME25": 30}):
        raise ValueError("Overlap analysis requires the complete benchmark composition")
    if {row["question_sha256"] for row in questions} != set(definition["evaluation_question_ids"]) or len(
        questions
    ) != 570:
        raise ValueError("Canonical evaluation questions differ from the overlap definition")
    pool, seen, overlaps = set(), set(), []
    for item in definition["overlaps"]:
        matched = bool(set(item["training_row_ids"]) & consumed)
        pool.add(item["question_sha256"])
        if matched:
            seen.add(item["question_sha256"])
        overlaps.append({**item, "consumed_in_this_run": matched})
    record = {
        "scope": "Exact normalized user-question overlap only; no semantic or pretraining exposure claim",
        "definition_sha256": sha256(Path(definition_path)),
        "native_manifest_sha256": sha256(Path(manifest_path)),
        "question_csv_sha256": sha256(Path(question_csv)),
        "actual_training_row_ids": len(consumed),
        "pool_overlap_questions": len(pool),
        "consumed_overlap_questions": len(seen),
        "overlaps": overlaps,
        "views": {
            name: subset_metrics(questions, excluded)
            for name, excluded in [("full", set()), ("exclude_consumed", seen), ("exclude_training_pool", pool)]
        },
        "analysis_source_sha256": sha256(Path(__file__)),
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--definition", type=Path, required=True)
    apply.add_argument("--run", type=Path, required=True)
    apply.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "build":
        result = build_definition(args.manifest, args.tokenizer, args.output)
        print(json.dumps({"pool_overlap_questions": len(result["overlaps"])}))
    else:
        paths = list(args.run.glob("checkpoints/**/native-trajectories/manifest.json"))
        if len(paths) != 1:
            raise ValueError("Expected one native training manifest")
        summary = json.loads((args.result / "summary.json").read_text())
        steps = summary["training_steps"]
        consumed = {
            json.loads(line)["request"]["row_id"]
            for step in range(1, steps + 1)
            for line in (paths[0].parent / f"step-{step}.jsonl").read_text().splitlines()
        }
        result = apply_definition(
            args.definition,
            paths[0],
            consumed,
            args.result / f"questions-step{steps}.csv",
            args.result / "train-test-overlap.json",
        )
        print(json.dumps({"pool": result["pool_overlap_questions"], "consumed": result["consumed_overlap_questions"]}))


if __name__ == "__main__":
    main()
