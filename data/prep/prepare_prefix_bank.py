"""Prepare a balanced, identity-bound validation prompt set for a fixed prefix bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARKS = ("MATH-500", "AMC23", "AIME25")


def select_questions(rows, tokenizer, *, per_benchmark, prompt_cap, seed, template_kwargs):
    from verl.trainer.ppo.fixed_rollout import canonical, digest_json

    if any(type(value) is not int or value < 1 for value in (per_benchmark, prompt_cap)):
        raise ValueError("Prefix-bank question count and prompt cap must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("Prefix-bank selection seed must be a nonnegative integer")
    groups = {name: [] for name in BENCHMARKS}
    vocab_size = len(tokenizer)
    seen = set()
    excluded = {name: 0 for name in BENCHMARKS}
    for index, row in enumerate(rows):
        source = row["data_source"]
        if source not in groups:
            raise ValueError("Prefix-bank inputs must be the three supported math benchmarks")
        ids = list(
            tokenizer.apply_chat_template(
                canonical(row["prompt"]), tokenize=True, add_generation_prompt=True, **template_kwargs
            )
        )
        if not ids or any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError("Prefix-bank tokenizer returned empty or invalid prompt tokens")
        if len(ids) > prompt_cap:
            excluded[source] += 1
            continue
        uid = digest_json(
            {"source": source, "prompt_ids": ids, "answer": canonical(row["reward_model"]["ground_truth"])}
        )
        if uid in seen:
            raise ValueError("Prefix-bank inputs contain duplicate question identities")
        seen.add(uid)
        groups[source].append(
            {
                "source": source,
                "uid": uid,
                "prompt_id": uid,
                "prompt_ids": ids,
                "input_row": index,
                "selection_key": digest_json([seed, source, uid]),
            }
        )
    selected = []
    for source in BENCHMARKS:
        values = sorted(groups[source], key=lambda row: row["selection_key"])
        if len(values) < per_benchmark:
            raise ValueError(f"Not enough eligible prefix-bank questions in {source}")
        selected.extend(values[:per_benchmark])
    return selected, excluded


def prepare(args):
    import pyarrow.parquet as pq

    from verl.trainer.ppo.fixed_rollout import digest_file, model_signature, tokenizer_signature
    from verl.utils import hf_tokenizer

    if type(args.response_cap) is not int or args.response_cap < 1:
        raise ValueError("Prefix-bank response cap must be positive")
    if type(args.positions) is not int or args.positions < 2:
        raise ValueError("Prefix-bank stratified sampling needs at least two available positions")
    student, teacher = Path(args.student).resolve(), Path(args.teacher).resolve()
    tokenizer = hf_tokenizer(str(student))
    signature = tokenizer_signature(tokenizer)
    if tokenizer_signature(hf_tokenizer(str(teacher))) != signature:
        raise ValueError("Prefix-bank student and teacher must share token IDs and chat template")
    rows, inputs = [], []
    for filename in args.input:
        path = Path(filename).resolve()
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
        inputs.append({"path": str(path), "sha256": digest_file(path), "rows": table.num_rows})
    kwargs = json.loads(args.template_kwargs)
    questions, excluded = select_questions(
        rows,
        tokenizer,
        per_benchmark=args.per_benchmark,
        prompt_cap=args.prompt_cap,
        seed=args.seed,
        template_kwargs=kwargs,
    )
    models = {
        role: {"path": str(path), "fingerprint": model_signature(path)}
        for role, path in (("initial_student", student), ("teacher", teacher))
    }
    report = {
        "schema": "fixed-prefix-input-v1",
        "scope": "shared_validation_state_diagnostic",
        "questions": questions,
        "per_benchmark": args.per_benchmark,
        "selection_seed": args.seed,
        "selection": "smallest sha256(seed,benchmark,question_id)",
        "excluded_overlong": excluded,
        "prompt_cap": args.prompt_cap,
        "response_cap": args.response_cap,
        "positions_per_response": args.positions,
        "template_kwargs": kwargs,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "tokenizer_signature": signature,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "models": models,
        "inputs": inputs,
        "preparation_source_sha256": digest_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=30)
    parser.add_argument("--prompt-cap", type=int, default=1024)
    parser.add_argument("--response-cap", type=int, default=4096)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-kwargs", default="{}")
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps({"output": str(args.output.resolve()), "questions": len(result["questions"])}))


if __name__ == "__main__":
    main()
