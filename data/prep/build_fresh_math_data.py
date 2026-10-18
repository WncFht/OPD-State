"""Prepare the complete unique prompt pool, independently of training steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_prompts(rows, tokenizer, prompt_cap, template_kwargs, conflict_policy="exclude"):
    from verl.trainer.ppo.fixed_rollout import canonical, digest_json

    selected, records, seen = [], [], {}
    conflicts = set()
    overlong = duplicates = 0
    for index, row in enumerate(rows):
        raw = canonical(row["prompt"])
        ids = tokenizer.apply_chat_template(raw, tokenize=True, add_generation_prompt=True, **template_kwargs)
        ids = list(ids)
        if not ids:
            raise ValueError(f"Empty rendered prompt at input row {index}")
        if len(ids) > prompt_cap:
            overlong += 1
            continue
        identity = digest_json(ids)
        answer = digest_json(row["reward_model"]["ground_truth"])
        if identity in seen:
            if seen[identity] != answer:
                if conflict_policy == "error":
                    raise ValueError(f"Conflicting answers for identical prompt tokens: {identity}")
                conflicts.add(identity)
            duplicates += 1
            continue
        seen[identity] = answer
        selected.append(index)
        records.append(
            {
                "prompt_id": identity,
                "source_row": index,
                "prompt_ids": ids,
                "prompt_sha256": digest_json(raw),
                "data_source": row["data_source"],
            }
        )
    kept = [(i, row) for i, row in zip(selected, records, strict=True) if row["prompt_id"] not in conflicts]
    return (
        [i for i, _ in kept],
        [row for _, row in kept],
        {
            "original_rows": len(rows),
            "excluded_overlong": overlong,
            "duplicates_removed": duplicates,
            "conflicting_prompts_removed": len(conflicts),
            "conflicting_prompt_ids": sorted(conflicts),
            "unique_prompts": len(kept),
        },
    )


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from verl.trainer.ppo.fixed_rollout import digest_file, tokenizer_signature
    from verl.utils import hf_tokenizer

    if args.prompt_cap < 1:
        raise ValueError("Prompt cap must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = [pq.read_table(path) for path in args.input]
    table = pa.concat_tables(tables, promote_options="default")
    if "opd_prompt_id" in table.column_names:
        raise ValueError("Use original data rather than an already prepared prompt pool")
    tokenizer = hf_tokenizer(str(args.student.resolve()))
    kwargs = json.loads(args.template_kwargs)
    indices, records, counts = select_prompts(
        table.to_pylist(), tokenizer, args.prompt_cap, kwargs, args.conflict_policy
    )
    if not records:
        raise ValueError("No unique prompts remain after length filtering")
    ordered = table.take(pa.array(indices, type=pa.int64()))
    ordered = ordered.append_column("opd_prompt_id", pa.array([r["prompt_id"] for r in records]))
    ordered = ordered.append_column("opd_prompt_length", pa.array([len(r["prompt_ids"]) for r in records]))
    ordered = ordered.replace_schema_metadata(None)
    parquet = output / "ordered.parquet"
    pq.write_table(ordered, parquet)
    with (output / "prompts.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema": 1,
        "collection": "fresh",
        "conflict_policy": args.conflict_policy,
        "ordering": "source order; training sampler owns seeded epoch permutations",
        "identity": "SHA-256 of canonical rendered token IDs; no training-step or batch-size cap",
        **counts,
        "prompt_cap": args.prompt_cap,
        "apply_chat_template_kwargs": kwargs,
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "source_files": {str(p.resolve()): digest_file(p) for p in args.input},
        "ordered_parquet_sha256": digest_file(parquet),
        "prompts_jsonl_sha256": digest_file(output / "prompts.jsonl"),
        "builder_source_sha256": digest_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--prompt-cap", type=int, default=1024)
    parser.add_argument("--template-kwargs", default="{}")
    parser.add_argument(
        "--conflict-policy",
        choices=["exclude", "error"],
        default="exclude",
        help="Exclude every copy of conflicting prompts, or stop without writing training data",
    )
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
