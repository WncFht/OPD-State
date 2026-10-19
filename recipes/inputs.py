"""Prepare the actual model/tokenizer/data contract before GPU allocation."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from opd_state.artifacts import atomic_json, content_id, file_digest, inside
from opd_state.core.evaluation import parse_evaluation
from opd_state.integrations.verl.modules import backend_module

MODEL_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_templates/*.jinja",
)
ARCHITECTURE = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "rope_theta",
    "rope_scaling",
    "tie_word_embeddings",
    "hidden_act",
    "rms_norm_eps",
)


def model_artifact(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(f"Incomplete local model: {root}")
    files = sorted(
        {p for pattern in MODEL_PATTERNS for p in root.glob(pattern) if p.is_file()}
    )
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("Local model has no weights")
    records = [
        {
            "path": str(p.relative_to(root)),
            "sha256": file_digest(p),
            "bytes": p.stat().st_size,
        }
        for p in files
    ]
    # Preserve the existing rollout fingerprint, while the stronger artifact
    # identity also covers tokenizer and external chat-template bytes.
    role_files = {
        r["path"]: r["sha256"]
        for r in records
        if r["path"] == "config.json"
        or Path(r["path"]).suffix in (".safetensors", ".bin")
    }
    return {
        "path": str(root),
        "files": records,
        "identity": content_id(records),
        "role_fingerprint": content_id(role_files),
    }


def verify_model(artifact):
    if content_id(artifact["files"]) != artifact["identity"]:
        raise ValueError("Model artifact manifest changed")
    actual = model_artifact(artifact["path"])
    if actual != artifact:
        raise ValueError(f"External model/tokenizer bytes changed: {artifact['path']}")


def inspect_model_layout(artifact):
    """Compare checkpoint tensor names/shapes with a storage-free model instance."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    root = Path(artifact["path"])
    hf_config = AutoConfig.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            hf_config, attn_implementation="eager", trust_remote_code=False
        )
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    actual = {}
    safetensors = sorted(root.glob("*.safetensors"))
    files = safetensors or sorted(root.glob("pytorch_model*.bin"))
    for path in files:
        if safetensors:
            with safe_open(path, framework="pt", device="cpu") as stream:
                shapes = {
                    name: tuple(stream.get_slice(name).get_shape())
                    for name in stream.keys()
                }
        else:
            weights = torch.load(
                path, map_location="meta", weights_only=True, mmap=True
            )
            if not isinstance(weights, dict) or any(
                not isinstance(value, torch.Tensor) for value in weights.values()
            ):
                raise ValueError("HF checkpoint must contain only named tensors")
            shapes = {name: tuple(value.shape) for name, value in weights.items()}
        if set(actual).intersection(shapes):
            raise ValueError("Checkpoint shards repeat a tensor name")
        actual.update(shapes)
    if hf_config.tie_word_embeddings and "lm_head.weight" not in actual:
        expected.pop("lm_head.weight", None)
    if actual != expected:
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        wrong = [
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        ]
        raise ValueError(
            f"Checkpoint tensor layout differs: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}, shapes={wrong[:5]}"
        )
    del model
    return {
        "tensors": len(actual),
        "parameters": sum(math.prod(shape) for shape in actual.values()),
        "storage": "meta",
    }


def freeze_file(source, root, relative, records):
    source = Path(source).expanduser().resolve(strict=True)
    digest = file_digest(source)
    target = inside(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_digest(source) != digest or file_digest(target) != digest:
        raise ValueError("Input changed while copying")
    records.append({"path": relative, "sha256": digest, "bytes": target.stat().st_size})
    return target


def rendered_record(row, tokenizer, *, kwargs, prompt_cap, vocab_size):
    raw = row["prompt"]
    ids = list(
        tokenizer.apply_chat_template(
            raw, tokenize=True, add_generation_prompt=True, **kwargs
        )
    )
    if not ids or any(
        type(token) is not int or not 0 <= token < vocab_size for token in ids
    ):
        raise ValueError("Rendered prompt must contain legal token IDs")
    source = row.get("data_source")
    if not isinstance(source, str) or not source:
        raise ValueError("Data rows require a nonempty data_source")
    truth = row.get("reward_model", {}).get("ground_truth")
    if truth is None:
        raise ValueError("The current math protocol requires ground_truth")
    return {
        "prompt_id": content_id(ids),
        "generation_id": content_id([source, ids]),
        "prompt_length": len(ids),
        "within_budget": len(ids) <= prompt_cap,
        "source": source,
        "reference_id": content_id(truth),
        "rendered_input": tokenizer.decode(ids, skip_special_tokens=True),
        "row_id": str(row.get("opd_prompt_id", "")),
    }


def selected_records(records, config, maximum):
    """Match stock text RLHFDataset sample-before-filter selection exactly."""
    import numpy as np

    chosen = records
    if 0 < maximum < len(records):
        indices = (
            np.random.default_rng(config.seed).choice(
                len(records), size=maximum, replace=False
            )
            if config.shuffle
            else range(maximum)
        )
        chosen = [records[int(i)] for i in indices]
    if config.filter_overlong_prompts:
        chosen = [row for row in chosen if row["within_budget"]]
    elif any(not row["within_budget"] for row in chosen):
        raise ValueError("Selected input exceeds the untruncated prompt budget")
    return chosen


def validate_prepared_pool(manifest, rows, records, *, tokenizer_id, config):
    if (
        manifest.get("schema") != 1
        or manifest.get("collection") != "fresh"
        or manifest.get("tokenizer_signature") != tokenizer_id
        or manifest.get("prompt_cap") != config.max_prompt_length
        or manifest.get("apply_chat_template_kwargs")
        != dict(config.apply_chat_template_kwargs)
        or manifest.get("unique_prompts") != len(rows)
    ):
        raise ValueError(
            "Prepared prompt pool differs from the configured tokenizer/template/budget"
        )
    if len({r["prompt_id"] for r in records}) != len(records):
        raise ValueError("Prepared prompt pool contains duplicate rendered prompts")
    for row, record in zip(rows, records, strict=True):
        if (
            row.get("opd_prompt_id") != record["prompt_id"]
            or row.get("opd_prompt_length") != record["prompt_length"]
            or not record["within_budget"]
        ):
            raise ValueError(
                "Prepared prompt identity/length disagrees with actual tokenizer output"
            )


def validate_bank(bank, *, tokenizer_id, models, vocab_size, eos):
    expected = {
        "schema": "fixed-prefix-bank-v1",
        "vocab_size": vocab_size,
        "tokenizer_signature": tokenizer_id,
        "initial_student_fingerprint": models["student"]["role_fingerprint"],
        "teacher_fingerprint": models["teacher"]["role_fingerprint"],
    }
    if any(bank.get(k) != v for k, v in expected.items()):
        raise ValueError("Fixed bank model/tokenizer identity changed")
    questions, keys = {}, set()
    for row in bank.get("records", []):
        state = row["state_source"]
        if state not in {"on_initial_states", "on_teacher_states"}:
            raise ValueError("Fixed bank has an unsupported state source")
        ids, response = row["prompt_ids"], row["response_ids"]
        if (
            not ids
            or not response
            or any(
                type(v) is not int or not 0 <= v < vocab_size for v in ids + response
            )
            or eos in response[:-1]
        ):
            raise ValueError(
                "Fixed bank contains invalid tokens or continued after EOS"
            )
        positions, weights = row.get("positions"), row.get("weights")
        if (
            not positions
            or positions != sorted(set(positions))
            or any(type(i) is not int or not 0 <= i < len(response) for i in positions)
        ):
            raise ValueError("Fixed bank has invalid predictor positions")
        if (
            not weights
            or len(weights) != len(positions)
            or any(
                type(w) not in (int, float) or not math.isfinite(w) or w <= 0
                for w in weights
            )
            or not math.isclose(
                math.fsum(weights), len(response), abs_tol=1e-8, rel_tol=0
            )
        ):
            raise ValueError("Fixed bank weights do not cover its response")
        key = state, row["source"], row["uid"], row["sample"]
        if key in keys:
            raise ValueError("Duplicate fixed-bank response")
        keys.add(key)
        questions.setdefault(state, {})[key[1:3]] = ids
    if (
        set(questions) != {"on_initial_states", "on_teacher_states"}
        or questions["on_initial_states"] != questions["on_teacher_states"]
    ):
        raise ValueError("Fixed bank sources must cover identical raw questions")


def prepare_inputs(
    config,
    root,
    *,
    backend,
    tokenizer_loader=None,
    table_reader=None,
    model_inspector=inspect_model_layout,
):
    """Read local assets and preserve byte-identical parquet inputs for the backend."""
    if config.actor_rollout_ref.model.trust_remote_code:
        raise ValueError(
            "The admitted Qwen2 protocol does not execute remote model code"
        )
    overrides = config.actor_rollout_ref.model.override_config
    if overrides and set(overrides) - {"attn_implementation"}:
        raise ValueError(
            "Model architecture overrides require explicit paired-model admission"
        )
    if config.data.custom_cls.path or config.data.get("tool_config_path"):
        raise ValueError("Prepare currently binds the stock single-turn text dataset")
    if table_reader is None:
        import pyarrow.parquet as pq

        def table_reader(path):
            return pq.read_table(path).to_pylist()

    if tokenizer_loader is None:
        tokenizer_loader = backend_module(
            backend, "utils/tokenizer/tokenizer.py"
        ).hf_tokenizer
    identities = backend_module(backend, "trainer/ppo/fixed_rollout.py")
    models = {
        "student": model_artifact(config.actor_rollout_ref.model.path),
        "teacher": model_artifact(config.distillation.fsdp_teacher.model_path),
    }
    layouts = {role: model_inspector(model) for role, model in models.items()}
    config.actor_rollout_ref.model.path = models["student"]["path"]
    config.distillation.fsdp_teacher.model_path = models["teacher"]["path"]
    tokens, model_configs = {}, {}
    for role, artifact in models.items():
        tokens[role] = tokenizer_loader(
            artifact["path"], local_files_only=True, trust_remote_code=False
        )
        model_configs[role] = json.loads(
            (Path(artifact["path"]) / "config.json").read_text()
        )
    student, teacher = model_configs["student"], model_configs["teacher"]
    if student.get("model_type") != "qwen2" or any(
        student.get(k) != teacher.get(k) for k in ARCHITECTURE
    ):
        raise ValueError(
            "The native source models must be compatible dense Qwen2 assets"
        )
    tokenizer = tokens["student"]
    custom = config.actor_rollout_ref.model.custom_chat_template
    if custom is not None:
        for token in tokens.values():
            token.chat_template = custom
    tokenizer_id = identities.tokenizer_signature(tokenizer)
    if identities.tokenizer_signature(tokens["teacher"]) != tokenizer_id:
        raise ValueError("Student and teacher tokenizer/template identities differ")
    vocab_size = len(tokenizer)
    if any(
        type(getattr(tokenizer, key)) is not int
        or not 0 <= getattr(tokenizer, key) < vocab_size
        for key in ("eos_token_id", "pad_token_id")
    ):
        raise ValueError("The native protocol requires legal EOS and PAD token IDs")
    if (
        set(tokenizer.get_vocab().values()) != set(range(vocab_size))
        or student.get("vocab_size", 0) < vocab_size
    ):
        raise ValueError("Tokenizer support must be dense and fit both model heads")
    if any(
        model.get("max_position_embeddings", 0)
        < config.actor_rollout_ref.rollout.max_model_len
        for model in model_configs.values()
    ):
        raise ValueError("Model context does not cover the declared execution budget")
    root = Path(root).resolve()
    files = []
    kwargs = dict(config.data.apply_chat_template_kwargs)
    original_manifest = (
        Path(config.recipe.prepared_manifest).expanduser().resolve(strict=True)
    )
    manifest = json.loads(original_manifest.read_text())
    if len(config.data.train_files) != 1:
        raise ValueError(
            "The unique-prompt protocol requires one prepared training parquet"
        )
    train_file = freeze_file(
        config.data.train_files[0], root, "data/train/ordered.parquet", files
    )
    if file_digest(train_file) != manifest.get("ordered_parquet_sha256"):
        raise ValueError("Prepared training parquet checksum mismatch")
    train_manifest = freeze_file(
        original_manifest, root, "data/train/manifest.json", files
    )
    prompts = freeze_file(
        original_manifest.parent / "prompts.jsonl",
        root,
        "data/train/prompts.jsonl",
        files,
    )
    if file_digest(prompts) != manifest.get("prompts_jsonl_sha256"):
        raise ValueError("Prepared prompt ledger checksum mismatch")
    raw = table_reader(train_file)
    rendered = [
        rendered_record(
            row,
            tokenizer,
            kwargs=kwargs,
            prompt_cap=config.data.max_prompt_length,
            vocab_size=vocab_size,
        )
        for row in raw
    ]
    validate_prepared_pool(
        manifest, raw, rendered, tokenizer_id=tokenizer_id, config=config.data
    )
    ledger = [json.loads(line) for line in prompts.read_text().splitlines()]
    if len(ledger) != len(rendered):
        raise ValueError("Prompt ledger population differs from its parquet")
    previous_source = -1
    for original, record, entry in zip(raw, rendered, ledger, strict=True):
        if (
            entry.get("prompt_id") != record["prompt_id"]
            or content_id(entry.get("prompt_ids")) != record["prompt_id"]
            or entry.get("prompt_sha256") != content_id(original["prompt"])
            or entry.get("data_source") != record["source"]
            or type(entry.get("source_row")) is not int
            or entry["source_row"] <= previous_source
        ):
            raise ValueError(
                "Prompt ledger does not describe the actual ordered inputs"
            )
        previous_source = entry["source_row"]
    train = selected_records(rendered, config.data, config.data.train_max_samples)
    if len(train) < config.data.train_batch_size:
        raise ValueError("Selected training inputs cannot fill one global batch")
    config.data.train_files = [str(train_file)]
    config.recipe.prepared_manifest = str(train_manifest)
    config.trainer.native_data_manifest = str(train_manifest)
    selection = {"train": train, "validation": []}
    evaluation = parse_evaluation(
        OmegaConf.to_container(config.evaluation, resolve=True)
    )
    for name, benchmark in evaluation.benchmarks.items():
        if benchmark.budget.prompt_tokens != config.data.max_prompt_length:
            raise ValueError(
                "The stock evaluator currently shares the training prompt budget"
            )
        paths, rows = [], []
        for index, path in enumerate(benchmark.data.paths):
            target = freeze_file(
                path, root, f"data/validation/{name}-{index}.parquet", files
            )
            paths.append(str(target))
            rows.extend(table_reader(target))
        records = [
            rendered_record(
                row,
                tokenizer,
                kwargs=kwargs,
                prompt_cap=config.data.max_prompt_length,
                vocab_size=vocab_size,
            )
            for row in rows
        ]
        selected = selected_records(records, config.data, config.data.val_max_samples)
        if not selected or len(
            {(row["source"], row["generation_id"]) for row in selected}
        ) != len(selected):
            raise ValueError(
                "Validation inputs are empty or contain duplicate questions"
            )
        counts = dict(Counter(row["source"] for row in selected))
        if set(counts) != {"MATH-500", "AMC23", "AIME25"}:
            raise ValueError(
                "The stock math evaluator requires all three benchmark sources"
            )
        if benchmark.data.expected_counts and counts != benchmark.data.expected_counts:
            raise ValueError(
                f"Validation population differs from configured counts: {counts}"
            )
        selection["validation"].extend(selected)
        config.evaluation.benchmarks[name].data.paths = paths
        config.data.val_files = paths
    for suite_name, suite in evaluation.suites.items():
        bank_path = suite.diagnostics.kl.fixed_prefix_bank
        if bank_path:
            target = freeze_file(
                bank_path, root, f"data/banks/{suite_name}.json", files
            )
            bank = json.loads(target.read_text())
            validate_bank(
                bank,
                tokenizer_id=tokenizer_id,
                models=models,
                vocab_size=vocab_size,
                eos=tokenizer.eos_token_id,
            )
            config.evaluation.suites[suite_name].diagnostics.kl.fixed_prefix_bank = str(
                target
            )
            config.trainer.validation_kl.fixed_prefix_bank = str(target)
    selection_path = root / "data/selection.json"
    atomic_json(selection_path, selection)
    files.append(
        {
            "path": "data/selection.json",
            "sha256": file_digest(selection_path),
            "bytes": selection_path.stat().st_size,
        }
    )
    # Hash once more after tokenization to detect an input changing during preparation.
    for artifact in models.values():
        verify_model(artifact)
    result = {
        "schema_version": 1,
        "models": models,
        "files": files,
        "model_layouts": layouts,
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "token_ids": {
            key: getattr(tokenizer, key + "_token_id") for key in ("bos", "eos", "pad")
        },
        "selection_sha256": file_digest(selection_path),
        "selected_counts": {key: len(value) for key, value in selection.items()},
    }
    result["inputs_id"] = content_id(
        {
            "models": {role: value["identity"] for role, value in models.items()},
            "files": files,
            "tokenizer_id": tokenizer_id,
        }
    )
    return result


def verify_inputs(root, inputs):
    if inputs.get("schema_version") != 1:
        raise ValueError("Unsupported input manifest schema")
    expected = content_id(
        {
            "models": {
                role: value["identity"] for role, value in inputs["models"].items()
            },
            "files": inputs["files"],
            "tokenizer_id": inputs["tokenizer_id"],
        }
    )
    if expected != inputs["inputs_id"]:
        raise ValueError("Input manifest identity changed")
    for artifact in inputs["models"].values():
        verify_model(artifact)
    members = {row["path"] for row in inputs["files"]}
    if len(members) != len(inputs["files"]):
        raise ValueError("Duplicate input artifact member")
    actual = {
        str(p.relative_to(root))
        for p in (Path(root) / "data").rglob("*")
        if p.is_file()
    }
    if actual != members:
        raise ValueError("Prepared input membership changed")
    for row in inputs["files"]:
        if file_digest(inside(root, row["path"])) != row["sha256"]:
            raise ValueError(f"Prepared data/bank changed: {row['path']}")
    if file_digest(Path(root) / "data/selection.json") != inputs["selection_sha256"]:
        raise ValueError("Prepared selected population changed")
    return inputs
