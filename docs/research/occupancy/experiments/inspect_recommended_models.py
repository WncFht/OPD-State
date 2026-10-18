"""Read pinned public configuration/tokenizer files; never fetch model weights."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    ),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    ),
    ("hbx/JustRL-DeepSeek-1.5B", "0637e4096c789c67f9eecbe8355e0bdeddede1c2"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def token_content(value):
    return value.get("content") if isinstance(value, dict) else value


def collect():
    rows = []
    tokenizers = {}
    for model, revision in MODELS:
        files = {}
        for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            url = f"https://huggingface.co/{model}/resolve/{revision}/{filename}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            value = json.loads(raw)
            entry = {"sha256": digest(raw), "bytes": len(raw), "url": url}
            if filename == "config.json":
                entry["selected_fields"] = {
                    key: value.get(key)
                    for key in (
                        "model_type",
                        "architectures",
                        "vocab_size",
                        "hidden_size",
                        "num_hidden_layers",
                        "bos_token_id",
                        "eos_token_id",
                    )
                }
            elif filename == "tokenizer_config.json":
                entry["selected_fields"] = {
                    "tokenizer_class": value.get("tokenizer_class"),
                    **{
                        key: token_content(value.get(key))
                        for key in ("bos_token", "eos_token", "pad_token")
                    },
                    "chat_template_sha256": digest(
                        value.get("chat_template", "").encode()
                    ),
                }
            else:
                tokenizers[model] = value
            files[filename] = entry
        rows.append({"id": model, "revision": revision, "files": files})
    reference = tokenizers[MODELS[0][0]]
    comparisons = {
        model: {
            key: tokenizer.get(key) == reference.get(key)
            for key in (
                "model",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
            )
        }
        for model, tokenizer in tokenizers.items()
    }
    vocab = dict(reference["model"]["vocab"])
    vocab.update({entry["content"]: entry["id"] for entry in reference["added_tokens"]})
    configs = [row["files"]["tokenizer_config.json"]["selected_fields"] for row in rows]
    return {
        "date": "2026-10-18",
        "scope": "Pinned public metadata only; no model weights, inference, or training",
        "producer_sha256": digest(Path(__file__).read_bytes()),
        "models": rows,
        "tokenizer_matches_to_student": comparisons,
        "tokenizer_config_selected_fields_equal": all(
            value == configs[0] for value in configs
        ),
        "common_tokenizer": {
            "token_count": len(vocab),
            "max_token_id": max(vocab.values()),
            "special_ids": {
                token: vocab.get(token)
                for token in (
                    "<｜begin▁of▁sentence｜>",
                    "<｜end▁of▁sentence｜>",
                    "<think>",
                    "</think>",
                )
            },
        },
    }


if __name__ == "__main__":
    result = collect()
    output = ROOT / "results/recommended-models.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "models": len(result["models"]),
                "common_tokenizer": result["common_tokenizer"],
                "all_selected_tokenizer_components_match": all(
                    all(row.values())
                    for row in result["tokenizer_matches_to_student"].values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
