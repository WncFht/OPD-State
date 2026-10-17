"""Strict, read-only trajectories for fixed-teacher and initial-student OPD.

Only trajectory tokens are reused. Current student probabilities and the frozen
teacher score are computed by the existing trainer on every optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = 1


def canonical(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    return value


def digest_json(value):
    payload = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_signature(tokenizer):
    return digest_json(
        {
            "vocab": tokenizer.get_vocab(),
            "template": tokenizer.chat_template,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "bos": tokenizer.bos_token_id,
        }
    )


def model_signature(path):
    root = Path(path).resolve()
    files = sorted({*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin"), root / "config.json"})
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise ValueError(f"Model files are incomplete: {root}")
    return digest_json({p.name: digest_file(p) for p in files})


class FixedRolloutCache:
    """Bind an immutable cache to a source, tokenizer, target and sampling protocol."""

    def __init__(self, directory, *, expected=None):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA_VERSION:
            raise ValueError("Unsupported fixed-rollout cache schema")
        if self.manifest.get("source") not in {"cached_teacher", "cached_initial_student"}:
            raise ValueError("Invalid fixed-rollout cache source")
        for key, value in (expected or {}).items():
            if self.manifest.get(key) != canonical(value):
                raise ValueError(f"Cache protocol mismatch: {key}")
        database = self.directory / "trajectories.sqlite"
        if digest_file(database) != self.manifest["database_sha256"]:
            raise ValueError("Cache database checksum mismatch")
        self.connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        count = self.connection.execute("SELECT count(*) FROM trajectories").fetchone()[0]
        if count != self.manifest["records"]:
            self.connection.close()
            raise ValueError("Cache record count mismatch")
        self.read_count = 0

    def close(self):
        self.connection.close()

    def lookup(self, prompt_id, raw_prompt):
        row = self.connection.execute(
            "SELECT prompt_sha256,prompt_ids,response_ids,finish_reason FROM trajectories WHERE prompt_id=?",
            (str(prompt_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Fixed trajectory is missing: {prompt_id}")
        if row[0] != digest_json(raw_prompt):
            raise ValueError(f"Cached prompt content mismatch: {prompt_id}")
        prompt_ids, response_ids = json.loads(row[1]), json.loads(row[2])
        limit = self.manifest["vocab_size"]
        if (
            not prompt_ids
            or not response_ids
            or len(prompt_ids) > self.manifest["prompt_cap"]
            or len(response_ids) > self.manifest["response_cap"]
        ):
            raise ValueError(f"Cached token lengths are invalid: {prompt_id}")
        if any(type(token) is not int or not 0 <= token < limit for token in prompt_ids + response_ids):
            raise ValueError(f"Cached token IDs are invalid: {prompt_id}")
        eos = self.manifest["eos_token_id"]
        if eos in response_ids[:-1]:
            raise ValueError(f"Cached response continues after EOS: {prompt_id}")
        self.read_count += 1
        return prompt_ids, response_ids, row[3]

    def build_batch(self, prompts):
        """Pack tokens while retaining fresh UIDs and the current batch's metadata."""
        import numpy as np
        import torch

        from verl import DataProto

        ids = prompts.non_tensor_batch.get("opd_prompt_id")
        if ids is None:
            raise ValueError("Fixed rollout requires stable opd_prompt_id from the paired schedule")
        raw = prompts.non_tensor_batch["raw_prompt"]
        records = [self.lookup(key, value) for key, value in zip(ids, raw, strict=True)]
        bsz = len(records)
        pcap, rcap = self.manifest["prompt_cap"], self.manifest["response_cap"]
        pad = self.manifest["pad_token_id"]
        p = torch.full((bsz, pcap), pad, dtype=torch.long)
        r = torch.full((bsz, rcap), pad, dtype=torch.long)
        p_mask, r_mask = torch.zeros_like(p), torch.zeros_like(r)
        for i, (pi, ri, _) in enumerate(records):
            p[i, -len(pi) :] = torch.tensor(pi)
            r[i, : len(ri)] = torch.tensor(ri)
            p_mask[i, -len(pi) :] = 1
            r_mask[i, : len(ri)] = 1
        attention_mask = torch.cat([p_mask, r_mask], dim=-1)
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        positions.masked_fill_(attention_mask == 0, 0)
        multi_modal = np.empty(bsz, dtype=object)
        for i in range(bsz):
            multi_modal[i] = {}
        output = DataProto.from_dict(
            tensors={
                "prompts": p,
                "responses": r,
                "input_ids": torch.cat([p, r], dim=-1),
                "attention_mask": attention_mask,
                "position_ids": positions,
                "response_mask": r_mask,
                "rm_scores": torch.zeros((bsz, rcap)),
            },
            non_tensors={"multi_modal_inputs": multi_modal, "num_turns": np.full(bsz, 2, dtype=np.int32)},
            meta_info={"timing": {}, "fixed_source": self.manifest["source"]},
        )
        return output


def open_training_cache(config, tokenizer):
    directory = config.trainer.get("fixed_rollout_cache")
    if not directory:
        return None
    if config.actor_rollout_ref.rollout.n != 1 or config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("Fixed-rollout OPD requires n=1 and token_reward_direct")
    source = config.trainer.get("rollout_source")
    expected = {
        "source": source,
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "prompt_cap": config.data.max_prompt_length,
        "response_cap": config.data.max_response_length,
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "student_signature": model_signature(config.actor_rollout_ref.model.path),
        "teacher_signature": model_signature(config.distillation.fsdp_teacher.model_path),
        "apply_chat_template_kwargs": canonical(config.data.get("apply_chat_template_kwargs", {})),
    }
    train_files = list(config.data.train_files)
    if len(train_files) != 1:
        raise ValueError("Fixed-rollout training requires one verified paired schedule")
    expected["schedule_parquet_sha256"] = digest_file(train_files[0])
    return FixedRolloutCache(directory, expected=expected)
