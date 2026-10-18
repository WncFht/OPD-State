"""Driver-side native rollout, independent source weights and common scoring."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from verl.trainer.ppo.fixed_rollout import canonical, digest_file, digest_json, model_signature, pack_token_records
from verl.workers.rollout.native_executor import Request, generate
from verl.workers.rollout.native_pool import NativeModelPool, NativeRayBackend
from verl.workers.rollout.policy import Fusion, ModelRef, compile_policy, parse_policy


def prepare_native_config(config, tokenizer, processor):
    """Resolve the opt-in plan before Ray workers reserve GPU memory."""
    from omegaconf import OmegaConf, open_dict

    rollout = config.actor_rollout_ref.rollout
    if rollout.get("policy") is None:
        return False
    if rollout.get("generation_fusion") is not None:
        raise ValueError("generation_fusion is reserved for the model pool; configure policy.kind=logit_fusion")
    policy = OmegaConf.to_container(rollout.policy, resolve=True)
    parsed = parse_policy(policy, int(rollout.response_length))
    plan = compile_policy(
        policy,
        cap=int(rollout.response_length),
        batch_size=int(config.data.train_batch_size),
        seed=int(rollout.seed),
        batch_id=0,
    )
    if rollout.name != "vllm" or rollout.get("generation_backend", "vllm_native") != "vllm_native":
        raise ValueError("Flat native sources require vllm_native; no HF fallback is supported")
    if processor is not None or rollout.agent.default_agent_loop != "single_turn_agent" or rollout.multi_turn.enable:
        raise ValueError("Native source MVP is single-turn text-only")
    if rollout.n != 1 or config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("Native source training requires n=1 sampled-token OPD")
    if not config.distillation.fsdp_teacher.enabled or config.reward.reward_model.enable:
        raise ValueError("Native sources require the existing FSDP teacher and rule-based verifier")
    if config.trainer.get("fixed_rollout_cache") or rollout.get("logit_fusion", {}).get("enabled", False):
        raise ValueError("Native fresh policies cannot also enable legacy cache or HF fusion")
    if rollout.get("collection", {}).get("mode", "fresh") != "fresh":
        raise ValueError("Use the historical strict cache adapter for replay")
    if any(
        int(rollout.get(k, 1)) != 1
        for k in ["tensor_model_parallel_size", "pipeline_model_parallel_size", "data_parallel_size"]
    ):
        raise ValueError("Native source MVP requires TP=PP=DP=1 per inference replica")
    if (
        not rollout.do_sample
        or rollout.temperature != 1
        or rollout.top_p != 1
        or rollout.top_k != -1
        or rollout.get("repetition_penalty", 1) != 1
        or rollout.get("ignore_eos", False)
    ):
        raise ValueError("Native source comparison requires T=1, full support and ordinary EOS")
    model = config.actor_rollout_ref.model
    if (
        model.get("lora_rank", 0)
        or (model.get("lora") or {}).get("rank", 0)
        or model.get("lora_adapter_path")
        or rollout.get("quantization")
        or (rollout.get("mtp") or {}).get("enable", False)
        or (rollout.get("disaggregation") or {}).get("enabled", False)
    ):
        raise ValueError("Native source MVP excludes LoRA, quantization, MTP and disaggregated inference")
    if config.algorithm.get("rollout_correction") is not None:
        raise ValueError("Native source policies require rollout_correction=null; behavior scores are diagnostics")
    if config.trainer.resume_mode != "disable":
        raise ValueError("Native source visit counters currently require resume_mode=disable")
    if config.distillation.fsdp_teacher.get("unbiased_topk_tail", False) and plan.models != {"current_student"}:
        raise ValueError("Unbiased top-k tail requires fresh current-student actions")
    if config.distillation.fsdp_teacher.get("unbiased_topk_tail", False):
        raise ValueError("Native policies lack verified behavior scores for unbiased top-k tail; use policy=null")
    if rollout.checkpoint_engine.backend != "naive":
        raise ValueError("Native source placement currently uses the existing naive IPC checkpoint writer")

    paths = {
        "current_student": str(Path(model.path).resolve()),
        "initial_student": str(Path(model.path).resolve()),
        "teacher": str(Path(config.distillation.fsdp_teacher.model_path).resolve()),
    }
    base_config = json.loads((Path(paths["current_student"]) / "config.json").read_text())
    if base_config.get("model_type") != "qwen2":
        raise ValueError("Native source MVP is validated for the local dense Qwen2 pair")
    fields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
        "hidden_act",
        "rms_norm_eps",
    ]
    descriptors = {}
    prepared_manifest = config.trainer.get("native_data_manifest")
    if prepared_manifest:
        from verl.trainer.ppo.fixed_rollout import tokenizer_signature

        prepared = json.loads(Path(prepared_manifest).read_text())
        if (
            len(config.data.train_files) != 1
            or digest_file(config.data.train_files[0]) != prepared["ordered_parquet_sha256"]
            or tokenizer_signature(tokenizer) != prepared["tokenizer_signature"]
            or int(rollout.prompt_length) != prepared["prompt_cap"]
            or canonical(config.data.get("apply_chat_template_kwargs", {})) != prepared["apply_chat_template_kwargs"]
        ):
            raise ValueError("Native prepared prompt pool does not match the training data/tokenizer/template/cap")
    budget = float(rollout.gpu_memory_utilization)
    fraction = budget if isinstance(parsed, Fusion) else budget / max(len(plan.models), 1)
    for role in sorted(plan.models - {"current_student"}):
        c = json.loads((Path(paths[role]) / "config.json").read_text())
        if any(c.get(k) != base_config.get(k) for k in fields):
            raise ValueError(f"Native source checkpoint is incompatible with actor: {role}")
        from verl.trainer.ppo.fixed_rollout import tokenizer_signature
        from verl.utils.tokenizer import hf_tokenizer

        if tokenizer_signature(hf_tokenizer(paths[role])) != tokenizer_signature(tokenizer):
            raise ValueError(f"Native source tokenizer/template does not match: {role}")
        descriptors[role] = {
            "path": paths[role],
            "fingerprint": model_signature(paths[role]),
            "memory_fraction": fraction,
        }
    with open_dict(config):
        rollout.generation_models = descriptors
        rollout.generation_sync_namespace = uuid.uuid4().hex
        # The eval-only current engine may use the full budget when it is never
        # concurrent with the fixed training sources.
        rollout.gpu_memory_utilization = budget
        rollout.generation_purpose = "evaluation"
        config.trainer.native_generation_memory_budget = budget
        config.trainer.native_required_models = sorted(plan.models)
    return True


def epochs_for_budget(total_steps, batches_per_epoch):
    if type(total_steps) is not int or total_steps < 1 or batches_per_epoch < 1:
        raise ValueError("Native training needs a positive update budget and nonempty dataloader")
    return max(1, math.ceil(total_steps / batches_per_epoch))


class RolloutCoordinator:
    def __init__(self, config, tokenizer, model_pool, checkpoint_manager, reward_manager):
        from omegaconf import OmegaConf

        self.config, self.tokenizer, self.pool = config, tokenizer, model_pool
        self.checkpoint = checkpoint_manager
        self.reward_manager = reward_manager
        self.rollout = config.actor_rollout_ref.rollout
        self.policy = OmegaConf.to_container(self.rollout.policy, resolve=True)
        self.uses_actor = "current_student" in config.trainer.native_required_models
        parsed = parse_policy(self.policy, int(self.rollout.response_length))
        self.fusion = parsed if isinstance(parsed, Fusion) else None
        self.train_slot = f"fusion:{parsed.base}" if self.fusion is not None else "current_student"
        self.current_path = str(Path(config.actor_rollout_ref.model.path).resolve())
        self.version = None
        self.visits = defaultdict(int)
        self.eval_samples = defaultdict(int)
        self.eval_version = None
        self.trace_dir = Path(config.trainer.default_local_dir).parent / "native-trajectories"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "policy": self.policy,
            "generation_models": canonical(self.rollout.generation_models),
            "train_files": {str(p): digest_file(p) for p in config.data.train_files},
            "dataset_rows_are_not_limited_by_training_steps": True,
            "memory_budget": config.trainer.native_generation_memory_budget,
            "prompt_cap": int(self.rollout.prompt_length),
            "response_cap": int(self.rollout.response_length),
            "seed": int(self.rollout.seed),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "trainer_gpu_count": int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes),
            "verify_frozen_parameters": bool(self.rollout.generation_verify_frozen),
            "engines": model_pool.describe() if model_pool is not None and hasattr(model_pool, "describe") else {},
            "prepared_data_manifest_sha256": digest_file(config.trainer.native_data_manifest)
            if config.trainer.get("native_data_manifest")
            else None,
        }
        (self.trace_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def create(cls, config, tokenizer, main_manager, checkpoint_manager, reward_manager):
        pool = NativeModelPool(config, main_manager)
        asyncio.run(pool.initialize())
        return cls(config, tokenizer, pool, checkpoint_manager, reward_manager)

    def update_weights(self, global_steps=None):
        if not self.uses_actor:
            return {"source/rollout_sync": 0.0}
        return self.sync_current(global_steps)

    def sync_current(self, version, *, for_eval=False):
        slot = self.pool.eval_slot if for_eval else self.train_slot
        import ray

        expected_fusion = canonical(self.pool.fusion_config) if self.fusion is not None and not for_eval else None
        ray.get(
            self.checkpoint.actor_wg.set_native_rollout_target(
                self.pool.handles[slot], expected_fusion, "evaluation" if for_eval else "training"
            )
        )
        self.checkpoint.update_weights(version)
        self.version = int(version)
        self.pool.awake.add(slot)
        if self.fusion is not None:

            async def statuses():
                return await asyncio.gather(*(h.native_source_status.remote() for h in self.pool.handles[slot]))

            published = asyncio.run(statuses())
            if any(s["identity"] != ("current_student", self.current_path, int(version)) for s in published):
                raise RuntimeError("Actor publication did not reach the selected native engines")
            with (self.trace_dir / "publications.jsonl").open("a") as stream:
                stream.write(json.dumps({"version": int(version), "slot": slot, "sources": published}) + "\n")
        return {"source/rollout_sync": 1.0}

    def sleep_replicas(self):
        asyncio.run(self.pool.sleep_all())

    def __getattr__(self, name):
        return getattr(self.checkpoint, name)

    def begin_eval(self, version):
        self.sleep_replicas()
        self.sync_current(version, for_eval=True)
        self.eval_samples.clear()
        self.eval_version = version

    def end_eval(self, version, *, resume_training):
        self.sleep_replicas()
        self.eval_version = None
        if resume_training and self.uses_actor:
            self.sync_current(version)

    def _requests(self, batch, validate):
        requests = []
        for i, raw in enumerate(batch.non_tensor_batch["raw_prompt"]):
            messages = canonical(raw)
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **dict(self.config.data.get("apply_chat_template_kwargs", {})),
            )
            if len(ids) > self.rollout.prompt_length:
                raise ValueError("Canonical prompt exceeds the shared prompt budget")
            source = batch.non_tensor_batch.get("data_source", [""] * len(batch))[i]
            stable_id = digest_json([str(source), ids])
            if validate:
                sample = self.eval_samples[stable_id]
                self.eval_samples[stable_id] += 1
                request = Request(f"eval:{stable_id}", tuple(ids), sample=sample)
            else:
                explicit = batch.non_tensor_batch.get("opd_prompt_id")
                index = (
                    str(explicit[i])
                    if explicit is not None
                    else str(batch.non_tensor_batch.get("index", [""] * len(batch))[i])
                )
                key = f"train:{stable_id}"
                visit = self.visits[key]
                self.visits[key] += 1
                request = Request(key, tuple(ids), visit=visit, row_id=index)
            requests.append(request)
        return requests

    def generate_sequences(self, batch):
        from verl import DataProto

        validate = bool(batch.meta_info.get("validate", False))
        step = int(batch.meta_info["global_steps"])
        version = step if validate else step - 1
        cap = (
            int(self.rollout.val_kwargs.get("max_tokens") or self.rollout.response_length)
            if validate
            else int(self.rollout.response_length)
        )
        policy = {"kind": "single", "model": "current_student"} if validate else self.policy
        if validate and self.eval_version != step:
            raise RuntimeError("Evaluation must begin with an explicit latest-student synchronization")
        refs = {"current_student": ModelRef("current_student", self.current_path, version=version)}
        for role, d in self.rollout.generation_models.items():
            refs[role] = ModelRef(role, d.path, fingerprint=d.fingerprint)
        requests = self._requests(batch, validate)
        plan = compile_policy(policy, cap=cap, batch_size=len(requests), seed=int(self.rollout.seed), batch_id=step)
        sampling = {
            "temperature": (self.rollout.val_kwargs.temperature if self.rollout.val_kwargs.do_sample else 0)
            if validate
            else self.rollout.temperature,
            "top_p": self.rollout.val_kwargs.top_p if validate else self.rollout.top_p,
            "top_k": self.rollout.val_kwargs.top_k if validate else self.rollout.top_k,
            "logprobs": False,
        }
        backend = NativeRayBackend(self.pool, sampling, validate=validate)
        contexts = {
            role: int(self.rollout.max_model_len)
            if role == "current_student"
            else int(self.rollout.prompt_length + self.rollout.response_length)
            for role in refs
        }
        started = time.perf_counter()
        trajectories, timing = asyncio.run(
            generate(
                requests,
                plan,
                backend,
                refs=refs,
                context_limits=contexts,
                vocab_size=len(self.tokenizer),
                eos_token_id=self.tokenizer.eos_token_id,
                seed=int(self.rollout.seed),
                concurrent_groups=True,
            )
        )
        if not validate:
            self.sleep_replicas()
        records = [(list(r.request.prompt_ids), list(r.token_ids), r.termination) for r in trajectories]
        result = pack_token_records(
            records,
            prompt_cap=int(self.rollout.prompt_length),
            response_cap=cap,
            pad_token_id=self.tokenizer.pad_token_id,
            source="native_policy",
        )
        # The same existing verifier scores every source, including teacher and
        # mixed trajectories. Do not let the packer's historical zero scores
        # masquerade as actual task accuracy.
        result.batch.pop("rm_scores")
        scoring = DataProto(
            batch=result.batch,
            non_tensor_batch={**result.non_tensor_batch, **batch.non_tensor_batch},
            meta_info=dict(batch.meta_info),
        )
        rewards = self.reward_manager.compute_rm_score(scoring)
        result = result.union(rewards)
        result.meta_info["timing"] = {
            "native_generate": timing["seconds"],
            "native_score_and_pack": time.perf_counter() - started - timing["seconds"],
        }
        metrics = {
            "source/response_count": len(trajectories),
            "source/unique_prompts": len({r.request.prompt_id for r in trajectories}),
            "source/revisited_responses": sum(r.request.visit > 0 for r in trajectories),
            "source/output_tokens": sum(len(r.token_ids) for r in trajectories),
            "source/prefill_tokens": sum(g["input_tokens"] for g in timing["groups"]),
            "source/switched_responses": sum(len(r.segments) > 1 for r in trajectories),
            "source/eos_responses": sum(r.termination == "eos" for r in trajectories),
        }
        for role in plan.models:
            segments = [(r, s) for r in trajectories for s in r.segments if role in [i[0] for i in s.identities]]
            metrics[f"source/{role}/tokens"] = sum(s.end - s.start for _, s in segments)
            metrics[f"source/{role}/responses"] = len({r.request.key for r, _ in segments})
        result.meta_info["native_metrics"] = metrics
        audit = {
            "step": step,
            "validate": validate,
            "identities": sorted({i for r in trajectories for s in r.segments for i in s.identities}),
            "metrics": metrics,
            "generation": timing,
            "frozen_verifications": backend.verifications,
        }
        with (self.trace_dir / "batches.jsonl").open("a") as stream:
            stream.write(json.dumps(audit, separators=(",", ":")) + "\n")
        if not validate:
            path = self.trace_dir / f"step-{step}.jsonl"
            with path.open("x") as stream:
                for r in trajectories:
                    stream.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
        return result
