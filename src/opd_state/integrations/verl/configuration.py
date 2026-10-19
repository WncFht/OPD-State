"""Native execution constraints and the temporary stock validation bridge.

Scientific configuration is composed once. During extraction of the trainer,
derived stock fields are checked against the suite that owns their meaning.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

from opd_state.core.config import Objective, decode, positive
from opd_state.core.evaluation import KL, parse_evaluation

FULL_MATH_COUNTS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}


def plain(value):
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def evaluation_field(value, field, total_steps=None):
    """Project one live suite into existing hooks until the suite runner is extracted."""
    evaluation = parse_evaluation(plain(value))
    if len(evaluation.suites) > 1:
        raise ValueError("The stock validation bridge accepts one live suite")
    if not evaluation.suites:
        defaults = {
            "files": [],
            "initial": False,
            "frequency": -1,
            "kl": asdict(KL()),
            "require_full": False,
            "n": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
            "do_sample": True,
            "max_tokens": 31744,
            "context": 0,
        }
        return defaults[field]
    suite = next(iter(evaluation.suites.values()))
    if len(suite.benchmarks) != 1:
        raise ValueError(
            "The stock validation bridge accepts one grouped math benchmark"
        )
    benchmark = evaluation.benchmarks[suite.benchmarks[0]]
    trigger = suite.trigger
    if trigger.every_outer_steps and not trigger.training_end:
        raise ValueError("Periodic-only suites require the extracted suite runner")
    fields = {
        "files": benchmark.data.paths,
        "initial": trigger.initial,
        "frequency": trigger.every_outer_steps
        or (total_steps if trigger.training_end else -1),
        "kl": asdict(suite.diagnostics.kl),
        "require_full": benchmark.data.expected_counts == FULL_MATH_COUNTS,
        "n": benchmark.sampling.samples_per_case,
        "temperature": benchmark.sampling.temperature,
        "top_p": benchmark.sampling.top_p,
        "top_k": benchmark.sampling.top_k,
        "do_sample": benchmark.sampling.do_sample,
        "max_tokens": benchmark.budget.response_tokens,
        "context": benchmark.budget.context_tokens,
    }
    return fields[field]


@lru_cache(maxsize=8)
def _policy_module(path, digest):
    """Use the selected backend's existing pure policy parser during extraction."""
    name = "_opd_backend_policy_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_config(config, *, backend):
    """Reject unsupported scientific/execution combinations before input or CUDA work."""
    objective = decode(Objective, plain(config.distillation.opd.objective), "objective")
    evaluation = parse_evaluation(plain(config.evaluation))
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    teacher = config.distillation.fsdp_teacher
    trainer = config.trainer
    for name, value in (
        ("train_batch_size", config.data.train_batch_size),
        ("ppo_mini_batch_size", actor.ppo_mini_batch_size),
        ("ppo_micro_batch_size_per_gpu", actor.ppo_micro_batch_size_per_gpu),
        ("n_gpus_per_node", trainer.n_gpus_per_node),
        ("total_training_steps", trainer.total_training_steps),
        ("max_prompt_length", config.data.max_prompt_length),
        ("max_response_length", config.data.max_response_length),
    ):
        positive(value, name)
    if (
        actor.ppo_epochs != 1
        or actor.ppo_mini_batch_size != config.data.train_batch_size
    ):
        raise ValueError("OPD requires one complete update per fresh prompt batch")
    if config.data.train_batch_size % (
        trainer.n_gpus_per_node * actor.ppo_micro_batch_size_per_gpu
    ):
        raise ValueError("Global batch must divide GPU count times micro-batch size")
    if trainer.nnodes != 1 or actor.strategy != "fsdp" or trainer.use_v1:
        raise ValueError(
            "The verified backend uses single-node FSDP and the V0 trainer"
        )
    if trainer.resume_mode != "disable":
        raise ValueError("Equivalent resume is not yet admitted")
    if not teacher.enabled or config.distillation.enabled or config.critic.enable:
        raise ValueError(
            "Use the colocated FSDP teacher, without upstream distillation or critic"
        )
    if config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("OPD uses token_reward_direct without verifier advantages")
    if config.algorithm.rollout_correction is not None:
        raise ValueError(
            "The source comparison does not apply behavior importance correction"
        )
    if (
        config.algorithm.use_kl_in_reward
        or actor.use_kl_loss
        or actor.entropy_coeff != 0
    ):
        raise ValueError(
            "Additional KL or entropy objectives change the source protocol"
        )
    if (
        rollout.n != 1
        or not rollout.do_sample
        or rollout.temperature != 1
        or teacher.teacher_temperature != 1
    ):
        raise ValueError(
            "Training requires one response with student/teacher temperature 1"
        )
    if (
        rollout.top_p != 1
        or rollout.top_k != -1
        or rollout.get("repetition_penalty", 1) != 1
    ):
        raise ValueError("Training must retain the complete action support")
    if rollout.name != "vllm" or rollout.generation_backend != "vllm_native":
        raise ValueError("Select the native vLLM backend")
    if (
        any(
            rollout[key] != 1
            for key in (
                "tensor_model_parallel_size",
                "data_parallel_size",
                "pipeline_model_parallel_size",
            )
        )
        or actor.ulysses_sequence_parallel_size != 1
    ):
        raise ValueError("Verified replicas require TP=DP=PP=SP=1")
    if rollout.multi_turn.enable or config.actor_rollout_ref.model.lora_rank:
        raise ValueError("The verified source path is full-parameter, single-turn text")
    if (
        rollout.get("quantization")
        or rollout.ignore_eos
        or rollout.collection.mode != "fresh"
    ):
        raise ValueError("Use fresh, non-quantized generation with ordinary EOS")
    if teacher.teachers or config.trainer.get("fixed_rollout_cache"):
        raise ValueError("This protocol has one teacher and no trajectory replay")
    if config.actor_rollout_ref.model.use_fused_kernels:
        raise ValueError("State diagnostics require the unfused causal forward")
    if objective.kind == "sampled_opd":
        if (
            teacher.log_prob_top_k
            or teacher.metric_top_k
            or teacher.unbiased_topk_tail
            or actor.policy_loss.loss_mode != "vanilla"
        ):
            raise ValueError(
                "sampled_opd requires the original scalar token reward and vanilla surrogate"
            )
        path = Path(backend).resolve() / "verl/verl/workers/rollout/policy.py"
        module = _policy_module(
            str(path), hashlib.sha256(path.read_bytes()).hexdigest()
        )
        module.parse_policy(plain(rollout.policy), config.data.max_response_length)
    elif (
        not teacher.unbiased_topk_tail
        or rollout.policy is not None
        or actor.policy_loss.loss_mode != "opd_topk_tail"
    ):
        raise ValueError(
            "Unbiased head-tail currently requires stock fresh-student rollout"
        )
    for name, value in (
        ("learning rate", actor.optim.lr),
        ("gradient clip", actor.optim.clip_grad),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if actor.grad_clip != actor.optim.clip_grad:
        raise ValueError("Actor and optimizer gradient clip values must agree")
    if (
        rollout.prompt_length != config.data.max_prompt_length
        or rollout.response_length != config.data.max_response_length
    ):
        raise ValueError("Data and training generation length budgets must agree")
    if (
        rollout.max_model_len
        < config.data.max_prompt_length + config.data.max_response_length
    ):
        raise ValueError(
            "Generation context is shorter than the training sequence budget"
        )
    if (
        actor.ppo_max_token_len_per_gpu
        < rollout.prompt_length + rollout.response_length
    ):
        raise ValueError("The dynamic batch token budget must fit one full sequence")
    for field, actual in (
        ("files", list(config.data.val_files)),
        ("initial", trainer.val_before_train),
        ("frequency", trainer.test_freq),
        ("kl", plain(trainer.validation_kl)),
        ("require_full", trainer.math_eval_require_full),
        ("n", trainer.math_eval_samples),
    ):
        if actual != evaluation_field(
            config.evaluation, field, trainer.total_training_steps
        ):
            raise ValueError(
                f"Native validation {field} disagrees with its owning evaluation suite"
            )
    for field, native in (
        ("n", "n"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("do_sample", "do_sample"),
        ("max_tokens", "max_tokens"),
    ):
        if rollout.val_kwargs[native] != evaluation_field(config.evaluation, field):
            raise ValueError(
                f"Native validation sampling {native} disagrees with its benchmark"
            )
    for suite in evaluation.suites.values():
        kl = suite.diagnostics.kl
        if kl.enabled and (rollout.policy is None or kl.forward_batch_size != 1):
            raise ValueError(
                "Validated KL requires native sources and backbone batch 1"
            )
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        positive(rollout[key], key)
        override = rollout.get("generation_eval_" + key)
        if override is not None:
            positive(override, "generation_eval_" + key)
    for value in (
        rollout.gpu_memory_utilization,
        rollout.generation_eval_memory_utilization,
    ):
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(
                "Generation memory budgets must be finite fractions in (0,1)"
            )
    return objective, evaluation
