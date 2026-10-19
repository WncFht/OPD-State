"""Compare real-model PPO and score gradients at their common initial anchor.

This CPU reference uses BF16 forward and FP32 log-prob normalization over the
full model head, matching the ordinary sampled objective's support. It does
not establish FlashAttention cross-entropy, FSDP or GPU throughput equivalence.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path


def compare_tensors(reference, candidate):
    import torch

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Compared tensor layout changed")
    if torch.equal(reference, candidate):
        return {"exact": True, "max_absolute_error": 0.0}
    difference = (reference.double() - candidate.double()).abs()
    result = {"exact": False, "max_absolute_error": difference.max().item()}
    torch.testing.assert_close(candidate, reference, rtol=1e-6, atol=1e-8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--response-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--candidate-advantage", type=Path)
    args = parser.parse_args()
    if min(args.records, args.response_tokens, args.threads) < 1:
        parser.error("Positive record, token and thread budgets are required")
    root = args.prepared_run.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "driver.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VERL_DISABLE_FLASH_ATTN_CE"] = "1"
    os.environ["VERL_USE_EXTERNAL_PLUGINS"] = "none"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = [str(root / "source"), str(root / "source/backend/verl")]
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.trainer.ppo.opd_core_algos import compute_token_reward_direct_advantage
    from verl.utils.torch_functional import logprobs_from_logits

    from opd_state.artifacts import atomic_json, file_digest
    from tools.verify import verify_prepared

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.perf_counter()
    config, manifest, inputs = verify_prepared(root)
    candidate, candidate_sha = None, None
    if args.candidate_advantage is not None:
        frozen_candidate = output / "candidate_advantages.py"
        shutil.copyfile(args.candidate_advantage, frozen_candidate)
        candidate_sha = file_digest(frozen_candidate)
        if candidate_sha != file_digest(args.candidate_advantage):
            raise ValueError("Candidate changed while preparing its frozen comparison")
        spec = importlib.util.spec_from_file_location(
            "_opd_candidate_advantages", frozen_candidate
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        candidate = module.compute_token_reward_direct_advantage
    actor_settings = config.actor_rollout_ref.actor
    if (
        config.distillation.opd.objective.kind != "sampled_opd"
        or actor_settings.policy_loss.loss_mode != "vanilla"
        or actor_settings.loss_agg_mode != "token-mean"
        or actor_settings.optim.optimizer != "AdamW"
        or actor_settings.optim.optimizer_impl != "torch.optim"
        or actor_settings.optim.override_optimizer_config
    ):
        raise ValueError(
            "Reference requires the declared sampled PPO/token-mean/AdamW protocol"
        )
    path = root / "native-trajectories/step-1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][: args.records]
    if len(rows) != args.records:
        raise ValueError("Insufficient real trajectories for the requested reference")
    prompts = [row["request"]["prompt_ids"] for row in rows]
    # Deliberately unequal response lengths exercise the mask/global token mean.
    responses = [
        row["token_ids"][: max(1, args.response_tokens - index)]
        for index, row in enumerate(rows)
    ]
    width = max(len(p) + len(r) for p, r in zip(prompts, responses, strict=True))
    response_width = max(map(len, responses))
    ids = torch.full((len(rows), width), inputs["token_ids"]["pad"], dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((len(rows), response_width), dtype=torch.long)
    labels = torch.zeros_like(positions)
    mask = torch.zeros_like(positions, dtype=torch.bool)
    for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        sequence = prompt + response
        ids[index, : len(sequence)] = torch.tensor(sequence)
        attention[index, : len(sequence)] = 1
        positions[index, : len(response)] = torch.arange(
            len(prompt) - 1, len(sequence) - 1
        )
        labels[index, : len(response)] = torch.tensor(response)
        mask[index, : len(response)] = True
    atomic_json(
        output / "inputs.json",
        {
            "source_run": str(root),
            "source_id": manifest["source_id"],
            "config_id": manifest["config_id"],
            "configuration_sha256": manifest["configuration_sha256"],
            "inputs_id": inputs["inputs_id"],
            "trajectory_sha256": file_digest(path),
            "prompts": prompts,
            "responses": responses,
            "condition": "CPU, BF16 forward, FP32 normalization, full output head, initial anchor, unequal masks",
        },
    )

    def load(role, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            inputs["models"][role]["path"],
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        ).eval()
        print(f"Loaded {role} on CPU", flush=True)
        return model

    def scores(model):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(
                input_ids=ids, attention_mask=attention, use_cache=False
            ).logits
        chosen = logits[torch.arange(len(rows))[:, None], positions].float()
        return logprobs_from_logits(chosen, labels, inplace_backward=False)

    student = load("student", torch.float32)
    with torch.no_grad():
        old_logp = scores(student).detach()
        teacher = load("teacher", torch.bfloat16)
        teacher_logp = scores(teacher).detach()
    del teacher
    gc.collect()
    reward = teacher_logp - old_logp
    advantage, _ = compute_token_reward_direct_advantage(reward, mask)
    atomic_json(
        output / "scores.json",
        {
            "old_log_probs": old_logp.tolist(),
            "teacher_log_probs": teacher_logp.tolist(),
            "reward": reward.tolist(),
            "mask": mask.tolist(),
        },
    )
    parameters = dict(student.named_parameters())
    initial = {name: value.detach().clone() for name, value in parameters.items()}
    actor = OmegaConf.create(
        OmegaConf.to_container(config.actor_rollout_ref.actor, resolve=True)
    )
    actor.global_batch_info = {
        "dp_size": 1,
        "batch_num_tokens": int(mask.sum()),
        "global_batch_size": len(rows),
        "loss_scale_factor": None,
    }
    reference_grads, reference_parameters, reference_state = {}, {}, {}
    comparisons, measurements = {}, []
    for mode in (
        "native_ppo",
        "migrated_advantage" if candidate else "score_reference",
    ):
        student.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(initial[name])
        tick = time.perf_counter()
        current = scores(student)
        if not torch.equal(current.detach(), old_logp):
            raise ValueError(
                "Paired forward changed the common anchor log probabilities"
            )
        if mode == "native_ppo" or candidate is not None:
            selected_advantage = advantage
            if mode == "migrated_advantage":
                selected_advantage, _ = candidate(reward, mask)
            loss, _ = compute_policy_loss_vanilla(
                old_logp,
                current,
                selected_advantage,
                mask,
                loss_agg_mode="token-mean",
                config=actor,
            )
        else:
            loss = -(reward.detach() * current * mask).sum() / mask.sum()
        loss.backward()
        print(f"Computed all parameter gradients: {mode}", flush=True)
        for name, parameter in parameters.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(
                    f"Missing or nonfinite real parameter gradient: {name}"
                )
            if mode == "native_ppo":
                reference_grads[name] = parameter.grad.detach().clone()
            else:
                comparisons[name] = {
                    "gradient": compare_tensors(
                        reference_grads.pop(name), parameter.grad
                    )
                }
        norm = torch.nn.utils.clip_grad_norm_(
            parameters.values(), actor.optim.clip_grad
        )
        optimizer = torch.optim.AdamW(
            parameters.values(),
            lr=actor.optim.lr,
            weight_decay=actor.optim.weight_decay,
            betas=tuple(actor.optim.betas),
            foreach=False,
        )
        optimizer.step()
        for name, parameter in parameters.items():
            if mode == "native_ppo":
                reference_parameters[name] = parameter.detach().clone()
                reference_state[name] = copy.deepcopy(optimizer.state[parameter])
            else:
                comparisons[name]["updated_parameter"] = compare_tensors(
                    reference_parameters.pop(name), parameter
                )
                state = reference_state.pop(name)
                comparisons[name]["optimizer"] = {
                    key: compare_tensors(value, optimizer.state[parameter][key])
                    for key, value in state.items()
                }
        measurements.append(
            {
                "mode": mode,
                "loss": float(loss.detach()),
                "grad_norm": float(norm),
                "seconds": time.perf_counter() - tick,
            }
        )
        del optimizer, loss, current
        gc.collect()
        print(f"Checked optimizer step: {mode}", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU reference unexpectedly initialized CUDA")
    atomic_json(
        output / "result.json",
        {
            "scope": "Real initial-model CPU parameter-gradient and AdamW comparison at the common anchor; not FSDP/GPU equivalence",
            "candidate_advantage_sha256": candidate_sha,
            "passed": True,
            "source_id": manifest["source_id"],
            "config_id": manifest["config_id"],
            "settings": {
                "student_parameter_dtype": "float32",
                "forward_dtype": "bfloat16",
                "log_prob_dtype": "float32",
                "attention": "sdpa",
                "model_mode": "eval",
                "head_vocab_size": student.config.vocab_size,
                "tokenizer_vocab_size": inputs["vocab_size"],
                "optimizer": "AdamW",
                "foreach": False,
                "eps": 1e-8,
                "lr": actor.optim.lr,
                "weight_decay": actor.optim.weight_decay,
                "betas": list(actor.optim.betas),
                "clip_grad": actor.optim.clip_grad,
                "threads": args.threads,
                "steps_per_mode": 1,
                "scalar_loss_equality_required": False,
            },
            "models": {
                role: value["identity"] for role, value in inputs["models"].items()
            },
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(p.numel() for p in parameters.values()),
            "effective_response_positions": int(mask.sum()),
            "measurements": measurements,
            "comparisons": comparisons,
            "cuda_initialized": False,
            "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            "seconds": time.perf_counter() - started,
            "driver_sha256": file_digest(output / "driver.py"),
        },
    )


if __name__ == "__main__":
    main()
