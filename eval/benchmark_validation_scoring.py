"""Paired four-GPU KL scoring controls on fixed real response token IDs.

This benchmark loads a frozen pretrained pair, uses actual FSDP collectives,
and compares implementations on identical inputs. It is not a recomputation
of a discarded trained checkpoint or a training-quality experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import timedelta
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("legacy-input-128-8192", False, 128, 8192, 1, True),
    ("balanced-128-8192", True, 128, 8192, 1, False),
    ("balanced-512-full", True, 512, 151665, 1, False),
    ("balanced-1024-full", True, 1024, 151665, 1, False),
    ("balanced-2048-full", True, 2048, 151665, 1, False),
    ("balanced-1024-full-b2", True, 1024, 151665, 2, False),
    ("balanced-2048-full-b4", True, 2048, 151665, 4, False),
    ("balanced-2048-full-b8", True, 2048, 151665, 8, False),
    ("balanced-2048-full-b16", True, 2048, 151665, 16, False),
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def prepare(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    with args.records.open() as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["state_source"], row["source"]].append(row)
    selected = []
    for source in ("MATH-500", "AMC23", "AIME25"):
        rows = sorted(
            groups["on_validation_states", source], key=lambda r: (len(r["response_ids"]), r["uid"], r["sample"])
        )
        if len(rows) < args.per_benchmark:
            raise ValueError("Not enough real validation responses for the declared workload")
        selected.extend(rows[round(i * (len(rows) - 1) / (args.per_benchmark - 1))] for i in range(args.per_benchmark))
    for state in ("on_initial_states", "on_teacher_states"):
        rows = [r for (s, _), items in groups.items() if s == state for r in items]
        selected.extend(sorted(rows, key=lambda r: (r["source"], r["uid"], r["sample"]))[:4])
    path = directory / "records.jsonl"
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected))
    baseline = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    source = directory / "baseline"
    source.mkdir()
    for name in ("trainer/ppo/validation_kl.py", "workers/validation_kl.py"):
        target = source / ("core.py" if name.startswith("trainer") else "worker.py")
        target.write_bytes(subprocess.check_output(["git", "show", f"{baseline}:verl/verl/{name}"], cwd=ROOT))
    manifest = {
        "scope": "Frozen real-response workload; benchmark of pretrained p0/q, not discarded student2 weights",
        "baseline_commit": baseline,
        "input_path": str(args.records.resolve()),
        "input_sha256": digest(args.records),
        "selected_sha256": digest(path),
        "responses": len(selected),
        "effective_main_positions": sum(
            len(r["response_ids"]) for r in selected if r["state_source"] == "on_validation_states"
        ),
        "sampling": "Within each benchmark, deterministic response-length quantiles; four bank responses per source",
        "baseline_sources_sha256": {p.name: digest(p) for p in source.iterdir()},
    }
    if hasattr(args, "student") and hasattr(args, "teacher"):
        identity = load_module("validation_prepared_identity", ROOT / "verl/verl/trainer/ppo/fixed_rollout.py")
        manifest["models"] = {
            role: {"path": str(path.resolve()), "fingerprint": identity.model_signature(path)}
            for role, path in (("student", args.student), ("teacher", args.teacher))
        }
    (directory / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_workload(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "input-manifest.json").read_text())
    if digest(directory / "records.jsonl") != manifest["selected_sha256"]:
        raise ValueError("Frozen response workload changed")
    for name, expected in manifest["baseline_sources_sha256"].items():
        if digest(directory / "baseline" / name) != expected:
            raise ValueError("Baseline implementation changed")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if not records or len(records) != manifest["responses"]:
        raise ValueError("Frozen response count changed")
    return manifest, records


def sample_memory(directory, stopped):
    with (directory / "gpu.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["epoch_s", "gpu", "used_mib", "utilization_pct"])
        while not stopped.is_set():
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    "0,1,2,3",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if query.returncode == 0:
                for line in query.stdout.splitlines():
                    writer.writerow([time.time(), *[int(value.strip()) for value in line.split(",")]])
                stream.flush()
            stopped.wait(2)


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3" or int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("Run with torchrun on the reserved physical GPUs 0–3 only")
    directory = args.output.resolve()
    manifest, original = frozen_workload(directory)
    identity = load_module("validation_current_identity", ROOT / "verl/verl/trainer/ppo/fixed_rollout.py")
    for role, path in (("student", args.student), ("teacher", args.teacher)):
        expected = manifest.get("models", {}).get(role)
        if (
            expected is None
            or expected["path"] != str(path.resolve())
            or expected["fingerprint"] != identity.model_signature(path)
        ):
            raise ValueError("Prepared model identity changed: " + role)
    import torch
    import torch.distributed as dist
    from native_devices import matches_device, query_devices
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from verl.trainer.ppo.validation_kl import balanced_record_indices, full_vocab_divergences
    from verl.workers.validation_kl import score_causal_records

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3), device_id=device)
    properties = torch.cuda.get_device_properties(rank)
    observed_device = {
        "rank": rank,
        "uuid": str(properties.uuid),
        "name": properties.name,
        "total_bytes": properties.total_memory,
    }
    observed_devices = [None] * 4 if rank == 0 else None
    dist.gather_object(observed_device, observed_devices, dst=0)
    expected_devices = query_devices() if rank == 0 else None
    if rank == 0 and any(not matches_device(row["uuid"], expected_devices[row["rank"]]) for row in observed_devices):
        raise ValueError("Scoring benchmark ran on an unexpected physical GPU")
    records = list(original)
    while len(records) % 4:
        records.append(original[(len(records) - len(original)) % len(original)])
    baseline_core = load_module("validation_baseline_core", directory / "baseline/core.py")
    baseline_worker = load_module("validation_baseline_worker", directory / "baseline/worker.py")
    baseline_worker.full_vocab_divergences = baseline_core.full_vocab_divergences
    stopped, monitor = threading.Event(), None
    if rank == 0:
        monitor = threading.Thread(target=sample_memory, args=(directory, stopped), daemon=True)
        monitor.start()
    try:
        models = []
        for path in (args.student, args.teacher):
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model.eval()
            model.requires_grad_(False)
            model = FSDP(
                model,
                device_id=device,
                use_orig_params=True,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
                ),
            )
            models.append(model)
        student, teacher = models
        # Same quantized model logits, independently normalized in FP64.
        row = records[rank]
        positions = sorted(
            set((0, len(row["response_ids"]) // 3, 2 * len(row["response_ids"]) // 3, len(row["response_ids"]) - 1))
        )
        ids = torch.tensor([row["prompt_ids"] + row["response_ids"][: positions[-1]]], device=device)
        head_positions = torch.tensor(positions, device=device) + len(row["prompt_ids"]) - 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = [m(input_ids=ids, logits_to_keep=head_positions, use_cache=False).logits[0] for m in models]
        logp, logq = [value[:, :151665].double().log_softmax(-1) for value in (p, q)]
        reference = {
            "student_teacher_kl": (logp.exp() * (logp - logq)).sum(-1),
            "teacher_student_kl": (logq.exp() * (logq - logp)).sum(-1),
            "student_entropy": -(logp.exp() * logp).sum(-1),
            "teacher_entropy": -(logq.exp() * logq).sum(-1),
        }
        errors = {}
        for name, function, chunk in (
            ("baseline", baseline_core.full_vocab_divergences, 8192),
            ("full_block", full_vocab_divergences, 151665),
        ):
            actual = function(p, q, vocab_size=151665, chunk_size=chunk)
            errors[name] = {key: float((actual[key].double() - value).abs().max()) for key, value in reference.items()}
            if max(errors[name].values()) > 1e-5:
                raise ValueError("GPU full-vocabulary reduction exceeded the independent FP64 error bound")
        all_errors = [None] * 4 if rank == 0 else None
        dist.gather_object(errors, all_errors, dst=0)
        del p, q, logp, logq, reference, actual
        results, reference_rows = [], None
        for repetition in range(-1, args.repeats):
            cases = CASES if repetition < 1 else list(reversed(CASES))
            for name, balanced, position, vocabulary, forward_batch, legacy in cases:
                order = balanced_record_indices(records, 4) if balanced else tuple(range(len(records)))
                local_indices = order[rank * (len(records) // 4) : (rank + 1) * (len(records) // 4)]
                local = [records[index] for index in local_indices]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                dist.barrier()
                started = time.time()
                tick = time.perf_counter()
                scored, timings = (baseline_worker.score_causal_records if legacy else score_causal_records)(
                    student,
                    teacher,
                    local,
                    vocab_size=151665,
                    device=device,
                    position_chunk_size=position,
                    vocab_chunk_size=vocabulary,
                    **({} if legacy else {"forward_batch_size": forward_batch}),
                )
                torch.cuda.synchronize()
                elapsed = torch.tensor(time.perf_counter() - tick, device=device)
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                finished = time.time()
                gathered = [None] * 4 if rank == 0 else None
                dist.gather_object(
                    {
                        "indices": local_indices,
                        "rows": scored,
                        "timings": timings,
                        "allocated": allocated,
                        "reserved": reserved,
                    },
                    gathered,
                    dst=0,
                )
                failure, fatal = None, False
                if rank == 0:
                    restored = [None] * len(records)
                    for part in gathered:
                        for index, value in zip(part["indices"], part["rows"], strict=True):
                            restored[index] = value
                    restored = restored[: len(original)]
                    if reference_rows is None:
                        reference_rows = restored
                    differences = []
                    for expected, actual_row in zip(reference_rows, restored, strict=True):
                        if (
                            expected["record_sha256"],
                            expected["response_length"],
                            expected["evaluated_positions"],
                        ) != (
                            actual_row["record_sha256"],
                            actual_row["response_length"],
                            actual_row["evaluated_positions"],
                        ):
                            failure = "Scoring changed response identity, length or effective-position coverage"
                            fatal = True
                        differences.extend(
                            abs(actual_row["sums"][key] - value) / expected["response_length"]
                            for key, value in expected["sums"].items()
                        )
                    if max(differences) > 1e-5:
                        failure = "Paired response-mean statistics differed by more than 1e-5"
                    entry = {
                        "case": name,
                        "repetition": repetition,
                        "warmup": repetition == -1,
                        "started_epoch_s": started,
                        "finished_epoch_s": finished,
                        "seconds": float(elapsed),
                        "max_paired_mean_error": max(differences),
                        "allocated_bytes_by_rank": [item["allocated"] for item in gathered],
                        "reserved_bytes_by_rank": [item["reserved"] for item in gathered],
                        "rank_evaluated_positions": [
                            sum(row["evaluated_positions"] for row in part["rows"]) for part in gathered
                        ],
                        "worker_timings": [part["timings"] for part in gathered],
                        "passed": failure is None,
                        "rejection": failure,
                    }
                    results.append(entry)
                    (directory / "result.json").write_text(
                        json.dumps(
                            {
                                "state": "running",
                                "input": manifest,
                                "scope": "BF16 pretrained pair; FSDP1 without optimizer/CPU offload; no generation",
                                "fp64_errors_by_rank": all_errors,
                                "expected_devices": expected_devices,
                                "observed_devices": observed_devices,
                                "measurements": results,
                                "student_path": str(args.student.resolve()),
                                "teacher_path": str(args.teacher.resolve()),
                                "implementation_sha256": {
                                    str(Path(function.__code__.co_filename)): digest(function.__code__.co_filename)
                                    for function in (
                                        score_causal_records,
                                        full_vocab_divergences,
                                        balanced_record_indices,
                                    )
                                },
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                state = [failure if fatal or legacy else None]
                dist.broadcast_object_list(state, src=0)
                if state[0]:
                    raise ValueError(state[0])
        if rank == 0:
            result = json.loads((directory / "result.json").read_text())
            rejected = sorted({row["case"] for row in results if not row["passed"]})
            result["state"] = "passed" if not rejected else "completed_with_rejected_candidates"
            result["rejected_cases"] = rejected
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        stopped.set()
        if monitor:
            monitor.join(timeout=5)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--student", type=Path, default=ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--teacher", type=Path, default=ROOT.parent / "models/JustRL-DeepSeek-1.5B")
    parser.add_argument("--baseline-commit", default="b528c6d29425e368ff7add630bfc845f060cb8b8")
    parser.add_argument("--per-benchmark", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.per_benchmark < 2 or args.repeats < 1:
        parser.error("Workload quantiles and repeat count must be positive")
    if args.prepare_only:
        if args.records is None:
            parser.error("--records is required for CPU workload preparation")
        print(json.dumps(prepare(args)))
    else:
        run(args)


if __name__ == "__main__":
    main()
