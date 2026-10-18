"""Profile fixed-work native vLLM 0.10 iterations after an unprofiled control.

Generation ignores EOS only to hold output work constant. These are inference
diagnostics, not valid math trajectories, training results or throughput claims.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from native_devices import matches_device, query_devices
from native_profile_metrics import trace_metrics
from probe_batch_invariance import digest, worker_info
from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def install_markers(worker):
    import torch
    from native_profile_metrics import scheduled_work

    runner = worker.model_runner
    if hasattr(runner, "_opd_profile_original"):
        raise ValueError("Profiling wrapper is already installed")
    original = runner.execute_model
    runner._opd_profile_original, runner._opd_profile_steps = original, []

    def measured(output, *args, **kwargs):
        row = scheduled_work(output, runner.requests)
        number = len(runner._opd_profile_steps)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.profiler.record_function(f"OPD_STEP_{number}_{row['phase']}"):
            result = original(output, *args, **kwargs)
        end.record()
        runner._opd_profile_steps.append((row, begin, end))
        return result

    runner.execute_model = measured
    return {"installed": True, "runner_class": type(runner).__name__}


def remove_markers(worker):
    import torch

    runner = worker.model_runner
    torch.cuda.synchronize()
    result = [
        {**row, "cuda_stream_envelope_ms": begin.elapsed_time(end)} for row, begin, end in runner._opd_profile_steps
    ]
    runner.execute_model = runner._opd_profile_original
    del runner._opd_profile_original, runner._opd_profile_steps
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--role", choices=("initial_student", "teacher", "fusion"), required=True)
    parser.add_argument("--batch", type=int, choices=(4, 16), default=16)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--token-budget", type=int, choices=(64, 8192), default=8192)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.cap <= 128 or metadata.version("vllm") != "0.10.0+cu126":
        parser.error("This probe requires the pinned vLLM 0.10 stack and a short fixed-work cap")
    if any(used > 128 for _, used, _ in gpu_rows()):
        raise RuntimeError("Wait until every physical GPU 0–3 is idle")
    commit = subprocess.check_output(
        ["git", "rev-parse", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    frozen_helpers = {}
    for name in (
        "profile_native_kernels.py",
        "native_profile_metrics.py",
        "probe_batch_invariance.py",
        "native_devices.py",
        "run_native_acceptance.py",
    ):
        original = subprocess.check_output(["git", "show", f"{commit}:eval/{name}"], cwd=ROOT)
        if original != Path(__file__).with_name(name).read_bytes():
            raise ValueError("Profile driver/helper differs from its committed source: " + name)
        frozen_helpers[name] = original
    devices = query_devices()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = output / "code"
    code.mkdir()
    for name, raw in frozen_helpers.items():
        (code / name).write_bytes(raw)
    source = output / "source"
    source.mkdir()
    archive = subprocess.Popen(["git", "archive", commit, "verl"], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(source)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait():
        raise RuntimeError("Could not freeze the native training source")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(devices[0]["cuda_ordinal"]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USE_V1="1",
        VLLM_ATTENTION_BACKEND="FLASH_ATTN",
        VLLM_NO_USAGE_STATS="1",
        VLLM_TORCH_PROFILER_DIR=str(output / "traces"),
        VLLM_BATCH_INVARIANT="0",
        OMP_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_CACHE_ROOT=str(output / "compiler-cache/vllm"),
        TRITON_CACHE_DIR=str(output / "compiler-cache/triton"),
        TORCHINDUCTOR_CACHE_DIR=str(output / "compiler-cache/inductor"),
        PYTHONPATH=os.pathsep.join([str(source / "verl"), str(code), os.environ.get("PYTHONPATH", "")]),
    )
    sys.path.insert(0, str(source / "verl"))
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.trainer.ppo.fixed_rollout import model_signature
    from verl.workers.rollout.fusion import register_native_fusion

    paths = [ROOT.parent / "models" / name for name in ("DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B")]
    fingerprints = [model_signature(str(path)) for path in paths]
    tokenizer = AutoTokenizer.from_pretrained(paths[0], local_files_only=True)
    records = [json.loads(line) for line in args.prompts.read_text().splitlines()[: args.batch]]
    if len(records) != args.batch or len({row["prompt_id"] for row in records}) != args.batch:
        raise ValueError("Profile inputs must contain the complete distinct prompt batch")
    prompts = [{"prompt_token_ids": row["prompt_ids"]} for row in records]
    if any(not row["prompt_ids"] or len(row["prompt_ids"]) + args.cap > 5120 for row in records):
        raise ValueError("Profile prompt violates the common context cap")
    overrides = {}
    if args.role == "fusion":
        register_native_fusion()
        overrides = {
            "architectures": ["OPDNativeQwen2Fusion"],
            "opd_teacher_path": str(paths[1]),
            "opd_teacher_fingerprint": fingerprints[1],
            "opd_teacher_weight": 0.5,
        }
    command = {
        "arguments": {**vars(args), "output": str(output), "prompts": str(args.prompts)},
        "source_commit": commit,
        "devices": devices,
        "model_fingerprints": fingerprints,
        "prompts_file_sha256": digest(args.prompts),
        "driver_sha256": digest(Path(__file__)),
        "metrics_source_sha256": digest(Path(__file__).with_name("native_profile_metrics.py")),
        "frozen_helpers_sha256": {name: digest(code / name) for name in frozen_helpers},
        "ignore_eos": True,
        "vocab_size": len(tokenizer),
        "scope": __doc__,
    }
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")
    llm, installed, active = None, False, False
    try:
        tick = time.perf_counter()
        llm = LLM(
            model=str(paths[1] if args.role == "teacher" else paths[0]),
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=5120,
            max_num_seqs=16,
            max_num_batched_tokens=args.token_budget,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.4,
            enforce_eager=not args.graph,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16], "custom_ops": ["all"]}
            if args.graph
            else {},
            hf_overrides=overrides,
        )
        cold_seconds = time.perf_counter() - tick
        info = llm.collective_rpc(worker_info)[0]
        (output / "worker.json").write_text(json.dumps(info, indent=2) + "\n")
        if not matches_device(info["device_uuid"], devices[0]) or args.graph and not info["cuda_graph_enabled"]:
            raise RuntimeError("Native profile worker has the wrong physical GPU or graph state")
        layers = AutoConfig.from_pretrained(paths[0], local_files_only=True).num_hidden_layers
        regions = sorted(info["kv_regions"], key=lambda row: row["pointer"])
        if len(regions) != layers * (2 if args.role == "fusion" else 1) or any(
            a["pointer"] + a["bytes"] > b["pointer"] for a, b in zip(regions, regions[1:], strict=False)
        ):
            raise RuntimeError("Native profile model lacks complete independent KV regions")
        params = SamplingParams(
            temperature=0, max_tokens=args.cap, ignore_eos=True, allowed_token_ids=list(range(len(tokenizer)))
        )

        def generate():
            tick = time.perf_counter()
            values = llm.generate(prompts, params, use_tqdm=False)
            seconds = time.perf_counter() - tick
            if len(values) != args.batch or any(
                list(value.prompt_token_ids) != prompt["prompt_token_ids"]
                or len(value.outputs[0].token_ids) != args.cap
                for value, prompt in zip(values, prompts, strict=True)
            ):
                raise ValueError("Profile generation changed original prefixes or fixed work")
            return [list(value.outputs[0].token_ids) for value in values], seconds

        generate()
        controls = [generate(), generate()]
        llm.collective_rpc(install_markers)
        installed = True
        llm.start_profile()
        active = True
        profiled, profiled_seconds = generate()
        llm.stop_profile()
        active = False
        steps = llm.collective_rpc(remove_markers)[0]
        installed = False
        (output / "scheduler-steps.json").write_text(json.dumps(steps, indent=2) + "\n")
        (output / "responses.json").write_text(
            json.dumps({"controls": controls, "profiled": profiled}, indent=2) + "\n"
        )
        traces = list((output / "traces").glob("*.pt.trace.json.gz"))
        if len(traces) != 1:
            raise ValueError("Single-worker profile must export exactly one native trace")
        with gzip.open(traces[0], "rt") as stream:
            metrics = trace_metrics(json.load(stream), steps)
        (output / "kernel-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if set(metrics["overall"]["kernel_busy_union_us_by_device"]) != {"0"}:
            raise ValueError("Single-visible-device native trace has unexpected GPU kernel identities")
        expected_prefill = sum(len(row["prompt_ids"]) for row in records)
        expected_decode = args.batch * (args.cap - 1)
        actual_prefill = sum(row["prefill_queries"] for row in steps)
        actual_decode = sum(row["decode_queries"] for row in steps)
        if actual_prefill < expected_prefill or actual_decode < expected_decode:
            raise ValueError("Native trace omitted scheduled input queries")
        equal = all(values == profiled for values, _ in controls)
        passed = (
            equal
            and metrics["complete_attribution"]
            and all(metrics["by_phase"][p]["kernels"] > 0 for p in ("prefill", "decode"))
        )
        if args.token_budget == 64:
            passed = passed and metrics["by_phase"]["mixed"]["kernels"] > 0
        summary = {
            "scope": __doc__,
            "passed": passed,
            "output_tokens_unchanged": equal,
            "vllm": metadata.version("vllm"),
            "torch": torch.__version__,
            "cold_seconds": cold_seconds,
            "unprofiled_call_seconds": [seconds for _, seconds in controls],
            "profiled_call_seconds": profiled_seconds,
            "minimum_logical_input_queries": {"prefill": expected_prefill, "decode": expected_decode},
            "observed_scheduled_queries": {"prefill": actual_prefill, "decode": actual_decode},
            "trace_sha256": digest(traces[0]),
            "trace_file": str(traces[0]),
            "artifact_sha256": {
                name: digest(output / name)
                for name in (
                    "command.json",
                    "worker.json",
                    "inputs.json",
                    "scheduler-steps.json",
                    "responses.json",
                    "kernel-metrics.json",
                )
            },
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"passed": passed, "steps": len(steps), "kernel_events": metrics["overall"]["kernels"]}))
        return 0 if passed else 2
    finally:
        if llm is not None:
            try:
                if active:
                    llm.stop_profile()
            finally:
                try:
                    if installed:
                        llm.collective_rpc(remove_markers)
                finally:
                    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
