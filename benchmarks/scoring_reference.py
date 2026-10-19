"""Compare frozen and extracted scorers on real validation tokens and p0/q.

This is CPU numerical evidence for the same frozen workload, not a performance
claim for the discarded trained student or for four-GPU FSDP validation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path


def load(name, path, *, bindings=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if bindings:
        tree = ast.parse(Path(path).read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                node.names = [
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name) not in bindings
                ]
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, ast.ImportFrom) or node.names
        ]
        module.__dict__.update(bindings)
        exec(compile(tree, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, output = args.prepared_run.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "driver.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VERL_USE_EXTERNAL_PLUGINS"] = "none"
    sys.path[:0] = [str(root / "src"), str(root / "verl"), str(root)]
    import torch
    from transformers import AutoModelForCausalLM

    from opd_state.artifacts import atomic_json, file_digest
    from opd_state.kernels.causal_scoring import score_causal_records
    from recipes.inputs import verify_inputs
    from recipes.snapshots import verify_frozen_files

    torch.set_num_threads(args.threads)
    manifest = verify_frozen_files(run)
    inputs = json.loads((run / "inputs.json").read_text())
    verify_inputs(run, inputs)
    path = run / f"validation/kl/kl-tokens-step-{args.step}.jsonl"
    selected = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["state_source"] == "on_validation_states":
            selected.setdefault(record["source"], record)
    if set(selected) != {"MATH-500", "AMC23", "AIME25"}:
        raise ValueError("Reference workload must cover each math benchmark")
    records = [selected[source] for source in sorted(selected)]
    atomic_json(output / "records.json", records)
    for source, target in (
        ("trainer/ppo/validation_kl.py", "baseline_core.py"),
        ("workers/validation_kl.py", "baseline_worker.py"),
    ):
        shutil.copyfile(run / "source/backend/verl/verl" / source, output / target)
    baseline_core = load("_reference_kl_core", output / "baseline_core.py")
    baseline_worker = load(
        "_reference_kl_worker",
        output / "baseline_worker.py",
        bindings={"full_vocab_divergences": baseline_core.full_vocab_divergences},
    )
    if baseline_worker.score_causal_records.__module__ != baseline_worker.__name__:
        raise ValueError(
            "Select a pre-extraction frozen scorer as the independent baseline"
        )
    baseline_worker.full_vocab_divergences = baseline_core.full_vocab_divergences
    candidate_sources = {
        name: file_digest(root / name)
        for name in (
            "src/opd_state/kernels/causal_scoring.py",
            "src/opd_state/kernels/divergence.py",
            "src/opd_state/core/records.py",
        )
    }
    for name in candidate_sources:
        target = output / "candidate" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    models = []
    for role in ("student", "teacher"):
        model = (
            AutoModelForCausalLM.from_pretrained(
                inputs["models"][role]["path"],
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        models.append(model)
        print(f"Loaded actual {role} checkpoint on CPU", flush=True)
    versions = [[p._version for p in model.parameters()] for model in models]
    results = []
    for position_chunk, vocab_chunk in ((64, 8192), (1024, inputs["vocab_size"])):
        arguments = dict(
            vocab_size=inputs["vocab_size"],
            device="cpu",
            dtype=torch.bfloat16,
            position_chunk_size=position_chunk,
            vocab_chunk_size=vocab_chunk,
            forward_batch_size=1,
        )
        tick = time.perf_counter()
        reference, old_timing = baseline_worker.score_causal_records(
            *models, records, **arguments
        )
        candidate, new_timing = score_causal_records(*models, records, **arguments)
        if candidate != reference:
            raise ValueError(
                "Extracted real-model KL rows differ from the frozen implementation"
            )
        results.append(
            {
                "position_chunk": position_chunk,
                "vocab_chunk": vocab_chunk,
                "responses": len(candidate),
                "evaluated_positions": sum(
                    row["evaluated_positions"] for row in candidate
                ),
                "rows": candidate,
                "exact": True,
                "reference_timing": old_timing,
                "candidate_timing": new_timing,
                "seconds": time.perf_counter() - tick,
            }
        )
        print(
            f"Exact real-token KL parity: position={position_chunk}, vocabulary={vocab_chunk}",
            flush=True,
        )
    if versions != [[p._version for p in model.parameters()] for model in models]:
        raise ValueError("Read-only scorer modified model parameter versions")
    if any(p.grad is not None for model in models for p in model.parameters()):
        raise ValueError("Read-only scoring produced gradients")
    if torch.cuda.is_initialized():
        raise ValueError("CPU reference initialized CUDA")
    if candidate_sources != {
        name: file_digest(root / name) for name in candidate_sources
    }:
        raise ValueError("Candidate source changed during numerical comparison")
    atomic_json(
        output / "result.json",
        {
            "scope": "CPU real p0/q checkpoints on frozen student-validation prefixes; not trained-student accuracy or GPU/FSDP performance",
            "passed": True,
            "source_run_id": manifest["run_id"],
            "source_id": manifest["source_id"],
            "input_tokens_sha256": file_digest(path),
            "records_sha256": file_digest(output / "records.json"),
            "models": {
                role: value["identity"] for role, value in inputs["models"].items()
            },
            "candidate_sources": candidate_sources,
            "baseline_sources": {
                name: file_digest(output / name)
                for name in ("baseline_core.py", "baseline_worker.py")
            },
            "results": results,
            "cuda_initialized": False,
            "driver_sha256": hashlib.sha256(
                (output / "driver.py").read_bytes()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
