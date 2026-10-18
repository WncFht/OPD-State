"""Serial fixed-checkpoint vLLM stack and batch-invariance inference controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from probe_batch_invariance import digest, full_vocab_distance
from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def snapshot_sources(directory, source_commit=None):
    """Freeze helpers; queued runs also bind the driver to its committed bytes."""
    names = (
        "probe_batch_invariance.py",
        "audit_batch_invariance.py",
        "native_devices.py",
        "run_native_acceptance.py",
    )

    def read(name):
        if source_commit:
            return subprocess.check_output(["git", "show", f"{source_commit}:eval/{name}"], cwd=ROOT)
        return (ROOT / "eval" / name).read_bytes()

    if source_commit and read(Path(__file__).name) != Path(__file__).read_bytes():
        raise ValueError("Queued probe driver differs from its requested source commit")
    directory.mkdir()
    hashes = {}
    for name in names:
        destination = directory / name
        destination.write_bytes(read(name))
        hashes[name] = digest(destination)
    return hashes


def cases():
    rows = []
    for graph in (False, True):
        prefix = "graph" if graph else "eager"
        for version, role, invariant in (
            ("0.10.0+cu126", "p0", False),
            ("0.15.1", "p0", False),
            ("0.15.1", "q", False),
            ("0.15.1", "p0", True),
            ("0.15.1", "q", True),
        ):
            rows.append(
                {
                    "name": f"{prefix}-{version}-{role}-bi{int(invariant)}",
                    "version": version,
                    "role": role,
                    "invariant": invariant,
                    "graph": graph,
                    "weight": None,
                }
            )
        rows += [
            {
                "name": f"{prefix}-0.15.1-dual-lambda{weight}",
                "version": "0.15.1",
                "role": "p0",
                "invariant": True,
                "graph": graph,
                "weight": weight,
            }
            for weight in (0.0, 0.5, 1.0)
        ]
    return rows


def command(case, output, old_python, new_python, prompts):
    paths = {
        "p0": ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B",
        "q": ROOT.parent / "models/JustRL-DeepSeek-1.5B",
    }
    result = [
        str(old_python if case["version"].startswith("0.10.") else new_python),
        str(ROOT / "eval/probe_batch_invariance.py"),
        "--output",
        str(output),
        "--model",
        str(paths[case["role"]]),
        "--prompts",
        str(prompts),
        "--expected-vllm",
        case["version"],
        "--gpu",
        "0",
        "--cap",
        "256",
        "--boundary",
        "128",
        "--repeats",
        "2",
    ]
    if case["invariant"]:
        result += ["--batch-invariant"]
    if case["graph"]:
        result += ["--graph"]
    if case["weight"] is not None:
        result += ["--teacher", str(paths["q"]), "--teacher-weight", str(case["weight"])]
    return result


def fusion_reference(directory, case, summaries):
    import numpy as np

    prefix = "graph" if case["graph"] else "eager"
    a = summaries[f"{prefix}-0.15.1-p0-bi1"]
    b = summaries[f"{prefix}-0.15.1-q-bi1"]
    c = summaries[case["name"]]
    if (
        not a["probe_prefixes"] == b["probe_prefixes"] == c["probe_prefixes"]
        or not a["tokenizer_vocab_size"] == b["tokenizer_vocab_size"] == c["tokenizer_vocab_size"]
    ):
        raise ValueError("Fusion and single-model probes used different fixed states/vocabularies")
    if (
        a["tokenizer_vocab_sha256"] != b["tokenizer_vocab_sha256"]
        or a["tokenizer_vocab_sha256"] != c["tokenizer_vocab_sha256"]
    ):
        raise ValueError("Fusion references changed token ID meanings")
    merged = {**a["checkpoint_files_sha256"], **b["checkpoint_files_sha256"]}
    if merged != c["checkpoint_files_sha256"]:
        raise ValueError("Fusion references changed their frozen checkpoint files")
    paths = [
        directory / name / "logits-single.npy"
        for name in (f"{prefix}-0.15.1-p0-bi1", f"{prefix}-0.15.1-q-bi1", case["name"])
    ]
    p, q, actual = [np.load(path, allow_pickle=False) for path in paths]
    expected = (1 - case["weight"]) * p + case["weight"] * q
    metrics = full_vocab_distance(expected, actual)
    return {
        "scope": "Independent same-stack native single-model FP32 logit combination, all valid vocabulary rows",
        "teacher_weight": case["weight"],
        "inputs_sha256": {str(path): digest(path) for path in paths},
        "metrics": metrics,
        "max_logit_error_tolerance": 0.5,
        "max_tv_tolerance": 0.05,
        "passed": metrics["max_logit_error"] <= 0.5 and metrics["max_tv"] <= 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-python", type=Path, default=ROOT.parent / "envs/opd-state/bin/python")
    parser.add_argument(
        "--new-python", type=Path, default=Path("/root/.cache/opd-state/envs/inference-vllm015/bin/python")
    )
    parser.add_argument(
        "--prompts", type=Path, default=ROOT / "runs/native-rollout-implementation/fresh-math-data-v2/prompts.jsonl"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--source-commit", help="Read probe helpers from this commit and verify driver bytes")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    code_hashes = snapshot_sources(directory / "code", args.source_commit)
    state = {
        "state": "planned" if args.plan_only else "running",
        "scope": "Native fixed-model inference only; stack dependencies differ; no actor/training acceptance",
        "driver_sha256": digest(Path(__file__)),
        "source_commit": args.source_commit,
        "prompts_sha256": digest(args.prompts),
        "cases": [],
        "frozen_code_sha256": code_hashes,
    }

    def save():
        temp = directory / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(directory / "state.json")

    save()
    summaries, failed_properties = {}, []
    for case in cases():
        output = directory / case["name"]
        cmd = command(case, output, args.old_python, args.new_python, args.prompts)
        cmd[1] = str(directory / "code/probe_batch_invariance.py")
        item = {**case, "state": "planned", "command": cmd}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("Wait until every physical GPU 0–3 is idle")
        with (directory / f"{case['name']}.log").open("w") as stream:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env={**os.environ, "OPD_VERSION_PROBE_ID": case["name"] + "-" + directory.name},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            item.update(state="running", pid=process.pid, started_at=time.time())
            save()
            code = process.wait()
        item.update(exit_code=code, finished_at=time.time())
        if code not in (0, 2):
            item["state"], state["state"] = "execution_failed", "failed"
            save()
            return code
        audit_path = output / "independent-audit.json"
        subprocess.run(
            [
                sys.executable,
                str(directory / "code/audit_batch_invariance.py"),
                "--run",
                str(output),
                "--output",
                str(audit_path),
            ],
            cwd=ROOT,
            check=True,
        )
        receipt = json.loads(audit_path.read_text())
        summaries[case["name"]] = json.loads((output / "summary.json").read_text())
        if code == 2 and receipt["batch_order_replays_equal"]:
            raise ValueError("Native probe failure code does not agree with its retained observations")
        item["state"] = "measured" if code == 0 else "batch_property_failed"
        if code == 2:
            failed_properties.append(case["name"])
        if case["weight"] is not None:
            reference = fusion_reference(directory, case, summaries)
            (output / "fusion-reference-audit.json").write_text(json.dumps(reference, indent=2) + "\n")
            item["fusion_reference_passed"] = reference["passed"]
            if not reference["passed"]:
                failed_properties.append(case["name"] + ":fusion_reference")
        save()
    state["failed_properties"] = failed_properties
    state["state"] = (
        "planned" if args.plan_only else "completed_with_failed_properties" if failed_properties else "finished"
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
