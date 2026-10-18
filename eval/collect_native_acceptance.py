"""Audit native source routes/versions and independently rescore final metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from collect_math_experiment import collect

ROOT = Path(__file__).resolve().parents[1]


def load_policy():
    spec = importlib.util.spec_from_file_location("acceptance_policy", ROOT / "verl/verl/workers/rollout/policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit(run, output, steps, *, require_full=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    status = json.loads((run / "status.json").read_text())
    if status.get("exit_code") != 0 or status["state"] != "finished":
        raise ValueError("Native training process did not finish successfully")
    locations = list(run.glob("checkpoints/**/native-trajectories/manifest.json"))
    if len(locations) != 1:
        raise ValueError("Expected one native trajectory manifest")
    directory = locations[0].parent
    manifest = json.loads(locations[0].read_text())
    env = json.loads((run / "command.json").read_text())["environment"]
    cap = int(manifest.get("response_cap", env["MAX_RESP_LENGTH"]))
    seed = int(manifest.get("seed", 42))
    policy = load_policy()
    reports = []
    for step in range(1, steps + 1):
        path = directory / f"step-{step}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        plan = policy.compile_policy(manifest["policy"], cap=cap, batch_size=len(rows), seed=seed, batch_id=step)
        if len(rows) != int(env["MINI_BATCH_SIZE"]):
            raise ValueError("Native training response count differs from the global batch")
        keys = set()
        for row, entry in zip(rows, plan.requests, strict=True):
            request = row["request"]
            key = (request["prompt_id"], request["visit"], request["sample"])
            if key in keys or row["route"] != entry.route:
                raise ValueError("Native trajectories duplicate requests or disagree with global routing")
            keys.add(key)
            end = 0
            if not row["segments"] or len(row["segments"]) > len(entry.stages):
                raise ValueError("Native trajectory has an invalid number of stages")
            for segment, stage in zip(row["segments"], entry.stages, strict=False):
                identities = segment["identities"]
                if [i[0] for i in identities] != list(stage.models) or segment[
                    "teacher_weight"
                ] != stage.teacher_weight:
                    raise ValueError("Native segment used the wrong model or fusion coefficient")
                if segment["start"] != end or not 0 < segment["end"] - end <= stage.max_tokens:
                    raise ValueError("Native segment boundaries exceed the shared budget or contain a gap")
                for role, path_name, version in identities:
                    if role == "current_student":
                        if version != step - 1:
                            raise ValueError("Training generated from a stale/current future actor version")
                    else:
                        descriptor = manifest["generation_models"][role]
                        if path_name != descriptor["path"] or version != descriptor["fingerprint"]:
                            raise ValueError("Frozen generation identity changed")
                end = segment["end"]
            if end != len(row["token_ids"]) or end > cap:
                raise ValueError("Native stages do not cover exactly the valid training response")
        reports.append(
            {
                "step": step,
                "responses": len(rows),
                "response_tokens": sum(len(r["token_ids"]) for r in rows),
                "generation_identities": sorted(
                    {tuple(i) for r in rows for s in r["segments"] for i in s["identities"]}
                ),
                "trace_sha256": hashlib.sha256((directory / f"step-{step}.jsonl").read_bytes()).hexdigest(),
            }
        )
    batches = [json.loads(line) for line in (directory / "batches.jsonl").read_text().splitlines()]
    evaluations = [b for b in batches if b["validate"]]
    if not evaluations or any(b["step"] != steps for b in evaluations):
        raise ValueError("Native acceptance requires final-only evaluation")
    if any(any(i[0] != "current_student" or i[2] != steps for i in b["identities"]) for b in evaluations):
        raise ValueError("Evaluation did not use the final current student exclusively")
    summary = collect(run, output, steps=steps, interval=steps, require_full=require_full)
    with (run / "gpu.csv").open() as stream:
        telemetry = list(csv.DictReader(stream))
    peaks = {str(i): max(int(r["used_mib"]) for r in telemetry if int(r["gpu"]) == i) for i in range(4)}
    result = {
        "scope": "Real four-GPU trainer acceptance; short caps do not establish math accuracy",
        "training": reports,
        "evaluation_responses": sum(b["metrics"]["source/response_count"] for b in evaluations),
        "final_student_version": steps,
        "evaluation_model_identities": sorted({tuple(i) for b in evaluations for i in b["identities"]}),
        "whole_card_peak_mib": peaks,
        "status": status,
        "native_manifest_sha256": hashlib.sha256(locations[0].read_bytes()).hexdigest(),
        "math_summary": summary,
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "native-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = audit(args.run, args.output, args.steps, require_full=args.full)
    print(
        json.dumps(
            {
                "steps": args.steps,
                "evaluation_responses": result["evaluation_responses"],
                "whole_card_peak_mib": result["whole_card_peak_mib"],
            }
        )
    )


if __name__ == "__main__":
    main()
