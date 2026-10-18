"""Native same-model request-boundary checks followed by matched quality pairs.

Six short role/APC checks precede four complete eval16 runs: single versus two
requests at each APC setting. This does not modify the frozen 15-policy suite.
One seed per pair cannot establish equivalence or asymptotic convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from native_devices import query_devices, validate_visibility
from native_validation_subset import paths as subset_paths
from native_validation_subset import verify as verify_subset
from run_native_acceptance import gpu_rows
from run_native_followups import command as native_command
from run_validation_kl_gate import input_bindings, verify_bank

ROOT = Path(__file__).resolve().parents[1]


def cases():
    common = {
        "layout": "overlap",
        "verify_frozen": True,
        "steps": 2,
        "batch": 8,
        "response_cap": 64,
        "val_cap": 128,
        "formal": False,
    }
    result = [
        {**common, "name": f"short-{role}-apc{int(apc)}", "policy": f"{role}_request_split", "boundary": 32, "apc": apc}
        for apc in (False, True)
        for role in ("current", "initial", "teacher")
    ]
    for apc in (False, True):
        for policy in (
            ("student_fresh", "current_request_split") if not apc else ("current_request_split", "student_fresh")
        ):
            result.append(
                {
                    **common,
                    "name": f"formal-{policy}-apc{int(apc)}",
                    "policy": policy,
                    "boundary": 2048,
                    "apc": apc,
                    "steps": 50,
                    "batch": 64,
                    "response_cap": 4096,
                    "val_cap": 31744,
                    "formal": True,
                    "verify_frozen": False,
                }
            )
    return result


def command(case, output, source, validation, bank):
    result = native_command(case, output, source, validation)
    result += [
        f"actor_rollout_ref.rollout.enable_prefix_caching={case['apc']}",
        f"trainer.validation_kl.enabled={case['formal']}",
    ]
    if case["formal"]:
        result += [
            "--full-eval",
            "trainer.validation_kl.position_chunk_size=128",
            "trainer.validation_kl.vocab_chunk_size=8192",
            f"trainer.validation_kl.fixed_prefix_bank={bank}",
        ]
    return result


def audit_boundary(run, steps, *, require_split):
    manifests = list(Path(run).glob("checkpoints/**/native-trajectories/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Request-boundary control needs one native generation manifest")
    manifest = json.loads(manifests[0].read_text())
    policy = manifest["policy"]
    if require_split and (policy.get("keep_request_boundary") is not True or policy["first"] != policy["second"]):
        raise ValueError("Request-boundary control was compiled without the explicit same-model option")
    split = total = 0
    for step in range(1, steps + 1):
        for line in (manifests[0].parent / f"step-{step}.jsonl").read_text().splitlines():
            row = json.loads(line)
            segments = row["segments"]
            total += 1
            if require_split:
                boundary = policy["boundary"]["value"]
                if len(row["token_ids"]) > boundary:
                    if (
                        len(segments) != 2
                        or segments[0]["end"] != boundary
                        or segments[1]["start"] != boundary
                        or segments[0]["identities"] != segments[1]["identities"]
                    ):
                        raise ValueError("Same-model boundary changed the source/version or was optimized away")
                    split += 1
                elif len(segments) != 1:
                    raise ValueError("Stopped same-model response continued after its boundary")
            elif len(segments) != 1:
                raise ValueError("Single-request reference unexpectedly used multiple requests")
    if require_split and split == 0:
        raise ValueError("No response exercised the requested same-model boundary")
    return {
        "scope": "Actual native training trajectories; supplements the full budget/EOS/identity acceptance",
        "responses": total,
        "two_request_responses": split,
        "model_switches": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--bank-input", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    if not args.plan_only:
        code = subprocess.check_output(
            ["git", "show", source + ":verl/verl/workers/rollout/policy.py"], cwd=ROOT, text=True
        )
        if "keep_request_boundary" not in code:
            raise ValueError("Selected source predates the same-model request-boundary control")
    prepared, full_files = input_bindings(args.bank_input)
    if not args.plan_only:
        verify_bank(args.bank, prepared, args.bank_input)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    subset = directory / "validation-subset"
    runtime_python = ROOT.parent / "envs/opd-state/bin/python"
    preparation = [
        str(runtime_python),
        str(ROOT / "data/prep/build_validation_subset.py"),
        "--output",
        str(subset),
        "--per-benchmark",
        "2",
    ]
    for filename in full_files.values():
        preparation += ["--input", filename]
    state = {
        "state": "planned" if args.plan_only else "running",
        "source_commit": source,
        "scope": "Same-model execution control; flat policies; no changes to the frozen formal suite",
        "validation_preparation": preparation,
        "cases": [],
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        "bank_input_sha256": hashlib.sha256(args.bank_input.read_bytes()).hexdigest(),
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest() if args.bank.is_file() else None,
    }

    def save():
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(directory / "state.json")

    def checked(cmd, *, environment=None):
        try:
            subprocess.run(cmd, cwd=ROOT, env=environment, check=True)
        except BaseException as error:
            state.update(state="audit_failed", error=type(error).__name__ + ": " + str(error))
            save()
            raise

    save()
    if not args.plan_only:
        checked(preparation, environment={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    for case in cases():
        output = directory / case["name"]
        validation = list(full_files.values()) if case["formal"] else subset_paths(subset)
        cmd = command(case, output, source, validation, args.bank.resolve())
        item = {**case, "state": "planned", "command": cmd}
        state["cases"].append(item)
        save()
        if args.plan_only:
            continue
        if not case["formal"]:
            verify_subset(subset / "manifest.json")
        if any(used > 128 for _, used, _ in gpu_rows()):
            state["state"] = "resource_busy"
            save()
            raise RuntimeError("Wait for the serial physical GPU 0–3 slot")
        devices = query_devices()
        validate_visibility(devices, "0,1,2,3")
        identity = directory / f"{case['name']}-device-identities.json"
        identity.write_text(json.dumps(devices, indent=2) + "\n")
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NATIVE_DEVICE_IDENTITIES_SHA256": hashlib.sha256(identity.read_bytes()).hexdigest(),
            "NATIVE_DATA_MANIFEST": str(args.data_manifest.resolve()),
            "SWITCH_TOKENS": str(case["boundary"]),
        }
        with (directory / f"{case['name']}-driver.log").open("w") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            item.update(state="running", pid=process.pid, started_at=time.time())
            save()
            code = process.wait()
        item.update(state="finished" if code == 0 else "failed", exit_code=code, finished_at=time.time())
        save()
        if code:
            state["state"] = "failed"
            save()
            return code
        result = ROOT / "docs/experiment/results/native-rollout/request-boundary" / directory.name / case["name"]
        audits = [
            [
                sys.executable,
                str(ROOT / "eval/collect_native_acceptance.py"),
                "--run",
                str(output),
                "--output",
                str(result),
                "--steps",
                str(case["steps"]),
                *(["--full"] if case["formal"] else []),
            ],
            [
                sys.executable,
                str(ROOT / "eval/verify_native_layout.py"),
                "--run",
                str(output),
                "--output",
                str(result),
                "--steps",
                str(case["steps"]),
                "--device-identities",
                str(identity),
            ],
        ]
        if case["formal"]:
            audits.append(
                [
                    sys.executable,
                    str(ROOT / "eval/collect_validation_kl.py"),
                    "--run",
                    str(output),
                    "--output",
                    str(result),
                    "--bank",
                    str(args.bank.resolve()),
                    "--full",
                ]
            )
        for audit in audits:
            code = subprocess.call(audit, cwd=ROOT)
            item.setdefault("audit_exit_codes", []).append(code)
            save()
            if code:
                state["state"] = "audit_failed"
                save()
                return code
            if audit is audits[0]:
                checked(
                    [str(ROOT / "node_modules/.bin/prettier"), "--write", *map(str, result.glob("*.json"))],
                )
        try:
            boundary = audit_boundary(output, case["steps"], require_split=case["policy"] != "student_fresh")
        except ValueError as error:
            state.update(state="audit_failed", error=str(error))
            save()
            raise
        (result / "request-boundary-acceptance.json").write_text(json.dumps(boundary, indent=2) + "\n")
        checked(
            [str(ROOT / "node_modules/.bin/prettier"), "--write", *map(str, result.glob("*.json"))],
        )
        auth = {
            **env,
            "CUDA_VISIBLE_DEVICES": "",
            "NETRC": str(ROOT.parent / ".secrets/netrc"),
            "WANDB_CONFIG_DIR": str(ROOT.parent / "config/wandb"),
            "WANDB_BASE_URL": "https://api.wandb.ai",
        }
        auth.pop("WANDB_API_KEY", None)
        verify = [
            str(runtime_python),
            str(ROOT / "eval/verify_wandb_math.py"),
            "--result",
            str(result),
            "--expected-prefix-caching",
            str(case["apc"]).lower(),
        ]
        verify += (
            ["--kl-report", str(output / "validation/kl/kl-step-50.json")] if case["formal"] else ["--engineering"]
        )
        checked(verify, environment=auth)
        remote = json.loads((result / "wandb-verification.json").read_text())
        if remote["settings"]["enable_prefix_caching"] is not case["apc"]:
            state["state"] = "audit_failed"
            save()
            raise ValueError("Remote APC setting disagrees with the requested matched control")
        item["request_boundary_acceptance"] = boundary
        save()
    state["state"] = "planned" if args.plan_only else "finished"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
