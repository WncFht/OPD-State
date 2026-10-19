"""Run a full 570-question eval16 KL scaling gate after two engineering updates.

This is an engineering coverage/resource gate, not a 50-step quality comparison.
It reuses the prepared bank's full input-file bindings; it does not evaluate on
only the bank's selected 90 questions. GPU generation remains native vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_validation_kl import BENCHMARKS, digest
from run_native_acceptance import gpu_rows

ROOT = Path(__file__).resolve().parents[1]


def input_bindings(path):
    prepared = json.loads(Path(path).read_text())
    if prepared.get("schema") != "fixed-prefix-input-v1":
        raise ValueError("Full KL gate requires the original prepared bank input manifest")
    files = {}
    for record in prepared["inputs"]:
        filename = Path(record["path"])
        benchmark = filename.parent.name
        if benchmark not in BENCHMARKS or benchmark in files or record["rows"] != BENCHMARKS[benchmark]:
            raise ValueError("Full KL gate requires the complete 500/40/30 input files, not a selected subset")
        if digest(filename) != record["sha256"]:
            raise ValueError("Full validation input bytes changed since bank preparation")
        files[benchmark] = str(filename.resolve())
    if set(files) != set(BENCHMARKS):
        raise ValueError("Full KL gate must bind all three complete benchmarks")
    return prepared, files


def verify_bank(bank_path, prepared, prepared_path):
    bank = json.loads(Path(bank_path).read_text())
    expected = {
        "schema": "fixed-prefix-bank-v1",
        "input_sha256": digest(prepared_path),
        "vocab_size": prepared["vocab_size"],
        "tokenizer_signature": prepared["tokenizer_signature"],
        "initial_student_fingerprint": prepared["models"]["initial_student"]["fingerprint"],
        "teacher_fingerprint": prepared["models"]["teacher"]["fingerprint"],
    }
    if any(bank.get(key) != value for key, value in expected.items()) or len(bank["records"]) != 180:
        raise ValueError("Full KL gate requires the complete original 90-question, dual-source bank")
    expected_questions = {(row["source"], row["uid"]): row["prompt_ids"] for row in prepared["questions"]}
    keys = {(row["state_source"], row["source"], row["uid"]) for row in bank["records"]}
    expected_keys = {
        (state, source, uid)
        for state in ("on_initial_states", "on_teacher_states")
        for source, uid in expected_questions
    }
    if len(expected_questions) != 90 or len(keys) != 180 or keys != expected_keys:
        raise ValueError("Full KL bank dropped or duplicated prepared source/question identities")
    if any(
        row["sample"] != 0 or row["prompt_ids"] != expected_questions[row["source"], row["uid"]]
        for row in bank["records"]
    ):
        raise ValueError("Full KL bank changed the prepared raw prompt or response sample identity")
    return digest(bank_path)


def commands(run, result, bank, files, commit, runtime_python):
    train = [
        sys.executable,
        str(ROOT / "eval/run_native_acceptance.py"),
        "--source-commit",
        commit,
        "--output",
        str(run),
        "--policy",
        "student_fresh",
        "--steps",
        "2",
        "--batch",
        "8",
        "--prompt-cap",
        "1024",
        "--response-cap",
        "64",
        "--val-response-cap",
        "31744",
        "--memory-budget",
        "0.55",
        "--full-eval",
        "--wandb",
    ]
    for benchmark in BENCHMARKS:
        train.extend(["--validation", files[benchmark]])
    train.extend(
        [
            "actor_rollout_ref.rollout.enforce_eager=False",
            "++actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.custom_ops=[all]",
            "actor_rollout_ref.rollout.val_kwargs.n=16",
            "trainer.math_eval_samples=16",
            "trainer.validation_kl.enabled=True",
            "trainer.validation_kl.position_chunk_size=128",
            "trainer.validation_kl.vocab_chunk_size=8192",
            f"trainer.validation_kl.fixed_prefix_bank={bank}",
        ]
    )
    return {
        "train": train,
        "native_audit": [
            sys.executable,
            str(ROOT / "eval/collect_native_acceptance.py"),
            "--run",
            str(run),
            "--output",
            str(result),
            "--steps",
            "2",
            "--full",
        ],
        "kl_audit": [
            sys.executable,
            str(ROOT / "eval/collect_validation_kl.py"),
            "--run",
            str(run),
            "--output",
            str(result),
            "--bank",
            str(bank),
            "--full",
        ],
        "wandb": [
            str(runtime_python),
            str(ROOT / "eval/verify_wandb_math.py"),
            "--result",
            str(result),
            "--kl-report",
            str(run / "validation/kl/kl-step-2.json"),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--bank-input", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-python", type=Path, default=ROOT.parent / "envs/opd-state/bin/python")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--audit-only", action="store_true", help="Collect a successfully completed run without running training again"
    )
    args = parser.parse_args()
    if args.plan_only and args.audit_only:
        parser.error("Choose a plan or audit-only collection")
    run, result = args.output.resolve(), args.result.resolve()
    bank, data_manifest = args.bank.resolve(), args.data_manifest.resolve()
    prepared, files = input_bindings(args.bank_input)
    if not data_manifest.is_file():
        raise ValueError("Full KL gate requires the immutable training input manifest")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_commit + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:src/opd_state/integrations/verl/validation.py"], cwd=ROOT, check=True
    )
    if not args.plan_only and not args.runtime_python.is_file():
        raise ValueError("Full KL gate requires the existing training-environment Python for W&B verification")
    planned = commands(run, result, bank, files, commit, args.runtime_python)
    marker = run.with_name(run.name + ("-plan.json" if args.plan_only else "-gate.json"))
    if not args.audit_only and (run.exists() or marker.exists()):
        raise FileExistsError("Preserve the existing gate; audit completed evidence or use a new output path")
    bank_sha = verify_bank(bank, prepared, args.bank_input) if bank.is_file() else None
    if bank_sha is None and not args.plan_only:
        raise ValueError("Actual full-scale KL execution requires the completed fixed bank")
    state = {
        "state": "planned" if args.plan_only else "ready",
        "scope": "Two engineering updates; complete validation/resource coverage, not a training quality comparison",
        "source_commit": commit,
        "controller_pid": os.getpid(),
        "controller_sha256": digest(Path(__file__)),
        "bank_input_sha256": digest(args.bank_input),
        "bank_sha256": bank_sha,
        "bank_state": "ready" if bank_sha else "pending",
        "training_manifest_sha256": digest(data_manifest),
        "validation_files": files,
        "validation_questions": 570,
        "responses": 9120,
        "expected_bank_responses": 180,
        "commands": planned,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)

    def save(**updates):
        state.update(updates)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(marker)

    save()
    if args.plan_only:
        print(json.dumps({"plan": str(marker), "bank_state": state["bank_state"], "responses": 9120}))
        return 0
    env = dict(os.environ)
    env.update(
        NATIVE_DATA_MANIFEST=str(data_manifest),
        TRAIN_FILES="[" + str(data_manifest.parent / "ordered.parquet") + "]",
        VAL_N="16",
    )
    env.pop("WANDB_API_KEY", None)
    env.update(
        NETRC=str(ROOT.parent / ".secrets/netrc"),
        WANDB_CONFIG_DIR=str(ROOT.parent / "config/wandb"),
        WANDB_BASE_URL="https://api.wandb.ai",
    )
    if not args.audit_only:
        if any(used > 128 for _, used, _ in gpu_rows()):
            save(state="resource_busy")
            raise RuntimeError("GPU 0–3 are occupied; wait for the current experiment")
        log = run.with_name(run.name + "-driver.log")
        with log.open("w") as stream:
            process = subprocess.Popen(planned["train"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(state="running_gpu", child_pid=process.pid, started_at=time.time())
            code = process.wait()
        save(state="gpu_finished" if code == 0 else "gpu_failed", gpu_exit_code=code, gpu_finished_at=time.time())
        if code:
            return code
    try:
        for name in ("native_audit", "kl_audit"):
            save(state="running_" + name)
            subprocess.run(planned[name], cwd=ROOT, env=env, check=True)
            if name == "native_audit":
                summary = json.loads((result / "summary.json").read_text())
                if summary["source_commit"] != commit:
                    raise ValueError("Collected training source differs from the gate's requested immutable commit")
                subprocess.run(
                    [str(ROOT / "node_modules/.bin/prettier"), "--write", *map(str, sorted(result.glob("*.json")))],
                    cwd=ROOT,
                    check=True,
                )
        # These exact bytes will be bound by the W&B verifier and committed.
        subprocess.run(
            [str(ROOT / "node_modules/.bin/prettier"), "--write", *map(str, sorted(result.glob("*.json")))],
            cwd=ROOT,
            check=True,
        )
        save(state="verifying_wandb")
        subprocess.run(planned["wandb"], cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        subprocess.run(
            [str(ROOT / "node_modules/.bin/prettier"), "--write", str(result / "wandb-verification.json")],
            cwd=ROOT,
            check=True,
        )
        receipt = json.loads((result / "kl-collection.json").read_text())
        if (
            digest(args.bank_input) != state["bank_input_sha256"]
            or digest(data_manifest) != state["training_manifest_sha256"]
            or receipt.get("fixed_bank_sha256") != bank_sha
            or receipt["effective_vocab_size"] != prepared["vocab_size"]
        ):
            raise ValueError("Full KL input/model/vocabulary bindings changed during the gate")
        input_bindings(args.bank_input)
        if (
            receipt["responses"] != 9120
            or not receipt["all_validation_positions"]
            or not receipt["scoring_samples_cover_all_gpus"]
        ):
            raise ValueError("Full KL gate lacks all responses/positions or sampled scoring-phase GPU coverage")
        if any(
            receipt["coverage"].get(source, {}).get("responses") != 90
            for source in ("on_initial_states", "on_teacher_states")
        ):
            raise ValueError("Full KL gate lacks the complete shared dual-source fixed prefix bank")
        save(state="passed", finished_at=time.time(), result=str(result))
    except BaseException as error:
        save(state="audit_failed", error=str(error), failed_at=time.time())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
