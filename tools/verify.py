"""Verify sealed inputs and recompute completion from actual run evidence."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from opd_state.artifacts import content_id, file_digest
from opd_state.core.evaluation import due_suites, parse_evaluation
from opd_state.core.policy import compile_policy
from opd_state.integrations.verl.configuration import validate_config
from recipes.inputs import verify_inputs
from recipes.snapshots import verify_snapshot


def configuration_id(config, root, inputs, scratch):
    """Separate content/scientific settings from per-run locations and labels."""
    value = copy.deepcopy(OmegaConf.to_container(config, resolve=True))
    value.pop("run", None)
    value.pop("paths", None)
    for key in ("source_commit", "experiment_name"):
        value["trainer"].pop(key, None)
    env = (
        value.get("ray_kwargs", {})
        .get("ray_init", {})
        .get("runtime_env", {})
        .get("env_vars", {})
    )
    for key in ("OPD_RUN_ID", "OPD_CONFIG_ID", "WANDB_RUN_ID", "WANDB_NAME"):
        env.pop(key, None)
    replacements = [
        (row["path"], "@asset/" + row["identity"]) for row in inputs["models"].values()
    ]
    replacements += [(str(root), "@run"), (str(scratch), "@scratch")]

    def normalized(item):
        if isinstance(item, dict):
            return {k: normalized(v) for k, v in item.items()}
        if isinstance(item, list):
            return [normalized(v) for v in item]
        if isinstance(item, str):
            for original, replacement in replacements:
                item = item.replace(original, replacement)
        return item

    return content_id(normalized(value))


def verify_prepared(directory, *, verify_assets=True):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-run schema")
    if manifest.get("manifest_id") != content_id(
        {k: v for k, v in manifest.items() if k != "manifest_id"}
    ):
        raise ValueError("Prepared manifest identity changed")
    if (
        file_digest(root / "resolved.yaml") != manifest["configuration_sha256"]
        or file_digest(root / "inputs.json") != manifest["inputs_sha256"]
    ):
        raise ValueError("Prepared configuration or input manifest changed")
    source = verify_snapshot(root / "source")
    if source["source_id"] != manifest["source_id"]:
        raise ValueError("Prepared run points to a different source snapshot")
    inputs = json.loads((root / "inputs.json").read_text())
    if inputs["inputs_id"] != manifest["inputs_id"]:
        raise ValueError("Prepared input identity differs from its manifest")
    if verify_assets:
        verify_inputs(root, inputs)
    config = OmegaConf.load(root / "resolved.yaml")
    if OmegaConf.to_yaml(config, resolve=False).find("${") != -1:
        raise ValueError(
            "Prepared configuration must contain no unresolved interpolation"
        )
    if (
        Path(config.run.dir).resolve() != root
        or Path(config.run.source_dir).resolve() != root / "source"
    ):
        raise ValueError(
            "Prepared run cannot be moved or rebound without new preparation"
        )
    if (
        Path(config.run.backend).resolve() != root / "source/backend"
        or Path(config.run.repository).resolve() != root / "source"
    ):
        raise ValueError(
            "Prepared execution must use its frozen backend and application"
        )
    if (
        configuration_id(config, root, inputs, manifest["scratch"])
        != manifest["config_id"]
    ):
        raise ValueError("Prepared scientific configuration identity changed")
    validate_config(config, backend=config.run.backend)
    return config, manifest, inputs


def _json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_training(root, config, inputs):
    observations = _json_lines(root / "logs/metrics.jsonl")
    steps = config.trainer.total_training_steps
    updates = {}
    for row in observations:
        step, metrics = row.get("step"), row.get("data")
        if type(step) is not int or not isinstance(metrics, dict):
            raise ValueError("Malformed training metric observation")
        if "actor/loss" not in metrics and "actor/grad_norm" not in metrics:
            continue
        if step in updates or step not in range(1, steps + 1):
            raise ValueError("Duplicate or out-of-budget optimizer update")
        for key in ("actor/loss", "actor/grad_norm", "timing_s/update_actor"):
            if not _finite_number(metrics.get(key)):
                raise ValueError(f"Missing/nonfinite optimizer evidence: {key}")
        if metrics["actor/grad_norm"] < 0 or metrics["timing_s/update_actor"] <= 0:
            raise ValueError("Invalid gradient or update timing")
        updates[step] = metrics
    if set(updates) != set(range(1, steps + 1)):
        raise ValueError("Run does not contain every requested optimizer update")
    directory = root / "native-trajectories"
    manifest = json.loads((directory / "manifest.json").read_text())
    policy = OmegaConf.to_container(
        config.actor_rollout_ref.rollout.policy, resolve=True
    )
    if (
        manifest["policy"] != policy
        or manifest["trainer_gpu_count"] != config.trainer.n_gpus_per_node
    ):
        raise ValueError(
            "Observed source policy or GPU count differs from the prepared configuration"
        )
    if manifest["prepared_data_manifest_sha256"] != file_digest(
        config.recipe.prepared_manifest
    ):
        raise ValueError("Observed training prompt pool differs from preparation")
    selection = json.loads((root / "data/selection.json").read_text())
    expected = {row["row_id"]: row for row in selection["train"]}
    seen_visits = Counter()
    traces = []
    batch_size = config.data.train_batch_size
    cap = config.data.max_response_length
    batches = _json_lines(directory / "batches.jsonl")
    training_batches = [row for row in batches if row["validate"] is False]
    if Counter(row["step"] for row in training_batches) != Counter(range(1, steps + 1)):
        raise ValueError("Generation receipts must cover each training update once")
    receipts = {row["step"]: row for row in training_batches}
    for step in range(1, steps + 1):
        path = directory / f"step-{step}.jsonl"
        rows = _json_lines(path)
        if len(rows) != batch_size:
            raise ValueError(
                "A training update did not consume its complete global batch"
            )
        plan = compile_policy(
            policy,
            cap=cap,
            batch_size=batch_size,
            seed=config.actor_rollout_ref.rollout.seed,
            batch_id=step,
        )
        keys = set()
        for row, entry in zip(rows, plan.requests, strict=True):
            request = row["request"]
            expected_row = expected.get(request["row_id"])
            if (
                not expected_row
                or content_id(request["prompt_ids"]) != expected_row["prompt_id"]
            ):
                raise ValueError(
                    "Training consumed a prompt outside the selected input population"
                )
            if request["prompt_id"] != "train:" + expected_row["generation_id"]:
                raise ValueError("Training prompt source identity changed")
            key = request["prompt_id"], request["visit"], request["sample"]
            if (
                key in keys
                or request["visit"] != seen_visits[request["prompt_id"]]
                or request["sample"] != 0
            ):
                raise ValueError(
                    "Training visit/sample identity is duplicated or inconsistent"
                )
            keys.add(key)
            seen_visits[request["prompt_id"]] += 1
            if (
                row["route"] != entry.route
                or not row["segments"]
                or len(row["segments"]) > len(entry.stages)
            ):
                raise ValueError(
                    "Training route/stage count disagrees with its configured policy"
                )
            cursor = 0
            for segment, stage in zip(row["segments"], entry.stages):
                if (
                    segment["start"] != cursor
                    or not 0 < segment["end"] - cursor <= stage.max_tokens
                ):
                    raise ValueError(
                        "Segment boundaries violate the shared response budget"
                    )
                if [i[0] for i in segment["identities"]] != list(
                    stage.models
                ) or segment["teacher_weight"] != stage.teacher_weight:
                    raise ValueError(
                        "Segment used the wrong source or fusion coefficient"
                    )
                for role, path_name, version in segment["identities"]:
                    model = inputs["models"][
                        "teacher" if role == "teacher" else "student"
                    ]
                    required_version = (
                        step - 1
                        if role == "current_student"
                        else model["role_fingerprint"]
                    )
                    if (
                        Path(path_name).resolve() != Path(model["path"])
                        or version != required_version
                    ):
                        raise ValueError(
                            "Generation used stale actor or different frozen model weights"
                        )
                cursor = segment["end"]
            response = row["token_ids"]
            if cursor != len(response) or not 0 < len(response) <= cap:
                raise ValueError("Segments do not cover exactly the effective response")
            if any(
                type(t) is not int or not 0 <= t < inputs["vocab_size"]
                for t in response
            ):
                raise ValueError("Training contains illegal response tokens")
            eos = inputs["token_ids"]["eos"]
            if row["termination"] not in {"eos", "stop", "length"}:
                raise ValueError("Training has an unsupported termination reason")
            if eos in response[:-1] or (row["termination"] == "eos") != (
                response[-1] == eos
            ):
                raise ValueError(
                    "Training response continued after EOS or misreported termination"
                )
            if row["termination"] == "length" and len(response) != cap:
                raise ValueError("Length-terminated response did not use its budget")
        identities = sorted(
            {
                tuple(identity)
                for row in rows
                for segment in row["segments"]
                for identity in segment["identities"]
            }
        )
        receipt = receipts[step]
        if receipt["identities"] != [list(identity) for identity in identities]:
            raise ValueError(
                "Training generation receipt disagrees with trajectory weights"
            )
        for key, value in {
            "source/response_count": len(rows),
            "source/output_tokens": sum(len(row["token_ids"]) for row in rows),
            "source/unique_prompts": len({row["request"]["prompt_id"] for row in rows}),
        }.items():
            if receipt["metrics"].get(key) != value or updates[step].get(key) != value:
                raise ValueError(
                    "Training source metrics disagree with actual trajectories"
                )
        traces.append(
            {
                "step": step,
                "sha256": file_digest(path),
                "response_tokens": sum(len(r["token_ids"]) for r in rows),
            }
        )
    if {p.stem for p in directory.glob("step-*.jsonl")} != {
        f"step-{s}" for s in range(1, steps + 1)
    }:
        raise ValueError("Unexpected training trajectory versions")
    return observations, updates, traces


def verify_validation(root, config, inputs, observations):
    evaluation = parse_evaluation(
        OmegaConf.to_container(config.evaluation, resolve=True)
    )
    steps = config.trainer.total_training_steps
    due = {
        step: due_suites(evaluation, outer=step, final=step == steps)
        for step in range(steps + 1)
    }
    required = {step for step, suites in due.items() if suites}
    paths = {int(p.stem): p for p in (root / "validation").glob("*.jsonl")}
    if set(paths) != required:
        raise ValueError("Validation did not cover exactly the configured versions")
    batches = _json_lines(root / "native-trajectories/batches.jsonl")
    validation_batches = [row for row in batches if row["validate"]]
    if {row["step"] for row in validation_batches} != required:
        raise ValueError(
            "Generation receipts disagree with required validation versions"
        )
    for batch in validation_batches:
        expected = [
            "current_student",
            inputs["models"]["student"]["path"],
            batch["step"],
        ]
        if batch["identities"] != [expected]:
            raise ValueError(
                "Validation must use only the matching live student version"
            )
    selection = json.loads((root / "data/selection.json").read_text())["validation"]
    questions = {(r["source"], "eval:" + r["generation_id"]): r for r in selection}
    results = {}
    for step, path in paths.items():
        suite = next(iter(evaluation.suites.values()))
        benchmark = evaluation.benchmarks[suite.benchmarks[0]]
        n = benchmark.sampling.samples_per_case
        grouped = defaultdict(dict)
        uid_by_question, question_by_uid = {}, {}
        for row in _json_lines(path):
            key = row["data_source"], row["rollout_generation_prompt_id"]
            expected = questions.get(key)
            if (
                not expected
                or row["input"] != expected["rendered_input"]
                or content_id(row["gts"]) != expected["reference_id"]
            ):
                raise ValueError("Validation changed the frozen question or reference")
            if row["step"] != step or row["rollout_student_version"] != step:
                raise ValueError("Validation response has the wrong student version")
            uid = row.get("validation_uid")
            if (
                not isinstance(uid, str)
                or not uid
                or uid_by_question.setdefault(key, uid) != uid
                or question_by_uid.setdefault((key[0], uid), key) != key
            ):
                raise ValueError(
                    "Validation UID and question identities are inconsistent"
                )
            sample = row["rollout_generation_sample"]
            if (
                type(sample) is not int
                or sample not in range(n)
                or sample in grouped[key]
            ):
                raise ValueError(
                    "Validation sample identity is duplicated or outside its budget"
                )
            if type(row["score"]) not in (int, float, bool) or row["score"] not in (
                0,
                1,
            ):
                raise ValueError("Math evaluation requires binary finite scores")
            grouped[key][sample] = row["score"]
        if set(grouped) != set(questions) or any(
            set(values) != set(range(n)) for values in grouped.values()
        ):
            raise ValueError("Validation response population is incomplete")
        metrics, by_source = {}, defaultdict(list)
        for (source, _), values in grouped.items():
            correct = int(sum(values.values()))
            item = {f"mean@{n}": correct / n}
            for k in (1, 2, 4, 8, 16):
                if k <= n:
                    item[f"pass@{k}"] = 1 - (
                        math.comb(n - correct, k) if n - correct >= k else 0
                    ) / math.comb(n, k)
            by_source[source].append(item)
        for source, items in by_source.items():
            metrics[f"eval/{source}/questions"] = len(items)
            metrics[f"eval/{source}/samples_per_question"] = n
            for name in items[0]:
                metrics[f"eval/{source}/{name}"] = math.fsum(
                    item[name] for item in items
                ) / len(items)
        fields = next(iter(by_source.values()))[0]
        for name in fields:
            metrics[f"eval/macro/{name}"] = math.fsum(
                metrics[f"eval/{source}/{name}"] for source in by_source
            ) / len(by_source)
            metrics[f"eval/micro/{name}"] = math.fsum(
                item[name] for items in by_source.values() for item in items
            ) / len(questions)
        logged = [
            row["data"]
            for row in observations
            if row["step"] == step and "eval/macro/mean@" + str(n) in row["data"]
        ]
        if len(logged) != 1 or any(
            not _finite_number(logged[0].get(k))
            or not math.isclose(logged[0][k], v, rel_tol=0, abs_tol=1e-8)
            for k, v in metrics.items()
        ):
            raise ValueError(
                "Independent accuracy aggregation disagrees with the logged validation"
            )
        results[str(step)] = {
            "responses": n * len(questions),
            "sha256": file_digest(path),
            "metrics": metrics,
        }
        if suite.diagnostics.kl.enabled:
            report = root / f"validation/kl/kl-step-{step}.json"
            tokens = root / f"validation/kl/kl-tokens-step-{step}.jsonl"
            audit_path = root / "source/eval/audit_validation_kl.py"
            spec = importlib.util.spec_from_file_location("_opd_kl_audit", audit_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.audit(
                report,
                tokens,
                accuracy_path=path,
                bank_path=suite.diagnostics.kl.fixed_prefix_bank,
                samples=n,
                workers=config.trainer.n_gpus_per_node,
                required_counts=benchmark.data.expected_counts,
                replicates=5000,
            )
            if result["student_version"] != step:
                raise ValueError("KL audit refers to a different live student version")
            report_metrics = json.loads(report.read_text())["metrics"]
            if any(
                not _finite_number(logged[0].get(key))
                or not math.isclose(logged[0][key], value, rel_tol=1e-8, abs_tol=1e-8)
                for key, value in report_metrics.items()
            ):
                raise ValueError("Logged KL metrics disagree with the audited report")
            results[str(step)]["kl"] = result
    kl_steps = (
        required
        if any(suite.diagnostics.kl.enabled for suite in evaluation.suites.values())
        else set()
    )
    for pattern, prefix, suffix in (
        ("kl-step-*.json", "kl-step-", ".json"),
        ("kl-tokens-step-*.jsonl", "kl-tokens-step-", ".jsonl"),
    ):
        found = {
            int(path.name[len(prefix) : -len(suffix)])
            for path in (root / "validation/kl").glob(pattern)
        }
        if found != kl_steps:
            raise ValueError(
                "KL artifacts do not match the configured validation versions"
            )
    return results


def verify_run(directory):
    root = Path(directory).resolve()
    config, manifest, inputs = verify_prepared(root)
    exit_record = json.loads((root / "train-exit.json").read_text())
    if (
        exit_record.get("return_code") != 0
        or exit_record.get("run_id") != manifest["run_id"]
    ):
        raise ValueError("Backend process did not finish successfully for this run")
    resources = json.loads((root / "resources.json").read_text())
    if (
        resources.get("run_id") != manifest["run_id"]
        or resources.get("gpu_count") != config.trainer.n_gpus_per_node
    ):
        raise ValueError("Runtime allocation does not match the prepared run")
    observations, updates, traces = verify_training(root, config, inputs)
    validations = verify_validation(root, config, inputs, observations)
    if config.trainer.save_freq > 0:
        raise ValueError(
            "Checkpoint publication/reload must be integrated before enabling saving"
        )
    if list((root / "checkpoints").glob("global_step_*")):
        raise ValueError("A no-checkpoint run unexpectedly saved a checkpoint")
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "source_id": manifest["source_id"],
        "config_id": manifest["config_id"],
        "inputs_id": manifest["inputs_id"],
        "updates": len(updates),
        "training": traces,
        "validation": validations,
        "metrics_sha256": file_digest(root / "logs/metrics.jsonl"),
        "scope": "Source/input integrity, actual update evidence, routed trajectories and independent accuracy/KL artifact aggregation",
    }
    status = json.loads((root / "status.json").read_text())
    if status.get("state") == "completed":
        if (
            file_digest(root / "verification.json") != status.get("verification_sha256")
            or json.loads((root / "verification.json").read_text()) != result
        ):
            raise ValueError("Published completion evidence changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    result = (
        verify_prepared(args.directory)[1]
        if args.prepared
        else verify_run(args.directory)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
