"""Compare every native field against independently captured legacy launchers."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from opd_state.core.evaluation import due_suites, parse_evaluation
from recipes.configuration import compose_config, configure

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads((ROOT / "tests/oracles/configuration-baseline.json").read_text())
BASE = [
    "paths.model_root=/reference/models",
    "paths.data_root=/reference/data",
    "recipe.prepared_manifest=/reference/prepared/manifest.json",
    "run.dir=/reference/run",
    "run.name=reference",
    "evaluation=math-final",
]


def configuration(*overrides):
    return compose_config(ROOT, [*BASE, *overrides])


def expected_case(name):
    result = copy.deepcopy(ORACLE["base"])
    for key, value in ORACLE["cases"][name].items():
        node = result
        components = key.split(".")
        for component in components[:-1]:
            node = node[component]
        node[components[-1]] = copy.deepcopy(value)
    return result


def native_case(name):
    if name == "validation_fast":
        return configuration("evaluation=math-final-kl")
    if name == "formal_graph":
        return configuration("training=opd50-graph")
    boundary = name.rsplit("_", 1)
    if boundary[-1] in {"1024", "2048"}:
        return configuration(
            "rollout=" + boundary[0].replace("_", "-"),
            "actor_rollout_ref.rollout.policy.boundary.value=" + boundary[-1],
        )
    return configuration("rollout=" + name.replace("_", "-"))


@pytest.mark.parametrize("name", list(ORACLE["cases"]))
def test_complete_native_configuration_matches_independent_reference(name):
    expected = expected_case(name)
    config = native_case(name)
    actual = OmegaConf.to_container(config, resolve=True)
    # New project-only contracts do not replace or drop any native subtree.
    assert set(actual) - set(expected) == {"paths", "run", "recipe", "evaluation"}
    actual = {key: actual[key] for key in expected}
    objective = actual["distillation"].pop("opd")
    assert objective == {"objective": {"schema_version": 1, "kind": "sampled_opd"}}
    # These are the only deliberate path/identity relocations in this phase.
    assert actual["trainer"]["default_local_dir"] == "/reference/run/checkpoints"
    assert len(actual["trainer"]["source_commit"]) == 40
    assert actual["reward"]["custom_reward_function"]["path"] == str(
        ROOT / "verl/verl/utils/reward_score/opd_mixed_verify.py"
    )
    for key in ("default_local_dir", "source_commit"):
        actual["trainer"][key] = expected["trainer"][key]
    actual["reward"]["custom_reward_function"]["path"] = expected["reward"][
        "custom_reward_function"
    ]["path"]
    assert actual == expected


def test_cli_overrides_and_paths_with_spaces_remain_native():
    config = configuration(
        "paths.model_root='/models with spaces'",
        "actor_rollout_ref.actor.optim.lr=2e-6",
        "trainer.n_gpus_per_node=8",
        "evaluation=math-final-kl",
        "data.val_batch_size=128",
    )
    assert (
        config.actor_rollout_ref.model.path
        == "/models with spaces/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    assert config.actor_rollout_ref.actor.optim.lr == 2e-6
    assert config.data.train_batch_size == 64
    assert config.actor_rollout_ref.rollout.max_num_seqs == 64
    assert config.actor_rollout_ref.rollout.generation_eval_max_num_seqs == 128
    assert config.data.val_batch_size == 128


def test_evaluation_profile_does_not_change_learner_or_source():
    baseline = configuration("rollout=initial-teacher-fusion")
    fast = configuration("rollout=initial-teacher-fusion", "evaluation=math-final-kl")
    for key in ("actor", "model", "ref"):
        assert baseline.actor_rollout_ref[key] == fast.actor_rollout_ref[key]
    assert (
        baseline.actor_rollout_ref.rollout.policy
        == fast.actor_rollout_ref.rollout.policy
    )
    assert fast.trainer.test_freq == 50
    assert not fast.trainer.val_before_train


def test_empty_evaluation_has_no_inputs_or_triggers():
    config = configuration("evaluation=none")
    assert config.data.val_files == []
    assert config.trainer.test_freq == -1
    assert not config.trainer.validation_kl.enabled
    assert config.actor_rollout_ref.rollout.max_model_len == 5120


def test_periodic_and_final_reasons_do_not_duplicate_a_suite():
    config = configuration("+evaluation.suites.validation.trigger.every_outer_steps=10")
    evaluation = parse_evaluation(
        OmegaConf.to_container(config.evaluation, resolve=True)
    )
    assert config.trainer.test_freq == 10
    assert due_suites(evaluation, outer=10) == {"validation": ["periodic"]}
    assert due_suites(evaluation, outer=50, final=True) == {
        "validation": ["periodic", "training_end"]
    }
    assert due_suites(evaluation, outer=0) == {}


@pytest.mark.parametrize(
    "overrides",
    [
        ["++distillation.opd.objective.typo=true"],
        ["distillation.opd.objective.schema_version=2"],
        ["++evaluation.unknown=true"],
        ["++evaluation.suites.validation.diagnostics.kl.positon_chunk_size=1024"],
        ["actor_rollout_ref.actor.ppo_epochs=2"],
        ["actor_rollout_ref.actor.ppo_mini_batch_size=32"],
        ["trainer.n_gpus_per_node=3"],
        ["actor_rollout_ref.rollout.top_p=0.9"],
        ["actor_rollout_ref.rollout.temperature=0.7"],
        ["distillation.fsdp_teacher.log_prob_top_k=16"],
        ["actor_rollout_ref.actor.use_kl_loss=true"],
        ["actor_rollout_ref.rollout.policy.model=unknown"],
        [
            "rollout=student-teacher-fusion",
            "actor_rollout_ref.rollout.policy.teacher_weight=1.5",
        ],
        [
            "rollout=teacher-current",
            "actor_rollout_ref.rollout.policy.boundary.value=-1",
        ],
        ["trainer.test_freq=10"],
        ["actor_rollout_ref.rollout.val_kwargs.n=8"],
        [
            "evaluation=math-final-kl",
            "evaluation.suites.validation.diagnostics.kl.forward_batch_size=2",
        ],
        ["trainer.resume_mode=auto"],
    ],
)
def test_invalid_contracts_fail_before_input_or_worker_initialization(overrides):
    with pytest.raises((ValueError, TypeError)):
        configuration(*overrides)


def test_unknown_native_override_uses_hydra_error():
    with pytest.raises(ConfigCompositionException):
        configuration("actor_rollout_ref.actor.optim.learning_rate=1e-6")


def test_printable_unresolved_configuration_does_not_read_inputs():
    cfg = compose_config(ROOT, resolve=False)
    assert cfg.paths.model_root is None
    assert cfg.run.dir is None
    assert cfg.distillation.opd.objective.kind == "sampled_opd"


def test_lazy_configuration_keeps_its_selected_backend(tmp_path):
    config = compose_config(ROOT, BASE, resolve=False)
    other = tmp_path / "other"
    native = other / "verl/verl/trainer/config"
    native.mkdir(parents=True)
    (native / "ppo_trainer.yaml").write_text("{}\n")
    configure(other)
    OmegaConf.resolve(config)
    assert config.run.backend == str(ROOT)
    assert config.reward.custom_reward_function.path == str(
        ROOT / "verl/verl/utils/reward_score/opd_mixed_verify.py"
    )


def test_backend_identity_cannot_be_overridden_as_a_runtime_option():
    with pytest.raises(ValueError, match="derived"):
        configuration("run.backend=/unselected/backend")


def test_composition_requires_neither_training_imports_nor_real_assets():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,json; from recipes.configuration import compose_config; "
            "compose_config(sys.argv[1], json.loads(sys.argv[2])); "
            "assert not ({'torch','ray','vllm','transformers'} & set(sys.modules))",
            str(ROOT),
            json.dumps(BASE),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
