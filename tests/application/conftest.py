"""Synthetic local assets test file/lifecycle contracts, not GPU model behavior."""

import json
import shutil
from pathlib import Path

import pytest

from opd_state.artifacts import content_id, file_digest
from recipes.application import prepare
from recipes.configuration import compose_config
from recipes.inputs import prepare_inputs

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    chat_template = "fixture"
    eos_token_id = 15
    pad_token_id = 15
    bos_token_id = 1

    def get_vocab(self):
        return {str(i): i for i in range(16)}

    def __len__(self):
        return 16

    def apply_chat_template(self, messages, **_kwargs):
        return [1, 2, int(messages[0]["content"]) + 3]

    def decode(self, ids, **_kwargs):
        return ",".join(map(str, ids))


def row(index, source):
    return {
        "prompt": [{"role": "user", "content": str(index)}],
        "data_source": source,
        "reward_model": {"ground_truth": str(index)},
    }


@pytest.fixture
def prepared(tmp_path):
    repository = tmp_path / "repository"
    backend = tmp_path / "backend"
    for name, content in {
        "src/opd_state/__init__.py": "__version__ = 'fixture'\n",
        "recipes/__init__.py": "",
        "recipes/train.py": "from pathlib import Path\nimport os\nPath(os.environ['OPD_MANIFEST_PATH']).parent.joinpath('dispatch-proof.txt').write_text(__file__)\n",
        "tools/__init__.py": "",
        "tools/verify.py": "from pathlib import Path\nimport sys\nPath(sys.argv[1]).joinpath('verification-dispatch-proof.txt').write_text(__file__)\n",
        "configs/train.yaml": "{}\n",
    }.items():
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for name in (
        "workers/rollout/policy.py",
        "trainer/ppo/fixed_rollout.py",
        "trainer/ppo/math_eval.py",
    ):
        target = backend / "verl/verl" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "verl/verl" / name, target)
    scorer = backend / "verl/verl/utils/reward_score/opd_mixed_verify.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("def reward_func(*args, **kwargs): return 1\n")
    models = tmp_path / "models"
    for name in ("DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B"):
        model = models / name
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "vocab_size": 16,
                    "max_position_embeddings": 32768,
                }
            )
        )
        (model / "model.safetensors").write_bytes(name.encode())
        (model / "tokenizer.json").write_text("{}\n")
    pool = tmp_path / "pool"
    pool.mkdir()
    tokenizer = Tokenizer()
    rows = [row(index, "DAPO") for index in range(4)]
    ledger = []
    for index, item in enumerate(rows):
        ids = tokenizer.apply_chat_template(item["prompt"])
        identity = content_id(ids)
        item.update(opd_prompt_id=identity, opd_prompt_length=len(ids))
        ledger.append(
            {
                "prompt_id": identity,
                "prompt_ids": ids,
                "prompt_sha256": content_id(item["prompt"]),
                "source_row": index,
                "data_source": "DAPO",
            }
        )
    (pool / "ordered.parquet").write_text(json.dumps(rows))
    (pool / "prompts.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in ledger)
    )
    token_id = content_id(
        {
            "vocab": tokenizer.get_vocab(),
            "template": tokenizer.chat_template,
            "eos": 15,
            "pad": 15,
            "bos": 1,
        }
    )
    manifest = {
        "schema": 1,
        "collection": "fresh",
        "tokenizer_signature": token_id,
        "prompt_cap": 1024,
        "apply_chat_template_kwargs": {},
        "unique_prompts": 4,
        "ordered_parquet_sha256": file_digest(pool / "ordered.parquet"),
        "prompts_jsonl_sha256": file_digest(pool / "prompts.jsonl"),
    }
    (pool / "manifest.json").write_text(json.dumps(manifest))
    paths = []
    for index, source in enumerate(("MATH-500", "AMC23", "AIME25"), 4):
        path = tmp_path / (source + ".parquet")
        path.write_text(json.dumps([row(index, source)]))
        paths.append(str(path))
    directory = tmp_path / "run"
    config = compose_config(
        ROOT,
        [
            f"paths.model_root={models}",
            f"paths.data_root={tmp_path}",
            f"recipe.prepared_manifest={pool / 'manifest.json'}",
            f"run.dir={directory}",
            "run.prepare_only=true",
            "logging=local",
            "evaluation=math-final",
            "data.train_batch_size=4",
            "trainer.total_training_steps=2",
            "evaluation.benchmarks.math.data.paths=" + json.dumps(paths),
            "evaluation.benchmarks.math.data.expected_counts={MATH-500:1,AMC23:1,AIME25:1}",
        ],
    )
    config.reward.custom_reward_function.path = str(scorer)
    config.run.backend = str(backend)
    config.run.repository = str(repository)

    def inputs(config, root, *, backend):
        return prepare_inputs(
            config,
            root,
            backend=backend,
            tokenizer_loader=lambda *a, **k: Tokenizer(),
            table_reader=lambda path: json.loads(path.read_text()),
            model_inspector=lambda artifact: {"fixture": True},
        )

    prepared_manifest = prepare(
        config, backend=backend, repository=repository, input_preparer=inputs
    )
    return {
        "root": directory,
        "config": config,
        "manifest": prepared_manifest,
        "repository": repository,
        "backend": backend,
        "models": models,
        "inputs": inputs,
    }
