"""Frozen pre-migration routing and code-body references guard the extraction."""

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from opd_state.core.policy import compile_policy

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/oracles/rollout-extraction.json").read_text())


def test_all_frozen_routes_and_endpoints_match():
    for case in REFERENCE["cases"]:
        result = asdict(compile_policy(case["policy"], **case["arguments"]))
        digest = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == case["plan_sha256"], case


def test_extraction_preserves_algorithm_bodies():
    for path, reference in REFERENCE["modules"].items():
        tree = ast.parse((ROOT / path).read_text())
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        digest = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode()
        ).hexdigest()
        assert digest == reference["body_ast_sha256"], path


def test_core_and_execution_import_without_training_frameworks(tmp_path):
    code = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ray', 'verl', 'vllm'}:
        raise AssertionError('Unexpected backend dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
from opd_state.core.policy import compile_policy
from opd_state.rollout.executor import generate
from opd_state.rollout.lifecycle import GenerationState
assert compile_policy({'kind': 'single', 'model': 'teacher'}, cap=8, batch_size=4, seed=42, batch_id=1).models == {'teacher'}
assert GenerationState('teacher', '/q', 'fingerprint').identity == ('teacher', '/q', 'fingerprint')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "src")], cwd=tmp_path, check=True
    )
