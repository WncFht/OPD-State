"""Frozen numerical bodies and backend-free record/aggregation imports."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_scoring_bodies_match_pre_migration_reference():
    reference = json.loads((ROOT / "tests/oracles/scoring-extraction.json").read_text())
    for item in reference["definitions"]:
        tree = ast.parse((ROOT / item["target"]).read_text())
        node = next(
            node for node in tree.body if getattr(node, "name", None) == item["name"]
        )
        body = ast.Module(body=node.body, type_ignores=[])
        digest = hashlib.sha256(
            ast.dump(body, include_attributes=False).encode()
        ).hexdigest()
        assert digest == item["body_sha256"], item


def test_record_and_metric_modules_do_not_import_backend(tmp_path):
    code = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ray', 'verl', 'vllm'}:
        raise AssertionError('Unexpected backend dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
from opd_state.evaluation.math import estimate_pass_at_k
from opd_state.evaluation.prefixes import sample_prefix_positions
from opd_state.core.records import record_fingerprint
from opd_state.evaluation.kl_metrics import summarize_kl
assert estimate_pass_at_k(16, 1, 16) == 1
assert sum(x.weight for x in sample_prefix_positions(100)) == 100
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "src")], cwd=tmp_path, check=True
    )
