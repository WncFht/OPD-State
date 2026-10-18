"""Changing a diagnostic mask must not retain old sampler restrictions."""

import ast
import unittest
from pathlib import Path
from types import MethodType

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/utils.py"
tree = ast.parse(SOURCE.read_text())
method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "monkey_patch_compute_logits")
namespace = {"MethodType": MethodType, "Optional": object, "torch": type("Torch", (), {"Tensor": object})}
# Postpone only the extracted annotations so the check needs no GPU imports.
module = ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[]
)
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class MaskTests(unittest.TestCase):
    def test_reconfiguring_mask_restores_the_original_logits_before_applying_new_constraints(self):
        class Model:
            def compute_logits(self):
                return np.array([[1.0, 2.0, 3.0, 4.0]])

        model = Model()
        namespace["monkey_patch_compute_logits"](model, 3, [1])
        self.assertTrue(np.isneginf(model.compute_logits()[0, 1]))
        namespace["monkey_patch_compute_logits"](model, 3, [])
        self.assertEqual(model.compute_logits()[0, 1], 2)
        self.assertTrue(np.isneginf(model.compute_logits()[0, 3]))


if __name__ == "__main__":
    unittest.main()
