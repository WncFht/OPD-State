"""Execute the actual adapter target-selection method without the GPU stack."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/vllm_rollout.py"
tree = ast.parse(SOURCE.read_text())
adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerAdapter")
method = next(n for n in adapter.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "set_native_weight_target")
subset = ast.Module(
    body=[ast.ClassDef(name="Adapter", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]
)
namespace = {}
exec(compile(ast.fix_missing_locations(subset), str(SOURCE), "exec"), namespace)
Adapter = namespace["Adapter"]


class Config(dict):
    __getattr__ = dict.__getitem__


class NativeRoutingTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        obj = Adapter()
        obj.config = Config(
            policy={"kind": "logit_fusion"}, tensor_model_parallel_size=1, generation_sync_namespace="run-one"
        )
        obj.model_config = NS(path="/p")
        obj.replica_rank = 1
        obj.server_handle = "previous"
        status = {
            "source_role": "current_student",
            "source_path": "/p",
            "sync_namespace": "run-one",
            "fusion": None,
            "active": 0,
            "awake": False,
            "purpose": "training",
        }

        async def query():
            return status

        handle = NS(native_source_status=NS(remote=query))
        return obj, status, handle

    async def test_target_selection_uses_replica_rank_and_verifies_the_frozen_companion(self):
        obj, status, handle = self.fixture()
        await obj.set_native_weight_target(["other-rank", handle], None)
        self.assertIs(obj.server_handle, handle)
        fusion = {"teacher_path": "/q", "teacher_fingerprint": "sha", "teacher_weight": 0.5}
        status["fusion"] = fusion
        await obj.set_native_weight_target(["other-rank", handle], fusion)
        self.assertIs(obj.server_handle, handle)

    async def test_wrong_source_job_path_or_live_receiver_cannot_replace_the_existing_target(self):
        for key, value in [
            ("source_role", "initial_student"),
            ("source_role", "teacher"),
            ("source_path", "/q"),
            ("sync_namespace", "other-run"),
            ("active", 1),
            ("awake", True),
            ("fusion", {"teacher_weight": 0.5}),
            ("purpose", "evaluation"),
        ]:
            obj, status, handle = self.fixture()
            status[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                await obj.set_native_weight_target([None, handle], None)
            self.assertEqual(obj.server_handle, "previous")


if __name__ == "__main__":
    unittest.main()
