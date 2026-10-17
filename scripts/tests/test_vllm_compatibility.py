"""Exercise the actual lifecycle methods without importing the GPU stack.

The source methods are compiled unchanged from their AST. These tests cover RPC
ordering and error propagation; the GPU smoke run covers full module integration.
"""

import ast
import inspect
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
)


def load_lifecycle():
    tree = ast.parse(SOURCE.read_text())
    server = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer"
    )
    selected = {
        "_reset_prefix_cache",
        "abort_all_requests",
        "resume_generation",
        "_sleep_hybrid",
        "collective_rpc",
    }
    methods = [
        node
        for node in server.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name in selected
    ]
    if {node.name for node in methods} != selected:
        raise AssertionError("Lifecycle method coverage changed")
    subset = ast.Module(
        body=[
            ast.ClassDef(
                name="Lifecycle", bases=[], keywords=[], body=methods, decorator_list=[]
            )
        ],
        type_ignores=[],
    )
    logger = logging.getLogger("vllm-lifecycle-test")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    namespace = {
        "inspect": inspect,
        "Any": Any,
        "Callable": Callable,
        "logger": logger,
        "is_torch_npu_available": lambda **kw: False,
    }
    exec(compile(ast.fix_missing_locations(subset), str(SOURCE), "exec"), namespace)
    return namespace["Lifecycle"]


class LegacyEngine:
    def __init__(self):
        self.calls = []
        self.output_processor = SimpleNamespace(request_states={"a": None, "b": None})

    async def reset_prefix_cache(self, device=None):
        self.calls.append(("prefix", device))

    async def reset_mm_cache(self):
        self.calls.append(("mm",))

    async def abort(self, request_id):
        self.calls.append(("abort", request_id))
        self.output_processor.request_states.pop(request_id)

    async def sleep(self, level):
        self.calls.append(("sleep", level))


class CurrentEngine(LegacyEngine):
    async def reset_prefix_cache(self, *, reset_connector=False):
        self.calls.append(("prefix", reset_connector))

    async def pause_generation(self, *, wait_for_inflight_requests, clear_cache):
        self.calls.append(("pause", wait_for_inflight_requests, clear_cache))

    async def resume_generation(self):
        self.calls.append(("resume",))

    async def reset_encoder_cache(self):
        self.calls.append(("encoder",))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, engine):
        obj = load_lifecycle()()
        obj.engine = engine
        obj.node_rank = 0
        obj.config = SimpleNamespace(mtp=None)
        obj.lora_as_adapter = False
        return obj

    async def test_prefix_reset_matches_installed_interface(self):
        for engine, expected in [(LegacyEngine(), None), (CurrentEngine(), True)]:
            await self.fixture(engine)._reset_prefix_cache()
            self.assertEqual(engine.calls, [("prefix", expected)])

    async def test_collective_rpc_returns_generation_payload_unchanged(self):
        payload = [{"responses": [[1, 2]], "student_version": 4, "lambda": 0.5}]
        captured = {}

        class Engine:
            async def collective_rpc(self, **kwargs):
                captured.update(kwargs)
                return payload

        result = await self.fixture(Engine()).collective_rpc(
            "generate_logit_fusion",
            timeout=60,
            args=("prefix",),
            kwargs={"expected_version": 4},
        )
        self.assertIs(result, payload)
        self.assertEqual(
            captured,
            {
                "method": "generate_logit_fusion",
                "timeout": 60,
                "args": ("prefix",),
                "kwargs": {"expected_version": 4},
            },
        )

    async def test_legacy_abort_clears_requests_and_caches_without_pause(self):
        engine = LegacyEngine()
        obj = self.fixture(engine)
        result = await obj.abort_all_requests()
        await obj.resume_generation()
        self.assertEqual(result, {"aborted_count": 2, "request_ids": ["a", "b"]})
        self.assertEqual(
            engine.calls, [("abort", "a"), ("abort", "b"), ("prefix", None), ("mm",)]
        )

    async def test_current_pause_and_resume_preserve_rpc_contract(self):
        engine = CurrentEngine()
        obj = self.fixture(engine)
        result = await obj.abort_all_requests()
        await obj.resume_generation()
        self.assertEqual(result["aborted_count"], 2)
        self.assertEqual(engine.calls, [("pause", False, True), ("resume",)])

    async def test_abort_failure_propagates_before_weight_update(self):
        class BrokenEngine(LegacyEngine):
            async def abort(self, request_id):
                raise RuntimeError("fixture abort failed")

        engine = BrokenEngine()
        with self.assertRaisesRegex(RuntimeError, "fixture abort failed"):
            await self.fixture(engine).abort_all_requests()
        self.assertEqual(engine.calls, [])

    async def test_sleep_preserves_level_and_optional_encoder_reset(self):
        for engine, tail in [(LegacyEngine(), []), (CurrentEngine(), [("encoder",)])]:
            obj = self.fixture(engine)
            await obj._sleep_hybrid()
            self.assertEqual(engine.calls, [("sleep", 2), *tail])
            engine.calls.clear()
            obj.lora_as_adapter = True
            await obj._sleep_hybrid()
            self.assertEqual(engine.calls, [("sleep", 1), *tail])


if __name__ == "__main__":
    unittest.main()
