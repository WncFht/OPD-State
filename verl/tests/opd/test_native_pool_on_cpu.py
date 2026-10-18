"""Check actual pool routing, cancellation ownership and run isolation."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2] / "verl/workers/rollout"
NAME = "native_pool_contract"
package = types.ModuleType(NAME)
package.__path__ = [str(ROOT)]
sys.modules[NAME] = package


def load(name):
    spec = importlib.util.spec_from_file_location(NAME + "." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = load("native_pool")
E = sys.modules[NAME + ".native_executor"]
S = load("sync_endpoint")
Policy = sys.modules[NAME + ".policy"]


class RPC:
    def __init__(self, fn):
        self.remote = fn


class Server:
    def __init__(self, wait=None):
        self.calls = []
        self.wait = wait
        self.finished = False
        self.generate_native_batch = RPC(self.generate)

    async def generate(self, calls, identity, params):
        self.calls.append((calls, identity, params))
        if self.wait:
            await self.wait.wait()
        self.finished = True
        return [
            {
                **c,
                "token_ids": [c["seed"]],
                "identity": identity,
                "finish_reason": "length",
                "stop_reason": None,
                "log_probs": None,
            }
            for c in calls
        ]


class NativePoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_routes_survive_replica_sharding_and_empty_replicas(self):
        servers = [Server() for _ in range(4)]
        pool = NS(handles={"teacher": servers})
        backend = P.NativeRayBackend(pool, {"temperature": 1})
        ref = Policy.ModelRef("teacher", "/q", fingerprint="sha")
        calls = [E.Call(str(i), (str(i), 0, 0), (1,), 2, i) for i in range(6)]
        stage = Policy.Stage(("teacher",), 2)
        out = await backend.generate(stage, calls, [ref])
        self.assertEqual([c.request_id for c in out], [str(i) for i in range(6)])
        self.assertEqual([c.token_ids for c in out], [(i,) for i in range(6)])
        self.assertEqual([len(s.calls[0][0]) for s in servers], [2, 2, 1, 1])
        for s in servers:
            s.calls.clear()
        await backend.generate(stage, calls[:1], [ref])
        self.assertEqual([len(s.calls) for s in servers], [1, 0, 0, 0])

    async def test_cancellation_keeps_native_work_owned_until_drain(self):
        stop = asyncio.Event()
        server = Server(stop)

        async def drain():
            stop.set()
            return True

        pool = NS(handles={"teacher": [server]}, abort_and_drain=drain)
        backend = P.NativeRayBackend(pool, {})
        ref = Policy.ModelRef("teacher", "/q", fingerprint="sha")
        call = E.Call("a", ("a", 0, 0), (1,), 2, 1)
        task = asyncio.create_task(backend.generate(Policy.Stage(("teacher",), 2), [call], [ref]))
        while not server.calls:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(server.finished)
        self.assertTrue(backend.pending)
        self.assertTrue(await backend.abort_and_drain(["a"]))
        self.assertTrue(server.finished)
        self.assertFalse(backend.pending)


class NativeEndpointTests(unittest.TestCase):
    def test_separate_ray_clusters_with_equal_job_ids_have_distinct_addresses(self):
        a = S.weight_sync_endpoint("01000000", 0, 0, "a" * 32)
        b = S.weight_sync_endpoint("01000000", 0, 0, "b" * 32)
        self.assertNotEqual(a, b)
        self.assertLess(len(a.removeprefix("ipc://").encode()), 108)
        self.assertEqual(
            S.weight_sync_endpoint("01000000", 2, 0), "ipc:///tmp/rl-colocate-zmq-01000000-replica-2-rank-0.sock"
        )
        for invalid in ["a/b", "x" * 41, "中文", "a\x00"]:
            with self.assertRaises(ValueError):
                S.weight_sync_endpoint("01000000", 0, 0, invalid)


if __name__ == "__main__":
    unittest.main()
