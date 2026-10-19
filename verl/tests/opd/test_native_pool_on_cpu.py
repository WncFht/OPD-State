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
import opd_state.rollout.executor as E
S = load("sync_endpoint")
import opd_state.core.policy as Policy


class RPC:
    def __init__(self, fn):
        self.remote = fn


class Server:
    def __init__(self, wait=None, *, replica_rank=0):
        self.replica_rank = replica_rank
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
                "execution": {
                    "replica_rank": self.replica_rank,
                    "device_uuid": f"unit-gpu-{self.replica_rank}",
                    "pid": 100 + self.replica_rank,
                    "local_device": 0,
                    "visible_devices": str(self.replica_rank),
                },
            }
            for c in calls
        ]


class Replica:
    def __init__(self, slot, rank, awake, events):
        self.slot, self.rank, self.awake, self.events = slot, rank, awake, events
        self._server_handle = NS(native_source_status=RPC(self.status), reopen_native_generation=RPC(self.reopen))

    async def status(self):
        return {"awake": self.awake}

    async def reopen(self):
        if not self.awake:
            raise RuntimeError("Cannot admit requests on a sleeping replica")
        self.events.append((self.slot, self.rank, "reopen"))

    async def wake_up(self):
        self.awake = True
        self.events.append((self.slot, self.rank, "wake"))

    async def sleep(self):
        self.awake = False
        self.events.append((self.slot, self.rank, "sleep"))


class NativeActivationTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        events = []
        pool = P.NativeModelPool.__new__(P.NativeModelPool)
        pool.actor_slots = {"current_student", "eval:current_student"}
        pool.awake = set()
        pool.replicas = {
            role: [Replica(role, i, role == "current_student", events) for i in range(4)]
            for role in ["current_student", "teacher"]
        }
        pool.handles = {role: [r._server_handle for r in replicas] for role, replicas in pool.replicas.items()}
        return pool, events

    async def test_unused_current_replicas_sleep_and_only_selected_frozen_replicas_wake(self):
        pool, events = self.fixture()
        await pool.prepare({"current_student": (0, 1), "teacher": (2, 3)})
        self.assertEqual([r.awake for r in pool.replicas["current_student"]], [True, True, False, False])
        self.assertEqual([r.awake for r in pool.replicas["teacher"]], [False, False, True, True])
        self.assertEqual({e[:2] for e in events if e[2] == "wake"}, {("teacher", 2), ("teacher", 3)})
        self.assertEqual(
            {e[:2] for e in events if e[2] == "reopen"},
            {("current_student", 0), ("current_student", 1), ("teacher", 2), ("teacher", 3)},
        )
        await pool.sleep_all()
        self.assertFalse(any(r.awake for replicas in pool.replicas.values() for r in replicas))
        self.assertFalse(pool.awake)

    async def test_sleeping_current_requires_actor_publication_and_cannot_be_blindly_woken(self):
        pool, events = self.fixture()
        pool.replicas["current_student"][0].awake = False
        with self.assertRaisesRegex(RuntimeError, "fresh actor publication"):
            await pool.prepare({"current_student": (0,)})
        self.assertFalse(any(e[2] == "wake" for e in events))

    async def test_invalid_selection_fails_before_any_replica_mutation(self):
        for indices in [(), (-1,), (4,), (0, 0), (False,)]:
            pool, events = self.fixture()
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                await pool.prepare({"teacher": indices})
            self.assertEqual(events, [])
        with self.assertRaisesRegex(ValueError, "no initialized slot"):
            await pool.prepare({"missing": (0,)})
        self.assertEqual(events, [])


class NativePoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_split_backend_routes_only_selected_devices_and_keeps_original_order(self):
        handles = {role: [Server(replica_rank=i) for i in range(4)] for role in ["current_student", "teacher"]}
        prepared = []

        async def prepare(slots):
            prepared.append(slots)

        backend = P.NativeRayBackend(NS(handles=handles, layout="split", prepare=prepare), {})
        stages = {role: Policy.Stage((role,), 2) for role in handles}
        await backend.prepare_phase([(stages["current_student"], 7), (stages["teacher"], 1)])
        self.assertEqual(prepared, [{"current_student": (0, 1, 2), "teacher": (3,)}])
        self.assertEqual(backend.placement_audit[0]["requests"], {"current_student": 7, "teacher": 1})
        for role, count in [("current_student", 7), ("teacher", 1)]:
            ref = Policy.ModelRef(
                role, "/" + role, **({"version": 0} if role == "current_student" else {"fingerprint": "sha"})
            )
            calls = [E.Call(f"{role}-{i}", (str(i), 0, 0), (1,), 2, i) for i in range(count)]
            output = await backend.generate(stages[role], calls, [ref])
            self.assertEqual([r.request_id for r in output], [c.request_id for c in calls])
            self.assertEqual([r.token_ids for r in output], [(i,) for i in range(count)])
        self.assertEqual([len(s.calls[0][0]) if s.calls else 0 for s in handles["current_student"]], [3, 2, 2, 0])
        self.assertEqual([len(s.calls[0][0]) if s.calls else 0 for s in handles["teacher"]], [0, 0, 0, 1])
        self.assertEqual(
            {(r["slot"], r["replica_rank"]): r["requests"] for r in backend.execution_audit},
            {("current_student", 0): 3, ("current_student", 1): 2, ("current_student", 2): 2, ("teacher", 3): 1},
        )

    async def test_wrong_replica_receipt_cannot_be_accepted_as_the_selected_worker(self):
        backend = P.NativeRayBackend(NS(handles={"teacher": [Server(replica_rank=3)]}), {})
        ref = Policy.ModelRef("teacher", "/q", fingerprint="sha")
        with self.assertRaisesRegex(RuntimeError, "physical worker/replica"):
            await backend.generate(Policy.Stage(("teacher",), 2), [E.Call("a", ("a", 0, 0), (1,), 2, 1)], [ref])

    async def test_eval_always_uses_pure_current_all_devices_and_empty_phase_does_nothing(self):
        prepared = []

        async def prepare(slots):
            prepared.append(slots)

        slot = "eval:current_student"
        backend = P.NativeRayBackend(
            NS(
                handles={slot: [Server(replica_rank=i) for i in range(4)]},
                eval_slot=slot,
                layout="split",
                prepare=prepare,
            ),
            {},
            validate=True,
        )
        await backend.prepare_phase([])
        self.assertEqual(prepared, [])
        await backend.prepare_phase([(Policy.Stage(("current_student",), 2), 16)])
        self.assertEqual(prepared, [{slot: (0, 1, 2, 3)}])
        self.assertEqual(backend.placement_audit[0]["layout"], "overlap")
        with self.assertRaisesRegex(ValueError, "pure current"):
            await backend.prepare_phase([(Policy.Stage(("teacher",), 2), 1)])

    async def test_global_routes_survive_replica_sharding_and_empty_replicas(self):
        servers = [Server(replica_rank=i) for i in range(4)]
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
