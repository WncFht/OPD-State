"""Exercise production server RPC ordering with a controlled async engine."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2] / "verl/workers/rollout"
NAME = "native_rpc_contract"
for suffix, path in [("", ROOT), (".vllm_rollout", ROOT / "vllm_rollout")]:
    package = types.ModuleType(NAME + suffix)
    package.__path__ = [str(path)]
    sys.modules[NAME + suffix] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(NAME + "." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load("vllm_rollout.native_server", ROOT / "vllm_rollout/native_server.py")
W = load("vllm_rollout.native_weights", ROOT / "vllm_rollout/native_weights.py")


class Config(dict):
    __getattr__ = dict.__getitem__


class Engine:
    def __init__(self):
        self.calls = []
        self.output_processor = NS(request_states={})
        self.version = 0
        self.write_serial = 0
        self.frozen_digests = {}
        self.parameter_addresses = "stable-addresses"
        self.done = asyncio.Event()

    async def abort(self, request_id):
        self.calls.append(("abort", request_id))
        self.done.set()

    async def collective_rpc(self, method, **kwargs):
        self.calls.append((method,))
        if method == "native_source_status":
            return [
                {
                    "actor_version": self.version,
                    "write_serial": self.write_serial,
                    "frozen_digests": dict(self.frozen_digests),
                    "parameter_addresses": self.parameter_addresses,
                }
            ]
        if method == "native_request_barrier":
            return [{"native_device_idle": True}]
        args = kwargs["kwargs"]
        fusion = args.get("fusion")
        return [
            {
                "role": args["role"],
                "worker_execution": {
                    "device_uuid": "unit-gpu-0",
                    "local_device": 0,
                    "pid": 100,
                    "visible_devices": "0",
                },
                "path": args["path"],
                "fusion": fusion,
                "companion": ("teacher", fusion["teacher_path"], fusion["teacher_fingerprint"]) if fusion else None,
                "frozen_digests": dict(self.frozen_digests),
                "parameter_addresses": self.parameter_addresses if args.get("verify_addresses") else None,
            }
        ]


class Server(M.NativePolicyServerMixin):
    def __init__(self, role="teacher"):
        self.replica_rank = 0
        self.engine = Engine()
        self.config = Config(
            policy={"kind": "single", "model": role},
            generation_role=role,
            generation_fingerprint=None if role == "current_student" else "sha",
            max_model_len=10,
        )
        self.model_config = NS(path="/q" if role == "teacher" else "/p")
        self.native_state = None
        self.generated = []
        self.block = False
        self.reset_result = None

    async def _reset_prefix_cache(self):
        self.engine.calls.append(("reset",))
        return self.reset_result

    async def _generate_one(self, **kwargs):
        rid = kwargs["request_id"]
        self.generated.append(kwargs)
        self.engine.output_processor.request_states[rid] = True
        try:
            if self.block:
                await self.engine.done.wait()
            count = kwargs["sampling_params"].pop("max_tokens", 2)
            return NS(
                token_ids=[2] * count,
                log_probs=None,
                num_preempted=None,
                extra_fields={"native_finish_reason": "length", "native_stop_reason": None},
            )
        finally:
            self.engine.output_processor.request_states.pop(rid, None)


class NativeServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_physical_worker_identity_rejects_native_initialization(self):
        server = Server()
        original = server.engine.collective_rpc

        async def without_device(method, **kwargs):
            result = await original(method, **kwargs)
            result[0].pop("worker_execution", None)
            return result

        server.engine.collective_rpc = without_device
        with self.assertRaisesRegex(RuntimeError, "physical CUDA device"):
            await server.initialize_native_source()

    async def test_failed_native_barrier_keeps_idle_engine_closed_to_reuse(self):
        s = await self.fixture()
        original = s.engine.collective_rpc

        async def stalled_barrier(method, **kwargs):
            if method == "native_request_barrier":
                await asyncio.Event().wait()
            return await original(method, **kwargs)

        s.engine.collective_rpc = stalled_barrier
        with self.assertRaises(TimeoutError):
            await s.drain_native_requests(timeout=0.001)
        self.assertTrue(s.native_state.drain_pending)
        self.assertFalse(s.native_state.active)
        with self.assertRaises(RuntimeError):
            await s.reopen_native_generation()
        s.engine.collective_rpc = original
        self.assertTrue(await s.drain_native_requests())
        await s.reopen_native_generation()

    async def test_abort_deadline_includes_a_stalled_native_abort_rpc(self):
        s = await self.fixture()

        async def stalled(request_id):
            await asyncio.Event().wait()

        s.engine.abort = stalled
        s.native_state.admit("stalled", s.native_state.identity)
        with self.assertRaises(TimeoutError):
            await s.drain_native_requests(abort=True, timeout=0.001)
        self.assertFalse(s.native_state.accepting)
        self.assertTrue(s.native_state.active)
        self.assertNotIn(("native_request_barrier",), s.engine.calls)

    async def test_abort_joins_iterators_when_engine_does_not_wake_the_output_queue(self):
        s = await self.fixture()
        s.block = True

        async def silent_abort(request_id):
            s.engine.calls.append(("abort", request_id))

        s.engine.abort = silent_abort
        task = asyncio.create_task(s.generate([1], {}, "silent"))
        await asyncio.sleep(0)
        self.assertTrue(s.native_state.active)
        self.assertTrue(await s.drain_native_requests(abort=True))
        self.assertTrue(task.cancelled())
        self.assertFalse(s.native_state.active)
        self.assertFalse(s._native_generating_tasks)
        self.assertFalse(s.engine.output_processor.request_states)

    async def test_graph_receiver_refuses_changed_parameter_addresses_after_an_update(self):
        s = Server("current_student")
        s.config["enforce_eager"] = False
        await s.initialize_native_source()
        await s.begin_native_update(0)
        s.engine.write_serial = 1
        await s.commit_native_update(0)
        await s.begin_native_update(1)
        s.engine.version, s.engine.write_serial = 1, 2
        s.engine.parameter_addresses = "reallocated-addresses"
        with self.assertRaisesRegex(RuntimeError, "addresses changed"):
            await s.commit_native_update(1)
        self.assertFalse(s.native_state.valid)

    async def paired(self):
        s = Server("current_student")
        s.config["generation_fusion"] = {"teacher_path": "/q", "teacher_fingerprint": "sha", "teacher_weight": 0.5}
        s.engine.frozen_digests = {"teacher": "unchanged"}
        await s.initialize_native_source()
        await s.begin_native_update(0)
        s.engine.write_serial = 1
        await s.commit_native_update(0)
        return s

    async def test_pair_requires_both_identities_and_actual_coefficient_before_sampling(self):
        s = await self.paired()
        with self.assertRaisesRegex(RuntimeError, "companion"):
            await s.generate_native_batch([self.call()], ("current_student", "/p", 0), {})
        self.assertFalse(s.generated)
        rows = await s.generate_native_batch(
            [self.call()],
            ("current_student", "/p", 0),
            {},
            expected_companion=("teacher", "/q", "sha"),
            teacher_weight=0.5,
        )
        self.assertEqual(rows[0]["identities"], (("current_student", "/p", 0), ("teacher", "/q", "sha")))
        self.assertEqual(rows[0]["teacher_weight"], 0.5)
        self.assertEqual(rows[0]["frozen_digests"], {"teacher": "unchanged"})

    async def test_pair_rejects_frozen_parameter_mutation_during_actor_publication(self):
        s = await self.paired()
        await s.begin_native_update(1)
        s.engine.version, s.engine.write_serial = 1, 2
        s.engine.frozen_digests["teacher"] = "corrupted"
        with self.assertRaisesRegex(RuntimeError, "frozen model parameters"):
            await s.commit_native_update(1)
        self.assertFalse(s.native_state.valid)

    async def test_pair_rejects_mutated_frozen_weights_instead_of_publishing_a_batch(self):
        s = await self.paired()
        s.engine.frozen_digests["teacher"] = "corrupted"
        with self.assertRaisesRegex(RuntimeError, "frozen model parameters"):
            await s.generate_native_batch(
                [self.call()],
                ("current_student", "/p", 0),
                {},
                expected_companion=("teacher", "/q", "sha"),
                teacher_weight=0.5,
            )
        self.assertFalse(s.native_state.valid)

    async def fixture(self, role="teacher"):
        s = Server(role)
        await s.initialize_native_source()
        return s

    def call(self, rid="a", count=2):
        return {"request_id": rid, "prompt_ids": [1, 3], "max_tokens": count, "seed": 42}

    async def test_legacy_requests_bypass_the_new_identity_protocol(self):
        s = Server()
        s.config["policy"] = None
        await s.initialize_native_source()
        output = await s.generate([1], {"max_tokens": 2}, "legacy")
        self.assertEqual(output.token_ids, [2, 2])
        self.assertNotIn("generation_identity", output.extra_fields)

    async def test_native_batch_preserves_raw_reasons_identity_and_sampling_input(self):
        s = await self.fixture()
        params = {"temperature": 1.0, "top_p": 1.0}
        rows = await s.generate_native_batch([self.call()], s.native_state.identity, params)
        self.assertEqual(rows[0]["execution"]["device_uuid"], "unit-gpu-0")
        self.assertEqual(rows[0]["execution"]["replica_rank"], 0)
        self.assertEqual(rows[0]["identity"], ("teacher", "/q", "sha"))
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertIsNone(rows[0]["stop_reason"])
        self.assertEqual(params, {"temperature": 1.0, "top_p": 1.0})
        self.assertFalse(s.native_state.active)

    async def test_wrong_source_and_excessive_stage_budget_fail_before_generation(self):
        s = await self.fixture()
        with self.assertRaises(RuntimeError):
            await s.generate_native_batch([self.call()], ("initial_student", "/q", "sha"), {})
        for count in [0, 9]:
            with self.assertRaises(ValueError):
                await s.generate_native_batch([self.call(count=count)], s.native_state.identity, {})
        self.assertFalse(s.generated)

    async def test_actor_can_only_publish_the_worker_acknowledged_version(self):
        s = await self.fixture("current_student")
        await s.begin_native_update(0)
        with self.assertRaises(RuntimeError):
            await s.generate_native_batch([self.call()], ("current_student", "/p", 0), {})
        s.engine.write_serial = 1
        await s.commit_native_update(0)
        await s.begin_native_update(1)
        with self.assertRaises(RuntimeError):
            await s.commit_native_update(1)
        await s.fail_native_update()
        await s.begin_native_update(1)
        s.engine.version = 1
        s.engine.write_serial = 2
        await s.commit_native_update(1)
        self.assertEqual(s.native_state.identity, ("current_student", "/p", 1))

    async def test_same_version_restore_cannot_publish_an_old_worker_acknowledgement(self):
        s = await self.fixture("current_student")
        await s.begin_native_update(0)
        with self.assertRaises(RuntimeError):
            await s.commit_native_update(0)
        s.engine.write_serial = 1
        await s.commit_native_update(0)
        await s.begin_native_update(0)
        with self.assertRaises(RuntimeError):
            await s.commit_native_update(0)
        s.engine.write_serial = 2
        await s.commit_native_update(0)

    async def test_frozen_server_refuses_actor_stream_without_touching_worker(self):
        s = await self.fixture()
        n = len(s.engine.calls)
        with self.assertRaises(RuntimeError):
            await s.begin_native_update(1)
        self.assertEqual(len(s.engine.calls), n)

    async def test_abort_waits_for_generator_output_map_and_native_barrier(self):
        s = await self.fixture()
        s.block = True
        task = asyncio.create_task(s.generate([1], {}, "long"))
        await asyncio.sleep(0)
        self.assertTrue(s.native_state.active)
        with self.assertRaises(RuntimeError):
            await s.begin_native_update(1)
        self.assertTrue(await s.drain_native_requests(abort=True))
        await task
        self.assertFalse(s.native_state.active)
        self.assertFalse(s.engine.output_processor.request_states)
        self.assertEqual(s.engine.calls[-3:], [("abort", "long"), ("native_request_barrier",), ("reset",)])
        await s.reopen_native_generation()

    async def test_failed_actual_cache_reset_does_not_acknowledge_drain(self):
        s = await self.fixture()
        s.reset_result = False
        with self.assertRaises(RuntimeError):
            await s.drain_native_requests()
        self.assertFalse(s.native_state.accepting)

    async def test_zero_timeout_does_not_reuse_an_active_engine(self):
        s = await self.fixture()
        s.native_state.admit("stuck", s.native_state.identity)
        with self.assertRaises(TimeoutError):
            await s.drain_native_requests(timeout=0)
        self.assertFalse(s.native_state.accepting)


class NativeWeightTests(unittest.TestCase):
    def test_named_tensor_transaction_rejects_missing_duplicate_and_unknown_parameters(self):
        worker = NS(_native_generation_role="current_student", _native_expected_weight_names={"a", "b"})
        received = set()
        W.check_actor_stream(worker, ["a"], received, final=False)
        with self.assertRaises(RuntimeError):
            W.check_actor_stream(worker, [], received, final=True)
        for names in [["a"], ["unknown"]]:
            with self.assertRaises(RuntimeError):
                W.check_actor_stream(worker, names, received, final=False)
        W.check_actor_stream(worker, ["b"], received, final=True)

    def test_fixed_source_rejects_actor_stream_before_consuming_any_weights(self):
        for role in ["teacher", "initial_student"]:
            with self.assertRaises(RuntimeError):
                W.check_actor_stream(NS(_native_generation_role=role), ["a"], set(), final=True)

    def test_legacy_weight_loader_is_unchanged(self):
        W.check_actor_stream(NS(), ["legacy-name"], set(), final=True)


if __name__ == "__main__":
    unittest.main()
