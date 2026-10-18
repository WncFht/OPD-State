"""Validate experimental staged dispatch against the actual token executor.

These are protocol/lifecycle tests, not proof of CUDA transfers or model logits.
"""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "verl/workers/rollout"
NAME = "native_staged_contract"
package = types.ModuleType(NAME)
package.__path__ = [str(ROOT)]
sys.modules[NAME] = package


def load(name):
    spec = importlib.util.spec_from_file_location(NAME + "." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S = load("native_staged")
E = load("native_executor")
N = load("native_pool")
P = sys.modules[NAME + ".policy"]
REFS = {role: P.ModelRef(role, "/" + role, fingerprint="sha-" + role) for role in ("initial_student", "teacher")}


class Delegate:
    def __init__(self):
        self.execution_audit, self.placement_audit, self.verifications = [], [], []
        self.calls = []
        self.role = None
        self.eos = False
        self.wrong_identity = False
        self.drained = False

    async def prepare_phase(self, stages):
        self.placement_audit.append(stages)

    async def generate(self, stage, calls, refs):
        self.calls.append((self.role, stage, calls, refs))
        await asyncio.sleep(0)
        identity = refs[0].identity if not self.wrong_identity else ("current_student", "/container", 999)
        token = 9 if self.eos and self.role == "initial_student" else 10 if self.role == "initial_student" else 20
        return [
            E.Completion(
                call.request_id,
                call.prompt_ids,
                (token,) * call.max_tokens if token != 9 else (10, 9),
                (identity,),
                "length",
            )
            for call in calls
        ]

    async def abort_and_drain(self, ids):
        self.drained = True
        return True


class StagedTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_native_ray_backend_preserves_physical_receipts_through_logical_dispatch(self):
        active = {"role": None, "version": -1}

        class Server:
            def __init__(self, rank):
                async def generate(calls, identity, sampling):
                    del sampling
                    return [
                        {
                            **call,
                            "token_ids": [10 if active["role"] == "initial_student" else 20] * call["max_tokens"],
                            "identity": identity,
                            "finish_reason": "length",
                            "stop_reason": None,
                            "log_probs": None,
                            "execution": {"replica_rank": rank, "device_uuid": f"unit-card-{rank}"},
                        }
                        for call in calls
                    ]

                self.generate_native_batch = types.SimpleNamespace(remote=generate)

        class Pool:
            layout = "overlap"
            handles = {"current_student": [Server(0), Server(1)]}

            async def prepare(self, slots):
                self.selection = slots

        delegate = N.NativeRayBackend(Pool(), {"temperature": 0})

        async def activate(ref):
            active.update(role=ref.role, version=active["version"] + 1)
            return {
                "source_identity": ref.identity,
                "runtime_ref": P.ModelRef("current_student", "/container", version=active["version"]),
                "committed": True,
            }

        backend = S.StagedNativeBackend(delegate, activate, "/container")
        policy = {
            "kind": "batch_mix",
            "components": [{"policy": {"kind": "single", "model": role}, "weight": 0.5} for role in REFS],
        }
        outputs, _ = await self.execute(backend, policy)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(backend.execution_audit), 4)
        self.assertEqual({row["logical_phase"] for row in backend.execution_audit}, {0})
        self.assertEqual({row["logical_slot"] for row in backend.execution_audit}, set(REFS))
        self.assertTrue(
            all(
                row["slot"] == "current_student" and row["logical_identity"] == REFS[row["logical_slot"]].identity
                for row in backend.execution_audit
            )
        )
        self.assertEqual({row["replica_rank"] for row in backend.execution_audit}, {0, 1})
        self.assertEqual(sum(row["requests"] for row in backend.execution_audit), 4)

    def fixture(self, corrupt=None):
        delegate = Delegate()
        publications = []

        async def activate(ref):
            delegate.role = ref.role
            publications.append(ref.identity)
            receipt = {
                "source_identity": ref.identity,
                "runtime_ref": P.ModelRef("current_student", "/container", version=len(publications)),
                "committed": True,
            }
            if corrupt:
                corrupt(receipt)
            return receipt

        return S.StagedNativeBackend(delegate, activate, "/container"), delegate, publications

    async def execute(self, backend, policy, size=4):
        requests = [E.Request(str(i), (1, 2), sample=i) for i in range(size)]
        return await E.generate(
            requests,
            P.compile_policy(policy, cap=4, batch_size=size, seed=42, batch_id=1),
            backend,
            refs=REFS,
            context_limits={role: 20 for role in REFS},
            vocab_size=30,
            eos_token_id=9,
            seed=42,
            concurrent_groups=True,
        )

    async def test_single_switch_and_mixture_keep_exact_prefixes_source_labels_and_quotas(self):
        for first, second in (("initial_student", "teacher"), ("teacher", "initial_student")):
            backend, delegate, publications = self.fixture()
            outputs, _ = await self.execute(
                backend,
                {
                    "kind": "sequence_switch",
                    "first": first,
                    "second": second,
                    "boundary": {"kind": "response_tokens", "value": 2},
                },
            )
            expected = (10, 10, 20, 20) if first == "initial_student" else (20, 20, 10, 10)
            self.assertTrue(all(row.token_ids == expected for row in outputs))
            self.assertEqual(publications, [REFS[first].identity, REFS[second].identity])
            self.assertEqual(delegate.calls[1][2][0].prompt_ids, (1, 2) + expected[:2])
            self.assertEqual(
                [segment.identities for segment in outputs[0].segments],
                [(REFS[first].identity,), (REFS[second].identity,)],
            )
        backend, delegate, _ = self.fixture()
        policy = {
            "kind": "batch_mix",
            "components": [{"policy": {"kind": "single", "model": role}, "weight": 0.5} for role in REFS],
        }
        outputs, _ = await self.execute(backend, policy, 5)
        self.assertEqual(sum(row.token_ids == (10,) * 4 for row in outputs), 3)
        self.assertEqual(sum(row.token_ids == (20,) * 4 for row in outputs), 2)
        self.assertEqual([row.request.prompt_id for row in outputs], list(map(str, range(5))))
        self.assertEqual([call[1].models for call in delegate.calls], [("current_student",), ("current_student",)])

    async def test_eos_at_boundary_never_activates_second_checkpoint(self):
        backend, delegate, publications = self.fixture()
        delegate.eos = True
        outputs, _ = await self.execute(
            backend,
            {
                "kind": "sequence_switch",
                "first": "initial_student",
                "second": "teacher",
                "boundary": {"kind": "response_tokens", "value": 2},
            },
        )
        self.assertEqual(publications, [REFS["initial_student"].identity])
        self.assertTrue(all(row.token_ids == (10, 9) and row.termination == "eos" for row in outputs))

    async def test_incomplete_wrong_source_and_regressed_version_are_rejected_and_drained(self):
        corruptions = [
            lambda r: r.update(committed=False),
            lambda r: r.update(source_identity=REFS["teacher"].identity),
            lambda r: r.update(runtime_ref=P.ModelRef("current_student", "/wrong", version=1)),
        ]
        for corrupt in corruptions:
            backend, delegate, _ = self.fixture(corrupt)
            with self.assertRaisesRegex(RuntimeError, "publication"):
                await self.execute(backend, {"kind": "single", "model": "initial_student"})
            self.assertTrue(delegate.drained)
        backend, delegate, _ = self.fixture(
            lambda r: r.update(runtime_ref=P.ModelRef("current_student", "/container", version=0))
        )
        with self.assertRaisesRegex(RuntimeError, "new committed"):
            await self.execute(
                backend,
                {
                    "kind": "sequence_switch",
                    "first": "initial_student",
                    "second": "teacher",
                    "boundary": {"kind": "response_tokens", "value": 2},
                },
            )
        self.assertTrue(delegate.drained)

    async def test_native_version_is_checked_before_logical_identity_translation(self):
        backend, delegate, _ = self.fixture()
        delegate.wrong_identity = True
        with self.assertRaisesRegex(RuntimeError, "different mutable"):
            await self.execute(backend, {"kind": "single", "model": "teacher"})
        self.assertTrue(delegate.drained)

    async def test_cancellation_waits_for_real_publication_before_drain_ack(self):
        delegate = Delegate()
        entered, release = asyncio.Event(), asyncio.Event()

        async def activate(ref):
            entered.set()
            await release.wait()
            return {
                "source_identity": ref.identity,
                "runtime_ref": P.ModelRef("current_student", "/container", version=1),
                "committed": True,
            }

        backend = S.StagedNativeBackend(delegate, activate, "/container")
        task = asyncio.create_task(self.execute(backend, {"kind": "single", "model": "teacher"}))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(delegate.drained)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(delegate.drained)

    async def test_fusion_and_online_student_are_rejected_before_activation(self):
        backend, _, publications = self.fixture()
        for models in (("initial_student", "teacher"), ("current_student",)):
            with self.assertRaises(ValueError):
                await backend.prepare_phase(((P.Stage(models, 4), 4),))
        self.assertEqual(publications, [])


if __name__ == "__main__":
    unittest.main()
