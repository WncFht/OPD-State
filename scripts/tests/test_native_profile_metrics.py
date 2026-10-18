"""Check scheduling boundaries and causal kernel attribution without CUDA."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as Obj

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from native_profile_metrics import scheduled_work, trace_metrics, union_us
from run_native_profile_controls import cases, command


def event(name, ts, dur, *, cat="cpu_op", tid=1, args=None):
    return {
        "name": name,
        "ts": ts,
        "dur": dur,
        "cat": cat,
        "tid": tid,
        "pid": 1,
        "ph": "X",
        "args": args or {},
    }


def fixture():
    steps = [
        {"phase": "prefill", "prefill_queries": 8, "decode_queries": 0},
        {"phase": "decode", "prefill_queries": 0, "decode_queries": 2},
    ]
    events = [
        event("OPD_STEP_0_prefill", 0, 10, cat="user_annotation"),
        event("cudaLaunchKernel", 2, 1, cat="cuda_runtime", args={"correlation": 101}),
        event("OPD_STEP_1_decode", 20, 10, cat="user_annotation"),
        event("aten::mm", 21, 4, args={"External id": 22}),
        event(
            "prefill_kernel",
            100,
            10,
            cat="kernel",
            tid=7,
            args={"correlation": 101, "device": 0},
        ),
        event(
            "decode_kernel",
            110,
            7,
            cat="kernel",
            tid=7,
            args={"External id": 22, "device": 0},
        ),
        event(
            "decode_kernel",
            112,
            8,
            cat="kernel",
            tid=8,
            args={"External id": 22, "device": 0},
        ),
        event(
            "memcpy",
            130,
            10,
            cat="gpu_memcpy",
            tid=8,
            args={"External id": 22, "device": 0},
        ),
    ]
    return {"traceEvents": events}, steps


class NativeProfileMetricsTests(unittest.TestCase):
    def test_profile_matrix_keeps_stack_work_and_roles_explicit(self):
        matrix = cases()
        self.assertEqual(len(matrix), 10)
        for graph in (False, True):
            group = [case for case in matrix if case["graph"] == graph]
            self.assertEqual(
                {(case["role"], case["batch"], case["token_budget"]) for case in group},
                {
                    ("initial_student", 4, 8192),
                    ("initial_student", 16, 8192),
                    ("teacher", 16, 8192),
                    ("fusion", 16, 8192),
                    ("initial_student", 16, 64),
                },
            )
            for case in group:
                cmd = command(
                    case, Path("out"), "commit", Path("prompts"), Path("python")
                )
                self.assertEqual(cmd[cmd.index("--cap") + 1], "32")
                self.assertEqual("--graph" in cmd, graph)
                self.assertEqual(cmd[cmd.index("--source-commit") + 1], "commit")
                self.assertEqual(
                    int(cmd[cmd.index("--token-budget") + 1]), case["token_budget"]
                )

    def test_chunk_boundary_and_mixed_batch_use_actual_computed_positions(self):
        cached = Obj(req_ids=["decode", "tail"], num_computed_tokens=[10, 7])
        output = Obj(
            scheduled_new_reqs=[
                Obj(req_id="new", prompt_token_ids=[1] * 6, num_computed_tokens=0)
            ],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"new": 3, "decode": 1, "tail": 3},
            total_num_scheduled_tokens=7,
        )
        requests = {
            "decode": Obj(prompt_token_ids=[1] * 8),
            "tail": Obj(prompt_token_ids=[1] * 8),
        }
        row = scheduled_work(output, requests)
        self.assertEqual(
            (row["phase"], row["prefill_queries"], row["decode_queries"]),
            ("mixed", 4, 3),
        )
        output.total_num_scheduled_tokens = 8
        with self.assertRaisesRegex(ValueError, "conserve"):
            scheduled_work(output, requests)

    def test_empty_and_single_query_prefill_are_not_mislabeled_decode(self):
        output = Obj(
            scheduled_new_reqs=[
                Obj(req_id="tail", prompt_token_ids=[1] * 8, num_computed_tokens=7)
            ],
            scheduled_cached_reqs=Obj(req_ids=[], num_computed_tokens=[]),
            num_scheduled_tokens={"tail": 1},
            total_num_scheduled_tokens=1,
        )
        self.assertEqual(scheduled_work(output, {})["phase"], "prefill")
        output.num_scheduled_tokens, output.total_num_scheduled_tokens = {}, 0
        self.assertEqual(scheduled_work(output, {})["phase"], "empty")

    def test_asynchronous_kernels_follow_launch_links_and_overlap_is_not_added(self):
        trace, steps = fixture()
        result = trace_metrics(trace, steps)
        self.assertTrue(result["complete_attribution"])
        self.assertEqual(result["by_phase"]["prefill"]["kernel_duration_sum_us"], 10)
        self.assertEqual(result["by_phase"]["decode"]["kernel_duration_sum_us"], 15)
        self.assertEqual(
            result["by_phase"]["decode"]["kernel_busy_union_us_by_device"], {"0": 10}
        )
        self.assertEqual(result["overall"]["kernels"], 3)
        self.assertEqual(result["overall"]["kernel_busy_union_us_by_device"], {"0": 20})

    def test_missing_or_ambiguous_launch_links_are_retained_as_unattributed(self):
        for ambiguity in (False, True):
            trace, steps = fixture()
            if ambiguity:
                trace["traceEvents"].append(
                    event("other_launch", 22, 1, args={"correlation": 101})
                )
            else:
                trace["traceEvents"][4]["args"]["correlation"] = 999
            result = trace_metrics(trace, steps)
            self.assertFalse(result["complete_attribution"])
            self.assertEqual(len(result["unattributed_kernels"]), 1)
            self.assertEqual(result["by_phase"]["prefill"]["kernels"], 0)

    def test_mixed_iteration_keeps_its_kernels_in_a_separate_group(self):
        trace, steps = fixture()
        steps[0].update(phase="mixed", decode_queries=2)
        trace["traceEvents"][0]["name"] = "OPD_STEP_0_mixed"
        result = trace_metrics(trace, steps)
        self.assertEqual(result["by_phase"]["mixed"]["prefill_queries"], 8)
        self.assertEqual(result["by_phase"]["mixed"]["decode_queries"], 2)
        self.assertEqual(result["by_phase"]["mixed"]["kernel_duration_sum_us"], 10)
        self.assertEqual(result["by_phase"]["prefill"]["kernels"], 0)

    def test_cpu_only_trace_missing_marker_or_wrong_phase_cannot_pass(self):
        base, steps = fixture()
        for change in ("no_gpu", "missing", "phase"):
            trace = copy.deepcopy(base)
            if change == "no_gpu":
                trace["traceEvents"] = [
                    e for e in trace["traceEvents"] if e["cat"] != "kernel"
                ]
            elif change == "missing":
                trace["traceEvents"].pop(0)
            else:
                trace["traceEvents"][0]["name"] = "OPD_STEP_0_decode"
            with self.assertRaises(ValueError):
                trace_metrics(trace, steps)

    def test_union_handles_nested_adjacent_and_invalid_intervals(self):
        self.assertEqual(union_us([(2, 3), (0, 5), (5, 8), (12, 13)]), 9)
        self.assertEqual(union_us([]), 0)
        with self.assertRaises(ValueError):
            union_us([(2, 1)])


if __name__ == "__main__":
    unittest.main()
