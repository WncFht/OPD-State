"""Check installed scheduler and CPU trace interfaces, without GPU inference."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from native_profile_metrics import scheduled_work, trace_metrics


class NativeProfileRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertFalse(torch.cuda.is_initialized())

    def test_installed_scheduler_schema_supports_chunked_and_cached_queries(self):
        new = NewRequestData(
            req_id="prefill",
            prompt_token_ids=[1] * 5,
            mm_inputs=[],
            mm_hashes=[],
            mm_positions=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=3,
            lora_request=None,
        )
        cached = CachedRequestData(
            req_ids=["decode"],
            resumed_from_preemption=[False],
            new_token_ids=[[]],
            new_block_ids=[([],)],
            num_computed_tokens=[5],
        )
        output = SchedulerOutput(
            scheduled_new_reqs=[new],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"prefill": 2, "decode": 1},
            total_num_scheduled_tokens=3,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        row = scheduled_work(output, {"decode": SimpleNamespace(prompt_token_ids=[1] * 5)})
        self.assertEqual((row["phase"], row["prefill_queries"], row["decode_queries"]), ("mixed", 2, 1))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_cpu_profiler_exports_marker_but_cannot_pass_cuda_acceptance(self):
        values = torch.ones((16, 16))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            with torch.profiler.record_function("OPD_STEP_0_prefill"):
                result = values @ values
        self.assertEqual(float(result.sum()), 4096)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-trace.json"
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
        markers = [event for event in trace["traceEvents"] if event.get("name") == "OPD_STEP_0_prefill"]
        self.assertEqual(len(markers), 1)
        self.assertIn("External id", markers[0]["args"])
        with self.assertRaisesRegex(ValueError, "No actual CUDA kernel"):
            trace_metrics(trace, [{"phase": "prefill", "prefill_queries": 16, "decode_queries": 0}])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
