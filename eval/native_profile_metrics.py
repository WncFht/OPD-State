"""Classify actual scheduled queries and attribute Kineto kernels by launch IDs.

GPU timestamps may follow their CPU launch scope. Wall-clock containment of a
GPU kernel inside a CPU scope is deliberately not used for attribution.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict


def scheduled_work(output, requests):
    states = {row.req_id: (len(row.prompt_token_ids), row.num_computed_tokens) for row in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.num_computed_tokens):
        raise ValueError("Scheduled cached request positions do not match their identities")
    for uid, computed in zip(cached.req_ids, cached.num_computed_tokens, strict=True):
        states[uid] = len(requests[uid].prompt_token_ids), computed
    prompt, decode, rows = 0, 0, []
    for uid, count in output.num_scheduled_tokens.items():
        length, computed = states[uid]
        if any(type(value) is not int or value < 0 for value in (length, computed, count)) or length == 0:
            raise ValueError("Scheduled query counts must be nonnegative and prompts nonempty")
        prefill = min(count, max(0, length - computed))
        prompt += prefill
        decode += count - prefill
        rows.append({"request_id": uid, "prompt_length": length, "computed": computed, "queries": count})
    if prompt + decode != output.total_num_scheduled_tokens:
        raise ValueError("Scheduled query classification does not conserve total work")
    phase = "mixed" if prompt and decode else "prefill" if prompt else "decode" if decode else "empty"
    return {"phase": phase, "prefill_queries": prompt, "decode_queries": decode, "requests": rows}


def union_us(intervals):
    result, right = 0.0, None
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("GPU event intervals must be finite and ordered")
        result += end - start if right is None or start >= right else max(0, end - right)
        right = end if right is None else max(right, end)
    return result


def trace_metrics(trace, steps):
    events = trace["traceEvents"]
    markers = {}
    for event in events:
        match = re.fullmatch(r"OPD_STEP_(\d+)_(prefill|decode|mixed|empty)", event.get("name", ""))
        if not match:
            continue
        number = int(match[1])
        if event.get("ph") != "X" or number in markers or event.get("dur", -1) < 0:
            raise ValueError("Profiler contains duplicate or incomplete scheduled-step markers")
        markers[number] = event
        if number >= len(steps) or match[2] != steps[number]["phase"]:
            raise ValueError("Profiler step markers disagree with actual scheduler records")
    if set(markers) != set(range(len(steps))):
        raise ValueError("Profiler trace does not contain every recorded scheduler step")

    links = {key: defaultdict(set) for key in ("External id", "correlation")}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") == "kernel":
            continue
        if "ts" not in event or "dur" not in event:
            continue
        owners = {
            number
            for number, marker in markers.items()
            if event.get("pid") == marker.get("pid")
            and event.get("tid") == marker.get("tid")
            and marker["ts"] <= event["ts"]
            and event["ts"] + event["dur"] <= marker["ts"] + marker["dur"] + 1e-6
        }
        for key, table in links.items():
            value = event.get("args", {}).get(key)
            if type(value) is int and value > 0:
                table[value].update(owners)

    assigned, unknown, kernels = defaultdict(list), [], []
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        if event.get("dur", -1) < 0 or not math.isfinite(event["dur"]) or not math.isfinite(event["ts"]):
            raise ValueError("GPU kernel trace has invalid timestamps")
        kernels.append(event)
        owners = set()
        for key, table in links.items():
            owners.update(table.get(event.get("args", {}).get(key), ()))
        if len(owners) == 1:
            assigned[next(iter(owners))].append(event)
        else:
            unknown.append({"name": event["name"], "candidate_steps": sorted(owners)})
    if not kernels:
        raise ValueError("No actual CUDA kernel events were captured")

    def totals(values):
        by_device = defaultdict(list)
        for event in values:
            device = event.get("args", {}).get("device")
            if type(device) is not int:
                raise ValueError("GPU kernel trace lacks its actual device identity")
            by_device[device].append((event["ts"], event["ts"] + event["dur"]))
        return {
            "kernels": len(values),
            "kernel_duration_sum_us": math.fsum(event["dur"] for event in values),
            "kernel_busy_union_us_by_device": {
                str(device): union_us(intervals) for device, intervals in by_device.items()
            },
        }

    per_step = []
    for number, row in enumerate(steps):
        per_step.append({**row, "cpu_execute_scope_us": markers[number]["dur"], **totals(assigned[number])})
    phases = {}
    for phase in ("prefill", "decode", "mixed", "empty"):
        selected = [i for i, row in enumerate(steps) if row["phase"] == phase]
        phases[phase] = {
            "steps": len(selected),
            "prefill_queries": sum(steps[i]["prefill_queries"] for i in selected),
            "decode_queries": sum(steps[i]["decode_queries"] for i in selected),
            **totals([event for i in selected for event in assigned[i]]),
        }
    names = defaultdict(lambda: {"kernels": 0, "kernel_duration_sum_us": 0.0})
    for event in kernels:
        names[event["name"]]["kernels"] += 1
        names[event["name"]]["kernel_duration_sum_us"] += event["dur"]
    return {
        "scope": "Actual CUDA kernel events correlated to CPU launches; memcpy/memset excluded",
        "complete_attribution": not unknown,
        "unattributed_kernels": unknown,
        "overall": totals(kernels),
        "by_phase": phases,
        "by_step": per_step,
        "kernel_names": dict(names),
        "limitations": [
            "Profiler instrumentation changes runtime; this is not an unprofiled throughput benchmark",
            "Concurrent kernel durations overlap; duration sums are not GPU wall time",
            "Mixed scheduled iterations are retained, without proportional prefill/decode attribution",
            "CUDA-event stream envelopes include dependencies and idle gaps, beyond kernel execution",
        ],
    }
