"""Diagnose split-versus-uninterrupted greedy decisions using native top-5 scores.

Candidate logprobs are pre-temperature model scores, not the delta distribution
used by greedy sampling. They diagnose close decisions, not full-vocabulary TV.
"""

import argparse
import hashlib
import json
import signal
from dataclasses import asdict
from pathlib import Path

from probe_vllm_composition import Engine, gpu_snapshot
from rollout_composition import Call, Request, stable_seed

ROOT = Path(__file__).resolve().parents[1]


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned diagnostic received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.gpu, args.memory_fraction, args.context, args.mode = 0, 0.1, 5120, "correctness"
    before = gpu_snapshot(0)
    assert before["total_mib"] - before["used_mib"] > before["total_mib"] * 0.1 + 4096
    prior = json.loads((ROOT / "runs/composition-probe-20261018-correctness-v4/metadata.json").read_text())
    records = [
        json.loads(x)
        for x in (ROOT / "runs/rollout-sources/paired-schedule/prompts.jsonl").read_text().splitlines()[:16]
    ]
    requests = [Request(str(r["prompt_id"]), tuple(r["prompt_ids"])) for r in records]
    engine = Engine("student", Path(prior["models"]["student"]), args, prior["vocab_size"], prior["eos_token_id"])
    raw = []

    def run(calls):
        result = engine.call("generate", calls=calls, controls={"diagnostic_logprobs": True})
        raw.append(
            {
                "calls": [asdict(c) for c in calls],
                "outputs": [asdict(r) for r in result["records"]],
                "candidate_logprobs": result["diagnostic_logprobs"],
            }
        )
        return result

    try:
        full = run([Call(r.key, r.prompt_ids, 128, stable_seed(42, r.key, "student", 0)) for r in requests])
        first = run([Call(r.key, r.prompt_ids, 64, 42) for r in requests])
        continuation = [
            Call(r.key, r.prompt_ids + out.token_ids, 128 - len(out.token_ids), 42)
            for r, out in zip(requests, first["records"], strict=True)
            if out.finish_reason == "length" and out.token_ids[-1] != prior["eos_token_id"]
        ]
        second = run(continuation)
        suffixes = {r.key: (r, lp) for r, lp in zip(second["records"], second["diagnostic_logprobs"], strict=True)}
        differences = []
        equal = 0
        for i, (direct, start) in enumerate(zip(full["records"], first["records"], strict=True)):
            suffix, suffix_lp = suffixes.get(start.key, (None, []))
            combined = start.token_ids + (suffix.token_ids if suffix else ())
            if combined == direct.token_ids:
                equal += 1
                continue
            at = next(j for j, (a, b) in enumerate(zip(direct.token_ids, combined, strict=False)) if a != b)
            a, b = direct.token_ids[at], combined[at]
            lp_a = full["diagnostic_logprobs"][i][at]
            combined_lp = first["diagnostic_logprobs"][i] + suffix_lp
            lp_b = combined_lp[at]
            common = set(lp_a) & set(lp_b)
            differences.append(
                {
                    "response_index": i,
                    "first_divergence": at,
                    "shared_input_token_prefix_verified": direct.prompt_ids + direct.token_ids[:at]
                    == start.prompt_ids + combined[:at],
                    "direct_token": a,
                    "split_token": b,
                    "direct_logprob_a": lp_a.get(a),
                    "direct_logprob_b": lp_a.get(b),
                    "split_logprob_a": lp_b.get(a),
                    "split_logprob_b": lp_b.get(b),
                    "direct_candidate_log_odds_gap": lp_a[a] - lp_a[b] if b in lp_a else None,
                    "split_candidate_log_odds_gap": lp_b[b] - lp_b[a] if a in lp_b else None,
                    "max_common_top5_logprob_difference": max(abs(lp_a[t] - lp_b[t]) for t in common),
                }
            )
        result = {
            "scope": "Native BF16 top-5 numerical diagnostic; no full-vocabulary error bound is claimed.",
            "responses": len(requests),
            "bitwise_equal_responses": equal,
            "differences": differences,
            "vllm_enable_v1_multiprocessing": False,
            "eager": True,
            "prefix_caching": False,
            "baseline": before,
            "engine": engine.ready,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(), ROOT / "eval/probe_vllm_composition.py"]
            },
        }
        (args.output / "numerics.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "raw-scores.json").write_text(json.dumps(raw) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        engine.close()
        (args.output / "cleanup.json").write_text(json.dumps(gpu_snapshot(0), indent=2) + "\n")


if __name__ == "__main__":
    main()
