"""Exercise real native server cancellation, CUDA IPC and APC weight changes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def server_class():
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class ProbeServer(vLLMHttpServer):
        async def full_logits(self, prompts, identity, request_prefix, vocab):
            import numpy as np

            async def one(index, ids):
                rid = f"{request_prefix}-{index}"
                self.native_state.admit(rid, tuple(identity))
                try:
                    stream = self.engine.generate(
                        TokensPrompt(prompt_token_ids=ids),
                        SamplingParams(max_tokens=1, temperature=0, logprobs=vocab),
                        rid,
                    )
                    final = None
                    async for output in stream:
                        final = output
                    scores = final.outputs[0].logprobs[0]
                    return {
                        "logits": np.array([scores[i].logprob for i in range(vocab)], dtype=np.float32),
                        "cached_tokens": final.num_cached_tokens,
                    }
                finally:
                    self.native_state.complete(rid)

            return await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))

        async def native_progress(self):
            return {
                "active": len(self.native_state.active),
                "frontend_requests": len(self.engine.output_processor.request_states),
                "output_tokens": sum(
                    len(r.detokenizer.output_token_ids)
                    for r in self.engine.output_processor.request_states.values()
                    if r.detokenizer is not None
                ),
            }

    return ProbeServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from native_source_snapshot import freeze

    source = freeze(ROOT, args.output, Path(__file__))
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VERL_USE_EXTERNAL_PLUGINS="none",
        RAY_ADDRESS="local",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        RAY_TMPDIR="/tmp/opdrt-" + uuid.uuid4().hex[:10],
    )
    import numpy as np
    import ray
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from safetensors import safe_open

    from verl.trainer.ppo.fixed_rollout import model_signature
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.sync_endpoint import weight_sync_endpoint
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    paths = [str(ROOT.parent / "models" / p) for p in ["DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B"]]
    tokenizer = hf_tokenizer(paths[0])
    vocab = len(tokenizer)
    records = [
        json.loads(x)
        for x in (ROOT / "runs/native-rollout-implementation/fresh-math-data-v2/prompts.jsonl")
        .read_text()
        .splitlines()[:2]
    ]
    prompts = [r["prompt_ids"] for r in records]
    fingerprints = [model_signature(p) for p in paths]
    with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    with open_dict(config):
        config.actor_rollout_ref.model.path = paths[0]
        rollout = config.actor_rollout_ref.rollout
        rollout.name = "vllm"
        rollout.policy = {"kind": "single", "model": "current_student"}
        rollout.tensor_model_parallel_size = 1
        rollout.gpu_memory_utilization = 0.12
        rollout.max_model_len = 5120
        rollout.prompt_length = 1024
        rollout.response_length = 4096
        rollout.max_num_batched_tokens = 2048
        rollout.max_num_seqs = 2
        rollout.enforce_eager = not args.graph
        rollout.enable_prefix_caching = True
        rollout.load_format = "auto"
        rollout.preserve_source_weights = True
        rollout.generation_verify_frozen = True
        rollout.generation_purpose = "training"
        rollout.generation_sync_namespace = uuid.uuid4().hex
        rollout.logprobs_mode = "raw_logits"
        rollout.cudagraph_capture_sizes = [1, 2]
        rollout.engine_kwargs.vllm = {"max_logprobs": vocab, "compilation_config": {"custom_ops": ["all"]}}
        config.trainer.n_gpus_per_node = 1
    ray.init(num_cpus=8, num_gpus=1, object_store_memory=1 << 30, include_dashboard=False)
    servers = []
    receipts = []
    try:
        cls = ray.remote(server_class())
        for index, role in enumerate(["current_student", "teacher"]):
            rcfg = OmegaConf.create(OmegaConf.to_container(rollout, resolve=True))
            rcfg.generation_role = role
            rcfg.generation_fingerprint = None if index == 0 else fingerprints[1]
            mcfg = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True))
            mcfg.path = paths[index]
            handle = cls.options(max_concurrency=32, num_cpus=1).remote(
                config=rcfg,
                model_config=mcfg,
                rollout_mode=RolloutMode.HYBRID if index == 0 else RolloutMode.COLOCATED,
                workers=[],
                replica_rank=index,
                node_rank=0,
                gpus_per_node=1,
                nnodes=1,
                cuda_visible_devices=str(args.gpu),
            )
            servers.append(handle)
            ray.get(handle.launch_server.remote())
        current, teacher = servers
        job = ray.get_runtime_context().get_job_id()
        endpoint = weight_sync_endpoint(job, 0, 0, rollout.generation_sync_namespace)

        def weights(path):
            for file in sorted(Path(path).glob("*.safetensors")):
                with safe_open(str(file), framework="pt", device="cpu") as reader:
                    for name in reader.keys():
                        yield name, reader.get_tensor(name).to("cuda:0")

        def publish(path, version):
            ray.get(current.begin_native_update.remote(version))
            ref = current.collective_rpc.remote(
                "update_weights_from_ipc", kwargs={"global_steps": version, "use_shm": False}
            )
            asyncio.run(BucketedWeightSender(endpoint, bucket_size_mb=128).async_send_weights(weights(path)))
            ray.get(ref, timeout=60)
            ray.get(current.clear_kv_cache.remote())
            ray.get(current.commit_native_update.remote(version))
            status = ray.get(current.native_source_status.remote())
            assert status["identity"] == ("current_student", paths[0], version)
            receipts.append(status)

        def probe(handle, identity, label):
            rows = ray.get(handle.full_logits.remote(prompts, identity, label, vocab))
            return np.stack([r["logits"] for r in rows]), [r["cached_tokens"] for r in rows]

        publish(paths[0], 0)
        baseline, _ = probe(current, ("current_student", paths[0], 0), "baseline")
        cached, cache_hits = probe(current, ("current_student", paths[0], 0), "cached")
        cached_error = float(np.max(np.abs(baseline - cached)))
        cached_tv = float(
            (torch.from_numpy(baseline).softmax(-1) - torch.from_numpy(cached).softmax(-1)).abs().sum(-1).max() / 2
        )
        (args.output / "cache-comparison.json").write_text(
            json.dumps({"cache_hits": cache_hits, "max_logit_error": cached_error, "max_tv": cached_tv}, indent=2)
            + "\n"
        )
        # APC computes a shorter suffix, changing BF16 kernel shapes. Preserve
        # the existing cross-kernel bound rather than assuming bitwise equality.
        assert sum(cache_hits) > 0 and cached_error <= 0.5 and cached_tv <= 0.05
        reference, _ = probe(teacher, ("teacher", paths[1], fingerprints[1]), "teacher")
        publish(paths[1], 1)
        changed, after_cache = probe(current, ("current_student", paths[0], 1), "changed")
        assert sum(after_cache) == 0 and np.max(np.abs(changed - reference)) < 1e-5
        assert np.max(np.abs(changed - baseline)) > 0.1
        publish(paths[0], 2)
        restored, _ = probe(current, ("current_student", paths[0], 2), "restored")
        assert np.array_equal(restored, baseline)
        calls = [
            {
                "request_id": "long-native",
                "key": ["long", 0, 0],
                "prompt_ids": prompts[0],
                "max_tokens": 4096,
                "seed": 42,
            }
        ]
        long_ref = current.generate_native_batch.remote(
            calls, ("current_student", paths[0], 2), {"temperature": 0, "ignore_eos": True, "allowed_token_ids": [13]}
        )
        deadline = time.monotonic() + 30
        while True:
            progress = ray.get(current.native_progress.remote())
            if progress["active"] and progress["output_tokens"] >= 4:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Native cancellation test never observed live decode")
            time.sleep(0.05)
        try:
            ray.get(current.begin_native_update.remote(3))
        except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError) as error:
            assert "active" in str(error)
        else:
            raise RuntimeError("Weight mutation was admitted during native decode")
        assert ray.get(current.drain_native_requests.remote(abort=True, timeout=30)) is True
        try:
            aborted = ray.get(long_ref, timeout=10)
            aborted_reason = aborted[0]["finish_reason"]
            assert aborted_reason == "abort"
        except (ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError) as error:
            assert "CancelledError" in str(error) or "cancel" in str(error).lower()
            aborted_reason = "cancelled_iterator_after_native_abort"
        empty = ray.get(current.native_progress.remote())
        assert empty["active"] == empty["frontend_requests"] == 0
        ray.get(current.reopen_native_generation.remote())
        final, _ = probe(current, ("current_student", paths[0], 2), "after-abort")
        assert np.array_equal(final, baseline)
        publish(paths[0], 3)
        result = {
            "graph": args.graph,
            "vocab": vocab,
            "cache_hits_before_update": cache_hits,
            "apc_max_logit_error": cached_error,
            "apc_max_tv": cached_tv,
            "cache_hits_after_update": after_cache,
            "hot_update_max_logit_error_vs_teacher": float(np.max(np.abs(changed - reference))),
            "restore_logits_bitwise_equal": True,
            "decode_before_abort": progress,
            "aborted_reason": aborted_reason,
            "drained": empty,
            "post_abort_logits_bitwise_equal": True,
            "publications": receipts,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for handle in servers:
            ray.kill(handle, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
