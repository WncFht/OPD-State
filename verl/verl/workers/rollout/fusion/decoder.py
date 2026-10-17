"""Reference and static-KV logit fusion with an explicit shared token stream.

The teacher is a frozen model. The student is a separate inference replica whose
parameters are copied in place from each committed actor update. Evaluation uses
the existing student-only vLLM engine, not this decoder.
"""

from __future__ import annotations

import math
import time

import torch


def flash_kv_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """Use physical KV lengths and left padding without materializing GQA copies."""
    from flash_attn import flash_attn_with_kvcache

    lengths, leftpad = kwargs.get("fusion_kv_lengths"), kwargs.get("fusion_kv_leftpad")
    if lengths is None or leftpad is None:
        raise RuntimeError("Flash KV fusion requires explicit cache lengths and left padding")
    output = flash_attn_with_kvcache(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        cache_seqlens=lengths,
        cache_leftpad=leftpad,
        softmax_scale=scaling,
        causal=query.shape[2] > 1,
        num_splits=1,
    )
    return output, None


def fuse_logits(student, teacher, weight, *, validate=True):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Fusion weight must be finite and in [0,1]")
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("Fusion requires matching [batch,vocabulary] logits")
    dtype = torch.promote_types(torch.promote_types(student.dtype, teacher.dtype), torch.float32)
    if weight == 0:
        mixed = student.to(dtype)
    elif weight == 1:
        mixed = teacher.to(dtype)
    else:
        mixed = student.to(dtype) * (1 - weight) + teacher.to(dtype) * weight
    if validate:
        if mixed.isnan().any() or mixed.isposinf().any() or not mixed.isfinite().any(dim=-1).all():
            raise ValueError("Fusion logits contain invalid values or empty shared support")
    return mixed


class StaticDecodeContext:
    """Fixed storage for decode; prefill stays eager and all positions are explicit."""

    def __init__(self, student, teacher, batch_size, max_length, *, capture, flash_kv=False):
        from transformers import StaticCache

        self.student, self.teacher = student, teacher
        self.device = next(student.parameters()).device
        self.dtype = next(student.parameters()).dtype
        self.batch_size, self.max_length = batch_size, max_length
        if next(teacher.parameters()).dtype != self.dtype:
            raise ValueError("Static fusion requires matching model dtypes")
        self.caches = [StaticCache(config=model.config, max_cache_len=max_length) for model in (student, teacher)]
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        self.position_ids = torch.zeros_like(self.input_ids)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.mask = torch.full((batch_size, 1, 1, max_length), -float("inf"), dtype=self.dtype, device=self.device)
        self.capture = capture
        self.flash_kv = flash_kv
        self.kv_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.kv_leftpad = torch.zeros_like(self.kv_lengths)
        if flash_kv:
            from transformers import AttentionInterface

            if self.device.type != "cuda" or self.dtype not in {torch.bfloat16, torch.float16}:
                raise ValueError("Flash KV fusion requires CUDA and BF16/FP16 models")
            AttentionInterface.register("opd_fusion_flash_kv", flash_kv_attention)
        self.graph = None
        self.outputs = None
        for model in (student, teacher):
            kinds = getattr(model.config, "layer_types", None)
            if kinds is not None and any(kind != "full_attention" for kind in kinds):
                raise ValueError("Static fusion currently requires full attention layers")
            rope = getattr(model.config, "rope_scaling", None) or {}
            if capture and rope.get("rope_type", rope.get("type", "default")) != "default":
                raise ValueError("CUDA Graph fusion currently requires fixed default RoPE")
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA Graph decoding requires a CUDA device")

    def _decode(self):
        results = []
        for model, cache in zip((self.student, self.teacher), self.caches, strict=True):
            result = self._forward(
                model,
                input_ids=self.input_ids,
                attention_mask={"full_attention": self.mask},
                position_ids=self.position_ids,
                cache_position=self.cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            results.append(result.logits[:, -1])
        return results

    def _forward(self, model, **kwargs):
        if not self.flash_kv:
            return model(**kwargs)
        previous = model.config._attn_implementation
        model.config._attn_implementation = "opd_fusion_flash_kv"
        try:
            return model(**kwargs, fusion_kv_lengths=self.kv_lengths, fusion_kv_leftpad=self.kv_leftpad)
        finally:
            model.config._attn_implementation = previous

    def prefill(self, input_ids, attention_mask, position_ids):
        for cache in self.caches:
            cache.reset()
        p = input_ids.shape[1]
        self.kv_lengths.fill_(p)
        self.kv_leftpad.copy_((attention_mask == 0).sum(-1))
        keys = torch.arange(self.max_length, device=self.device)
        query = torch.arange(p, device=self.device)
        allowed = (keys[None, :] <= query[:, None])[None, None]
        full_mask = torch.zeros((self.batch_size, self.max_length), dtype=torch.bool, device=self.device)
        full_mask[:, :p] = attention_mask.bool()
        allowed = allowed & full_mask[:, None, None, :]
        # Unattended padding queries may see all keys, while valid queries keep
        # strict causal/padding masks. This avoids all-masked SDPA rows.
        allowed = allowed | (~attention_mask.bool())[:, None, :, None]
        mask = torch.zeros(allowed.shape, dtype=self.dtype, device=self.device).masked_fill_(~allowed, -float("inf"))
        outputs = []
        positions = torch.arange(p, device=self.device)
        for model, cache in zip((self.student, self.teacher), self.caches, strict=True):
            result = self._forward(
                model,
                input_ids=input_ids,
                attention_mask={"full_attention": mask},
                position_ids=position_ids,
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            outputs.append(result.logits[:, -1])
        if self.capture and self.graph is None:
            # Capture at an unused last slot. Real decode never writes the final
            # slot: the last response token is returned without another forward.
            self.cache_position.fill_(self.max_length - 1)
            self.kv_lengths.fill_(self.max_length)
            self.mask.fill_(-float("inf"))
            self.mask[:, :, :, -1] = 0
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._decode()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = self._decode()
            torch.cuda.current_stream(self.device).wait_stream(stream)
        return outputs

    def decode(self, token, attention_mask, position_ids, physical_position):
        self.input_ids.copy_(token)
        self.position_ids.copy_(position_ids)
        self.cache_position.fill_(physical_position)
        self.kv_lengths.fill_(physical_position + 1)
        self.mask.masked_fill_(attention_mask[:, None, None, :], 0)
        self.mask.masked_fill_(~attention_mask[:, None, None, :], -float("inf"))
        if self.graph is not None:
            self.graph.replay()
            return self.outputs
        return self._decode()


class LogitFusionDecoder:
    def __init__(
        self,
        student,
        teacher,
        *,
        vocab_size,
        eos_token_id,
        pad_token_id,
        weight=0.5,
        backend="dynamic",
        prompt_cap=1024,
        response_cap=4096,
        completion_check_interval=16,
    ):
        if student is teacher:
            raise ValueError("Student and frozen teacher must be separate model objects")
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Fusion weight must be in [0,1]")
        if backend not in {"dynamic", "static", "cuda_graph", "cuda_graph_flash"}:
            raise ValueError("Unknown fusion decoder backend")
        if prompt_cap < 1 or response_cap < 1 or completion_check_interval < 1:
            raise ValueError("Token caps and completion interval must be positive")
        self.student, self.teacher = student.eval(), teacher.eval()
        self.student.requires_grad_(False)
        self.teacher.requires_grad_(False)
        self.device = next(student.parameters()).device
        if next(teacher.parameters()).device != self.device:
            raise ValueError("Fusion models must reside on the same device")
        if not 0 <= eos_token_id < vocab_size or not 0 <= pad_token_id < vocab_size:
            raise ValueError("EOS and padding IDs must belong to the shared vocabulary")
        self.vocab_size, self.eos, self.pad = vocab_size, eos_token_id, pad_token_id
        self.weight, self.backend = float(weight), backend
        self.prompt_cap, self.response_cap = prompt_cap, response_cap
        self.completion_check_interval = completion_check_interval
        self.contexts = {}
        self.student_version = None
        self._pending_version = None
        parameters = dict(self.student.named_parameters())
        self._parameters = parameters
        self._seen = set()
        self._teacher_probe = next(teacher.parameters()).reshape(-1)[:32].detach().clone()

    @classmethod
    def from_paths(cls, student_path, teacher_path, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        p_tok, q_tok = [AutoTokenizer.from_pretrained(path) for path in (student_path, teacher_path)]
        if p_tok.get_vocab() != q_tok.get_vocab() or p_tok.eos_token_id != q_tok.eos_token_id:
            raise ValueError("Fusion requires identical token identities and EOS")
        device = kwargs.pop("device", "cuda")
        models = [
            AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(
                device
            )
            for path in (student_path, teacher_path)
        ]
        return cls(
            *models,
            vocab_size=len(p_tok),
            eos_token_id=p_tok.eos_token_id,
            pad_token_id=p_tok.pad_token_id if p_tok.pad_token_id is not None else p_tok.eos_token_id,
            **kwargs,
        )

    @torch.no_grad()
    def load_student_weights(self, weights, *, version, final):
        if version is None or not isinstance(version, int) or version < 0:
            raise ValueError("A nonnegative actor weight version is required")
        if self._pending_version is None:
            self._pending_version = version
            self._seen.clear()
        if self._pending_version != version:
            raise ValueError("Actor weight buckets span different versions")
        for name, tensor in weights:
            if name not in self._parameters:
                raise ValueError(f"Unknown fusion student parameter: {name}")
            target = self._parameters[name]
            if target.shape != tensor.shape:
                raise ValueError(f"Fusion student weight shape mismatch: {name}")
            target.copy_(tensor)
            self._seen.add(name)
        if final:
            missing = self._parameters.keys() - self._seen
            if missing:
                raise ValueError(f"Fusion student weight stream is incomplete: {sorted(missing)[:8]}")
            self.student_version = version
            self._pending_version = None
            if self.backend == "dynamic":
                self.contexts.clear()

    def _inputs(self, prompts):
        if not prompts or any(not p or len(p) > self.prompt_cap for p in prompts):
            raise ValueError("Fusion prompts must be nonempty and fit the prompt cap")
        if any(type(token) is not int or not 0 <= token < self.vocab_size for p in prompts for token in p):
            raise ValueError("Fusion prompt contains an invalid token ID")
        bsz, width = len(prompts), max(map(len, prompts))
        inputs = torch.full((bsz, width), self.pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(inputs)
        for i, prompt in enumerate(prompts):
            inputs[i, -len(prompt) :] = torch.tensor(prompt, device=self.device)
            mask[i, -len(prompt) :] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0).masked_fill(mask == 0, 0)
        return inputs, mask, positions

    @torch.inference_mode()
    def generate(self, prompts, *, max_new_tokens=None, temperature=1.0, seed=42, expected_version=None):
        if self._pending_version is not None:
            raise RuntimeError("Fusion cannot generate from an incomplete weight stream")
        if expected_version is not None and self.student_version != expected_version:
            raise RuntimeError(f"Stale fusion student weights: {self.student_version} != {expected_version}")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        cap = self.response_cap if max_new_tokens is None else max_new_tokens
        if not 1 <= cap <= self.response_cap:
            raise ValueError("Requested response exceeds fusion cache capacity")
        if not torch.equal(next(self.teacher.parameters()).reshape(-1)[:32], self._teacher_probe):
            raise RuntimeError("Frozen teacher parameters changed")
        started = time.perf_counter()
        inputs, prefix_mask, positions = self._inputs(prompts)
        bsz, width = inputs.shape
        total_capacity = self.prompt_cap + self.response_cap
        full_mask = torch.zeros((bsz, total_capacity), dtype=torch.bool, device=self.device)
        full_mask[:, :width] = prefix_mask.bool()
        caches, context = None, None
        if self.backend == "dynamic":
            from transformers import DynamicCache

            caches = [DynamicCache(config=model.config) for model in (self.student, self.teacher)]
            output = []
            for model, cache in zip((self.student, self.teacher), caches, strict=True):
                output.append(
                    model(
                        input_ids=inputs,
                        attention_mask=prefix_mask,
                        position_ids=positions,
                        past_key_values=cache,
                        cache_position=torch.arange(width, device=self.device),
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    ).logits[:, -1]
                )
        else:
            if bsz not in self.contexts:
                self.contexts[bsz] = StaticDecodeContext(
                    self.student,
                    self.teacher,
                    bsz,
                    total_capacity,
                    capture=self.backend.startswith("cuda_graph"),
                    flash_kv=self.backend == "cuda_graph_flash",
                )
            context = self.contexts[bsz]
            output = context.prefill(inputs, prefix_mask, positions)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        response = torch.full((bsz, cap), self.pad, dtype=torch.long, device=self.device)
        behavior_lp = torch.zeros((bsz, cap), dtype=torch.float32, device=self.device)
        lengths = torch.full((bsz,), cap, dtype=torch.long, device=self.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        finite = torch.ones(bsz, dtype=torch.bool, device=self.device)
        logical_lengths = prefix_mask.sum(dim=-1)
        for step in range(cap):
            logits = fuse_logits(
                output[0][:, : self.vocab_size], output[1][:, : self.vocab_size], self.weight, validate=step == 0
            )
            finite &= ~logits.isnan().any(dim=-1) & ~logits.isposinf().any(dim=-1) & logits.isfinite().any(dim=-1)
            if temperature == 0:
                chosen = logits.argmax(dim=-1)
                lp = torch.zeros(bsz, device=self.device)
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                lp = probabilities.gather(1, chosen[:, None]).squeeze(-1).log()
            finite &= lp.isfinite()
            active = ~finished
            chosen = torch.where(active, chosen, self.pad)
            response[:, step] = chosen
            behavior_lp[:, step] = torch.where(active, lp, 0)
            stopped = active & (chosen == self.eos)
            lengths = torch.where(stopped, step + 1, lengths)
            finished |= stopped
            if step + 1 == cap or (step % self.completion_check_interval == 0 and bool(finished.all())):
                break
            full_mask[:, width + step] = active
            token = chosen[:, None]
            token_positions = (logical_lengths - 1 + full_mask[:, width : width + step + 1].sum(-1))[:, None]
            if context is not None:
                output = context.decode(token, full_mask, token_positions, width + step)
            else:
                output = []
                for model, cache in zip((self.student, self.teacher), caches, strict=True):
                    output.append(
                        model(
                            input_ids=token,
                            attention_mask=full_mask[:, : width + step + 1].long(),
                            position_ids=token_positions,
                            cache_position=torch.tensor([width + step], device=self.device),
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                            return_dict=True,
                        ).logits[:, -1]
                    )
        if not bool(finite.all()):
            raise FloatingPointError("Nonfinite logits encountered during fused decoding")
        ids, logps, lengths, stopped = (
            response.cpu().tolist(),
            behavior_lp.cpu().tolist(),
            lengths.cpu().tolist(),
            finished.cpu().tolist(),
        )
        return {
            "responses": [row[:n] for row, n in zip(ids, lengths, strict=True)],
            "behavior_log_probs": [row[:n] for row, n in zip(logps, lengths, strict=True)],
            "finish_reasons": ["eos" if value else "length" for value in stopped],
            "seconds": time.perf_counter() - started,
            "student_version": self.student_version,
            "lambda": self.weight,
            "backend": self.backend,
        }
