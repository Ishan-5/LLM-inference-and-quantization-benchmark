from __future__ import annotations

import gc
import math
import time
from typing import Any

import torch

from ..config import AppConfig, BackendSpec
from .base import BaseBackend, GenerationRequest, GenerationResult, PerplexityResult


def _dtype(name: str) -> torch.dtype:
    name = (name or "float16").lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Transformers 5 renamed torch_dtype to dtype; support both."""
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


class HuggingFaceBackend(BaseBackend):
    """Plain PyTorch/Transformers baseline, also reused for AWQ and GPTQ."""

    name = "hf"
    loader_name = "AutoModelForCausalLM"

    def __init__(self, spec: BackendSpec, cfg: AppConfig) -> None:
        super().__init__(spec, cfg)
        self.device: Any = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id,
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            **_dtype_kwargs(_dtype(self.spec.dtype)),
            device_map=self.spec.device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=self.spec.trust_remote_code,
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.device = self.model.device
        self.loaded = True

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not self.loaded:
            raise RuntimeError("Backend must be loaded before generate().")
        results = [
            GenerationResult(request_id=r.request_id, prompt_tokens=0, output_tokens=0)
            for r in requests
        ]
        try:
            texts = [self.format_prompt(r.prompt, r.use_chat_template) for r in requests]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.spec.max_model_len,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            prompt_lengths = attention_mask.sum(dim=1).tolist()
            row_limits = [
                min(r.max_new_tokens, max(1, self.spec.max_model_len - int(prompt_lengths[i])))
                for i, r in enumerate(requests)
            ]
            for result, plen in zip(results, prompt_lengths):
                result.prompt_tokens = int(plen)

            eos_id = self.tokenizer.eos_token_id
            past_key_values = None
            finished = [False] * len(requests)
            t0 = time.perf_counter()

            for step in range(max(row_limits)):
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = outputs.logits[:, -1, :].float()
                next_ids = torch.argmax(logits, dim=-1)
                now = time.perf_counter()
                offset = now - t0

                any_active = False
                for i in range(len(requests)):
                    if finished[i]:
                        continue
                    any_active = True
                    token_id = int(next_ids[i])
                    results[i].token_ids.append(token_id)
                    results[i].token_times.append(offset)
                    if results[i].ttft_s is None:
                        results[i].ttft_s = offset
                    if token_id == eos_id or len(results[i].token_ids) >= row_limits[i]:
                        finished[i] = True
                        results[i].finished_at = now

                if not any_active:
                    break

                input_ids = next_ids.unsqueeze(1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)],
                    dim=1,
                )
                past_key_values = outputs.past_key_values

            end = time.perf_counter()
            for i, result in enumerate(results):
                result.e2e_latency_s = (result.finished_at or end) - t0
                result.output_tokens = len(result.token_ids)
                result.text = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
            return results
        except Exception as exc:
            for result in results:
                result.error = f"{type(exc).__name__}: {exc}"
                result.e2e_latency_s = 0.0
            return results

    def perplexity(self, texts: list[str]) -> PerplexityResult:
        window = min(self.spec.max_model_len, self.cfg.quality.perplexity_window)
        stride = max(1, window // 2)
        total_nll = 0.0
        total_tokens = 0
        self.model.eval()
        with torch.inference_mode():
            for text in texts:
                ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                for start in range(0, max(1, len(ids) - 1), stride):
                    chunk = ids[start : start + window]
                    if len(chunk) < 8:
                        continue
                    inputs = chunk.unsqueeze(0).to(self.device)
                    logits = self.model(input_ids=inputs).logits[0, :-1, :].float()
                    targets = inputs[0, 1:]
                    nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
                    total_nll += float(nll)
                    total_tokens += int(targets.numel())
        if total_tokens == 0:
            return PerplexityResult(None, None, 0, "No scorable tokens.")
        mean_nll = total_nll / total_tokens
        return PerplexityResult(math.exp(mean_nll), mean_nll, total_tokens)

    def score_choices(self, prompts: list[str], choices: list[str]) -> list[float]:
        if len(prompts) != len(choices):
            raise ValueError("prompts and choices must have equal length")
        scores: list[float] = []
        max_len = self.spec.max_model_len
        self.model.eval()
        with torch.inference_mode():
            for prompt, choice in zip(prompts, choices):
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                choice_ids = self.tokenizer(choice, add_special_tokens=False)["input_ids"]
                if not choice_ids:
                    scores.append(float("-inf"))
                    continue
                ids = (prompt_ids + choice_ids)[-max_len:]
                keep = min(len(choice_ids), len(ids) - 1)
                tensor = torch.tensor([ids], device=self.device)
                logits = self.model(input_ids=tensor).logits[0, :-1, :].float()
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                targets = tensor[0, 1:]
                picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                scores.append(float(picked[-keep:].sum()))
        return scores

    def close(self) -> None:
        super().close()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class AwqBackend(HuggingFaceBackend):
    name = "awq"
    loader_name = "AutoAWQForCausalLM"

    def load(self) -> None:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id, trust_remote_code=self.spec.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoAWQForCausalLM.from_quantized(
            self.spec.model_id,
            device_map=self.spec.device_map,
            fuse_layers=self.spec.fuse_layers,
            safetensors=self.spec.use_safetensors,
            trust_remote_code=self.spec.trust_remote_code,
        )
        self.model.eval()
        self.model.config.use_cache = True
        # AWQ wrappers do not expose a ``device`` attribute; infer it from weights.
        self.device = next(self.model.parameters()).device
        self.loaded = True


class GptqBackend(HuggingFaceBackend):
    name = "gptq"
    loader_name = "AutoGPTQForCausalLM"

    def load(self) -> None:
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id, trust_remote_code=self.spec.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoGPTQForCausalLM.from_quantized(
            self.spec.model_id,
            device_map=self.spec.device_map,
            use_safetensors=self.spec.use_safetensors,
            trust_remote_code=self.spec.trust_remote_code,
        )
        self.model.eval()
        self.model.config.use_cache = True
        # GPTQ wrappers do not always expose a ``device`` attribute.
        self.device = next(self.model.parameters()).device
        self.loaded = True
