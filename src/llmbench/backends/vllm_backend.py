from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Any

from ..config import AppConfig, BackendSpec
from .base import BaseBackend, GenerationRequest, GenerationResult, PerplexityResult


class VLLMBackend(BaseBackend):
    """vLLM async engine: continuous batching + paged KV cache."""

    name = "vllm"
    supports_perplexity = True
    supports_logprob_scoring = True

    def __init__(self, spec: BackendSpec, cfg: AppConfig) -> None:
        super().__init__(spec, cfg)
        self.engine: Any = None
        self.SamplingParams: Any = None
        self.loop: Any = None

    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams

        # vLLM renamed AsyncLLM -> AsyncLLMEngine in newer releases (e.g. 0.34).
        try:
            from vllm import AsyncLLM as AsyncLLMEngine
        except ImportError:
            from vllm import AsyncLLMEngine

        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id, trust_remote_code=self.spec.trust_remote_code
        )
        args = AsyncEngineArgs(
            model=self.spec.model_id,
            dtype=self.spec.dtype,
            max_model_len=self.spec.max_model_len,
            tensor_parallel_size=self.spec.tensor_parallel_size,
            gpu_memory_utilization=self.spec.gpu_memory_utilization,
            trust_remote_code=self.spec.trust_remote_code,
            disable_log_stats=True,
        )
        # vLLM 0.34 starts its output handler only when the engine is created
        # inside a running event loop. Keep one loop for all requests.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def build_engine() -> Any:
            return AsyncLLMEngine.from_engine_args(args)

        try:
            self.engine = self.loop.run_until_complete(build_engine())
        except Exception:
            self.loop.close()
            self.loop = None
            raise
        self.loaded = True

    def _run(self, coro: Any) -> Any:
        if self.loop is None:
            return asyncio.run(coro)
        return self.loop.run_until_complete(coro)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not self.loaded:
            raise RuntimeError("Backend must be loaded before generate().")
        t0 = time.perf_counter()

        async def generate_all() -> list[GenerationResult]:
            tasks = [self._generate_one(t0, req) for req in requests]
            return await asyncio.gather(*tasks)

        return self._run(generate_all())

    async def _generate_one(self, t0: float, req: GenerationRequest) -> GenerationResult:
        result = GenerationResult(request_id=req.request_id)
        prompt = self.format_prompt(req.prompt, req.use_chat_template)
        try:
            result.prompt_tokens = len(
                self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            max_tokens = min(req.max_new_tokens, max(1, self.spec.max_model_len - result.prompt_tokens))
            params = self.SamplingParams(
                max_tokens=max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                ignore_eos=False,
            )
            seen = 0
            output_text = ""
            request_id = f"bench-{req.request_id}-{uuid.uuid4().hex}"
            async for output in self.engine.generate(prompt, params, request_id):
                if not output.outputs:
                    continue
                completion = output.outputs[0]
                token_ids = list(getattr(completion, "token_ids", []))
                if len(token_ids) > seen:
                    arrived = time.perf_counter() - t0
                    if result.ttft_s is None:
                        result.ttft_s = arrived
                    result.token_times.extend([arrived] * (len(token_ids) - seen))
                    result.token_ids = token_ids
                    seen = len(token_ids)
                output_text = completion.text
            result.text = output_text
            result.output_tokens = len(result.token_ids)
            result.e2e_latency_s = time.perf_counter() - t0
            result.finished_at = result.e2e_latency_s
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.e2e_latency_s = time.perf_counter() - t0
        return result

    def _prompt_logprobs(self, prompt: str) -> list[dict[int, Any]]:
        """Return vLLM prompt logprobs, or raise if this build does not expose them."""
        if self.SamplingParams is None:
            raise RuntimeError("vLLM not loaded")

        async def collect() -> list[dict[int, Any]] | None:
            params = self.SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
            request_id = f"score-{uuid.uuid4().hex}"
            last = None
            async for output in self.engine.generate(prompt, params, request_id):
                last = output
            if last is None:
                return None
            return getattr(last, "prompt_logprobs", None)

        values = self._run(collect())
        if not values:
            raise RuntimeError("prompt_logprobs unavailable in this vLLM build")
        return values

    @staticmethod
    def _first_logprob(entry: Any) -> float | None:
        if not isinstance(entry, dict) or not entry:
            return None
        value = next(iter(entry.values()))
        logprob = getattr(value, "logprob", None)
        if logprob is None and isinstance(value, (int, float)):
            logprob = float(value)
        return None if logprob is None else float(logprob)

    def perplexity(self, texts: list[str]) -> PerplexityResult:
        total_nll = 0.0
        total_tokens = 0
        notes: list[str] = []
        for text in texts:
            try:
                prompt = text[: self.spec.max_model_len * 3]
                entries = self._prompt_logprobs(prompt)
                values = [self._first_logprob(e) for e in entries[1:]]
                values = [v for v in values if v is not None]
                if not values:
                    raise RuntimeError("no prompt logprobs returned")
                total_nll += -sum(values)
                total_tokens += len(values)
            except Exception as exc:
                notes.append(str(exc))
        if total_tokens == 0:
            return PerplexityResult(None, None, 0, "; ".join(notes[:2]) or "No tokens scored")
        mean_nll = total_nll / total_tokens
        return PerplexityResult(
            math.exp(mean_nll),
            mean_nll,
            total_tokens,
            "partial" if notes else "",
        )

    def score_choices(self, prompts: list[str], choices: list[str]) -> list[float]:
        scores: list[float] = []
        for prompt, choice in zip(prompts, choices):
            full = (prompt + choice)[-self.spec.max_model_len * 3 :]
            try:
                entries = self._prompt_logprobs(full)
                choice_ids = self.tokenizer(choice, add_special_tokens=False)["input_ids"]
                keep = max(1, min(len(choice_ids), len(entries) - 1))
                values = [self._first_logprob(e) for e in entries[-keep:]]
                scores.append(sum(v for v in values if v is not None))
            except Exception:
                scores.append(float("-inf"))
        return scores

    def close(self) -> None:
        try:
            if self.engine is not None:
                self.engine.shutdown()
        except Exception:
            pass
        try:
            if self.loop is not None:
                self.loop.close()
        except Exception:
            pass
        self.loop = None
        self.engine = None
        super().close()
