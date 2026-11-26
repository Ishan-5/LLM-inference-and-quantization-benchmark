from __future__ import annotations

import gc
import math
import time
from typing import Any

from ..config import AppConfig, BackendSpec
from .base import BaseBackend, GenerationRequest, GenerationResult, PerplexityResult


class LlamaCppBackend(BaseBackend):
    """llama.cpp via gguf, using its chat completion streaming API."""

    name = "llamacpp"
    supports_perplexity = True
    supports_logprob_scoring = True

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        if not self.spec.gguf_file:
            raise ValueError("llamacpp backend requires gguf_file in config")
        path = hf_hub_download(repo_id=self.spec.model_id, filename=self.spec.gguf_file)
        self._model_path = path
        self.model = Llama(
            model_path=path,
            n_ctx=self.spec.n_ctx,
            n_gpu_layers=self.spec.n_gpu_layers,
            n_batch=self.spec.n_batch,
            chat_format=self.spec.chat_format,
            verbose=False,
        )
        # Logits for every token are only needed by quality scoring. Keeping the
        # serving model on the fast path avoids paying that cost during inference.
        self.score_model: Any = None
        self.loaded = True

    def _token_len(self, text: str) -> int:
        try:
            return len(
                self.model.tokenize(text.encode("utf-8"), add_bos=True, special=True)
            )
        except Exception:
            return max(1, len(text) // 4)

    def _raw_prompt(self, text: str) -> str:
        template = self.spec.extra.get(
            "raw_chat_template",
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
        )
        return template.format(text=text)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not self.loaded:
            raise RuntimeError("Backend must be loaded before generate().")
        results: list[GenerationResult] = []
        t0 = time.perf_counter()
        for req in requests:
            result = GenerationResult(request_id=req.request_id)
            try:
                prompt = self._raw_prompt(req.prompt)
                result.prompt_tokens = self._token_len(prompt)
                max_tokens = min(
                    req.max_new_tokens, max(1, self.spec.n_ctx - result.prompt_tokens)
                )
                stream = self.model.create_chat_completion(
                    messages=[{"role": "user", "content": req.prompt}],
                    max_tokens=max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stream=True,
                )
                text = ""
                completion_tokens = 0
                for chunk in stream:
                    choices = chunk.get("choices") or []
                    if not choices:
                        usage = chunk.get("usage") or {}
                        completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        now = time.perf_counter() - t0
                        if result.ttft_s is None:
                            result.ttft_s = now
                        result.token_times.append(now)
                        text += piece
                    usage = chunk.get("usage") or {}
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                result.text = text
                result.output_tokens = (
                    completion_tokens
                    if completion_tokens > 0
                    else max(1, self._token_len(text))
                )
                result.token_ids = []
                result.e2e_latency_s = time.perf_counter() - t0
                result.finished_at = result.e2e_latency_s
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.e2e_latency_s = time.perf_counter() - t0
            results.append(result)
        return results

    def _ensure_score_model(self) -> Any:
        """Load a second copy with per-token logits only when quality is measured."""
        if self.score_model is None:
            from llama_cpp import Llama

            self.score_model = Llama(
                model_path=self._model_path,
                n_ctx=self.spec.n_ctx,
                n_gpu_layers=self.spec.n_gpu_layers,
                n_batch=self.spec.n_batch,
                chat_format=self.spec.chat_format,
                logits_all=True,
                verbose=False,
            )
        return self.score_model

    def _token_ids(self, text: str) -> list[int]:
        return self.model.tokenize(text.encode("utf-8"), add_bos=True, special=True)

    def _token_logprobs(self, text: str) -> list[float]:
        """Score P(token_i | previous tokens) without llama.cpp's top-logprobs loop."""
        import numpy as np

        model = self._ensure_score_model()
        tokens = self._token_ids(text)[: self.spec.n_ctx]
        if len(tokens) < 2:
            return []
        model.reset()
        model.eval(tokens)
        logits = model._scores[: len(tokens), :]
        values: list[float] = []
        for i, token_id in enumerate(tokens[1:]):
            row = logits[i]
            row_max = float(np.max(row))
            log_norm = row_max + float(np.log(np.sum(np.exp(row - row_max))))
            values.append(float(row[token_id]) - log_norm)
        return values

    def perplexity(self, texts: list[str]) -> PerplexityResult:
        total_nll = 0.0
        total_tokens = 0
        notes: list[str] = []
        for text in texts:
            try:
                values = self._token_logprobs(text[: self.spec.n_ctx * 3])
                if not values:
                    raise RuntimeError("no tokens scored")
                total_nll += -sum(values)
                total_tokens += len(values)
            except Exception as exc:
                notes.append(str(exc))
        if total_tokens == 0:
            return PerplexityResult(None, None, 0, "; ".join(notes[:2]) or "No tokens scored")
        mean_nll = total_nll / total_tokens
        return PerplexityResult(math.exp(mean_nll), mean_nll, total_tokens, "partial" if notes else "")

    def score_choices(self, prompts: list[str], choices: list[str]) -> list[float]:
        scores: list[float] = []
        for prompt, choice in zip(prompts, choices):
            try:
                choice_tokens = max(1, len(self._token_ids(choice)) - 1)
                values = self._token_logprobs(prompt + choice)[-choice_tokens:]
                scores.append(sum(values))
            except Exception:
                scores.append(float("-inf"))
        return scores

    def close(self) -> None:
        self.score_model = None
        super().close()
        gc.collect()
