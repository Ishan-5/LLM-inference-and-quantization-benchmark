from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import AppConfig, BackendSpec


@dataclass
class GenerationRequest:
    """A single user request in the synthetic workload."""

    prompt: str
    request_id: int
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = True


@dataclass
class GenerationResult:
    """Request-level measurements produced by every backend."""

    request_id: int
    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float | None = None
    e2e_latency_s: float = 0.0
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finished_at: float | None = None
    error: str | None = None


@dataclass
class PerplexityResult:
    perplexity: float | None
    nll: float | None
    tokens: int
    note: str = ""


@dataclass
class MMLUResult:
    accuracy: float | None
    correct: int
    total: int
    predictions: list[int] = field(default_factory=list)
    note: str = ""


class BaseBackend(ABC):
    """Common interface so every serving setup is measured the same way."""

    name = "base"
    supports_perplexity = True
    supports_logprob_scoring = True

    def __init__(self, spec: "BackendSpec", cfg: "AppConfig") -> None:
        self.spec = spec
        self.cfg = cfg
        self.model: Any = None
        self.tokenizer: Any = None
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model/tokenizer into memory."""

    @abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Generate for a batch of requests and record arrival times."""

    def perplexity(self, texts: list[str]) -> PerplexityResult:
        raise NotImplementedError(f"{self.name} does not expose token logprobs for perplexity.")

    def score_choices(self, prompts: list[str], choices: list[str]) -> list[float]:
        """Return log P(choice | prompt) for each pair."""
        raise NotImplementedError(f"{self.name} does not expose continuation logprobs.")

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded = False

    # -- Shared helpers -------------------------------------------------
    def format_prompt(self, text: str, use_chat_template: bool = True) -> str:
        if not use_chat_template:
            return text
        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def error_result(request: GenerationRequest, exc: Exception) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            e2e_latency_s=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )
