from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseBackend

if TYPE_CHECKING:
    from ..config import AppConfig, BackendSpec


def build_backend(name: str, spec: "BackendSpec", cfg: "AppConfig") -> BaseBackend:
    name = name.lower()
    if name == "hf":
        from .hf_backend import HuggingFaceBackend

        return HuggingFaceBackend(spec, cfg)
    if name in {"vllm", "vllm_awq"}:
        from .vllm_backend import VLLMBackend

        backend = VLLMBackend(spec, cfg)
        if name == "vllm_awq":
            backend.name = "vllm_awq"
        return backend
    if name == "llamacpp":
        from .llamacpp_backend import LlamaCppBackend

        return LlamaCppBackend(spec, cfg)
    if name == "awq":
        from .hf_backend import AwqBackend

        return AwqBackend(spec, cfg)
    if name == "gptq":
        from .hf_backend import GptqBackend

        return GptqBackend(spec, cfg)
    raise ValueError(
        f"Unknown backend '{name}'. Choose from: hf, vllm, vllm_awq, llamacpp, awq, gptq"
    )
