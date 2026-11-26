from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendSpec:
    """One serving setup to benchmark."""

    name: str
    model_id: str
    dtype: str = "float16"
    max_model_len: int = 2048
    device_map: str = "auto"
    trust_remote_code: bool = False
    # llama.cpp
    gguf_file: str | None = None
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    n_batch: int = 512
    chat_format: str | None = None
    # vLLM
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    # AutoAWQ / AutoGPTQ
    fuse_layers: bool = True
    use_safetensors: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    prompts_per_level: int = 48
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    repeats: int = 1
    warmup_requests: int = 4
    input_tokens: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    timeout_s: float = 600.0
    results_dir: str = "results"


@dataclass
class QualityConfig:
    perplexity_chars: int = 6000
    perplexity_window: int = 1024
    mmlu_subjects: list[str] = field(
        default_factory=lambda: [
            "abstract_algebra",
            "anatomy",
            "college_computer_science",
            "high_school_mathematics",
            "miscellaneous",
        ]
    )
    mmlu_examples: int = 40
    seed: int = 42
    max_acceptable_quality_loss_pp: float = 2.0


@dataclass
class AppConfig:
    benchmark: BenchmarkConfig
    quality: QualityConfig
    backends: dict[str, BackendSpec]
    results_dir: str = "results"

    def backend(self, name: str) -> BackendSpec:
        if name not in self.backends:
            available = ", ".join(sorted(self.backends))
            raise KeyError(f"Unknown backend '{name}'. Available: {available}")
        return self.backends[name]


def _backend_spec(name: str, raw: dict[str, Any]) -> BackendSpec:
    known = {
        "model_id",
        "dtype",
        "max_model_len",
        "device_map",
        "trust_remote_code",
        "gguf_file",
        "n_ctx",
        "n_gpu_layers",
        "n_batch",
        "chat_format",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "fuse_layers",
        "use_safetensors",
    }
    values = {k: v for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    return BackendSpec(name=name, extra=extra, **values)


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config into typed dataclasses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    bench_raw = raw.get("benchmark", {})
    quality_raw = raw.get("quality", {})
    results_dir = raw.get("results_dir", bench_raw.get("results_dir", "results"))

    benchmark = BenchmarkConfig(
        **{k: v for k, v in bench_raw.items() if k in BenchmarkConfig.__dataclass_fields__}
    )
    quality = QualityConfig(
        **{k: v for k, v in quality_raw.items() if k in QualityConfig.__dataclass_fields__}
    )
    backends = {
        name: _backend_spec(name, spec or {}) for name, spec in (raw.get("backends") or {}).items()
    }
    if not backends:
        raise ValueError("Config has no backends defined.")
    return AppConfig(benchmark=benchmark, quality=quality, backends=backends, results_dir=results_dir)
