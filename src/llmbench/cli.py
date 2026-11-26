from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .backends import build_backend
from .config import AppConfig, load_config
from .data import make_workload_prompts
from .hardware import env_report, get_device_info
from .report import build_report
from .runner import run_benchmark, run_quality
from .utils import write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _parse_backends(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _spawn(kind: str, backend: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "llmbench.cli",
        "_worker",
        "--kind",
        kind,
        "--backend",
        backend,
        "--config",
        args.config,
    ]
    if getattr(args, "results_dir", None):
        cmd += ["--results-dir", args.results_dir]
    if getattr(args, "limit_prompts", None):
        cmd += ["--limit-prompts", str(args.limit_prompts)]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _run_worker(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.results_dir:
        cfg.results_dir = args.results_dir
        cfg.benchmark.results_dir = args.results_dir
    results_dir = Path(cfg.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.benchmark.seed)

    spec = cfg.backend(args.backend)
    env_report(raw_dir / f"env_{args.backend}.json")

    backend = build_backend(args.backend, spec, cfg)
    print(f"[{args.backend}] loading {spec.model_id}", flush=True)
    backend.load()
    try:
        if args.kind in {"benchmark", "both"}:
            n = args.limit_prompts or cfg.benchmark.prompts_per_level
            prompts = make_workload_prompts(n, cfg.benchmark.input_tokens, cfg.benchmark.seed)
            print(f"[{args.backend}] benchmark: {n} prompts", flush=True)
            rows = run_benchmark(backend, prompts, cfg, limit_prompts=args.limit_prompts)
            write_json(raw_dir / f"benchmark_{args.backend}.json", rows)
            print(
                f"[{args.backend}] best out tok/s = "
                f"{max((r.get('out_tok_s') or 0) for r in rows):.1f}",
                flush=True,
            )
        if args.kind in {"quality", "both"}:
            print(f"[{args.backend}] quality eval", flush=True)
            quality = run_quality(backend, cfg)
            write_json(raw_dir / f"quality_{args.backend}.json", quality)
            print(
                f"[{args.backend}] MMLU = {quality.get('mmlu_accuracy_pct')}%, "
                f"PPL = {quality.get('perplexity')}",
                flush=True,
            )
    finally:
        backend.close()


def _doctor(args: argparse.Namespace) -> None:
    info = get_device_info()
    print(f"llmbench {__version__}")
    print("-" * 48)
    for key, value in info.items():
        print(f"{key:26}: {value}")
    missing = [k for k in ("vllm", "llama_cpp", "autoawq", "auto_gptq") if not info.get(k)]
    if missing:
        print("\nMissing optional backends: " + ", ".join(missing))
        print("Install only the ones you plan to run; see requirements-*.txt")


def _prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = load_config(args.config)
    names = _parse_backends(args.backends)
    patterns = ["*.json", "*.txt", "*.py", "tokenizer*", "*.safetensors", "*.bin"]
    for name in names:
        spec = cfg.backend(name)
        print(f"[{name}] preparing {spec.model_id}")
        if spec.gguf_file:
            path = hf_hub_download(repo_id=spec.model_id, filename=spec.gguf_file)
            print(f"  GGUF: {path}")
        else:
            path = snapshot_download(repo_id=spec.model_id, allow_patterns=patterns)
            print(f"  snapshot: {path}")

    print("Preparing fixed eval data...")
    try:
        from .data import load_mmlu, load_perplexity_texts

        load_perplexity_texts(cfg.quality.perplexity_chars, cfg.quality.seed)
        load_mmlu(cfg.quality.mmlu_subjects, cfg.quality.mmlu_examples, cfg.quality.seed)
        print("  eval data cached")
    except Exception as exc:
        print(f"  eval data unavailable, bundled fallback will be used: {exc}")


def _maybe_report(args: argparse.Namespace) -> None:
    if getattr(args, "no_report", False):
        return
    cfg = load_config(args.config)
    results_dir = args.results_dir or cfg.results_dir
    payload = build_report(results_dir, cfg)
    if payload.get("winner"):
        print(f"\nReport written to {Path(results_dir) / 'tradeoff_table.md'}")
        print(f"Winner: {payload['winner']['backend']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark small LLM inference setups and quantization tradeoffs.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check GPU, CUDA, and installed backends").set_defaults(func=_doctor)

    prepare = sub.add_parser("prepare", help="Download model and eval assets")
    prepare.add_argument("--config", default="configs/benchmark.yaml")
    prepare.add_argument("--backends", default="hf,vllm,llamacpp,awq")
    prepare.set_defaults(func=_prepare)

    benchmark = sub.add_parser("benchmark", help="Run serving benchmark for each backend")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")
    benchmark.add_argument("--backends", default="hf,vllm,llamacpp,awq")
    benchmark.add_argument("--results-dir", default=None)
    benchmark.add_argument("--limit-prompts", type=int, default=None)
    benchmark.add_argument("--no-report", action="store_true")
    benchmark.set_defaults(func=lambda a: _run_suite("benchmark", a))

    quality = sub.add_parser("quality", help="Run perplexity + MMLU for each backend")
    quality.add_argument("--config", default="configs/benchmark.yaml")
    quality.add_argument("--backends", default="hf,vllm,llamacpp,awq")
    quality.add_argument("--results-dir", default=None)
    quality.add_argument("--no-report", action="store_true")
    quality.set_defaults(func=lambda a: _run_suite("quality", a))

    all_cmd = sub.add_parser("all", help="Benchmark + quality + report for each backend")
    all_cmd.add_argument("--config", default="configs/benchmark.yaml")
    all_cmd.add_argument("--backends", default="hf,vllm,llamacpp,awq")
    all_cmd.add_argument("--results-dir", default=None)
    all_cmd.add_argument("--limit-prompts", type=int, default=None)
    all_cmd.add_argument("--no-report", action="store_true")
    all_cmd.set_defaults(func=lambda a: _run_suite("both", a))

    report = sub.add_parser("report", help="Build tradeoff table from saved results")
    report.add_argument("--config", default="configs/benchmark.yaml")
    report.add_argument("--results-dir", default=None)
    report.set_defaults(func=lambda a: build_report(a.results_dir or load_config(a.config).results_dir, load_config(a.config)))

    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--kind", choices=["benchmark", "quality", "both"], required=True)
    worker.add_argument("--backend", required=True)
    worker.add_argument("--config", default="configs/benchmark.yaml")
    worker.add_argument("--results-dir", default=None)
    worker.add_argument("--limit-prompts", type=int, default=None)
    worker.set_defaults(func=_run_worker)

    return parser


def _run_suite(kind: str, args: argparse.Namespace) -> None:
    backends = _parse_backends(args.backends)
    if not backends:
        raise SystemExit("No backends selected.")
    for backend in backends:
        _spawn(kind, backend, args)
    _maybe_report(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result: Any = args.func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
