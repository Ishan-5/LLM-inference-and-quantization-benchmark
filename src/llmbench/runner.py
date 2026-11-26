from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .backends.base import BaseBackend, GenerationRequest
from .config import AppConfig
from .data import CHOICE_LABELS, format_mmlu_prompt, load_mmlu, load_perplexity_texts
from .hardware import MemoryMonitor
from .metrics import summarize
from .utils import chunks


def _requests(
    texts: list[str],
    start_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            prompt=text,
            request_id=start_id + i,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for i, text in enumerate(texts)
    ]


def run_benchmark(
    backend: BaseBackend,
    prompts: list[str],
    cfg: AppConfig,
    limit_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Run all concurrency levels and return one row per (level, repeat)."""
    bench = cfg.benchmark
    pool = prompts[: limit_prompts or bench.prompts_per_level]
    if not pool:
        raise ValueError("Empty prompt pool")

    warmup_n = min(bench.warmup_requests, len(pool))
    backend.generate(_requests(pool[:warmup_n], -1000, bench.max_new_tokens, bench.temperature, bench.top_p))

    rows: list[dict[str, Any]] = []
    next_id = 1
    for level in bench.concurrency_levels:
        for repeat in range(bench.repeats):
            offset = (level + repeat) % len(pool)
            selected = pool[offset:] + pool[:offset]
            requests = _requests(
                selected,
                next_id,
                bench.max_new_tokens,
                bench.temperature,
                bench.top_p,
            )
            next_id += len(requests)

            with MemoryMonitor() as memory:
                t0 = time.perf_counter()
                results: list[Any] = []
                for batch in chunks(requests, level):
                    results.extend(backend.generate(batch))
                wall = time.perf_counter() - t0

            metrics = summarize(results, wall)
            metrics.update(
                {
                    "backend": backend.name,
                    "concurrency": level,
                    "repeat": repeat,
                    "dtype": backend.spec.dtype,
                    "model_id": backend.spec.model_id,
                    "max_new_tokens": bench.max_new_tokens,
                    "input_tokens_target": bench.input_tokens,
                    "errors": metrics.get("errors", 0),
                }
            )
            metrics.update(memory.summary())
            rows.append(metrics)
    return rows


def run_quality(backend: BaseBackend, cfg: AppConfig) -> dict[str, Any]:
    """Perplexity + small MMLU subset, logged with any backend limitations."""
    quality = cfg.quality
    row: dict[str, Any] = {
        "backend": backend.name,
        "model_id": backend.spec.model_id,
        "dtype": backend.spec.dtype,
    }

    texts = load_perplexity_texts(quality.perplexity_chars, quality.seed)
    try:
        ppl = backend.perplexity(texts)
        row.update(
            {
                "perplexity": ppl.perplexity,
                "perplexity_nll": ppl.nll,
                "perplexity_tokens": ppl.tokens,
                "perplexity_note": ppl.note,
            }
        )
    except Exception as exc:
        row.update(
            {
                "perplexity": None,
                "perplexity_nll": None,
                "perplexity_tokens": 0,
                "perplexity_note": f"{type(exc).__name__}: {exc}",
            }
        )

    examples = load_mmlu(quality.mmlu_subjects, quality.mmlu_examples, quality.seed)
    prompts = [format_mmlu_prompt(example) for example in examples]
    flat_prompts: list[str] = []
    flat_choices: list[str] = []
    for prompt in prompts:
        for choice in CHOICE_LABELS:
            flat_prompts.append(prompt)
            flat_choices.append(choice)

    predictions: list[int] = []
    note = ""
    try:
        scores = backend.score_choices(flat_prompts, flat_choices)
        if len(scores) != len(flat_prompts):
            raise RuntimeError(
                f"Expected {len(flat_prompts)} scores, got {len(scores)}"
            )
        for i in range(len(examples)):
            group = scores[i * 4 : (i + 1) * 4]
            predictions.append(max(range(4), key=lambda j: group[j]))
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"

    correct = sum(
        1
        for i, pred in enumerate(predictions)
        if pred == int(examples[i].get("answer", -1))
    )
    total = len(examples)
    row.update(
        {
            "mmlu_accuracy_pct": round(100.0 * correct / total, 2) if predictions else None,
            "mmlu_correct": correct,
            "mmlu_total": total,
            "mmlu_note": note,
            "mmlu_predictions": predictions,
            "mmlu_subjects": sorted({str(e["subject"]) for e in examples}),
        }
    )
    return row
