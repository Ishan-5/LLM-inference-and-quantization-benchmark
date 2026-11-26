from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from .backends.base import GenerationResult


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile. Returns None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize(results: Iterable["GenerationResult"], wall_s: float) -> dict[str, float | int | None]:
    """Aggregate request-level timings into a comparison-friendly row."""
    rows = list(results)
    ok = [r for r in rows if not r.error]
    errors = len(rows) - len(ok)

    ttfts = [r.ttft_s * 1000 for r in ok if r.ttft_s is not None]
    e2es = [r.e2e_latency_s * 1000 for r in ok]
    tpots: list[float] = []
    for r in ok:
        if r.ttft_s is not None and r.output_tokens > 1:
            tpots.append(((r.e2e_latency_s - r.ttft_s) / (r.output_tokens - 1)) * 1000)

    prompt_tokens = sum(r.prompt_tokens for r in ok)
    output_tokens = sum(r.output_tokens for r in ok)
    denom = max(wall_s, 1e-9)

    return {
        "requests": len(rows),
        "errors": errors,
        "wall_s": round(wall_s, 4),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "out_tok_s": round(output_tokens / denom, 2),
        "total_tok_s": round((prompt_tokens + output_tokens) / denom, 2),
        "req_s": round(len(ok) / denom, 3),
        "ttft_p50_ms": round(percentile(ttfts, 50) or 0.0, 2),
        "ttft_p95_ms": round(percentile(ttfts, 95) or 0.0, 2),
        "e2e_p50_ms": round(percentile(e2es, 50) or 0.0, 2),
        "e2e_p95_ms": round(percentile(e2es, 95) or 0.0, 2),
        "tpot_p50_ms": round(percentile(tpots, 50) or 0.0, 2),
        "tpot_p95_ms": round(percentile(tpots, 95) or 0.0, 2),
    }
