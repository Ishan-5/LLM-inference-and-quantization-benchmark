from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import flatten_records, fmt, read_json, write_json

_NUMERIC_FIELDS = [
    "out_tok_s",
    "total_tok_s",
    "req_s",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "peak_gpu_mib",
    "mean_gpu_mib",
    "peak_rss_mib",
    "torch_peak_allocated_mib",
    "prompt_tokens",
    "output_tokens",
    "wall_s",
]

_EXPLANATIONS = {
    "vllm": [
        "**Continuous batching** admits new requests as soon as a sequence finishes, so the GPU is not idle while waiting for the slowest member of a static batch.",
        "**PagedAttention** stores the KV cache in fixed-size blocks, reducing fragmentation and letting more concurrent sequences stay resident.",
        "Expect the largest gain at concurrency 4–16; at concurrency 1 the advantage is mostly kernel/runtime overhead.",
    ],
    "vllm_awq": [
        "**Same engine, different weights:** compare this row directly with `vllm` FP16 to isolate quantization from serving-engine effects.",
        "**4-bit AWQ weights** reduce memory traffic and free VRAM for KV cache inside the same paged-attention runtime.",
        "Any throughput difference from `vllm` FP16 is a kernel/quantization effect, not a batching effect.",
    ],
    "awq": [
        "**4-bit weights** cut memory traffic, which matters because single-token decoding is usually memory-bandwidth bound rather than compute bound.",
        "AWQ protects salient channels during quantization, so quality loss is usually small compared with naive INT4.",
        "On a T4 the custom 4-bit kernels may be slower than BF16 at large batch sizes even though memory use drops sharply.",
    ],
    "gptq": [
        "**4-bit weights** reduce memory bandwidth and free VRAM for a longer KV cache.",
        "GPTQ uses second-order error compensation; quality is close to AWQ on many small models but kernel support varies by GPU.",
        "If GPTQ loses to AWQ here, the reason is usually kernel efficiency rather than a fundamental quantization-quality difference.",
    ],
    "llamacpp": [
        "**GGUF + quantized kernels** minimize runtime overhead and can run fully or partially on CPU/GPU.",
        "It is often excellent for a single user or low concurrency, but it does not provide vLLM-style continuous batching across arbitrary requests.",
        "If the build fell back to CPU, treat that number as a separate deployment option, not a like-for-like GPU comparison.",
    ],
    "hf": [
        "The plain Transformers path is the honest baseline: standard PyTorch kernels and static batching.",
        "It has no KV-cache paging and no continuous batching, so tail latency grows quickly as concurrency rises.",
        "Its value is as the control condition that quantifies what the optimized runtimes actually buy you.",
    ],
}


def _load_rows(results_dir: Path, prefix: str) -> list[dict[str, Any]]:
    raw_dir = results_dir / "raw"
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob(f"{prefix}_*.json")):
        rows.extend(flatten_records(read_json(path)))
    return [r for r in rows if isinstance(r, dict) and r.get("backend")]


def _avg(rows: list[dict[str, Any]], key: str) -> Any:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not values:
        return None
    avg = sum(values) / len(values)
    return round(avg, 2)


def _wilson_ci(correct: Any, total: Any) -> list[float] | None:
    """Wilson 95% interval for a binomial proportion, in percentage points."""
    if not isinstance(correct, int) or not isinstance(total, int) or total <= 0:
        return None
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(100 * (center - half), 1), round(100 * (center + half), 1)]


_T_CRITICAL = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    15: 2.145,
    20: 2.093,
    30: 2.045,
}


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]


def _sample_sd(values: list[float]) -> float | None:
    """Sample standard deviation; undefined for fewer than two repeats."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(math.sqrt(variance), 2)


def _mean_ci95(values: list[float]) -> list[float] | None:
    """Student-t 95% interval for the mean of a small repeat sample."""
    if len(values) < 2:
        return None
    n = len(values)
    t = next(
        (critical for size, critical in sorted(_T_CRITICAL.items(), reverse=True) if size <= n),
        1.96,
    )
    if n > max(_T_CRITICAL):
        t = 1.96
    mean = sum(values) / n
    sd = _sample_sd(values) or 0.0
    half = t * sd / math.sqrt(n)
    return [round(mean - half, 2), round(mean + half, 2)]


def _spread_text(mean: Any, sd: Any, digits: int = 1) -> str:
    if mean is None:
        return "n/a"
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _summarize_best_level(perf_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in perf_rows:
        grouped.setdefault(str(row["backend"]), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for backend, rows in grouped.items():
        max_level = max(int(r.get("concurrency", 0)) for r in rows)
        final = [r for r in rows if int(r.get("concurrency", 0)) == max_level]
        item: dict[str, Any] = {
            "backend": backend,
            "concurrency": max_level,
            "repeats": len(final),
            "model_id": final[0].get("model_id"),
            "dtype": final[0].get("dtype"),
        }
        for key in _NUMERIC_FIELDS:
            item[key] = _avg(final, key)
        for key in ("out_tok_s", "total_tok_s", "ttft_p95_ms", "e2e_p95_ms", "peak_gpu_mib"):
            values = _numbers(final, key)
            sd = _sample_sd(values)
            item[f"{key}_sd"] = sd
            item[f"{key}_ci95"] = _mean_ci95(values)
            item[f"{key}_spread_text"] = _spread_text(item.get(key), sd, 1)
        summary[backend] = item
    return summary


def _summarize_series(perf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One point per backend/concurrency, with repeat spread for chart error bars."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in perf_rows:
        key = (str(row["backend"]), int(row.get("concurrency", 0)))
        grouped.setdefault(key, []).append(row)

    series: list[dict[str, Any]] = []
    for (backend, level), rows in grouped.items():
        point: dict[str, Any] = {"backend": backend, "concurrency": level, "repeats": len(rows)}
        for key in ("out_tok_s", "total_tok_s", "ttft_p95_ms", "e2e_p95_ms", "peak_gpu_mib"):
            values = _numbers(rows, key)
            point[key] = _avg(rows, key)
            point[f"{key}_sd"] = _sample_sd(values)
            point[f"{key}_ci95"] = _mean_ci95(values)
        series.append(point)
    return sorted(series, key=lambda r: (r["backend"], r["concurrency"]))



def _merge_quality(summary: dict[str, dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    by_backend = {str(r["backend"]): r for r in quality_rows}
    for backend, row in summary.items():
        q = by_backend.get(backend, {})
        row["perplexity"] = q.get("perplexity")
        row["mmlu_accuracy_pct"] = q.get("mmlu_accuracy_pct")
        row["mmlu_correct"] = q.get("mmlu_correct")
        row["mmlu_total"] = q.get("mmlu_total")
        ci = _wilson_ci(q.get("mmlu_correct"), q.get("mmlu_total"))
        row["mmlu_ci95"] = ci
        row["mmlu_ci95_text"] = f"{ci[0]:.1f}–{ci[1]:.1f}" if ci else None
        row["perplexity_note"] = q.get("perplexity_note", "")
        row["mmlu_note"] = q.get("mmlu_note", "")


def _add_derived(summary: dict[str, dict[str, Any]], baseline_name: str) -> None:
    base = summary.get(baseline_name)
    base_out = base.get("out_tok_s") if base else None
    base_mmlu = base.get("mmlu_accuracy_pct") if base else None
    base_ppl = base.get("perplexity") if base else None
    for row in summary.values():
        out = row.get("out_tok_s")
        row["speedup_vs_baseline"] = (
            round(out / base_out, 2) if base_out and out and base_out > 0 else None
        )
        mmlu = row.get("mmlu_accuracy_pct")
        ppl = row.get("perplexity")
        row["delta_mmlu_pp"] = round(mmlu - base_mmlu, 2) if mmlu is not None and base_mmlu is not None else None
        row["delta_ppl"] = round(ppl - base_ppl, 3) if ppl is not None and base_ppl is not None else None


def _pick_winner(
    summary: dict[str, dict[str, Any]], cfg: AppConfig, baseline_name: str
) -> dict[str, Any] | None:
    candidates = [r for r in summary.values() if r.get("out_tok_s")]
    if not candidates:
        return None
    baseline = summary.get(baseline_name, {})
    base_mmlu = baseline.get("mmlu_accuracy_pct")
    if base_mmlu is not None:
        allowed = cfg.quality.max_acceptable_quality_loss_pp
        with_quality = [r for r in candidates if r.get("mmlu_accuracy_pct") is not None]
        eligible = [
            r
            for r in (with_quality or candidates)
            if r.get("mmlu_accuracy_pct") is None
            or (r["mmlu_accuracy_pct"] - base_mmlu) >= -allowed
        ]
        if eligible:
            candidates = eligible
    return max(candidates, key=lambda r: r["out_tok_s"])


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if key.startswith("delta_") and isinstance(value, (int, float)):
                cells.append(f"{value:+,.2f}" if key == "delta_mmlu_pp" else f"{value:+,.3f}")
            else:
                cells.append(fmt(value, 2))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(results_dir: str | Path, cfg: AppConfig) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    perf_rows = _load_rows(results_dir, "benchmark")
    quality_rows = _load_rows(results_dir, "quality")

    env_files = sorted((results_dir / "raw").glob("env_*.json"))
    env_info: dict[str, Any] = {}
    if env_files:
        env_info = read_json(env_files[0])

    baseline_name = "hf" if any(r["backend"] == "hf" for r in perf_rows) else (
        perf_rows[0]["backend"] if perf_rows else "hf"
    )
    summary = _summarize_best_level(perf_rows)
    _merge_quality(summary, quality_rows)
    _add_derived(summary, baseline_name)
    winner = _pick_winner(summary, cfg, baseline_name)

    summary_columns = [
        ("backend", "Setup"),
        ("concurrency", "Concurrency"),
        ("repeats", "Repeats"),
        ("out_tok_s_spread_text", "Out tok/s ±SD"),
        ("total_tok_s", "Total tok/s"),
        ("e2e_p95_ms_spread_text", "E2E p95 ±SD"),
        ("ttft_p95_ms", "TTFT p95 ms"),
        ("peak_gpu_mib", "Peak GPU MiB"),
        ("mmlu_accuracy_pct", "MMLU %"),
        ("mmlu_ci95_text", "MMLU 95% CI"),
        ("perplexity", "WikiText PPL"),
        ("delta_mmlu_pp", "ΔMMLU"),
        ("delta_ppl", "ΔPPL"),
        ("speedup_vs_baseline", f"Speedup vs {baseline_name}"),
    ]
    detail_columns = [
        ("backend", "Setup"),
        ("concurrency", "Concurrency"),
        ("out_tok_s", "Out tok/s"),
        ("ttft_p50_ms", "TTFT p50 ms"),
        ("ttft_p95_ms", "TTFT p95 ms"),
        ("e2e_p50_ms", "E2E p50 ms"),
        ("e2e_p95_ms", "E2E p95 ms"),
        ("tpot_p95_ms", "TPOT p95 ms"),
        ("peak_gpu_mib", "Peak GPU MiB"),
        ("errors", "Errors"),
    ]

    summary_rows = sorted(summary.values(), key=lambda r: str(r["backend"]))
    detail_rows = sorted(
        perf_rows,
        key=lambda r: (str(r["backend"]), int(r.get("concurrency", 0)), int(r.get("repeat", 0))),
    )

    gpu = env_info.get("gpu_name") or env_info.get("gpu") or "unknown GPU"
    vram = env_info.get("gpu_vram_mib")
    gpu_line = f"**Hardware:** {gpu}" + (f" ({fmt(vram, 0)} MiB VRAM)" if vram else "")

    lines = [
        "# Inference & quantization tradeoff report",
        "",
        gpu_line,
        f"**Baseline:** `{baseline_name}` · **Concurrent users tested:** "
        + (", ".join(str(x) for x in cfg.benchmark.concurrency_levels) or "n/a"),
        "",
    ]
    if not summary_rows:
        lines += [
            "> No benchmark results found yet. Run `python -m llmbench.cli all --backends hf,vllm,llamacpp,awq` first.",
            "",
        ]
    else:
        lines += [
            "## Headline table (highest concurrency tested)",
            "",
            _md_table(summary_rows, summary_columns),
            "",
            "## Per-concurrency detail",
            "",
            _md_table(detail_rows, detail_columns),
            "",
        ]

    if winner:
        baseline = summary.get(baseline_name, {})
        same_weights = winner.get("model_id") == baseline.get("model_id")
        if same_weights:
            quality_text = (
                f"using the same FP16 weights as `{baseline_name}`; identical MMLU is expected "
                "with greedy decoding and is not evidence of a quality advantage"
            )
        else:
            quality_text = (
                "with MMLU reported as a point estimate plus a 95% interval; see the caveats "
                "before interpreting quality deltas"
            )
        vram_note = (
            "which for vLLM includes preallocated KV cache rather than model weights alone"
            if str(winner["backend"]) == "vllm"
            else "sampled during the measured window"
        )
        lines += [
            "## Recommendation",
            "",
            f"Pick **`{winner['backend']}`** for the latency/throughput-sensitive path: it delivered "
            f"**{fmt(winner.get('out_tok_s'), 1)} output tokens/sec** at concurrency {winner.get('concurrency')}, "
            f"{quality_text}. Peak GPU memory was **{fmt(winner.get('peak_gpu_mib'), 0)} MiB**, "
            f"{vram_note}.",
            "",
            "### Why this setup wins",
            "",
        ]
        for bullet in _EXPLANATIONS.get(str(winner["backend"]), []):
            lines.append(f"- {bullet}")
        lines += [
            "- vLLM beating plain Transformers at high concurrency is a known result; the useful "
            "contribution here is the reproducible harness and the explicit uncertainty reporting.",
            "",
            "### What to put on a resume",
            "",
        ]
        hf_row = summary.get("hf", {})
        awq_row = summary.get("awq", {})
        hf_vram = hf_row.get("peak_gpu_mib")
        awq_vram = awq_row.get("peak_gpu_mib")
        if hf_vram and awq_vram:
            vram_saved = round(100 * (1 - awq_vram / hf_vram))
            vram_phrase = f"and quantified **AWQ's {vram_saved}% VRAM reduction vs HF in the same engine**"
        else:
            vram_phrase = "and measured VRAM/quality tradeoffs across backends"

        vllm_fp16 = summary.get("vllm", {})
        vllm_awq = summary.get("vllm_awq", {})
        fp16_out = vllm_fp16.get("out_tok_s")
        awq_out = vllm_awq.get("out_tok_s")
        if fp16_out and awq_out:
            same_engine_phrase = (
                f"; same-engine vLLM FP16 {fmt(fp16_out, 1)} vs AWQ {fmt(awq_out, 1)} "
                f"output tok/s ({awq_out / fp16_out:.2f}×)"
            )
        else:
            same_engine_phrase = ""
        lines += [
            f"> Built a reproducible harness for {len(summary)} inference setups at 1–"
            f"{max(cfg.benchmark.concurrency_levels) if cfg.benchmark.concurrency_levels else 'N'} "
            f"concurrent users on {gpu}; measured **{winner['backend']}** at "
            f"**{fmt(winner.get('out_tok_s'), 1)} output tok/s** "
            f"({fmt(winner.get('speedup_vs_baseline'), 2)}× HF) {vram_phrase}{same_engine_phrase}.",
            "",
        ]
    else:
        lines += ["## Recommendation", "", "Run the benchmark to generate a data-backed recommendation.", ""]

    mmlu_ns = sorted(
        {int(r["mmlu_total"]) for r in quality_rows if isinstance(r.get("mmlu_total"), int)}
    )
    mmlu_n_text = ", ".join(str(x) for x in mmlu_ns) or "unknown"
    if cfg.benchmark.repeats > 1:
        repeat_caveat = (
            f"- {cfg.benchmark.repeats} repeats were measured per concurrency level. The headline table "
            "shows sample SD and the chart shows ±SD whiskers; a Student-t 95% interval is also stored "
            "in `summary.json`. Small n still means wide intervals."
        )
    else:
        repeat_caveat = (
            "- There was one measured repeat per concurrency level, so p95 has no error bars. "
            "`configs/benchmark_v2.yaml` completed a 3-repeat run in `results_v2/`."
        )
    if any(str(r.get("backend")) == "vllm_awq" for r in perf_rows):
        confound_caveat = (
            "- `vllm_awq` is included. Compare it directly with `vllm` FP16 to isolate quantization "
            "from serving-engine effects. HF vs AWQ still isolates quantization inside the HF engine."
        )
    else:
        confound_caveat = (
            "- AWQ ran through the HF engine while vLLM ran FP16 through vLLM. Cross-engine comparisons "
            "confound engine and quantization; only HF vs AWQ isolates quantization. "
            "`configs/benchmark_v2.yaml` completed a same-engine vLLM AWQ run in `results_v2/`."
        )
    lines += [
        "## Caveats",
        "",
        "- A T4 is memory-bandwidth constrained; rankings can change on A100/H100 class GPUs.",
        "- GGUF and 4-bit kernels are build-sensitive. Record the exact package versions from "
        f"`{results_dir.as_posix()}/raw/env_*.json`.",
        f"- MMLU used a fixed subset of n={mmlu_n_text} per backend. Wilson 95% intervals are in the "
        "headline table; point-estimate deltas are not statistically distinguishable at this size.",
        f"- The configured {cfg.quality.max_acceptable_quality_loss_pp:g}-pp quality gate is smaller than "
        f"one question at n={mmlu_n_text}. It is a screening heuristic, not a significance test.",
        repeat_caveat,
        confound_caveat,
        "- vLLM's peak GPU memory includes preallocated KV cache. It is not a model-weight-only footprint "
        "and should not be used as a direct memory-efficiency comparison.",
        "- Streaming chunk counts are used as token arrival times for llama.cpp; vLLM/HF use native token IDs.",
        "- On native Windows, vLLM is unavailable and GPTQ wheels may lack CUDA kernels. Check "
        "`raw/env_*.json` and `raw/quality_*.json` before treating GPTQ quality numbers as valid.",
        "- If a backend processes concurrent requests sequentially, its throughput rows are repeated "
        "single-stream measurements, not shared-server batching.",
        "",
    ]

    report_text = "\n".join(lines)
    report_path = results_dir / "tradeoff_table.md"
    report_path.write_text(report_text, encoding="utf-8")

    csv_path = results_dir / "summary.csv"
    fieldnames = sorted({k for r in perf_rows for k in r.keys()} | {"mmlu_accuracy_pct", "perplexity"})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in perf_rows:
            merged = dict(row)
            q = next((q for q in quality_rows if q.get("backend") == row.get("backend")), {})
            merged["mmlu_accuracy_pct"] = q.get("mmlu_accuracy_pct")
            merged["perplexity"] = q.get("perplexity")
            writer.writerow(merged)

    payload = {
        "baseline": baseline_name,
        "winner": winner,
        "summary": summary_rows,
        "series": _summarize_series(perf_rows),
        "detail": detail_rows,
        "quality": quality_rows,
        "env": env_info,
    }
    write_json(results_dir / "summary.json", payload)
    return payload
