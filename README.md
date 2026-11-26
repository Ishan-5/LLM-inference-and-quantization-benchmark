# LLM Inference & Quantization Benchmark

A portfolio-grade benchmark for the question AI infra teams actually ask:

> **How fast and cheap can we run a useful small model, and what quality do we give up?**

This project serves **one** open model (`Qwen2.5-1.5B-Instruct`) through four serving setups, applies the **same synthetic concurrent workload** to each, measures latency/throughput/memory, then runs a **fixed quality eval** (WikiText perplexity + a small MMLU subset). It produces a tradeoff table and a data-backed recommendation.

It is deliberately the opposite of a model-router project: this is systems/infra, not "which API to call."

---

## Why this is a strong project

- **Hiring-relevant skills:** continuous batching, KV-cache behavior, quantization, memory bandwidth, p95 latency, GPU memory accounting.
- **Reproducible:** fixed seed, fixed prompts, fixed eval subset, exact package versions saved in `results/raw/env_*.json`.
- **Cheap:** a free Colab/Kaggle **T4 16 GB** is enough.
- **Not just tool output:** the report explains *why* the winner wins (batching, KV cache, memory traffic), which is the part that separates a project from a blog post.

---

## The four setups

| Setup | Backend implementation | What it demonstrates |
|---|---|---|
| `hf` | Plain `transformers` + PyTorch, FP16 | Honest baseline; static batching, no KV paging |
| `vllm` | vLLM async engine, FP16 | PagedAttention + continuous batching under concurrency |
| `llamacpp` | `llama-cpp-python` with Q4_K_M GGUF | Lightweight runtime, quantized kernels, GPU-offload behavior |
| `awq` | AutoAWQ 4-bit | Activation-aware weight quantization; memory-bandwidth win |
| `gptq` | AutoGPTQ INT4 (optional 5th) | Alternative 4-bit family; useful comparison to AWQ |

All use the same base model so the comparison is about **serving strategy**, not model capability.

---

## Metrics

| Metric | Why it matters |
|---|---|
| Output tokens/sec | Main serving throughput number |
| Total tokens/sec | Includes prompt processing; useful for long-input workloads |
| TTFT p50 / p95 | Time to first token; perceived responsiveness |
| E2E p50 / p95 | Full request latency; SLA-relevant tail number |
| TPOT p95 | Inter-token latency; streaming smoothness |
| Peak GPU MiB | Deployment cost / whether the model fits alongside other work |
| WikiText perplexity | Generic language-model quality regression |
| MMLU subset accuracy | Task-level quality regression, reported in percentage points |

Quality columns are reported as deltas against the `hf` baseline: `ΔMMLU` in percentage points and `ΔPPL` as absolute perplexity change.

---

## Quickstart

### Option A — Colab T4 (recommended)

1. In this folder, create a zip of the project (or clone your repo into Colab).
2. In Colab: **Runtime → Change runtime type → T4 GPU**.
3. Upload/unzip the project so `README.md` and `scripts/` are visible.
4. Run:

```bash
!bash scripts/bootstrap_colab.sh
!python -m llmbench.cli doctor
```

5. Smoke test first:

```bash
!python -m llmbench.cli all --backends hf,vllm,llamacpp,awq --limit-prompts 8
```

6. Full run after the smoke test passes:

```bash
!bash scripts/run_all.sh
```

Or open `notebooks/01_t4_quickstart.ipynb` and run the cells in order.

### Option B — Kaggle T4

```bash
!bash scripts/bootstrap_kaggle.sh
!bash scripts/run_all.sh
```

Kaggle sometimes ships older CUDA tooling. If `llama-cpp-python` fails to build with CUDA, the bootstrap script falls back to a prebuilt cu124 wheel. Check `python -m llmbench.cli doctor` to confirm `llama_cpp` is importable. If it falls back to CPU, label that result separately.

### Option C — Local Windows with an NVIDIA GPU

Yes, this works, but with constraints. See `LOCAL_LAPTOP.md`.

For an **RTX 3050 Laptop 4 GB**, use the reduced `Qwen2.5-0.5B` config:

```powershell
powershell -File scripts\bootstrap_windows_gpu.ps1
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m llmbench.cli doctor
.\.venv\Scripts\python.exe -m llmbench.cli all --config configs\laptop_rtx3050.yaml --backends hf,llamacpp,awq,gptq
```

`vLLM` is not supported natively on Windows; use WSL2 or Colab for that setup. Keep the T4/1.5B config for final portfolio numbers.

A reduced local run has already been collected in `results_laptop/tradeoff_table.md`. On the RTX 3050 Laptop 4 GB, `llamacpp` (GGUF Q4_K_M) was fastest at ~93 output tok/s with MMLU 30% and WikiText PPL 23.64; plain HF was ~61 tok/s, MMLU 20%, PPL 23.66. Treat this as a local validation, not the final portfolio result: vLLM is absent, AWQ runs unfused on Windows, llama.cpp is measured as repeated single-stream requests, and GPTQ's Windows wheel has no working CUDA kernels, so its local quality row is invalid.

### Final T4 result

The final portfolio run is in `results/tradeoff_table.md` (Tesla T4 16 GB, Qwen2.5-1.5B-Instruct, concurrency 1/4/8/16, four backends, all benchmark rows with 0 errors).

| Setup | Out tok/s @16 | MMLU % | WikiText PPL | Peak GPU MiB | Speedup vs HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vllm` | **809.97** | **57.50** | **28.89** | 12,968 | **2.13×** |
| `hf` | 379.90 | 57.50 | 28.90 | 4,380 | 1.00× |
| `awq` | 222.10 | 47.50 | 30.81 | **2,588** | 0.58× |
| `llamacpp` | 5.26 | 55.00 | 29.93 | 450 | 0.01× |

**Recommendation:** use `vllm` for latency/throughput-sensitive serving. It is 2.13× faster than plain Transformers at concurrency 16 with no MMLU loss and slightly lower perplexity. Use `awq` only when VRAM is the binding constraint and a 10-point MMLU drop is acceptable. `llamacpp` is a low-memory option, not a competitive shared server in this run. GPTQ was skipped in the final run because `auto_gptq` was unavailable in the Colab environment.

### Option D — Local Windows / no GPU

You can validate the harness with a tiny smoke test, but **do not use those numbers in the report**.

```powershell
pip install -r requirements.txt -r requirements-llamacpp.txt
pip install -e .
python -m llmbench.cli doctor
powershell -File scripts\run_all.ps1
```

---

## Commands reference

```bash
# 1. Check hardware and optional backends
python -m llmbench.cli doctor

# 2. Download models + eval data (skipping this is fine; loaders download on demand)
python -m llmbench.cli prepare --backends hf,vllm,llamacpp,awq

# 3. Only serving benchmark
python -m llmbench.cli benchmark --backends hf,vllm,llamacpp,awq

# 4. Only quality eval
python -m llmbench.cli quality --backends hf,vllm,llamacpp,awq

# 5. Everything + auto report
python -m llmbench.cli all --backends hf,vllm,llamacpp,awq

# 6. Rebuild the report from saved raw JSON
python -m llmbench.cli report
```

Useful flags:

- `--limit-prompts 8` — fast smoke test.
- `--backends hf,gptq` — include/exclude setups.
- `--config configs/benchmark.yaml` — change concurrency, token lengths, prompts, quality gate.
- `--results-dir results_v2` — keep an independent run for comparison.

Each backend runs in a **separate subprocess** on purpose. Mixing vLLM, AWQ, and Transformers in one Python process causes CUDA allocator/graph conflicts.

---

## How to read the result

`results/tradeoff_table.md` contains:

1. **Headline table** at the highest concurrency tested.
2. **Per-concurrency detail** so you can see where a backend overtakes another.
3. **Recommendation** selected by throughput subject to a configurable quality gate (`max_acceptable_quality_loss_pp`).
4. **Why this wins** with the mechanism: batching, KV cache, or memory bandwidth.
5. **Resume line** generated from your actual numbers.

Typical interpretation on a T4:

- **vLLM usually wins concurrent throughput** because continuous batching keeps the GPU busy and paged KV cache reduces fragmentation. Its benefit is small at concurrency 1 and large at 8–16.
- **AWQ/GPTQ usually win memory** because 4-bit weights cut weight traffic and VRAM footprint. On a bandwidth-bound T4 this can also improve single-stream latency, but custom kernels may lose at large batches.
- **llama.cpp is strong for low concurrency** and simple deployment. If it runs on CPU, treat it as a different deployment class.
- **Plain HF is the control**, not the punchline. Its growing tail latency is what makes the other numbers meaningful.

Do **not** claim "quantization is always faster." The correct claim is: *on this hardware, at this concurrency, with this kernel stack, setup X delivered Y tokens/sec at Z quality delta.*

---

## Fairness rules baked into the harness

- Same model family and tokenizer lineage for all setups.
- Same deterministic prompt pool, seed, input length target, and max new tokens.
- Greedy decoding (`temperature=0`) so quality scoring is stable.
- Warmup requests before measurement.
- GPU memory sampled during the measured window, not just after load.
- Separate subprocess per backend to avoid allocator contamination.

Known approximation: llama.cpp streaming reports content chunks, so token arrival times are chunk-based. vLLM and HF/quant backends use native token IDs. This is documented in the generated caveats.

---

## Suggested 3–4 week plan

See `PROJECT_PLAN.md` for the full checklist. Short version:

- **Week 1:** environment, smoke test, baseline HF numbers, fix instrumentation.
- **Week 2:** vLLM + llama.cpp, concurrency sweep 1/4/8/16, memory sampling.
- **Week 3:** AWQ + optional GPTQ, perplexity + MMLU subset, repeat runs for p95 stability.
- **Week 4:** plots, mechanism analysis, README/report polish, resume line, repo cleanup.

---

## Final deliverable you can show

- `results/tradeoff_table.md` — the tradeoff table and recommendation.
- `results/summary.csv` — raw aggregate numbers for plotting.
- `results/raw/*.json` — request-level aggregates, quality rows, environment versions.
- `README.md` + `PROJECT_PLAN.md` — methodology and why the result happened.

Resume template:

> Benchmarked 4 inference setups under 16 concurrent users on a T4; found **vLLM** gives **2.13×** throughput over plain Transformers at **0.00 pp** MMLU change, while **AWQ** cuts peak VRAM by **80%**.

Numbers come from `results/tradeoff_table.md`.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Colab/kaggle GPU not actually T4 | `doctor` prints the GPU name; save it in env JSON |
| llama.cpp built for CPU | Label it clearly; rerun with CUDA wheel/build |
| MMLU subset too noisy | Use 40–100 examples and report it as regression detection only |
| p95 unstable | Set `repeats: 2` or `3` in config |
| Kernel/version drift | Pin versions in `results/raw/env_*.json`; do not compare across stacks blindly |
| Quantized model repo changes | Use the exact `model_id` and GGUF filename in `configs/benchmark.yaml` |

---

## Folder map

```text
configs/benchmark.yaml          Experiment matrix and model IDs
src/llmbench/
  cli.py                        Command-line entry point
  config.py                     Typed YAML config
  data.py                       Fixed prompts, perplexity text, MMLU subset
  metrics.py                    p50/p95, throughput, TPOT
  hardware.py                   GPU/RSS sampling and environment report
  runner.py                     Benchmark + quality orchestration
  report.py                     Tradeoff table and recommendation
  backends/                     HF, vLLM, llama.cpp, AWQ, GPTQ adapters
scripts/                        Bootstrap, full run, Windows smoke test
notebooks/                      Colab quickstart
results/                        Generated after you run (gitignored raw files)
PROJECT_PLAN.md                 Week-by-week execution plan
NEXT_STEPS.md                   Your action checklist
```

Start with `NEXT_STEPS.md`.
