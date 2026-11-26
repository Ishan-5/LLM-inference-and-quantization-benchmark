# Results template

Fill this in after running `python -m llmbench.cli report`. Prefer the generated `results/tradeoff_table.md`; use this file as a manually curated summary for your README.

## Hardware

- GPU:
- VRAM:
- CUDA:
- Torch:
- Transformers:
- vLLM:
- llama-cpp-python:
- AutoAWQ:
- AutoGPTQ:

## Tradeoff table

| Setup | Precision | Concurrency | Out tok/s | TTFT p95 ms | E2E p95 ms | Peak VRAM MiB | MMLU % | ΔMMLU pp | PPL | ΔPPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF | FP16 | 16 | — | — | — | — | — | — | — | — |
| vLLM | FP16 | 16 | — | — | — | — | — | — | — | — |
| llama.cpp GGUF | Q4_K_M | 16 | — | — | — | — | — | — | — | — |
| AWQ | INT4 | 16 | — | — | — | — | — | — | — | — |
| GPTQ | INT4 | 16 | — | — | — | — | — | — | — | — |

## Recommendation

**Winner:**

**Why (pick 2–3):**
- Batching:
- KV cache:
- Memory bandwidth:
- Kernel/runtime maturity:

## Resume line

> Benchmarked 4 inference setups under 16 concurrent users on a T4; found ______ gives ______× throughput at ______-point MMLU loss and ______% lower peak VRAM.

## What did NOT work / caveats

-
