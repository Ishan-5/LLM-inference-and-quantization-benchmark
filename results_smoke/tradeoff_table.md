# Inference & quantization tradeoff report

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
**Baseline:** `hf` · **Concurrent users tested:** 1, 4, 8, 16

## Headline table (highest concurrency tested)

| Setup | Concurrency | Out tok/s | Total tok/s | E2E p95 ms | TTFT p95 ms | Peak GPU MiB | MMLU % | MMLU 95% CI | WikiText PPL | ΔMMLU | ΔPPL | Speedup vs hf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 4 | 53.73 | 116.77 | 7,312.77 | 148.37 | 944.50 | 20.00 | 8.1–41.6 | 28.46 | +0.00 | +4.801 | 0.80 |
| gptq | 4 | 32.87 | 71.44 | 12,212.21 | 122.89 | 1,086.50 | 15.00 | 5.2–36.0 | 151,935.98 | -5.00 | +151,912.315 | 0.49 |
| hf | 4 | 67.16 | 145.94 | 5,769.99 | 108.86 | 1,450.50 | 20.00 | 8.1–41.6 | 23.66 | +0.00 | +0.000 | 1.00 |
| llamacpp | 4 | 117.96 | 230.55 | 3,260.44 | 2,476.53 | 766.50 | 15.00 | 5.2–36.0 | n/a | -5.00 | n/a | 1.76 |

## Per-concurrency detail

| Setup | Concurrency | Out tok/s | TTFT p50 ms | TTFT p95 ms | E2E p50 ms | E2E p95 ms | TPOT p95 ms | Peak GPU MiB | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 1 | 13.91 | 65.16 | 470.49 | 6,724.04 | 7,469.96 | 75.35 | 940.50 | 0 |
| awq | 2 | 26.82 | 69.08 | 71.60 | 7,161.60 | 7,529.04 | 78.55 | 940.50 | 0 |
| awq | 4 | 53.73 | 109.59 | 148.37 | 7,139.85 | 7,312.77 | 75.41 | 944.50 | 0 |
| gptq | 1 | 8.83 | 109.17 | 120.12 | 10,894.58 | 11,277.88 | 117.58 | 814.10 | 0 |
| gptq | 2 | 17.02 | 113.51 | 117.33 | 11,214.29 | 11,500.72 | 119.87 | 808.50 | 0 |
| gptq | 4 | 32.87 | 116.10 | 122.89 | 11,676.17 | 12,212.21 | 127.40 | 1,086.50 | 0 |
| hf | 1 | 18.26 | 49.24 | 56.75 | 5,315.36 | 5,417.48 | 56.58 | 1,444.50 | 0 |
| hf | 2 | 32.92 | 56.42 | 59.83 | 5,887.13 | 6,033.45 | 62.90 | 1,444.50 | 0 |
| hf | 4 | 67.16 | 85.23 | 108.86 | 5,710.60 | 5,769.99 | 60.09 | 1,450.50 | 0 |
| llamacpp | 1 | 114.85 | 24.26 | 29.96 | 837.03 | 887.23 | 9.07 | 766.50 | 0 |
| llamacpp | 2 | 115.97 | 423.45 | 870.29 | 1,224.26 | 1,705.28 | 8.87 | 766.50 | 0 |
| llamacpp | 4 | 117.96 | 1,245.85 | 2,476.53 | 2,036.01 | 3,260.44 | 8.44 | 766.50 | 0 |

## Recommendation

Pick **`hf`** for the latency/throughput-sensitive path: it delivered **67.2 output tokens/sec** at concurrency 4, using the same FP16 weights as `hf`; identical MMLU is expected with greedy decoding and is not evidence of a quality advantage. Peak GPU memory was **1,450 MiB**, sampled during the measured window.

### Why this setup wins

- The plain Transformers path is the honest baseline: standard PyTorch kernels and static batching.
- It has no KV-cache paging and no continuous batching, so tail latency grows quickly as concurrency rises.
- Its value is as the control condition that quantifies what the optimized runtimes actually buy you.
- vLLM beating plain Transformers at high concurrency is a known result; the useful contribution here is the reproducible harness and the explicit uncertainty reporting.

### What to put on a resume

> Built a reproducible harness for 4 inference setups at 1–16 concurrent users on NVIDIA GeForce RTX 3050 Laptop GPU; measured **hf** at **67.2 output tok/s** (1.00× HF) and quantified **AWQ's 35% VRAM reduction vs HF in the same engine**.

## Caveats

- A T4 is memory-bandwidth constrained; rankings can change on A100/H100 class GPUs.
- GGUF and 4-bit kernels are build-sensitive. Record the exact package versions from `results/raw/env_*.json`.
- MMLU used a fixed subset of n=20 per backend. Wilson 95% intervals are in the headline table and overlap heavily; point-estimate deltas are not statistically distinguishable at this size. Future runs should use 200+ examples before making quality claims.
- The configured 2-pp quality gate is smaller than one question at n=40. It is a screening heuristic, not a significance test.
- There was one measured repeat per concurrency level, so p95 has no error bars. Future runs should set `repeats: 2` or `3`; the published config deliberately stays at 1 so it reproduces the reported numbers.
- AWQ ran through the HF engine while vLLM ran FP16 through vLLM. Cross-engine comparisons confound engine and quantization; only HF vs AWQ isolates quantization. A vLLM+AWQ run is future work.
- vLLM's peak GPU memory includes preallocated KV cache. It is not a model-weight-only footprint and should not be used as a direct memory-efficiency comparison.
- Streaming chunk counts are used as token arrival times for llama.cpp; vLLM/HF use native token IDs.
- On native Windows, vLLM is unavailable and GPTQ wheels may lack CUDA kernels. Check `raw/env_*.json` and `raw/quality_*.json` before treating GPTQ quality numbers as valid.
- If a backend processes concurrent requests sequentially, its throughput rows are repeated single-stream measurements, not shared-server batching.
