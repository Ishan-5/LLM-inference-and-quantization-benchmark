# Inference & quantization tradeoff report

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
**Baseline:** `hf` · **Concurrent users tested:** 1, 2, 4

## Headline table (highest concurrency tested)

| Setup | Concurrency | Repeats | Out tok/s ±SD | Total tok/s | E2E p95 ±SD | TTFT p95 ms | Peak GPU MiB | MMLU % | MMLU 95% CI | WikiText PPL | ΔMMLU | ΔPPL | Speedup vs hf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 4 | 1 | 34.6 | 75.82 | 12270.7 | 351.26 | 944.50 | 20.00 | 8.1–41.6 | 28.46 | +0.00 | +4.798 | 0.57 |
| gptq | 4 | 1 | 25.6 | 55.77 | 15443.8 | 218.85 | 1,162.50 | 15.00 | 5.2–36.0 | 151,935.98 | -5.00 | +151,912.315 | 0.42 |
| hf | 4 | 1 | 61.2 | 133.41 | 6614.8 | 128.42 | 1,450.50 | 20.00 | 8.1–41.6 | 23.66 | +0.00 | +0.000 | 1.00 |
| llamacpp | 4 | 1 | 93.1 | 182.75 | 4189.1 | 3,188.64 | 766.50 | 30.00 | 14.5–51.9 | 23.64 | +10.00 | -0.020 | 1.52 |

## Per-concurrency detail

| Setup | Concurrency | Out tok/s | TTFT p50 ms | TTFT p95 ms | E2E p50 ms | E2E p95 ms | TPOT p95 ms | Peak GPU MiB | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 1 | 9.44 | 83.29 | 137.80 | 10,007.72 | 10,964.06 | 114.27 | 940.50 | 0 |
| awq | 2 | 17.06 | 89.86 | 106.05 | 10,942.51 | 12,457.72 | 130.21 | 940.50 | 0 |
| awq | 4 | 34.57 | 101.70 | 351.26 | 10,670.42 | 12,270.71 | 125.47 | 944.50 | 0 |
| gptq | 1 | 6.57 | 151.91 | 197.59 | 14,556.03 | 15,765.33 | 164.24 | 808.50 | 0 |
| gptq | 2 | 13.23 | 153.62 | 168.69 | 14,591.92 | 15,315.10 | 159.70 | 880.50 | 0 |
| gptq | 4 | 25.57 | 176.17 | 218.85 | 15,011.23 | 15,443.77 | 160.33 | 1,162.50 | 0 |
| hf | 1 | 15.85 | 61.29 | 75.59 | 6,103.69 | 6,667.53 | 69.41 | 1,444.50 | 0 |
| hf | 2 | 28.83 | 67.93 | 94.12 | 6,634.04 | 7,119.70 | 74.27 | 1,444.50 | 0 |
| hf | 4 | 61.17 | 62.12 | 128.42 | 6,325.43 | 6,614.82 | 68.96 | 1,450.50 | 0 |
| llamacpp | 1 | 92.81 | 41.31 | 48.27 | 1,016.81 | 1,113.30 | 11.28 | 766.50 | 0 |
| llamacpp | 2 | 93.43 | 538.46 | 1,090.17 | 1,546.51 | 2,083.94 | 10.67 | 766.50 | 0 |
| llamacpp | 4 | 93.13 | 1,595.73 | 3,188.64 | 2,584.39 | 4,189.07 | 10.72 | 766.50 | 0 |

## Recommendation

Pick **`llamacpp`** for the latency/throughput-sensitive path: it delivered **93.1 output tokens/sec** at concurrency 4, with MMLU reported as a point estimate plus a 95% interval; see the caveats before interpreting quality deltas. Peak GPU memory was **766 MiB**, sampled during the measured window.

### Why this setup wins

- **GGUF + quantized kernels** minimize runtime overhead and can run fully or partially on CPU/GPU.
- It is often excellent for a single user or low concurrency, but it does not provide vLLM-style continuous batching across arbitrary requests.
- If the build fell back to CPU, treat that number as a separate deployment option, not a like-for-like GPU comparison.
- vLLM beating plain Transformers at high concurrency is a known result; the useful contribution here is the reproducible harness and the explicit uncertainty reporting.

### What to put on a resume

> Built a reproducible harness for 4 inference setups at 1–4 concurrent users on NVIDIA GeForce RTX 3050 Laptop GPU; measured **llamacpp** at **93.1 output tok/s** (1.52× HF) and quantified **AWQ's 35% VRAM reduction vs HF in the same engine**.

## Caveats

- A T4 is memory-bandwidth constrained; rankings can change on A100/H100 class GPUs.
- GGUF and 4-bit kernels are build-sensitive. Record the exact package versions from `results/raw/env_*.json`.
- MMLU used a fixed subset of n=20 per backend. Wilson 95% intervals are in the headline table; point-estimate deltas are not statistically distinguishable at this size.
- The configured 3-pp quality gate is smaller than one question at n=20. It is a screening heuristic, not a significance test.
- There was one measured repeat per concurrency level, so p95 has no error bars. The published config stays at `repeats: 1` for reproducibility; `configs/benchmark_v2.yaml` is prepared with 3 repeats.
- AWQ ran through the HF engine while vLLM ran FP16 through vLLM. Cross-engine comparisons confound engine and quantization; only HF vs AWQ isolates quantization. `configs/benchmark_v2.yaml` is prepared with a vLLM+AWQ run.
- vLLM's peak GPU memory includes preallocated KV cache. It is not a model-weight-only footprint and should not be used as a direct memory-efficiency comparison.
- Streaming chunk counts are used as token arrival times for llama.cpp; vLLM/HF use native token IDs.
- On native Windows, vLLM is unavailable and GPTQ wheels may lack CUDA kernels. Check `raw/env_*.json` and `raw/quality_*.json` before treating GPTQ quality numbers as valid.
- If a backend processes concurrent requests sequentially, its throughput rows are repeated single-stream measurements, not shared-server batching.
