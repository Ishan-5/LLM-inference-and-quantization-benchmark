# Inference & quantization tradeoff report

**Hardware:** Tesla T4 (14,913 MiB VRAM)
**Baseline:** `hf` · **Concurrent users tested:** 1, 4, 8, 16

## Headline table (highest concurrency tested)

| Setup | Concurrency | Repeats | Out tok/s ±SD | Total tok/s | E2E p95 ±SD | TTFT p95 ms | Peak GPU MiB | MMLU % | MMLU 95% CI | WikiText PPL | ΔMMLU | ΔPPL | Speedup vs hf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 16 | 3 | 210.4 ± 2.2 | 429.61 | 9965.7 ± 101.7 | 124.05 | 2,588.20 | 50.50 | 43.6–57.4 | 30.81 | +1.00 | +1.911 | 0.59 |
| hf | 16 | 3 | 358.7 ± 9.6 | 733.80 | 6019.2 ± 273.8 | 73.11 | 4,380.20 | 49.50 | 42.6–56.4 | 28.90 | +0.00 | +0.000 | 1.00 |
| vllm | 16 | 3 | 813.5 ± 2.7 | 1,664.13 | 2525.1 ± 13.5 | 132.65 | 12,968.20 | 49.50 | 42.6–56.4 | 28.89 | +0.00 | -0.011 | 2.27 |
| vllm_awq | 16 | 3 | 1384.8 ± 9.8 | 2,827.82 | 1480.4 ± 11.1 | 109.36 | 12,946.20 | 51.00 | 44.1–57.8 | 30.79 | +1.50 | +1.889 | 3.86 |

## Per-concurrency detail

| Setup | Concurrency | Out tok/s | TTFT p50 ms | TTFT p95 ms | E2E p50 ms | E2E p95 ms | TPOT p95 ms | Peak GPU MiB | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 1 | 14.35 | 63.84 | 94.25 | 9,101.01 | 9,229.08 | 72.17 | 2,172.20 | 0 |
| awq | 1 | 14.38 | 62.84 | 88.00 | 9,043.23 | 9,233.33 | 72.21 | 2,172.20 | 0 |
| awq | 1 | 14.50 | 62.65 | 88.99 | 8,864.20 | 9,163.20 | 71.67 | 2,172.20 | 0 |
| awq | 4 | 50.55 | 281.73 | 282.46 | 10,258.10 | 10,460.20 | 80.14 | 2,174.20 | 0 |
| awq | 4 | 50.68 | 280.92 | 285.08 | 10,191.91 | 10,392.01 | 79.58 | 2,174.20 | 0 |
| awq | 4 | 50.84 | 280.81 | 282.44 | 10,177.78 | 10,380.06 | 79.68 | 2,174.20 | 0 |
| awq | 8 | 106.87 | 75.43 | 96.63 | 9,743.27 | 9,804.75 | 76.51 | 2,268.20 | 0 |
| awq | 8 | 106.73 | 74.80 | 124.88 | 9,658.11 | 9,815.30 | 76.71 | 2,268.20 | 0 |
| awq | 8 | 107.31 | 77.62 | 92.44 | 9,539.06 | 9,812.56 | 76.69 | 2,268.20 | 0 |
| awq | 16 | 212.27 | 122.39 | 123.71 | 9,634.85 | 9,853.78 | 76.64 | 2,588.20 | 0 |
| awq | 16 | 210.93 | 123.25 | 124.53 | 9,694.22 | 9,991.12 | 77.69 | 2,588.20 | 0 |
| awq | 16 | 207.97 | 122.29 | 123.90 | 9,834.81 | 10,052.28 | 78.20 | 2,588.20 | 0 |
| hf | 1 | 27.92 | 33.25 | 51.79 | 4,393.61 | 5,110.40 | 39.98 | 3,742.20 | 0 |
| hf | 1 | 27.81 | 32.97 | 51.70 | 4,468.29 | 5,091.63 | 39.81 | 3,742.20 | 0 |
| hf | 1 | 27.85 | 33.53 | 43.81 | 4,444.12 | 5,128.97 | 40.23 | 3,742.20 | 0 |
| hf | 4 | 95.41 | 41.46 | 59.59 | 5,161.32 | 5,813.34 | 45.46 | 3,762.20 | 0 |
| hf | 4 | 96.37 | 37.22 | 51.25 | 5,111.67 | 5,788.02 | 45.29 | 3,762.20 | 0 |
| hf | 4 | 95.36 | 37.28 | 54.96 | 5,243.23 | 5,785.15 | 45.25 | 3,762.20 | 0 |
| hf | 8 | 190.65 | 38.97 | 47.42 | 5,256.26 | 5,736.25 | 44.87 | 3,960.20 | 0 |
| hf | 8 | 186.16 | 41.20 | 52.97 | 5,464.30 | 5,885.22 | 45.99 | 3,960.20 | 0 |
| hf | 8 | 181.22 | 38.72 | 62.96 | 5,470.62 | 6,073.69 | 47.52 | 3,960.20 | 0 |
| hf | 16 | 351.69 | 72.64 | 73.72 | 5,505.18 | 6,218.92 | 48.39 | 4,380.20 | 0 |
| hf | 16 | 354.89 | 70.60 | 72.28 | 5,402.37 | 6,131.59 | 47.71 | 4,380.20 | 0 |
| hf | 16 | 369.60 | 71.46 | 73.33 | 5,311.91 | 5,707.05 | 44.36 | 4,380.20 | 0 |
| vllm | 1 | 60.83 | 46.09 | 55.15 | 2,050.11 | 2,326.95 | 17.95 | 12,964.20 | 0 |
| vllm | 1 | 63.02 | 32.19 | 50.42 | 2,049.19 | 2,093.54 | 16.18 | 12,964.20 | 0 |
| vllm | 1 | 63.21 | 33.90 | 47.59 | 2,022.70 | 2,051.74 | 15.87 | 12,964.20 | 0 |
| vllm | 4 | 245.13 | 62.58 | 87.39 | 2,076.00 | 2,093.07 | 15.91 | 12,964.20 | 0 |
| vllm | 4 | 245.66 | 58.37 | 79.11 | 2,074.40 | 2,084.83 | 15.88 | 12,964.20 | 0 |
| vllm | 4 | 245.63 | 59.24 | 77.78 | 2,071.85 | 2,091.41 | 15.90 | 12,964.20 | 0 |
| vllm | 8 | 460.14 | 74.74 | 88.94 | 2,212.22 | 2,231.00 | 16.87 | 12,964.20 | 0 |
| vllm | 8 | 457.72 | 88.68 | 118.39 | 2,212.52 | 2,246.48 | 16.85 | 12,964.20 | 0 |
| vllm | 8 | 461.68 | 69.30 | 75.92 | 2,206.22 | 2,211.86 | 16.84 | 12,964.20 | 0 |
| vllm | 16 | 814.13 | 110.10 | 134.02 | 2,496.89 | 2,528.17 | 18.85 | 12,968.20 | 0 |
| vllm | 16 | 815.86 | 98.15 | 118.65 | 2,492.24 | 2,510.38 | 18.95 | 12,968.20 | 0 |
| vllm | 16 | 810.61 | 117.35 | 145.29 | 2,498.15 | 2,536.85 | 18.97 | 12,968.20 | 0 |
| vllm_awq | 1 | 125.92 | 39.10 | 47.65 | 1,018.20 | 1,039.81 | 7.87 | 12,942.20 | 0 |
| vllm_awq | 1 | 124.82 | 31.51 | 58.70 | 1,026.37 | 1,044.90 | 7.91 | 12,942.20 | 0 |
| vllm_awq | 1 | 127.25 | 34.42 | 46.85 | 1,003.84 | 1,014.39 | 7.65 | 12,942.20 | 0 |
| vllm_awq | 4 | 482.02 | 60.76 | 90.45 | 1,053.52 | 1,083.83 | 7.87 | 12,942.20 | 0 |
| vllm_awq | 4 | 477.62 | 64.64 | 95.49 | 1,062.23 | 1,090.03 | 7.94 | 12,942.20 | 0 |
| vllm_awq | 4 | 483.41 | 59.15 | 64.67 | 1,054.71 | 1,067.78 | 7.90 | 12,942.20 | 0 |
| vllm_awq | 8 | 866.79 | 79.98 | 105.05 | 1,175.35 | 1,200.24 | 8.70 | 12,942.20 | 0 |
| vllm_awq | 8 | 874.70 | 74.39 | 77.75 | 1,172.14 | 1,177.16 | 8.69 | 12,942.20 | 0 |
| vllm_awq | 8 | 871.34 | 71.53 | 108.22 | 1,168.16 | 1,198.84 | 8.66 | 12,942.20 | 0 |
| vllm_awq | 16 | 1,375.03 | 111.46 | 125.10 | 1,486.20 | 1,491.95 | 10.86 | 12,946.20 | 0 |
| vllm_awq | 16 | 1,384.75 | 102.44 | 105.80 | 1,477.88 | 1,479.46 | 10.84 | 12,946.20 | 0 |
| vllm_awq | 16 | 1,394.72 | 93.72 | 97.18 | 1,466.42 | 1,469.72 | 10.93 | 12,946.20 | 0 |

## Recommendation

Pick **`vllm_awq`** for the latency/throughput-sensitive path: it delivered **1,384.8 output tokens/sec** at concurrency 16, with MMLU reported as a point estimate plus a 95% interval; see the caveats before interpreting quality deltas. Peak GPU memory was **12,946 MiB**, sampled during the measured window.

### Why this setup wins

- **Same engine, different weights:** compare this row directly with `vllm` FP16 to isolate quantization from serving-engine effects.
- **4-bit AWQ weights** reduce memory traffic and free VRAM for KV cache inside the same paged-attention runtime.
- Any throughput difference from `vllm` FP16 is a kernel/quantization effect, not a batching effect.
- vLLM beating plain Transformers at high concurrency is a known result; the useful contribution here is the reproducible harness and the explicit uncertainty reporting.

### What to put on a resume

> Built a reproducible harness for 4 inference setups at 1–16 concurrent users on Tesla T4; measured **vllm_awq** at **1,384.8 output tok/s** (3.86× HF) and quantified **AWQ's 41% VRAM reduction vs HF in the same engine**; same-engine vLLM FP16 813.5 vs AWQ 1,384.8 output tok/s (1.70×).

## Caveats

- A T4 is memory-bandwidth constrained; rankings can change on A100/H100 class GPUs.
- GGUF and 4-bit kernels are build-sensitive. Record the exact package versions from `results/raw/env_*.json`.
- MMLU used a fixed subset of n=200 per backend. Wilson 95% intervals are in the headline table; point-estimate deltas are not statistically distinguishable at this size.
- The configured 2-pp quality gate is smaller than one question at n=200. It is a screening heuristic, not a significance test.
- 3 repeats were measured per concurrency level. The headline table shows sample SD and the chart shows ±SD whiskers; a Student-t 95% interval is also stored in `summary.json`. Small n still means wide intervals.
- `vllm_awq` is included. Compare it directly with `vllm` FP16 to isolate quantization from serving-engine effects. HF vs AWQ still isolates quantization inside the HF engine.
- vLLM's peak GPU memory includes preallocated KV cache. It is not a model-weight-only footprint and should not be used as a direct memory-efficiency comparison.
- Streaming chunk counts are used as token arrival times for llama.cpp; vLLM/HF use native token IDs.
- On native Windows, vLLM is unavailable and GPTQ wheels may lack CUDA kernels. Check `raw/env_*.json` and `raw/quality_*.json` before treating GPTQ quality numbers as valid.
- If a backend processes concurrent requests sequentially, its throughput rows are repeated single-stream measurements, not shared-server batching.
