# Inference & quantization tradeoff report

**Hardware:** Tesla T4 (14,913 MiB VRAM)
**Baseline:** `hf` · **Concurrent users tested:** 1, 4, 8, 16

## Headline table (highest concurrency tested)

| Setup | Concurrency | Out tok/s | Total tok/s | E2E p95 ms | TTFT p95 ms | Peak GPU MiB | MMLU % | WikiText PPL | ΔMMLU | ΔPPL | Speedup vs hf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 16 | 222.10 | 453.52 | 9,264.46 | 130.31 | 2,588.20 | 47.50 | 30.81 | -10.00 | +1.908 | 0.58 |
| hf | 16 | 379.90 | 777.11 | 5,733.60 | 76.33 | 4,380.20 | 57.50 | 28.90 | +0.00 | +0.000 | 1.00 |
| llamacpp | 16 | 5.26 | 10.00 | 365,643.37 | 345,755.96 | 450.20 | 55.00 | 29.93 | -2.50 | +1.031 | 0.01 |
| vllm | 16 | 809.97 | 1,656.85 | 2,523.64 | 115.07 | 12,968.20 | 57.50 | 28.89 | +0.00 | -0.011 | 2.13 |

## Per-concurrency detail

| Setup | Concurrency | Out tok/s | TTFT p50 ms | TTFT p95 ms | E2E p50 ms | E2E p95 ms | TPOT p95 ms | Peak GPU MiB | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awq | 1 | 15.63 | 57.80 | 84.85 | 8,297.42 | 8,510.75 | 66.44 | 2,172.20 | 0 |
| awq | 4 | 54.22 | 281.93 | 286.35 | 9,607.77 | 9,747.92 | 74.50 | 2,174.20 | 0 |
| awq | 8 | 117.81 | 77.55 | 86.46 | 8,677.94 | 9,051.05 | 70.68 | 2,268.20 | 0 |
| awq | 16 | 222.10 | 127.80 | 130.31 | 9,208.50 | 9,264.46 | 71.92 | 2,588.20 | 0 |
| hf | 1 | 30.52 | 30.36 | 43.29 | 4,003.57 | 4,660.13 | 36.46 | 3,742.20 | 0 |
| hf | 4 | 104.13 | 34.69 | 43.59 | 4,766.68 | 5,357.97 | 41.92 | 3,762.20 | 0 |
| hf | 8 | 203.79 | 39.38 | 42.93 | 4,967.83 | 5,260.39 | 41.08 | 3,960.20 | 0 |
| hf | 16 | 379.90 | 74.10 | 76.33 | 5,360.70 | 5,733.60 | 44.58 | 4,380.20 | 0 |
| llamacpp | 1 | 5.20 | 3,896.10 | 5,740.63 | 24,281.89 | 25,340.05 | 162.01 | 450.20 | 0 |
| llamacpp | 4 | 5.13 | 39,147.76 | 79,158.92 | 57,152.79 | 99,397.40 | 166.19 | 450.20 | 0 |
| llamacpp | 8 | 5.22 | 85,100.19 | 170,754.19 | 104,730.70 | 190,780.68 | 161.86 | 450.20 | 0 |
| llamacpp | 16 | 5.26 | 178,375.22 | 345,755.96 | 198,332.30 | 365,643.37 | 160.90 | 450.20 | 0 |
| vllm | 1 | 55.57 | 47.79 | 55.63 | 2,296.78 | 2,577.19 | 19.94 | 12,964.20 | 0 |
| vllm | 4 | 245.31 | 56.69 | 70.17 | 2,078.31 | 2,092.90 | 15.96 | 12,964.20 | 0 |
| vllm | 8 | 458.70 | 72.98 | 100.03 | 2,215.86 | 2,242.79 | 16.90 | 12,964.20 | 0 |
| vllm | 16 | 809.97 | 110.61 | 115.07 | 2,515.51 | 2,523.64 | 19.11 | 12,968.20 | 0 |

## Recommendation

Pick **`vllm`** for the latency/throughput-sensitive path: it delivered **810.0 output tokens/sec** at concurrency 16 within the 2-point quality gate (ΔMMLU +0.00 pp). Peak GPU memory was **12,968 MiB**.

### Why this setup wins

- **Continuous batching** admits new requests as soon as a sequence finishes, so the GPU is not idle while waiting for the slowest member of a static batch.
- **PagedAttention** stores the KV cache in fixed-size blocks, reducing fragmentation and letting more concurrent sequences stay resident.
- Expect the largest gain at concurrency 4–16; at concurrency 1 the advantage is mostly kernel/runtime overhead.

### What to put on a resume

> Benchmarked 4 inference setups under 16 concurrent users; found **vllm** gives **2.13×** throughput versus plain Transformers at **0.00** points of MMLU change.

## Caveats

- A T4 is memory-bandwidth constrained; rankings can change on A100/H100 class GPUs.
- GGUF and 4-bit kernels are build-sensitive. Record the exact package versions from `results/raw/env_*.json`.
- MMLU here is a small fixed subset, not a full leaderboard score. Use it only to detect regression, not to claim state of the art.
- Streaming chunk counts are used as token arrival times for llama.cpp; vLLM/HF use native token IDs.
- On native Windows, vLLM is unavailable and GPTQ wheels may lack CUDA kernels. Check `raw/env_*.json` and `raw/quality_*.json` before treating GPTQ quality numbers as valid.
- If a backend processes concurrent requests sequentially, its throughput rows are repeated single-stream measurements, not shared-server batching.
