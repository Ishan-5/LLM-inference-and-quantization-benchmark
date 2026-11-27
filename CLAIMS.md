# Claims ledger

Use this table when presenting the project. Every number in the README or dashboard should map to a row here. The **v2 T4 run is the primary portfolio result**; v1 remains a reproducible baseline that includes llama.cpp.

| Claim | Status | Evidence | Caveat / scope |
|---|---|---|---|
| `vllm_awq` delivered **1,384.83 out tok/s @16**, **3.86× HF** | Primary | [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md), [`results_v2/raw/benchmark_vllm_awq.json`](results_v2/raw/benchmark_vllm_awq.json), [`results_v2/raw/benchmark_hf.json`](results_v2/raw/benchmark_hf.json) | 3 repeats (SD ±9.85). vLLM peak includes KV-cache preallocation. |
| Same-engine vLLM AWQ is **1.70×** vLLM FP16 (1,384.8 vs 813.5 out tok/s) | Primary | [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md), [`results_v2/raw/benchmark_vllm.json`](results_v2/raw/benchmark_vllm.json), [`results_v2/raw/benchmark_vllm_awq.json`](results_v2/raw/benchmark_vllm_awq.json) | Engine held constant; difference is AWQ weights/kernels, not batching. Do **not** read it as a VRAM saving. |
| vLLM FP16 is **2.27× HF** and cuts E2E p95 from **6,019 ms to 2,525 ms** | Primary | [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md), [`results_v2/raw/benchmark_vllm.json`](results_v2/raw/benchmark_vllm.json) | Known continuous-batching/PagedAttention result, not novelty. Same T4 workload. |
| AWQ uses **41% less peak VRAM than HF** in the HF engine (2,588 vs 4,380 MiB) | Primary | [`results_v2/raw/benchmark_awq.json`](results_v2/raw/benchmark_awq.json), [`results_v2/raw/benchmark_hf.json`](results_v2/raw/benchmark_hf.json) | Same engine isolates quantization. Do **not** say 80%; that compares AWQ to vLLM's preallocated cache. |
| HF-AWQ throughput is lower than HF-FP16 (210 vs 359 out tok/s) | Primary | [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md) | On this T4/kernel stack. "Quantization is slower" is not a universal claim: vLLM-AWQ is faster than vLLM-FP16 here. |
| MMLU at **n=200** shows no statistically distinguishable deltas (49.5–51.0% point estimates; all intervals overlap) | Primary | [`results_v2/raw/quality_*.json`](results_v2/raw/quality_awq.json), [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md) | Wilson 95% CIs overlap. Do not claim a quality win/loss from point estimates. |
| AWQ has higher WikiText PPL (~30.8 vs ~28.9 FP16) | Primary | [`results_v2/tradeoff_table.md`](results_v2/tradeoff_table.md), [`results_v2/raw/quality_awq.json`](results_v2/raw/quality_awq.json), [`results_v2/raw/quality_vllm_awq.json`](results_v2/raw/quality_vllm_awq.json) | Generic language-quality regression, not an MMLU significance result. |
| Performance error bars are reported from **3 repeats** | Primary | [`results_v2/summary.json`](results_v2/summary.json), [`docs/index.html`](docs/index.html) | Headline table shows sample SD; chart shows ±SD whiskers; t-CIs are in `summary.json`. Small n still means uncertainty. |
| v1 llama.cpp used **450 MiB** and is a deployment tradeoff | Supplemental | [`results/tradeoff_table.md`](results/tradeoff_table.md), [`results/raw/benchmark_llamacpp.json`](results/raw/benchmark_llamacpp.json) | Measured path processes concurrent requests sequentially. Not part of the v2 headline. |
| GPTQ excluded from final portfolio numbers | Supplemental | [`results/raw/env_*.json`](results/raw/env_awq.json), [`results_laptop/tradeoff_table.md`](results_laptop/tradeoff_table.md) | `auto_gptq` unavailable in Colab; Windows GPTQ quality invalid without CUDA kernels. |
| Dashboard is live and shows v2 data | Done | [GitHub Pages](https://ishan-5.github.io/LLM-inference-and-quantization-benchmark/), [`docs/data/summary.json`](docs/data/summary.json) | `docs/data/summary.json` is a snapshot of `results_v2/summary.json`; regenerate after any rerun. |
| Reproducibility configs | Done | [`configs/benchmark_v2.yaml`](configs/benchmark_v2.yaml), [`configs/benchmark.yaml`](configs/benchmark.yaml) | v2 reproduces the primary table; v1 reproduces the original table. |
| Automated checks | Done | [`tests/test_report.py`](tests/test_report.py), [`tests/test_config.py`](tests/test_config.py) | Run locally with `python -m unittest discover -s tests -v`. |

## Rule of thumb

- **Publishable now:** same-engine speed comparisons, throughput/tail-latency numbers, the 41% HF-engine VRAM saving, environment evidence, and uncertainty intervals.
- **Not publishable as findings:** statistically significant MMLU deltas, an 80% VRAM cut, a universal "quantization is slower" claim, vLLM VRAM savings from AWQ, or cross-engine confounded comparisons.
- **Framing:** the novelty is the reproducible harness, same-engine comparisons, and explicit uncertainty reporting — not "vLLM beats Transformers".
