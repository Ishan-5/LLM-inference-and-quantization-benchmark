# Claims ledger

Use this table when presenting the project. Every number in the README or dashboard should map to a row here.

| Claim | Status | Evidence | Caveat / scope |
|---|---|---|---|
| vLLM delivered **809.97 out tok/s @16**, **2.13× HF** | Published | [`results/tradeoff_table.md`](results/tradeoff_table.md), [`results/raw/benchmark_vllm.json`](results/raw/benchmark_vllm.json), [`results/raw/benchmark_hf.json`](results/raw/benchmark_hf.json) | One repeat; p95 has no error bars. Known result, not novelty. |
| vLLM E2E p95 **2,524 ms** vs HF **5,734 ms** | Published | [`results/tradeoff_table.md`](results/tradeoff_table.md) | Same workload, same GPU; one repeat. vLLM memory includes KV-cache preallocation. |
| AWQ uses **41% less peak VRAM than HF** (2,588 vs 4,380 MiB) | Published | [`results/raw/benchmark_awq.json`](results/raw/benchmark_awq.json), [`results/raw/benchmark_hf.json`](results/raw/benchmark_hf.json) | Same HF engine, so quantization is isolated. Do **not** say 80%; that compares AWQ to vLLM's preallocated cache. |
| AWQ throughput **222 tok/s** vs HF **380 tok/s** | Published | [`results/tradeoff_table.md`](results/tradeoff_table.md) | On this T4 and kernel stack. Cross-engine comparison with vLLM is confounded. |
| vLLM and HF both scored **57.5% MMLU** | Published | [`results/raw/quality_vllm.json`](results/raw/quality_vllm.json), [`results/raw/quality_hf.json`](results/raw/quality_hf.json) | Same FP16 weights + greedy decoding. Expected; not evidence of a quality advantage. |
| AWQ MMLU point estimate **47.5%** | Published but weak | [`results/raw/quality_awq.json`](results/raw/quality_awq.json), [`results/tradeoff_table.md`](results/tradeoff_table.md) | n=40, Wilson 95% CI 32.9–62.5 overlaps all others. Treat as noise at this sample size. |
| llama.cpp used **450 MiB** and is a deployment tradeoff | Published | [`results/raw/benchmark_llamacpp.json`](results/raw/benchmark_llamacpp.json) | Measured path processes concurrent requests sequentially; rows are repeated single-stream measurements. |
| GPTQ excluded from final portfolio numbers | Published | [`results/raw/env_*.json`](results/raw/env_*.json), [`results_laptop/tradeoff_table.md`](results_laptop/tradeoff_table.md) | `auto_gptq` unavailable in Colab; Windows GPTQ quality invalid without CUDA kernels. |
| Dashboard is live | Done | [GitHub Pages](https://ishan-5.github.io/LLM-inference-and-quantization-benchmark/), [`docs/data/summary.json`](docs/data/summary.json) | `docs/data/summary.json` is a snapshot. Regenerate it after any results rerun. |
| Same-engine vLLM FP16 vs AWQ | **Prepared, not run** | [`configs/benchmark_v2.yaml`](configs/benchmark_v2.yaml), [`COLAB_V2_STEPS.md`](COLAB_V2_STEPS.md) | Run on T4 before making a same-engine quantization claim. |
| MMLU intervals with n=200 | **Prepared, not run** | [`configs/benchmark_v2.yaml`](configs/benchmark_v2.yaml) | Published run remains n=40. Do not quote v2 numbers until `results_v2/` exists. |
| Performance error bars | **Prepared, not run** | [`configs/benchmark_v2.yaml`](configs/benchmark_v2.yaml), [`docs/index.html`](docs/index.html) | Report already supports ±SD and t-CI; chart shows whiskers once v2 data is loaded. |
| Automated checks | Done | [`tests/test_report.py`](tests/test_report.py), [`tests/test_config.py`](tests/test_config.py) | Run locally with `python -m unittest discover -s tests -v`. |

## Rule of thumb

- **Publishable now:** throughput, tail latency, VRAM, environment, harness reproducibility.
- **Not publishable as findings:** MMLU deltas at n=40, p95 variance, any vLLM-vs-AWQ quantization claim.
- **After v2:** same-engine quantization and quality/error-bar claims become defensible.
