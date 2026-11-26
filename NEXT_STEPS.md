# What you have to do next

## Current status

- **Done:** final portfolio run on Colab T4 with `configs/benchmark.yaml` (Qwen2.5-1.5B, concurrency 1/4/8/16, `hf`, `vllm`, `llamacpp`, `awq`). See `results/tradeoff_table.md`. All benchmark rows have 0 errors, non-zero output tokens, and GPU memory readings.
- **Winner:** `vllm` at 809.97 out tok/s @16, 57.50% MMLU, PPL 28.89, 12,968 MiB peak, 2.13× faster than HF.
- **Done locally:** reduced RTX 3050 Laptop run with `hf`, `llamacpp`, `awq`, and `gptq`. See `results_laptop/tradeoff_table.md`.
- **Not valid locally:** GPTQ quality. The Windows wheel has no CUDA kernels; its PPL/MMLU rows must be ignored.
- **Not available locally:** vLLM. It needs WSL2 or Linux.
- **Skipped in final run:** GPTQ, because `auto_gptq` was unavailable in the Colab environment. GPTQ is optional and does not affect the valid four-backend comparison.

Nothing is left to run unless you want optional extra repeats or a GPTQ comparison. The instructions below are kept for reproducing the T4 run.

## 1. Choose where to run

Recommended for final portfolio numbers: **Colab with a T4 GPU**.

- Zip this folder or push it to GitHub.
- Open Colab → Runtime → Change runtime type → **T4 GPU**.
- Upload/clone the folder so `scripts/` and `configs/` are present.

You can also run a reduced version locally on the RTX 3050 Laptop 4 GB. See `LOCAL_LAPTOP.md`. It uses `configs/laptop_rtx3050.yaml` and Qwen2.5-0.5B so everything fits in 4 GB. Native Windows cannot run vLLM; use WSL2 or Colab for that setup.

## 2. Install

In a Colab cell:

```bash
!bash scripts/bootstrap_colab.sh
```

If you use Kaggle:

```bash
!bash scripts/bootstrap_kaggle.sh
```

## 3. Verify the GPU

```bash
!python -m llmbench.cli doctor
```

You want to see `gpu_name: Tesla T4`, `cuda_available: True`, and the optional backends listed. If a backend shows `None`, install only its `requirements-*.txt`.

## 4. Smoke test (important)

```bash
!python -m llmbench.cli all --backends hf,vllm,llamacpp,awq --limit-prompts 8
```

Open `results/raw/benchmark_*.json`. Confirm:
- `errors` is 0,
- `output_tokens` is not 0,
- `peak_gpu_mib` is not `null`,
- TTFT is smaller than E2E latency.

If something fails, fix that backend before the full run.

## 5. Full run

```bash
!bash scripts/run_all.sh
```

Expected time on T4: roughly 45–90 minutes depending on downloads and whether all five backends are included.

To include GPTQ as a fifth setup:

```bash
!python -m llmbench.cli all --backends hf,vllm,llamacpp,awq,gptq
!python -m llmbench.cli report
```

## 6. Read and polish the output

- `results/tradeoff_table.md` — auto-generated tradeoff table + recommendation.
- `results/summary.csv` — numbers for charts.
- `results/raw/*.json` — evidence and package versions.

Then edit `README.md`:
1. Replace the placeholder resume line with your generated one.
2. Add a 5–8 sentence "What I found" section using your actual numbers.
3. Add 2–3 plots from `summary.csv` if you want a stronger portfolio artifact.

## 7. If something breaks

| Symptom | Action |
|---|---|
| `No module named llmbench` | Run `pip install -e .` or set `PYTHONPATH=src` |
| vLLM install fails | Skip it with `--backends hf,llamacpp,awq`; retry later |
| llama.cpp is CPU-only | Install CUDA wheel/build; if impossible, label it as CPU, don't compare as GPU |
| AWQ OOM | Lower `gpu_memory_utilization` for vLLM or run backends one at a time (the CLI already does this) |
| MMLU numbers look random | Increase `mmlu_examples` to 100; treat it as regression detection only |
| p95 jumps around | Set `repeats: 2` or `3` in `configs/benchmark.yaml` |

## 8. Minimum viable portfolio result

If you only have 2–3 hours, run this instead:

```bash
!python -m llmbench.cli all --backends hf,vllm,awq --limit-prompts 16
!python -m llmbench.cli report
```

That still gives you three setups, concurrency 1/4/8/16, quality delta, and a recommendation.
