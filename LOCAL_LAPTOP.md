# Running locally on the RTX 3050 Laptop

## Hardware detected here

- GPU: **NVIDIA GeForce RTX 3050 Laptop GPU**
- VRAM: **4 GB**
- Driver: **610.91**, CUDA UMD 13.3
- Python: 3.10.2
- Current PyTorch: **2.13.0+cu130** (installed in the project `.venv`; CUDA is working)

So: **yes, you can do a meaningful version here**, but not the identical 1.5B/T4 run.

---

## The two constraints

### 1. 4 GB VRAM is too small for the 1.5B FP16 baseline

`Qwen2.5-1.5B-Instruct` in FP16 is about **3.1 GB of weights alone**. Once you add the KV cache, activations, and CUDA context, plain HF will very likely OOM on 4 GB.

Solution: use `configs/laptop_rtx3050.yaml`, which switches **all setups** to:

`Qwen/Qwen2.5-0.5B-Instruct`

Official quantized variants exist and were verified:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct-AWQ`
- `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4`
- `bartowski/Qwen2.5-0.5B-Instruct-GGUF` → `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

Same model family, same tokenizer lineage, same workload — still a fair serving comparison.

### 2. vLLM does not support native Windows

Native Windows can run:

- `hf`
- `llamacpp`
- `awq`
- `gptq`

For `vLLM`, choose one:

- **Colab/Kaggle/T4** — easiest, recommended for final numbers
- **WSL2 with CUDA** — possible, more setup

---

## Verified working stack on this laptop

The following combination was installed and import-verified:

- `torch==2.13.0+cu130` — CUDA works, `cuda_capability 8.6`
- `transformers==4.51.3` — pinned for AWQ/GPTQ compatibility
- `datasets==3.6.0`, `huggingface-hub==0.36.2`, `peft==0.11.1`
- `triton-windows==3.8.0.post29` — satisfies AWQ Triton kernels on Windows
- `autoawq==0.2.9` — imports with a deprecation warning and loads quantized weights, but the Windows build skips fused kernels (`awq_ext` missing). Treat local AWQ speed as an unfused reference, not a tuned AWQ result.
- `auto-gptq==0.7.1` — imports, but the Windows wheel has no CUDA kernels. The Triton fallback crashes with `triton-windows`, and the non-Triton path returns invalid logits, so **GPTQ quality results on this laptop are not trustworthy**. Use GPTQ speed only as a rough signal; get real GPTQ numbers from the Linux/T4 run.
- `llama-cpp-python==0.2.88` prebuilt **cu121** wheel — this is the working wheel on this Ryzen 5 5600H. It loads the GGUF with GPU offload and runs generation/quality checks. The newer `0.3.37` cu130/cu124 wheels crash with a CPU illegal-instruction error, so do not use them here.

Exact pins are saved in `requirements-local-windows.txt`.

## Local reduced run (native Windows)

### Step 1 — install CUDA PyTorch

```powershell
powershell -File scripts\bootstrap_windows_gpu.ps1
```

This creates a project-local `.venv`, installs `torch==2.13.0+cu130` from the PyTorch CUDA index, and leaves your global Python untouched.

Verify:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m llmbench.cli doctor
```

You must see:

```text
cuda_available : True
gpu_name       : NVIDIA GeForce RTX 3050 Laptop GPU
```

If it still says `False`, stop and fix PyTorch before benchmarking.

Use `.venv\Scripts\python.exe` for every command below.

### Step 2 — smoke test

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m llmbench.cli all `
  --config configs\laptop_rtx3050.yaml `
  --backends hf,llamacpp,awq,gptq `
  --limit-prompts 8
```

### Step 3 — full local run

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m llmbench.cli all `
  --config configs\laptop_rtx3050.yaml `
  --backends hf,llamacpp,awq,gptq `
  --results-dir results_laptop
```

Results land in `results_laptop/`, separate from the T4 run.

Shortcut for the same flow:

```powershell
powershell -File scripts\run_local_laptop.ps1
powershell -File scripts\run_local_laptop.ps1 --smoke
```

---

## What the local run is good for

- Validating the harness end to end
- Seeing quantization tradeoffs on real hardware
- Getting a laptop-specific table: `results_laptop/tradeoff_table.md`
- Learning how concurrency 1/2/4 behaves on a 4 GB GPU

## What the local run is **not** good for

- Claiming 1.5B inference performance
- Comparing directly against a T4 table
- Representing production server throughput — a 60 W laptop GPU is thermally and power limited
- Treating llama.cpp concurrency rows like a shared server. This adapter runs each request in sequence, so its throughput is a single-stream number repeated for the concurrency column. Use the T4 vLLM rows for true concurrent serving.

Best approach: **use the local run to prove the methodology, then use Colab T4/1.5B for the final portfolio table.** Mention both in the README: "validated locally on RTX 3050 4 GB; final numbers collected on T4 16 GB."
