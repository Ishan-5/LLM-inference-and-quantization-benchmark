# Colab v2 run — prepared, not yet executed

This run closes the three documented gaps **without overwriting the published `results/`**:

| Gap | Published run | v2 |
|---|---|---|
| MMLU size | n=40 | **n=200** |
| Repeats per concurrency | 1 | **3** |
| Same-engine quantization | HF FP16 vs HF AWQ only | **vLLM FP16 vs vLLM AWQ** |

Expected T4 time: **~90–150 minutes** (3× repeats + 5× MMLU + one extra backend).

---

## Step 1 — get the updated repo into Colab

Either clone the repo, or zip the project from Windows and upload it.

```python
!git clone https://github.com/Ishan-5/LLM-inference-and-quantization-benchmark.git /content/llmbench
%cd /content/llmbench
```

If you already have an old copy in the session, pull first:

```python
%cd /content/llmbench
!git pull
```

Confirm the new files exist:

```python
!ls configs/benchmark_v2.yaml scripts/run_all_v2.sh COLAB_V2_STEPS.md
```

---

## Step 2 — runtime and install

Runtime → Change runtime type → **T4 GPU**.

```python
!nvidia-smi
!bash scripts/bootstrap_colab.sh
```

Verify:

```python
import os
os.environ["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!python -m llmbench.cli doctor
```

You want: `Tesla T4`, `cuda_available: True`, and non-`None` `vllm`, `llama_cpp`, `autoawq`.

---

## Step 3 — smoke test (do not skip)

```python
!python -m llmbench.cli all \
  --config configs/benchmark_v2.yaml \
  --backends hf,vllm,llamacpp,awq,vllm_awq \
  --limit-prompts 8 \
  --results-dir results_v2_smoke
```

Validate:

```python
import json, glob, os

for p in sorted(glob.glob("results_v2_smoke/raw/benchmark_*.json")):
    rows = json.load(open(p, encoding="utf-8"))
    print(
        os.path.basename(p),
        "errors=", sum(r.get("errors", 0) for r in rows),
        "out_tokens=", sum(r.get("output_tokens", 0) for r in rows),
        "gpu_ok=", all(r.get("peak_gpu_mib") is not None for r in rows),
        "rows=", len(rows),
    )
```

Expected for every backend: `errors=0`, `out_tokens>0`, `gpu_ok=True`, `rows=12`
(4 levels × 3 repeats).

If `vllm_awq` fails to load, stop and fix it before the full run. Do not substitute the
HF-AWQ row for it — the whole point is the same-engine comparison.

### If `llama_cpp` is missing

If the smoke run stops with `ModuleNotFoundError: No module named 'llama_cpp'`, install the
prebuilt CUDA wheel and re-run only the remaining backends:

```python
!python -m pip install -q "llama-cpp-python==0.3.38" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
!python -m llmbench.cli doctor

!python -m llmbench.cli all --config configs/benchmark_v2.yaml --backends llamacpp,awq,vllm_awq --limit-prompts 8 --results-dir results_v2_smoke
```

`hf` and `vllm` rows from the first attempt are still valid; do not repeat them just to
fix llama.cpp. Re-run the validation cell and continue only when all five files exist.

`RuntimeError: Event loop is closed` from vLLM during shutdown, after `[vllm] quality eval`
has printed, is harmless cleanup noise. Confirm with the JSON validation cell instead.

### If llama.cpp smoke runs for over an hour

The measured llama.cpp path processes concurrent requests sequentially, so the c=8/c=16
smoke batches can take far longer than the other backends. llama.cpp is **not required**
for the v2 gaps (n=200 MMLU, repeats, same-engine quantization). Interrupt the cell and
continue with only the missing backends:

```python
!python -m llmbench.cli all --config configs/benchmark_v2.yaml --backends awq,vllm_awq --limit-prompts 8 --results-dir results_v2_smoke
```

Then expect 4 benchmark files: `hf`, `vllm`, `awq`, `vllm_awq`. Use the v2 run command
below with `BACKENDS=hf,vllm,awq,vllm_awq` to skip llama.cpp in the full run too.

---

## Step 4 — full v2 run

```python
!bash scripts/run_all_v2.sh
```

If llama.cpp was skipped in smoke, use this instead:

```python
!BACKENDS=hf,vllm,awq,vllm_awq bash scripts/run_all_v2.sh
```

Equivalent explicit command:

```python
!python -m llmbench.cli all \
  --config configs/benchmark_v2.yaml \
  --backends hf,vllm,llamacpp,awq,vllm_awq \
  --results-dir results_v2
```

Keep the tab active. Backends run one at a time in separate processes.

---

## Step 5 — validate and read

```python
import json, glob, os

for p in sorted(glob.glob("results_v2/raw/benchmark_*.json")):
    rows = json.load(open(p, encoding="utf-8"))
    print(
        os.path.basename(p),
        "errors=", sum(r.get("errors", 0) for r in rows),
        "out_tokens=", sum(r.get("output_tokens", 0) for r in rows),
        "rows=", len(rows),
    )

for p in sorted(glob.glob("results_v2/raw/quality_*.json")):
    q = json.load(open(p, encoding="utf-8"))
    print(os.path.basename(p), "MMLU n=", q.get("mmlu_total"), "note=", q.get("mmlu_note") or "ok")
```

Expected: `errors=0`, `out_tokens>0`, `rows=12` per backend (4 levels × 3 repeats), `MMLU n=200`.

Then open the report:

```python
from pathlib import Path
print(Path("results_v2/tradeoff_table.md").read_text(encoding="utf-8"))
```

Look for the new columns: **MMLU 95% CI**, **Out tok/s ±SD**, **E2E p95 ±SD**.

---

## Step 6 — download and hand back

```python
!zip -rq results_v2.zip results_v2
from google.colab import files
files.download("results_v2.zip")
```

Upload `results_v2.zip` back here. I will then:

1. generate the v2 report and dashboard data,
2. add error bars to the charts,
3. update `CLAIMS.md`,
4. commit and push.

The published `results/` stays untouched until you decide whether v2 becomes the new headline.
