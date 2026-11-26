# Colab T4 step-by-step

Use this for the **final portfolio numbers** (Qwen2.5-1.5B, vLLM included).
The local RTX 3050 results are already in `results_laptop/`; do not mix them with these.

**Status:** completed. The downloaded `results_final.zip` was extracted into `results/`; the final report is `results/tradeoff_table.md`. These steps are kept for reproduction.

---

## Step 1 — make the upload zip on your Windows PC

Open PowerShell in the project folder and run:

```powershell
$items = @(
  "README.md",
  "NEXT_STEPS.md",
  "pyproject.toml",
  "configs",
  "scripts",
  "src",
  "notebooks",
  "requirements*.txt"
)
Compress-Archive -Path $items -DestinationPath "$env:USERPROFILE\Downloads\llmbench_colab.zip" -Force
```

This excludes `.venv` and the local results. Upload `llmbench_colab.zip` from your Downloads folder.

---

## Step 2 — Colab runtime

1. Go to https://colab.research.google.com and create a **new Python 3 notebook**.
2. Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.
3. Verify GPU:

```python
!nvidia-smi
```

You should see `Tesla T4`.

---

## Step 3 — upload and enter the project

```python
from google.colab import files
uploaded = files.upload()   # choose llmbench_colab.zip
```

```python
!unzip -o llmbench_colab.zip -d /content/llmbench
%cd /content/llmbench
```

Confirm you see `configs`, `scripts`, and `src`:

```python
!ls
```

---

## Step 4 — install

```python
!bash scripts/bootstrap_colab.sh
```

This takes several minutes. If the runtime restarts, reconnect and run just these two cells again:

```python
%cd /content/llmbench
!bash scripts/bootstrap_colab.sh
```

---

## Step 5 — verify

```python
import os
os.environ["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!python -m llmbench.cli doctor
```

You want:

- `cuda_available : True`
- `gpu_name : Tesla T4`
- `vllm`, `llama_cpp`, and `autoawq` not `None`

If `vllm` is `None`, stop and fix that before the benchmark.

---

## Step 6 — download models and eval data

```python
!python -m llmbench.cli prepare --config configs/benchmark.yaml --backends hf,vllm,llamacpp,awq
```

---

## Step 7 — smoke test (do not skip)

```python
!python -m llmbench.cli all --config configs/benchmark.yaml --backends hf,vllm,llamacpp,awq --limit-prompts 8 --results-dir results_smoke
```

Validate the smoke files:

```python
import json, glob, os

for p in sorted(glob.glob("results_smoke/raw/benchmark_*.json")):
    rows = json.load(open(p, encoding="utf-8"))
    print(
        os.path.basename(p),
        "errors=", sum(r.get("errors", 0) for r in rows),
        "out_tokens=", sum(r.get("output_tokens", 0) for r in rows),
        "gpu_ok=", all(r.get("peak_gpu_mib") is not None for r in rows),
    )
```

Expected: `errors= 0`, `out_tokens=` greater than 0, `gpu_ok= True` for every backend.

Also read the smoke table:

```python
from pathlib import Path
print(Path("results_smoke/tradeoff_table.md").read_text(encoding="utf-8"))
```

---

## Step 8 — full run

Keep the tab active. This can take 45–90 minutes.

```python
!bash scripts/run_all.sh
```

Equivalent explicit command if you prefer:

```python
!python -m llmbench.cli all --config configs/benchmark.yaml --backends hf,vllm,llamacpp,awq --results-dir results
```

`run_all.sh` already prepares assets and rebuilds the report.

---

## Step 9 — check final files

```python
import json, glob, os

for p in sorted(glob.glob("results/raw/benchmark_*.json")):
    rows = json.load(open(p, encoding="utf-8"))
    print(
        os.path.basename(p),
        "errors=", sum(r.get("errors", 0) for r in rows),
        "out_tokens=", sum(r.get("output_tokens", 0) for r in rows),
        "gpu_ok=", all(r.get("peak_gpu_mib") is not None for r in rows),
    )
```

Open the final report:

```python
from pathlib import Path
print(Path("results/tradeoff_table.md").read_text(encoding="utf-8"))
```

---

## Step 10 — download results

```python
!zip -rq results_final.zip results
from google.colab import files
files.download("results_final.zip")
```

Copy `results/tradeoff_table.md` and `results/summary.csv` into your project when back on Windows.

---

## Optional Step 11 — add GPTQ

Only do this after the 4-backend run is finished and valid.

```python
!python -m llmbench.cli all --config configs/benchmark.yaml --backends hf,vllm,llamacpp,awq,gptq --results-dir results

!python -m llmbench.cli report --config configs/benchmark.yaml --results-dir results
```

If GPTQ fails to install or crashes, leave it out. The 4-backend table is the main result.

---

## If something breaks

| Problem | Fix |
|---|---|
| `No module named llmbench` | Re-run `%cd /content/llmbench`, then `!pip install -e .` |
| `vllm` is `None` in doctor | `!pip install -q vllm`, then re-run doctor |
| CUDA OOM | Edit `configs/benchmark.yaml`, change `gpu_memory_utilization: 0.85` to `0.75`, rerun that backend |
| Runtime disconnected | Reconnect, `%cd /content/llmbench`, rerun the failed command; raw files are overwritten safely |
| llama.cpp build fails | The bootstrap script already falls back to a prebuilt cu124 wheel; check doctor output |
| gptq install fails | Skip GPTQ; it is optional |
