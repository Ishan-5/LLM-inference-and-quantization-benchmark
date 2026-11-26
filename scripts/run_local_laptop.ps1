# Reduced local benchmark on RTX 3050 Laptop 4 GB.
# Native Windows: hf, llamacpp, awq, gptq. vLLM needs WSL2/Colab.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Virtual environment not found. Run scripts\bootstrap_windows_gpu.ps1 first."
}

$env:PYTHONPATH = "$root\src;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$smoke = $args -contains "--smoke"
$backends = "hf,llamacpp,awq,gptq"
$extra = @()
if ($smoke) { $extra += @("--limit-prompts", "8") }

& $py -m llmbench.cli doctor
& $py -m llmbench.cli prepare --config configs/laptop_rtx3050.yaml --backends $backends
& $py -m llmbench.cli all --config configs/laptop_rtx3050.yaml --backends $backends --results-dir results_laptop @extra

Write-Host "Done. Open results_laptop\tradeoff_table.md"
