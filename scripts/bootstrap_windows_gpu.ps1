# Native Windows setup for an NVIDIA laptop GPU.
# Verified target here: RTX 3050 Laptop 4 GB, driver 610.91 (CUDA 13.x capable).
# NOTE: vLLM does not support native Windows. Use WSL2 or Colab for vLLM.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvDir = Join-Path $root ".venv"
$py = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[0/5] Creating project-local virtual environment..."
    python -m venv $venvDir
}

Write-Host "[1/5] Installing CUDA-enabled PyTorch 2.13.0+cu130..."
& $py -m pip install --upgrade pip
& $py -m pip install --force-reinstall --no-deps `
    torch==2.13.0+cu130 `
    --index-url https://download.pytorch.org/whl/cu130 `
    --retries 10 --timeout 120

Write-Host "[2/5] Installing PyTorch dependencies and pinned local stack..."
& $py -m pip install -r requirements.txt `
    filelock typing-extensions sympy networkx jinja2 fsspec
& $py -m pip install -r requirements-local-windows.txt

Write-Host "[3/5] Installing the working CUDA llama.cpp wheel..."
# 0.3.37 cu130/cu124 fail on this Ryzen 5 5600H with a CPU illegal-instruction
# error. The 0.2.88 cu121 prebuilt wheel loads and runs with GPU offload.
$wheel = Join-Path $env:TEMP "llama_cpp_python-0.2.88-cp310-cp310-win_amd64.whl"
$url = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.88-cu121/llama_cpp_python-0.2.88-cp310-cp310-win_amd64.whl"
$expectedBytes = 396897941
if (-not (Test-Path $wheel) -or (Get-Item $wheel).Length -lt $expectedBytes) {
    curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 3 -C - -o $wheel $url
}
& $py -m pip install --ignore-installed --no-deps $wheel

Write-Host "[4/5] Installing this project..."
& $py -m pip install -e .

$env:PYTHONPATH = "$root\src;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
& $py -m llmbench.cli doctor

Write-Host "Done. Use .venv\Scripts\python.exe for all benchmark commands."
