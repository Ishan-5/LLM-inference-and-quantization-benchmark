# Local Windows smoke-test runner. Use a GPU machine for comparable numbers.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -m llmbench.cli doctor
python -m llmbench.cli prepare --config configs/benchmark.yaml --backends hf,llamacpp
python -m llmbench.cli all --config configs/benchmark.yaml --backends hf,llamacpp --limit-prompts 8

Write-Host "Done. Open results\tradeoff_table.md"
