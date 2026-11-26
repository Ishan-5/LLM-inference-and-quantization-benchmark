#!/usr/bin/env bash
# Full run: prepare assets, benchmark, quality eval, and build the report.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BACKENDS="${BACKENDS:-hf,vllm,llamacpp,awq}"
CONFIG="${CONFIG:-configs/benchmark.yaml}"

python -m llmbench.cli doctor
python -m llmbench.cli prepare --config "$CONFIG" --backends "$BACKENDS"
python -m llmbench.cli all --config "$CONFIG" --backends "$BACKENDS"

echo "Done. Open results/tradeoff_table.md"
