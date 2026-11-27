#!/usr/bin/env bash
# Prepared v2 run: closes MMLU-n, repeats, and same-engine quantization gaps.
# Does NOT overwrite the published results/ directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONFIG="${CONFIG:-configs/benchmark_v2.yaml}"
BACKENDS="${BACKENDS:-hf,vllm,llamacpp,awq,vllm_awq}"
RESULTS_DIR="${RESULTS_DIR:-results_v2}"

python -m llmbench.cli doctor
python -m llmbench.cli prepare --config "$CONFIG" --backends "$BACKENDS"
python -m llmbench.cli all \
  --config "$CONFIG" \
  --backends "$BACKENDS" \
  --results-dir "$RESULTS_DIR"

echo "Done. Open $RESULTS_DIR/tradeoff_table.md"
echo "Upload $RESULTS_DIR as results_v2.zip when back on Windows."
