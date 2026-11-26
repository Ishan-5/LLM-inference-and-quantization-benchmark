#!/usr/bin/env bash
# Kaggle GPU notebook cell: !bash scripts/bootstrap_kaggle.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-vllm.txt
python -m pip install -q -r requirements-awq.txt
python -m pip install -q -r requirements-gptq.txt

CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  python -m pip install -q -r requirements-llamacpp.txt || \
  python -m pip install -q llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

python -m pip install -q -e .
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m llmbench.cli doctor
