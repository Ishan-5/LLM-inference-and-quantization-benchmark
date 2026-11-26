#!/usr/bin/env bash
# Run from a Colab T4 runtime: !bash scripts/bootstrap_colab.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-vllm.txt
python -m pip install -q -r requirements-awq.txt
python -m pip install -q -r requirements-gptq.txt

# Build llama.cpp with CUDA support. If this fails, use the prebuilt wheel line below.
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  python -m pip install -q -r requirements-llamacpp.txt || \
  python -m pip install -q llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

python -m pip install -q -e .
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m llmbench.cli doctor
