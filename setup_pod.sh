#!/usr/bin/env bash
# One-shot setup for a RunPod GPU pod (template: RunPod PyTorch, CUDA 12.x, >=24GB VRAM).
# Creates two isolated envs (Unsloth for training, vLLM for serving) and builds the dataset.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME=/workspace/hf_cache
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
[ -f .env ] && set -a && source .env && set +a

# 1) Training env
uv venv .venv-train --python 3.12
uv pip install --python .venv-train unsloth huggingface_hub flash-linear-attention
uv pip install --python .venv-train causal-conv1d --no-build-isolation || echo "causal-conv1d optional, skipped"

# 2) Serving env (kept separate so torch versions never fight)
uv venv .venv-serve --python 3.12
uv pip install --python .venv-serve vllm httpx ninja

# 3) Dataset: Together's open tev1 pipeline (37,840 train / 4,568 dev)
if [ ! -f tev1/data/new-v1/instruction/train.jsonl ]; then
  [ -d tev1 ] || git clone --depth 1 https://github.com/togethercomputer/tev1
  (cd tev1 && uv sync --locked && PYTHONUTF8=1 uv run python fetch_sources.py \
    && PYTHONUTF8=1 uv run python build_all.py)
fi
wc -l tev1/data/new-v1/instruction/*.jsonl
echo "Setup done. Next: bash train.sh"
