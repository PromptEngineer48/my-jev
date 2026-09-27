#!/usr/bin/env bash
# Serve the fine-tuned model with an OpenAI-compatible API on port 8000.
# On RunPod expose port 8000; public URL: https://<POD_ID>-8000.proxy.runpod.net/v1
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export HF_HOME=/workspace/hf_cache PATH="$PWD/.venv-serve/bin:$PATH"
MODEL="${1:-$HF_REPO}"
.venv-serve/bin/vllm serve "$MODEL" \
  --served-model-name my-jev \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --api-key "${MY_JEV_API_KEY:-local}"
