#!/usr/bin/env bash
# Fine-tune + push to Hugging Face. Pass extra flags through, e.g. --max_train 200 for a smoke test.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export HF_HOME=/workspace/hf_cache HF_XET_HIGH_PERFORMANCE=1 PYTHONWARNINGS=ignore
.venv-train/bin/python train_unsloth.py --hf_repo "$HF_REPO" "$@" 2>&1 | tee train.log
