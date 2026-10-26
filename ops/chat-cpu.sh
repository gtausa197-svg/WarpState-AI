#!/usr/bin/env bash
set -euo pipefail
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
# CPU-only inference keeps both training GPUs untouched.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
exec .server-venv/bin/python -B -u chat.py "$@"
