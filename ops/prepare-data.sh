#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

export TMPDIR=/workspace/WarpState/tmp
export XDG_CACHE_HOME=/workspace/WarpState/.cache
export HF_HOME=/workspace/WarpState/rtx6000_40b/work/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /workspace/WarpState/rtx6000_40b
exec /usr/bin/nice -n 10 .data-venv/bin/python -u ops/prepare_data.py \
  --no-bootstrap \
  --stage prepare \
  --work-dir /workspace/WarpState/rtx6000_40b/work \
  --train-tokens 40000000000 \
  --val-tokens 1048576 \
  --workers 16
