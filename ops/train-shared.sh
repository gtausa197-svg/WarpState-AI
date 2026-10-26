#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
# Select physical GPU 1 by UUID; it appears as cuda:0 inside this process.
export CUDA_VISIBLE_DEVICES="${WARPSTATE_CUDA_DEVICE:-GPU-a32efffc-8eb9-fb3e-b951-6f8cea2edd91}"
cd /workspace/WarpState/rtx6000_40b
exec .server-venv/bin/python -u run_40b.py \
  --no-bootstrap --stage train --from-scratch \
  --work-dir /workspace/WarpState/rtx6000_40b/work \
  --seq-len 2048 --micro-batch 1 --grad-accum 64 \
  --activation-checkpointing --no-autocast-cache --loss-chunk-size 256 \
  --vram-limit-gib 33.5 --vram-headroom-gib 3 \
  --workers 4 --eval-batches 8 --eval-every 100 --save-every 50 \
  "$@"
