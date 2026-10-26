#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
export CUDA_VISIBLE_DEVICES=GPU-7091a369-8eef-e6c9-81b2-679a8e6a78d6,GPU-a32efffc-8eb9-fb3e-b951-6f8cea2edd91
cd /workspace/WarpState/rtx6000_40b
exec .server-venv/bin/python -u ops/launch_training.py \
  --processes 2 --work-dir /workspace/WarpState/rtx6000_40b/work -- \
  --no-bootstrap --stage train --reconfigure \
  --seq-len 2048 --micro-batch 4 --grad-accum 8 \
  --compile-core --no-activation-checkpointing --autocast-cache --loss-chunk-size 1024 \
  --vram-limit-gib 88 --vram-headroom-gib 3 \
  --workers 4 --eval-batches 8 --eval-micro-batch 1 --eval-every 100 --save-every 50 \
  "$@"
