#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
if [[ -f /workspace/WarpState/tmp/remote-setup/toolchain-env.sh ]]; then
  source /workspace/WarpState/tmp/remote-setup/toolchain-env.sh
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
cd /workspace/WarpState/rtx6000_40b
exec .server-venv/bin/python -u ops/launch_training.py \
  --processes 1 --work-dir /workspace/WarpState/rtx6000_40b/work -- \
  --no-bootstrap --stage train --reconfigure \
  --seq-len 2048 --micro-batch 16 --grad-accum 4 \
  --compile-core --compile-mode max-autotune-no-cudagraphs \
  --no-activation-checkpointing --autocast-cache \
  --loss-mode compiled-full --loss-chunk-size 1024 --async-checkpoint \
  --vram-limit-gib 150 --vram-headroom-gib 3 \
  --workers 4 --eval-batches 8 --eval-micro-batch 1 --eval-every 100 --save-every 50 \
  "$@"
