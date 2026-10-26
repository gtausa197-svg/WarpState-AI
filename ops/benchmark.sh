#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
cd /workspace/WarpState/rtx6000_40b
export CUDA_VISIBLE_DEVICES="${WARPSTATE_BENCH_DEVICES:-0}"
if [[ "${WARPSTATE_BENCH_PROCESSES:-1}" == 1 ]]; then
  exec .server-venv/bin/python -u ops/benchmark_training.py "$@"
else
  exec .server-venv/bin/torchrun --standalone \
    --nproc-per-node="$WARPSTATE_BENCH_PROCESSES" \
    --log-dir=work/logs/torchrun-benchmark ops/benchmark_training.py "$@"
fi
