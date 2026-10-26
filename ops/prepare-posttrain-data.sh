#!/usr/bin/env bash
set -euo pipefail
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
exec rtx6000_40b/work/posttrain-data-venv/bin/python -u -B \
  prepare_posttrain_data.py \
  --recipe rtx6000_40b/configs/posttrain_data.json \
  --out rtx6000_40b/work/posttrain/data "$@"
