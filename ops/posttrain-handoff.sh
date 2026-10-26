#!/usr/bin/env bash
set -euo pipefail
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
exec rtx6000_40b/.server-venv/bin/python -B \
  rtx6000_40b/ops/posttrain_handoff.py \
  --config rtx6000_40b/configs/posttrain_10b.json "$@"
