#!/usr/bin/env bash
set -euo pipefail
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
exec rtx6000_40b/.server-venv/bin/python -B \
  rtx6000_40b/ops/dpo_handoff.py --config rtx6000_40b/configs/dpo_10b.json "$@"
