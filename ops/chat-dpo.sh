#!/usr/bin/env bash
set -euo pipefail
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
exec rtx6000_40b/.server-venv/bin/python -B rtx6000_40b/ops/posttrain_chat.py "$@"
