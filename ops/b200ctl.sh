#!/usr/bin/env bash
set -euo pipefail
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
exec /workspace/WarpState/rtx6000_40b/.server-venv/bin/supervisorctl \
  -c /workspace/WarpState/rtx6000_40b/ops/b200-supervisord.conf "$@"
