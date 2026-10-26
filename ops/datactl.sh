#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/workspace/WarpState/tmp
export PYTHONDONTWRITEBYTECODE=1
exec /usr/local/bin/supervisorctl \
  -c /workspace/WarpState/rtx6000_40b/ops/supervisord.conf "$@"
