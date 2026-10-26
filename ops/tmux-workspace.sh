#!/usr/bin/env bash
set -euo pipefail
source /workspace/WarpState/rtx6000_40b/ops/training-env.sh
source /workspace/WarpState/rtx6000_40b/ops/workspace-tools.sh
export TMUX_TMPDIR=/workspace/WarpState/rtx6000_40b/work/control
mkdir -p "$TMUX_TMPDIR"
LD_LIBRARY_PATH="/workspace/WarpState/.tools/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  exec /workspace/WarpState/.tools/usr/bin/tmux \
  -S "$TMUX_TMPDIR/b200-tmux.sock" -f /dev/null "$@"
