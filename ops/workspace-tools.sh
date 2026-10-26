#!/usr/bin/env bash
# Source this after training-env.sh when using workspace-local Git packages.
if [[ -x /workspace/WarpState/.tools/usr/bin/git ]]; then
  export PATH="/workspace/WarpState/.tools/usr/bin:$PATH"
  export GIT_EXEC_PATH=/workspace/WarpState/.tools/usr/lib/git-core
  export GIT_TEMPLATE_DIR=/workspace/WarpState/.tools/usr/share/git-core/templates
fi
