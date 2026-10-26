# Workspace-local tmux

The B200 host uses Ubuntu 24.04 amd64 packages extracted into the ignored
`/workspace/WarpState/.tools` directory. Downloads stay in ignored
`/workspace/WarpState/tmp/tmux-install`. No system package installation is
required. [tmux-workspace.sh](tmux-workspace.sh) supplies the local library
path and uses `work/control/b200-tmux.sock` with an empty tmux configuration.
It sources the usual environment scripts without changing their contents.

Package provenance comes from the host's authenticated Ubuntu APT indexes;
the downloaded package SHA256 hashes matched those indexes. Exact sources:

| Package | Version | Download |
| --- | --- | --- |
| `tmux` | `3.4-1ubuntu0.1` | [Ubuntu archive](http://archive.ubuntu.com/ubuntu/pool/main/t/tmux/tmux_3.4-1ubuntu0.1_amd64.deb) |
| `libevent-core-2.1-7t64` | `2.1.12-stable-9ubuntu2.1` | [Ubuntu archive](http://archive.ubuntu.com/ubuntu/pool/main/libe/libevent/libevent-core-2.1-7t64_2.1.12-stable-9ubuntu2.1_amd64.deb) |
| `libutempter0` | `1.2.1-3build1` | [Ubuntu archive](http://archive.ubuntu.com/ubuntu/pool/main/libu/libutempter/libutempter0_1.2.1-3build1_amd64.deb) |

The extracted dependencies are `libevent-core-2.1-7t64` and `libutempter0`.
The host already supplies the remaining direct tmux dependencies: `libc6`
(>= 2.38), `libsystemd0` (>= 221), and `libtinfo6`
(>= 6.4+20230603, < 6.5), together with their dependencies.

To reproduce the installation from the exact package URLs and verify the
expected bytes before extraction:

```bash
set -euo pipefail
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
source rtx6000_40b/ops/workspace-tools.sh
mkdir -p tmp/tmux-install .tools
cd tmp/tmux-install
curl --fail --location --output tmux_3.4-1ubuntu0.1_amd64.deb \
  http://archive.ubuntu.com/ubuntu/pool/main/t/tmux/tmux_3.4-1ubuntu0.1_amd64.deb
curl --fail --location --output libevent-core-2.1-7t64_2.1.12-stable-9ubuntu2.1_amd64.deb \
  http://archive.ubuntu.com/ubuntu/pool/main/libe/libevent/libevent-core-2.1-7t64_2.1.12-stable-9ubuntu2.1_amd64.deb
curl --fail --location --output libutempter0_1.2.1-3build1_amd64.deb \
  http://archive.ubuntu.com/ubuntu/pool/main/libu/libutempter/libutempter0_1.2.1-3build1_amd64.deb
sha256sum --check <<'SHA256'
2913e17aa61879d1f905aab6f374f153f1ece63aefba1b996e387325d1d5338d  tmux_3.4-1ubuntu0.1_amd64.deb
f8af913312f8de9cab6efcd2259918b3df92f5b7d501aba7ce9a55699fa452b8  libevent-core-2.1-7t64_2.1.12-stable-9ubuntu2.1_amd64.deb
46c0dfbfeca19ba0773398d654b7e7225b178f26b414f0e2f8250519551bfb32  libutempter0_1.2.1-3build1_amd64.deb
SHA256
for tmux_package in \
  tmux_3.4-1ubuntu0.1_amd64.deb \
  libevent-core-2.1-7t64_2.1.12-stable-9ubuntu2.1_amd64.deb \
  libutempter0_1.2.1-3build1_amd64.deb; do
  dpkg-deb -x "$tmux_package" /workspace/WarpState/.tools
done
/workspace/WarpState/rtx6000_40b/ops/tmux-workspace.sh -V
```

The verified binary reports `tmux 3.4`. To inspect sessions, run
`rtx6000_40b/ops/tmux-workspace.sh list-sessions`; attach with
`rtx6000_40b/ops/tmux-workspace.sh attach -t SESSION_NAME`.
