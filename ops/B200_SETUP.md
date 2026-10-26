# Single-B200 training environment

A new deployment starts **paused**. See [B200_HANDOFF.md](B200_HANDOFF.md)
for the latest checkpoint and current job status. Starting its Supervisor
does not start training. Use this configuration on the B200 host after
verifying the transferred dataset and checkpoint.

The repository and all environments, caches, data, logs, and temporary files
must remain under `/workspace/WarpState`. The Python 3.12 environment at
`rtx6000_40b/.server-venv` needs
[`training-requirements-resolved.txt`](training-requirements-resolved.txt)
and `supervisor==4.3.0`, combined in
[`b200-requirements.txt`](b200-requirements.txt). Source
[`training-env.sh`](training-env.sh) before provisioning. If present, the launcher also sources the generated
`/workspace/WarpState/tmp/remote-setup/toolchain-env.sh` for the host's
workspace-local compiler toolchain.

For a fresh environment on Ubuntu 24.04 x86-64 with Python 3.12 and a
CUDA-13-compatible driver:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
python3 -m venv rtx6000_40b/.server-venv
rtx6000_40b/.server-venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r rtx6000_40b/ops/b200-requirements.txt
rtx6000_40b/.server-venv/bin/python -m pip check
```

A copy of the existing environment is also usable when the OS architecture,
Python version and absolute environment path match. Verify its dependency
versions and CUDA execution on the destination; omit compiler caches from
the transfer. The system needs GCC/G++, libc and Python development headers
for compilation. Do not install an additional CUDA driver inside the project.

If the host has no Git, download and extract its distribution package under
the workspace instead of modifying system directories:

```bash
mkdir -p /workspace/WarpState/tmp/remote-setup /workspace/WarpState/.tools
cd /workspace/WarpState/tmp/remote-setup
apt-get download git
dpkg-deb -x git_*.deb /workspace/WarpState/.tools
cd /workspace/WarpState
source rtx6000_40b/ops/workspace-tools.sh
git --version
```

This uses the host's existing shared libraries. Source
[`workspace-tools.sh`](workspace-tools.sh) in subsequent shells to use this
Git installation. Transfer Git history with a bundle of all refs; exclude
private keys, local credentials and generated artifacts from that bundle.

[`train-b200.sh`](train-b200.sh) uses one visible GPU, defaulting to index 0;
an existing `CUDA_VISIBLE_DEVICES` value takes precedence. It selects
sequence length 2,048, microbatch 16, and accumulation 4, preserving
**131,072 tokens per optimizer update**. Matrix computation uses BF16
autocast; parameters, gradients, and full AdamW state remain FP32. Core
compilation uses `max-autotune-no-cudagraphs`, core activation checkpointing
is disabled, and full vocabulary loss is compiled. Full checkpoints use
immutable CPU staging followed by asynchronous writing. The 150 GiB allocator
cap requires another 3 GiB free at startup and excludes CUDA context and
library memory. Current measurements and validation are recorded in
[B200_PERFORMANCE.md](B200_PERFORMANCE.md).

Copy the complete prepared `work/data/` directory, including its original
manifest, and `work/run/latest.pt`. The original transferred checkpoint was
step 448 with 58,720,256 tokens consumed; consult the handoff for the latest
saved step. Keep the manifest's requested revision `main`; its immutable source revision is recorded separately. The runner
restores model, full optimizer, token cursor, learning-rate schedule, and
rank-zero random state. `--reconfigure` permits the transition from two
GPUs to one while checking the unchanged data, objective, and global batch.
Different grouping can produce floating-point rounding differences.

## Environment checks

On this host, setting `CUDA_VISIBLE_DEVICES=''` caused pytest collection
to hang. Keep normal device visibility and explicitly deselect the CUDA
training cases when running the CPU suite:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
rtx6000_40b/.server-venv/bin/python -m pytest \
  rtx6000_40b/test_server.py rtx6000_40b/test_training_execution.py \
  rtx6000_40b/test_training_launcher.py rtx6000_40b/test_loss_execution.py \
  rtx6000_40b/test_checkpoint_writer.py \
  rtx6000_40b/test_residual_execution.py rtx6000_40b/test_artifact_transfer.py \
  -k 'not gpu_full_state_resume_and_exact_partial_update and not single_gpu_compiled_full_state_resume and not two_gpu_full_state_resume_and_empty_final_rank and not one_to_two_gpu_reconfiguration_preserves_training_state' \
  --basetemp=/workspace/WarpState/tmp/b200-cpu-tests -q
```

This runs the CPU tests and performs device-availability queries during
collection. Small CUDA resume tests can be run separately with the B200
available, using their own temporary checkpoints:

```bash
rtx6000_40b/.server-venv/bin/python -m pytest rtx6000_40b/test_server.py \
  -k gpu_full_state_resume_and_exact_partial_update \
  --basetemp=/workspace/WarpState/tmp/b200-cuda-tests -q
```

The prepared corpus occupies 80,002,097,152 bytes, and the full checkpoint
about 24.06 GB. Keep at least 53 GiB additionally free for checkpoint writing
and atomic replacement. Generated files remain ignored by Git.

## Large artifact transfers

[`transfer_artifacts.py`](transfer_artifacts.py) sends large files over
independent SSH connections when one connection limits throughput. Run it
on the source host with the destination's already verified host-key file:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
python3 rtx6000_40b/ops/transfer_artifacts.py \
  /workspace/WarpState/rtx6000_40b/work/data/train.bin \
  /workspace/WarpState/rtx6000_40b/work/data/train.bin \
  --host ubuntu@HOST --port PORT \
  --key /workspace/WarpState/id_ed25519 \
  --known-hosts /workspace/WarpState/tmp/remote-setup/known_hosts \
  --connections 12 --chunk-mib 512 --compress \
  --expected-sha256 237a9ea6a90831769ec22fb8089c2a26c02fb991492629bb28f2b5b10d6b9114
```

Use the same command with the checkpoint's source/destination paths and
its actual SHA256. Copy the small data manifest, preparation state,
validation tokens and run metadata separately. The uploader retains
`.uploading`, `.upload-state.json`, `.upload-chunks/` and `.upload.lock`
sidecars next to each destination. These are inside ignored `work/`.
Interrupted uploads resume by rerunning the same command; completed chunks
are checked again before reuse. Full source and destination hashes must
match before the final filename appears. An existing different destination
is preserved. Keep training stopped until all artifacts have been verified.

## Controls on the remote host

Initialize the dedicated Supervisor once after provisioning:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
mkdir -p rtx6000_40b/work/control rtx6000_40b/work/logs
rtx6000_40b/.server-venv/bin/supervisord \
  -c rtx6000_40b/ops/b200-supervisord.conf
rtx6000_40b/ops/b200ctl.sh status
```

It runs as the invoking user and uses its own `b200-*` control files and
logs. It does not manage preparation or benchmarks. Training remains
stopped until explicitly started and never restarts automatically after
failure or a daemon restart.

When ready to resume on the B200:

```bash
cd /workspace/WarpState
nvidia-smi
rtx6000_40b/ops/b200ctl.sh start warpstate-train
rtx6000_40b/ops/b200ctl.sh status
tail -n 10 rtx6000_40b/work/logs/b200-train.log
tail -n 3 rtx6000_40b/work/run/metrics.jsonl
```

To pause and save a full checkpoint:

```bash
rtx6000_40b/ops/b200ctl.sh stop warpstate-train
```

The stop request waits for the current optimizer update and checkpoint,
with a 15-minute Supervisor timeout. The next start resumes `latest.pt`.
Initial compilation can take several minutes. Use `b200ctl.sh` for this
daemon; the original `datactl.sh` addresses the separate workstation
Supervisor. Do not run two launchers against the same `work` directory.
