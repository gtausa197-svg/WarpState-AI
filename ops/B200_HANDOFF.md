# Continue WarpState on the B200

This is the starting point for sessions running directly in
`/workspace/WarpState` on the **single visible B200** instance. Use the current
session state below; the original transfer details remain for provenance.

**Latest session state, September 10, 2026:** training is **STOPPED** at the
user's request. The graceful stop finished step **680 / 89,128,960 tokens**,
evaluated validation loss **4.39469**, and durably published the full checkpoint
at `work/run/latest.pt` before exit. This is the checkpoint for the next resume.

Checkpoint SHA256:
`bf689a10a030842699bfa3a25ff62e488cfa2d5b022b2b6e3bfc860359eb8b4f`.
Inspection verified all 2,005,111,424 finite FP32 parameters, 256 full AdamW
entries at step 680, FP32 moments, rank-zero RNG structure, unchanged data and
resume signature, and scheduled optimizer LR **0.000102**. The full inspection
is `work/validation/b200/paused-checkpoint-step-680.json`. The final CPU staging
took 4.93 seconds and background publication took 17.74 seconds; shutdown
waited for publication. No training GPU process remains.

Earlier snapshots `work/run/paused-step-488.pt` and
`work/run/validated-step-500.pt` remain preserved. The step-500 verification is
historical; use the newer step-680 report for the current resume state.
Resume only after the user requests it, using the controls below.

The B200 launcher now uses microbatch 16 / accumulation 4, compiled full loss,
BF16 kernel autotuning, and asynchronous full checkpoints. See
[B200_PERFORMANCE.md](B200_PERFORMANCE.md) for all measurements and validation,
and [B200_SCHEDULING.md](B200_SCHEDULING.md) for evidence linking recurring waits
to provider GPU scheduling. Check Supervisor status before starting another
job. The deployment record and handover tables below describe the original
step-448 transfer, not the current training cursor.

## First commands

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
source rtx6000_40b/ops/workspace-tools.sh
git status --short --branch
git log -3 --oneline
rtx6000_40b/ops/b200ctl.sh status
nvidia-smi
df -h /workspace/WarpState
rtx6000_40b/.server-venv/bin/python -m json.tool \
  rtx6000_40b/work/validation/b200/deployment.json
```

The generated `work/validation/b200/deployment.json` is the final deployment
record: use it to check artifact hashes, environment validation, and the
handover state. If it is missing or reports pending work, finish those checks
before training. Reading this document alone does not establish that a large
file transfer completed. Published `train.bin` and `latest.pt` must have
verified full SHA256 hashes; `.uploading` files are incomplete transfers.

The shell setup keeps caches and temporary files inside the workspace and
adds the project-local Git installation to `PATH`. Source both scripts in
each new shell. All source and Git history have been transferred; imported
history includes `a7b25a9`, followed by the handoff documentation commit. Use
`git log` for the actual current revision. Keep the repository on `main`,
follow the root [AGENTS.md](../../AGENTS.md), and commit completed source
changes using `WarpState Automation <warpstate@localhost>`. Generated data,
environments, checkpoints, and validation reports remain ignored.
This is an independent Git copy with no push remote configured; commits on
the B200 are local to that instance until explicitly synchronized elsewhere.

## Environment and preserved training state

| Item | Handover state |
| --- | --- |
| Hardware | One NVIDIA B200; Ubuntu 24.04; Python 3.12 |
| Python environment | `/workspace/WarpState/rtx6000_40b/.server-venv` |
| Dependencies | PyTorch `2.14.0+cu130`; pinned [training requirements](training-requirements-resolved.txt) and [B200 requirements](b200-requirements.txt) |
| Model | 2,005,111,424 parameters; original tracked 32,768-token tokenizer |
| Training corpus | `work/data/train.bin`: 40,000,000,000 `uint16` token IDs; 80,000,000,000 bytes |
| Validation corpus | `work/data/val.bin`: 1,048,576 `uint16` token IDs; 2,097,152 bytes |
| Checkpoint | `work/run/latest.pt`: full checkpoint at step **448**, with **58,720,256** tokens consumed |
| Restored state | Model, full FP32 AdamW optimizer, token cursor, learning-rate schedule, and rank-zero random state |
| Precision | **BF16 computation only**, as explicitly required by the user; parameters, gradients, and AdamW state stay FP32 |

Paths beginning with `work/` above are relative to `rtx6000_40b/`. Preserve
the copied data manifest and preparation state. The original request was
revision `main`; the data recipe pins the immutable upstream revision
separately. Changing the requested revision for this existing work directory
breaks its resume signature. The existing data requires no new download or
tokenization.

The host's GCC/G++ 13.2 and Python development headers passed actual Inductor
compilation. No additional compiler setup is needed. The launcher also
supports an optional `tmp/remote-setup/toolchain-env.sh` if the toolchain is
changed later. Provisioning details are in [B200_SETUP.md](B200_SETUP.md).

## Resume and pause controls

Use [b200ctl.sh](b200ctl.sh) with [b200-supervisord.conf](b200-supervisord.conf).
Both `autostart` and `autorestart` are disabled. Starting the Supervisor alone
leaves training stopped.

If `b200ctl.sh status` reports that the control socket is absent, first check
whether the daemon is already starting. Initialize it only when absent:

```bash
mkdir -p rtx6000_40b/work/control rtx6000_40b/work/logs
rtx6000_40b/.server-venv/bin/supervisord \
  -c rtx6000_40b/ops/b200-supervisord.conf
rtx6000_40b/ops/b200ctl.sh status
```

After deployment verification and a user request to resume training:

```bash
rtx6000_40b/ops/b200ctl.sh start warpstate-train
rtx6000_40b/ops/b200ctl.sh status
tail -n 30 rtx6000_40b/work/logs/b200-train.log
tail -n 5 rtx6000_40b/work/run/metrics.jsonl
```

Initial core compilation can take several minutes. Confirm the log restores
step 448, or a later checkpoint after subsequent training, and then reports
finite loss and completed optimizer updates. Measure throughput after warmup.
To follow the live log, use `tail -f` on
`rtx6000_40b/work/logs/b200-train.log`.

For interactive monitoring, the separate `warpstate-b200` tmux session has a
`training` window with the live training log and a second pane showing GPU
usage. Attach from the repository root:

```bash
rtx6000_40b/ops/tmux-workspace.sh attach -t warpstate-b200
```

Detach with `Ctrl-b`, then `d`. Training remains managed by Supervisor, so
detaching or closing a monitoring pane does not stop training. Continue to
use `b200ctl.sh` for start/stop controls. The wrapper keeps the tmux socket
under `work/control`; workspace-local installation details are recorded in
[tmux-workspace-install.md](tmux-workspace-install.md).

To pause and preserve the full continuation state:

```bash
rtx6000_40b/ops/b200ctl.sh stop warpstate-train
rtx6000_40b/ops/b200ctl.sh status
tail -n 30 rtx6000_40b/work/logs/b200-train.log
```

Stopping requests a graceful save after the current optimizer update and
allows up to 15 minutes. Verify the saved-checkpoint message and `STOPPED`
status. The next start resumes `latest.pt`. Preserve this checkpoint;
**never replace it with a fresh run or use `--from-scratch` to resume**.

The original `train-fast.sh` hardcodes the workstation's two GPU UUIDs.
The original `supervisord.conf`/`datactl.sh` belongs to that workstation,
specifies user `root`, and can automatically start preparation. Use the B200
controls on this instance. Run only one training launcher against `work`.

## Current single-GPU configuration

[train-b200.sh](train-b200.sh) selects one GPU and uses `--reconfigure` to
continue the existing full checkpoint with its data, objective, and global batch
unchanged:

| Setting | Value |
| --- | --- |
| Sequence length | 2,048 |
| Microbatch / accumulation | 16 / 4 |
| Tokens per optimizer update | **131,072** |
| Core compilation | Enabled; `max-autotune-no-cudagraphs` |
| Core activation checkpointing | Disabled |
| Loss | Compiled full vocabulary loss |
| BF16 autocast cache | Enabled |
| Allocator cap | 150 GiB |
| Additional required free VRAM at startup | 3 GiB beyond the cap |
| Required free SSD space at startup | At least 53 GiB beyond existing files |
| Save / evaluation interval | Every 50 / 100 updates |
| Checkpoint writing | Immutable CPU staging, then asynchronous serialization and atomic publication |

The allocator cap excludes CUDA context and library memory. The selected
preset measured about 72.5K median tokens/s and 139.3 GiB peak reserved memory.
Sustained throughput is much lower and varies with provider scheduling waits;
see [B200_PERFORMANCE.md](B200_PERFORMANCE.md) for complete measurements and
limitations. The dataset, 40B token budget, global batch and LR schedule remain
unchanged. Autotuning can extend initial compilation; eight compiler workers
are used by default.

Asynchronous saves need approximately 24 GB of additional host memory. CPU
staging finishes before the next update; only serialization and disk I/O overlap
training. One outstanding save bounds memory, failures stop the job, and graceful
exit waits for publication. Wait for the **Saved** message, not just **Staged**.
`--no-async-checkpoint` restores synchronous saves. CUDA graphs are validated
as an optional `--compile-mode reduce-overhead` override with persistent FP32
gradient buffers; they were not selected as the default. cuBLASLt preference
remains an optional experiment rather than a default override.

## Original deployment validation

The remote CPU suite passed **92 tests**, with **5 GPU cases deselected**.
This provider hung during collection with `CUDA_VISIBLE_DEVICES=''`; keep
normal GPU visibility and use the explicit test selection documented in
[B200_SETUP.md](B200_SETUP.md). Do not copy the workstation's empty-mask CPU
test command into this environment.

Both single-GPU full-state resume cases also passed (**2 passed, 3
deselected**, 80.37 seconds). Actual BF16 Inductor compilation passed with
one graph and 38 generated kernels; two fused AdamW updates retained finite
FP32 parameters, gradients and moments. These checks used small fixtures;
the production checkpoint was preserved. The large transfers' final hashes,
checkpoint inspection and final readiness state are recorded in
`work/validation/b200/deployment.json`; consult that record for completion.
Multi-GPU tests require more than this instance's one GPU.

The deployment checks above describe the original transfer. Current source
checks, full-state continuation tests, and production checkpoint validation are
recorded in [B200_PERFORMANCE.md](B200_PERFORMANCE.md). Read the current
checkpoint inspection and confirm job status before any later resume.
