# Resume the B200 checkpoint on the RTX workstation

This configuration continues the B200 training run on the local RTX PRO 6000
workstation.
The source checkpoint was saved on the B200 at **step 680 / 89,128,960
tokens**, with validation loss **4.39469** and SHA256:

```text
bf689a10a030842699bfa3a25ff62e488cfa2d5b022b2b6e3bfc860359eb8b4f
```

After the verified import, `work/run/latest.pt` is the local resume checkpoint.
Later local saves replace it atomically; the checksum above identifies the
imported step-680 checkpoint. [B200_HANDOFF.md](B200_HANDOFF.md) and the B200
performance documents record the remote run and its historical controls.
Use the workstation controls below on this host.
The B200 source history was imported through commit `752563e`. The generated
local migration record is `work/validation/local-resume/import.json`, relative
to `rtx6000_40b`; it records the verified checkpoint and continuation state.

[`transfer_artifacts.py`](transfer_artifacts.py) accepts `--pull` to retrieve
a remote source into a local destination with parallel SSH connections.
Use `--expected-sha256` with the checkpoint hash above, keep the private key
on the local host, and use the existing verified host-key file. The download
uses resumable chunks and verifies the entire file before publishing its
final name. No second download of the unchanged 40B-token corpus is needed.

The prepared 40,000,000,000 training tokens and 1,048,576 validation tokens
are unchanged. Their immutable upstream revision, tokenizer hash, split rule,
and preparation dependencies remain in
[`../configs/data_40b.json`](../configs/data_40b.json). The existing data
signature retains the original requested revision `main`, which was resolved
to the pinned upstream commit. Preserve that signature when resuming.

## Local execution

[`train-local-resume.sh`](train-local-resume.sh) requires an existing full
checkpoint and selects both physical workstation GPUs by UUID. Both GPUs
became available before the local resume. This launcher uses the measured
BF16 configuration and continues the existing run. Check both GPUs'
availability before every start:

| Setting | Value |
| --- | --- |
| Sequence length | 2,048 |
| Microbatch per GPU / gradient accumulation | 4 / 8 |
| GPUs / global tokens per optimizer update | 2 / 131,072 |
| Parameters, gradients, full AdamW moments | FP32 on the GPU |
| Forward/backward computation / residual activations | BF16 autocast / BF16 |
| Core activation checkpointing | Disabled |
| Core compilation / autocast weight cache | Enabled, default mode / enabled |
| Vocabulary loss | Compiled full loss |
| Gradient execution / communication | Compiled autograd with delayed waits/copies; FP32 reduction, 128 MiB buckets |
| Allocator cap / extra startup headroom per GPU | 88 GiB / 3 GiB |
| Checkpoint writing | Synchronous immutable CPU snapshot; background serialization and atomic replacement |
| Save / evaluation interval | Every 100 / 100 updates |
| Validation display | Fresh results only, labeled with their evaluation step |

The current preset includes gradient conversion/accumulation fusion and moves
collective waits with their dependent copies past independent backward GEMMs.
The repeated 20-update comparison measured **31,045 tokens/s** versus
**30,578 tokens/s** without those changes (**1.53% faster**), with **66.98 GiB
reserved per GPU**. Both GPU ranks remain data-parallel; communication overlaps
their local computation. See [OPTIMIZATION_20260912.md](OPTIMIZATION_20260912.md)
for the profile, numerical comparisons and exact two-GPU checkpoint/resume tests.
Shape-specific BF16 matrix plans are available through `--matmul-plan`. The
experimental fused-epilogue/TMA recipe measured 31,156 versus 31,022 tokens/s
with the same overlap mode in one comparison (+0.43%); that additional gain is
not yet confirmed across repeated windows, so the launcher keeps its current
matrix implementation. See [MATMUL_TUNING.md](MATMUL_TUNING.md) for the recipe,
tile tuning, persistent kernels, precision checks, and full-model measurements.
The earlier loss/residual changes are documented in
[RTX_PROFILE_20260910.md](RTX_PROFILE_20260910.md), and asynchronous checkpoints
in [RTX_ASYNC_TRAINING.md](RTX_ASYNC_TRAINING.md).

These rates exclude evaluation, checkpoint pauses and compilation. Compiled
autograd can spend approximately three minutes compiling its first full update;
even with cached kernels, each new process has graph-capture/setup costs.
The allocator cap excludes CUDA context/library allocations. Startup requires
91 GiB free on each GPU and does not reserve memory for other jobs.
Keep at least **53 GiB of free SSD space** beyond existing files for full
checkpoint writing and replacement.

Asynchronous saves use about **22.4 GiB of additional host memory** for one
full model/optimizer snapshot. GPU-to-CPU staging still pauses both workers;
serialization, file synchronization and publication overlap later updates.
There is at most one outstanding snapshot, and a new save waits for the
previous write if necessary. A logged `Staged step` is not yet durable: use
the corresponding `Saved ... (step ..., ... background write ...)` message.
Normal completion and graceful stop wait for the final write. A save failure
is reported to both workers at the next coordinated check, save or close and
stops training; a failure during an update can allow that update to finish.

Training step lines display `val ... (step ...)` only when that step has a
fresh validation result. Every step still records `val_loss`, `val_step` and
`val_tokens` in `metrics.jsonl`; these fields retain the latest measured
result between evaluations. Checkpoints retain the same validation metadata.
This adds no per-step evaluation work.
Startup still evaluates the resumed weights, and graceful stop/final completion
still evaluates and saves even when the step is not a multiple of 100.

`--reconfigure` preserves the model, full optimizer, token cursor, objective,
and learning-rate schedule while recording execution changes in the checkpoint.
The original B200 import restored rank-zero random states and seeded the new
rank reproducibly. Subsequent two-GPU resumes restore both ranks' states.
BF16 residual activations retain FP32 master parameters, gradients and Adam
moments, but change activation rounding. A small validation change at the
same saved weights is not training progress. Disabling them requires
`--no-bf16-residuals` with an explicit reconfiguration.

## Controls and current progress

Run from `/workspace/WarpState`:

```bash
rtx6000_40b/ops/datactl.sh status
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
rtx6000_40b/ops/datactl.sh start warpstate-train
tail -n 10 rtx6000_40b/work/logs/train.log
tail -n 3 rtx6000_40b/work/run/metrics.jsonl
```

The status command and latest metrics identify the actual running state and
current cursor. This document records the import and launcher configuration;
it does not imply that a job is running. Preparation and training both have
automatic startup and restart disabled. After changing Supervisor
configuration, use `datactl.sh reread` and `datactl.sh update` before an
explicit start.

Stop with `rtx6000_40b/ops/datactl.sh stop warpstate-train`. The launcher
requests a stop through its stop file, waits for the current update and full
checkpoint, and has a 900-second Supervisor stop allowance. Check the saved
step in the log before treating an in-memory update as durable.

## Resuming after a deliberate pause

Keep a deliberately paused run stopped until the user requests resumption.
As requested on 2026-09-14, the workstation Supervisor command includes
`--stop-at-step 76294`: the trainer finishes optimizer update 76,294, evaluates,
and publishes the full checkpoint before exiting. This boundary corresponds to
10,000,007,168 consumed tokens. No update 76,295 runs. Async checkpoint writing
is flushed before exit, and Supervisor has automatic restart disabled.

`--stop-at-step` is an absolute limit, so an intervening restart cannot move
the boundary. `--max-steps` still limits additional updates in one invocation;
if both are supplied, the earlier boundary wins. Neither option changes the
40B-token learning-rate schedule or the checkpoint's training signature.
The active boundary is also recorded in `work/run/settings.json` under
`runtime.stop_at_step` and `runtime.end_step`.

When the user requests training beyond this boundary, first remove or increase
`--stop-at-step` in the `warpstate-train` command in `ops/supervisord.conf`, then
run `ops/datactl.sh reread` and `ops/datactl.sh update warpstate-train` from the
package directory before starting the job. A restart retaining a boundary at
or below the loaded checkpoint step performs no further optimizer updates.

The generated `work/validation/paused-run.json` records the last verified
pause: checkpoint path, SHA256, step, tokens consumed, validation loss,
optimizer/RNG checks, execution settings, source commit and next update.
`work/validation/paused-run.sha256` provides a directly checkable checksum.
Both records are excluded from Git. Subsequent training replaces `latest.pt`,
so these records identify the last deliberate pause rather than live progress.

Before resuming a stopped run, inspect its saved state and available hardware:

```bash
cd /workspace/WarpState
rtx6000_40b/ops/datactl.sh status
cat rtx6000_40b/work/validation/paused-run.json
sha256sum -c rtx6000_40b/work/validation/paused-run.sha256
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

Use the existing full checkpoint at `work/run/latest.pt`, which contains
model weights, full AdamW state and both ranks' RNG states. Preserve
`work/data/`, `work/run/`, the `.server-venv` environment, tokenizer and
existing launchers. No data preparation or checkpoint conversion is needed.
Both workstation GPUs must have at least 91 GiB free before launch. The
launcher retains BF16 compute, the 131,072-token global batch, the 40B-token
schedule, asynchronous saves, and 100-step save/evaluation intervals.

When resumption is requested:

```bash
cd /workspace/WarpState
rtx6000_40b/ops/datactl.sh start warpstate-train
tmux attach -t warpstate-train
```

The launcher clears the previous stop request under its process lock. If the
host has rebooted and Supervisor is absent, initialize it once with
`/usr/local/bin/supervisord -c /workspace/WarpState/rtx6000_40b/ops/supervisord.conf`
before using the controls. Its training and preparation jobs start paused.
If the tmux monitor was lost during reboot, recreate it with:

```bash
tmux new-session -d -s warpstate-train -n training -c /workspace/WarpState \
  'tail -n 30 -F /workspace/WarpState/rtx6000_40b/work/logs/train.log'
```

Keep the active token files, their manifest/preparation state, the latest full
checkpoint, active environment, and controls required to continue training.
Git retains source, pinned dependency specifications, tokenizer, and input
provenance for rebuilding. Obsolete checkpoints, experimental outputs,
download staging, caches, and disposable test artifacts can be removed once
their replacement or source recipe is verified; generated runtime files stay
out of Git.
