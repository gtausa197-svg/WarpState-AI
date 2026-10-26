# B200 BF16 execution and checkpoint validation

Measurements from September 10, 2026 use disposable copies of the full
step-488 checkpoint (63,963,136 tokens), including FP32 AdamW moments.
The prepared dataset, tokenizer, model architecture, sequence length 2,048,
and global batch of 131,072 tokens remain unchanged. No FP8 or FP16 training
is used. Production resumed from step 488 with validation loss 4.79387 after
the user requested continuation; see the production validation section below.
The preserved input is `work/run/paused-step-488.pt`, SHA256
`69d37e902c41f541d55a07b3d10dcb7643721240cc6aae032455cd8904c64608`.

## Latest pause

Training stopped gracefully at the user's request after step **680 /
89,128,960 tokens**, with training loss **4.60232** and validation loss
**4.39469**. The full `work/run/latest.pt` checkpoint was published before the
process exited. Final staging took **4.93 seconds**, followed by **17.74
seconds** of background writing. All 256 optimizer entries are at step 680;
model/AdamW values are finite FP32, the schedule and data signature are
unchanged, and the saved optimizer LR is **0.000102**. SHA256 and the complete
inspection are recorded in `work/validation/b200/paused-checkpoint-step-680.json`
and the current [handoff](B200_HANDOFF.md).

The final 100 completed updates (581-680) averaged **16,501 wall-clock
tokens/s**, including between-update overhead, with a median **72,418 tokens/s**
and 18 slow updates. This window ends before the final pause evaluation/save.
Peak reserved memory stayed **139.55 GiB**. The earlier measurements below
remain historical comparisons. Production is now stopped; source settings,
prepared data and earlier preserved checkpoint snapshots remain unchanged.

## Selected launcher and measured tradeoffs

[train-b200.sh](train-b200.sh) now selects microbatch 16 / accumulation 4,
compiled full loss, `max-autotune-no-cudagraphs`, and asynchronous full
checkpoints. BF16 autocast, FP32 master weights/gradients/AdamW, the 40B token
budget, original learning-rate schedule, seed, and save/evaluation cadence
remain unchanged. Core activation checkpointing stays disabled and the
allocator cap remains 150 GiB. Eight compiler workers are the default;
initial autotuning can take several minutes.

Every benchmark below uses two warmup and twenty measured updates. Sustained
throughput divides measured tokens by measured update seconds, including
provider scheduling waits. Median speed describes a typical update; it is
not continuous throughput or a guarantee of exclusive GPU capacity. Loading,
compilation, validation and checkpoint writes are outside these measurements.

| Microbatch / accumulation | Loss | Core mode / BLAS preference | Sustained tokens/s | Median tokens/s | Peak reserved GiB |
| --- | --- | --- | ---: | ---: | ---: |
| 4 / 16 | Checkpointed | Default; stall diagnostics | 13,808 | 61,853 | 60.44 |
| 4 / 16 | Checkpointed | Final baseline control | 15,344 | 61,839 | 60.44 |
| 8 / 8 | Checkpointed | Default | 16,362 | 65,636 | 86.09 |
| 16 / 4 | Checkpointed | Default | 16,467 | 67,421 | 137.72 |
| 16 / 4 | Compiled full | Default | 16,698 | 71,370 | 141.25 |
| 16 / 4 | Compiled full | Autotune, no graphs | 20,752 | 72,499 | 139.31 |
| 16 / 4 | Compiled full | Default / cuBLASLt preference | 18,718 | 71,309 | 141.25 |
| 16 / 4 | Compiled full | CUDA graphs, corrected timer | 17,432 | 70,660 | 141.31 |

Autotuning gave the fastest typical update, about 17.2% above the initial
baseline. Peak allocated memory was 138.18 GiB. Its first warmup took 206.16
seconds including compilation/tuning and scheduling waits, excluded above.
The baseline diagnostic run encountered five slow measured updates, versus
three in the autotune run. Scheduling phase contributes substantially to
aggregate differences, so sustained gains cannot be attributed solely to the
chosen execution settings. The final baseline control repeated a median of
61,839 tokens/s, within 0.03% of the original diagnostic baseline despite
different scheduling waits; the typical-speed gain is repeatable.

cuBLASLt preference produced exactly the same losses and gradient norms as
the default compiled-full case and no meaningful median speed improvement.
It is available for comparison but is not forced in the B200 launcher. The
recorded preferences were `_BlasBackend.Cublas` and `_BlasBackend.Cublaslt`;
a preference does not guarantee every operation uses that backend. Current `CUBLAS_WORKSPACE_CONFIG=:4096:8`
provides 32 MiB workspace. See [PyTorch backend controls](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cuda.preferred_blas_library)
and [NVIDIA workspace guidance](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublassetworkspace).

## CUDA graph implementation and evidence

`--compile-mode reduce-overhead` now preallocates FP32 gradient buffers outside
capture and zeros them in place between updates. Without this, the installed
PyTorch rejected accumulation into graph-owned gradient outputs. Other modes
retain `zero_grad(set_to_none=True)`. No per-core step markers were inserted;
completed forward/backward microbatches use PyTorch's iteration tracking.

The full-model graph run recorded **56 captures**, **zero compiler skips**,
and **4,480 successful nonempty replays during measured updates**. Losses and
gradient norms match the default compiled-full trajectory exactly. The
optional [diagnostic helper](cudagraph_diagnostics.py) counts successful tree
replay calls using private PyTorch 2.14 APIs, restores its wrapper on exit,
and adds no CUDA operations or synchronization. Counts are not GPU timings.

The first exploratory graph benchmark excluded gradient zeroing from the
timer and logged capture details with `TORCH_LOGS=cudagraphs`. It is retained
as evidence but not used in the table. The corrected run includes gradient
zeroing, matching the production update timer. Future benchmark configurations
all use this timing. First-update peak allocation was 138.15 GiB; steady
allocator peaks of 34.69 GiB during replay do not represent the complete graph
memory footprint. Reserved memory, including graph pools, remained 141.31 GiB.

Graph mode is validated and available as an override, but was not selected as
the default because it did not improve typical update speed. It captures cores;
embedding, depth modulation, output projection, optimizer and the separately
compiled full-loss reduction are outside those core graphs. See
[PyTorch compile modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
and [CUDA graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints).

## Asynchronous full checkpoints

`--async-checkpoint` is optional in the general runner and enabled in the B200
launcher. It retains the existing `warpstate-server-v1` single-file checkpoint,
including model, complete AdamW state, RNG, token cursor, schedule/signature,
configuration and provenance. `--no-async-checkpoint` restores synchronous
saving. The asynchronous option currently requires a single training rank.

The main thread first creates a fully independent CPU snapshot. GPU-to-CPU
staging completes synchronously before the next optimizer update can mutate
live state. One background writer then serializes, flushes and fsyncs the file,
atomically replaces `latest.pt`, and fsyncs the parent directory. Only
serialization and disk I/O overlap training. Staging needs roughly 24 GB of
additional host memory; it does not allocate a full extra GPU checkpoint.

A new save waits for the previous writer before allocating another snapshot,
so only one snapshot/write can be outstanding. Background errors surface
between updates and again on flush/close. Normal exit, graceful stop and
training exceptions join any already-staged valid save before the pipeline
lock is released. Exceptions do not snapshot partially updated live weights.
Logs distinguish staging from completed publication and include snapshot
step/tokens. A **Staged** message alone is not a durable checkpoint.
This follows the stable-snapshot and bounded-outstanding-save principles in
[PyTorch's asynchronous checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
while preserving this repository's existing checkpoint format.

## Numerical and continuation checks

All 22 updates in each completed benchmark had finite loss and gradient norm.
Against the microbatch-4 trajectory, maximum absolute loss differences were
0.00006819 for microbatch 8, 0.00006199 for microbatch 16, 0.00031376 for compiled
full loss at microbatch 16, and 0.00006914 for its autotuned version. Maximum
relative gradient-norm differences were 0.2680%, 0.1688%, 1.1087%, and 0.1760%.
Executions preserve the objective but need not be bitwise equal across modes.
Checkpoint and dataset identities remained unchanged throughout the benchmarks.

The final CPU runner/loss/checkpoint/launcher selection passed **91 tests**
with nine GPU cases deselected. Default and autotuned compiled-full CUDA
continuation cases passed earlier; the graph and asynchronous cases then
passed **2/2** in 84.41 seconds. These regressions check exact model, full
optimizer and RNG state, canonical keys, FP32 state, learning-rate schedule,
finite validation, and the final partial update at token counts 32, 64 and 73.
The asynchronous case compares asynchronous pause/resume against synchronous
continuous training and verifies that the asynchronous path actually ran.
CPU writer tests cover immutable snapshots during AdamW mutation, atomic
publication and failure cleanup, bounded writes, persistent errors and both
normal/exceptional exit flushing. Only one GPU workload ran at a time.

## Production continuation and first asynchronous save

After the user requested continuation, production resumed under B200
Supervisor in tmux session `warpstate-b200`, window `training`, new pane `0.1`.
Initial validation loss matched the pause snapshot at **4.79387**. The first
update, step 489, took 8.11 seconds including cached compilation startup.
Steps **490-500** processed 1,441,792 tokens at **19,223 sustained tokens/s**,
with a median **72,509 tokens/s**, including provider waits in update timing.
Peak allocated/reserved memory was **138.18 / 139.55 GiB**. This window excludes
validation, CPU checkpoint staging and background disk-write overhead.

At step **500 / 65,536,000 tokens**, training loss was **5.0160** and validation
loss was **4.74471**. The different training batches explain why individual
training losses can be above five while the fixed validation loss is lower.
The complete snapshot took **35.15 seconds** to stage synchronously, followed
by **44.44 seconds** of background serialization/fsync/publication. Steps
**501-507** completed while that writer was active. These staging and write
times are observed wall times on this host; the staging wait was not separately
profiled and should not be attributed entirely to transfer bandwidth.

The published checkpoint retains all **2,005,111,424 FP32 parameters**, **256
full AdamW entries at step 500**, finite FP32 moments, one rank's RNG state,
the same data/origin, **131,072 tokens per update**, and scheduled optimizer
LR **0.000075**. The full SHA256 and inspection are recorded in
`work/validation/b200/optimized-checkpoint-step-500.json`. The checkpoint is
preserved as `work/run/validated-step-500.pt`; later saves advanced `latest.pt`.
The original `paused-step-488.pt` remained unchanged. Training continued after
this first-save validation, until the later step-680 pause recorded above.
Preparation, tokenizer, dataset, schedule and precision were preserved.

## Remaining bottleneck

The recurring extra 30-second waits are strongly associated with the provider's
HAIShare GPU scheduler, which waits for ownership beneath CUDA launch calls.
Data transfer took about 5-6 ms/update; AdamW and CPU quota throttling did not
account for sampled delays. See [B200_SCHEDULING.md](B200_SCHEDULING.md) for
evidence and uncertainty. No provider scheduling or interposition settings
were changed. Confirm exclusive compute with the provider; full visible VRAM
alone does not establish exclusive allocation.

## Reproduction

Use [b200ctl.sh](b200ctl.sh) to stop production gracefully and confirm the GPU
is free before benchmarking. Preserve the pause snapshot. Benchmarks copy
model and optimizer state into disposable CUDA storage and never save a
training checkpoint. Their learning rate stays fixed at the loaded value and
RNG is reseeded, so benchmark updates are timing experiments.

From `/workspace/WarpState`, choose a new output directory for each run:

```bash
source rtx6000_40b/ops/training-env.sh
source rtx6000_40b/ops/workspace-tools.sh
export TORCHINDUCTOR_COMPILE_THREADS=8
rtx6000_40b/ops/benchmark.sh \
  --checkpoint /workspace/WarpState/rtx6000_40b/work/run/paused-step-488.pt \
  --micro-batch 16 --seq-len 2048 --global-tokens-per-update 131072 \
  --compile-cores --compile-mode max-autotune-no-cudagraphs \
  --no-activation-checkpointing --autocast-cache --loss-mode compiled-full \
  --loss-chunk-size 1024 --vram-limit-gib 150 \
  --warmup-updates 2 --measured-updates 20 \
  --output-dir /workspace/WarpState/rtx6000_40b/work/benchmark-b200-reproduction
```

For backend comparisons use `--compile-mode default` and select
`--blas-backend default`, `--blas-backend cublas`, or `--blas-backend cublaslt`
in otherwise identical cases. The completed cuBLASLt case used the equivalent
`TORCH_BLAS_PREFER_CUBLASLT=1` environment preference, recorded in its metadata.
For graph comparison use `--compile-mode reduce-overhead --cudagraph-diagnostics`.
The same compile-mode override is accepted by `train-b200.sh` on a later
production launch; the general runner's async flag is independent.

Raw `summary.json` and rank records are under `work/benchmark-b200-*`;
`work/validation/b200/optimization-grid.json` aggregates comparisons.
Metadata records source hashes, input identities, versions, backend options,
precision and each update's measurement. Shared logs are shown in tmux session
`warpstate-b200`, window `optimization`; attach with
`rtx6000_40b/ops/tmux-workspace.sh attach -t warpstate-b200`.
