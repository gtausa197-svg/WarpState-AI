# RTX training optimization, 2026-09-12

The refreshed baseline delivers **30,799 tokens/s during optimizer updates** on
the two RTX PRO 6000 Blackwell Max-Q GPUs. The experiments below address
conversion and gradient accumulation, DDP communication, SwiGLU/RMSNorm, and
repeated core weight casts. The communication-overlap candidate beats the baseline in two full-model
comparisons. The workstation resume preset selects this mode; isolated kernel
improvements are not evidence of the same gain in complete training. See
[MATMUL_PROFILE_20260912.md](MATMUL_PROFILE_20260912.md) for the matrix breakdown.

## Protected training state

Production training was gracefully paused at step **40,081**, after
**5,253,496,832 tokens**. The next update is step 40,082. Its full resume checkpoint
is `work/run/latest.pt`, containing FP32 model parameters, AdamW moments, optimizer
step 40,081, and both ranks' RNG state. All inspected model and optimizer tensors
were finite. The model has 2,005,111,424 parameters.

The checkpoint is 24,061,689,997 bytes; its SHA256 at the pause was
`9745d93345dff7c389ac6fa65948dc78ec021c90f56b40e9caed8ddcd84e8c54`.
The generated verification record is
[`work/validation/rtx-opt-20260912/paused-checkpoint.json`](../work/validation/rtx-opt-20260912/paused-checkpoint.json).

The step-8,000, step-16,000, and step-40,000 archives under `work/archives/` are
immutable inputs. These experiments do not replace, re-export, or edit them.
The benchmark reads the production checkpoint through a read-only CPU mapping,
loads independent CUDA model/optimizer tensors, and writes no checkpoints.
Benchmark updates do not advance the production token cursor or RNG state.

Production remains paused while disposable experiments run. A later production
resume must use the retained full checkpoint and record any selected execution
changes through the runner's explicit reconfiguration mechanism.

## Baseline and profile method

The baseline resumes the full AdamW state from step 40,081 and uses deterministic
training-token offsets. It measures 20 updates after three warmup updates:

| Setting | Baseline |
| --- | --- |
| PyTorch / CUDA | `2.14.0+cu130` / `13.0` |
| Devices | Two RTX PRO 6000 Blackwell Max-Q GPUs |
| Sequence length / per-rank microbatch | 2,048 / 4 |
| Accumulation / global tokens per update | 8 / 131,072 |
| Core execution | `torch.compile`, default mode |
| Loss / residual activations | Compiled full-vocabulary loss / BF16 |
| Activation checkpointing | Off |
| DDP | FP32 gradients, 128 MiB buckets, bucket views |
| Optimizer | Fused AdamW, original FP32 state |
| Measured throughput | 30,799.17 tokens/s |
| Peak allocated / reserved memory, rank 0 | 65.49 / 66.44 GiB |

This throughput includes gradient zeroing, forward/backward, accumulation,
gradient clipping, and optimizer updates. It excludes periodic production
evaluation/checkpoint pauses and initial compilation. It must not be presented
as sustained wall-clock throughput over a long training run.

The separate profile records one complete update after two warmups. That
instrumented update took **4.1781 seconds**. Use the unprofiled 20-update baseline
for speed comparisons; the shorter profiled run has a different warmup history
and instrumentation overhead.

| GPU activity | Rank 0 summed milliseconds | Rank 1 summed milliseconds |
| --- | ---: | ---: |
| Matrix multiplication | 3,166.45 | 3,049.22 |
| Compiled elementwise operations and reductions | 512.12 | 515.05 |
| NCCL communication | 422.56 | 532.72 |
| Other tensor kernels | 206.16 | 206.27 |
| Local attention | 99.41 | 98.82 |
| Fused AdamW | 38.50 | 38.46 |
| Explicit copies and fills | 2.96 | 3.08 |

Matrix multiplication accounts for 71.19% and 68.62% of summed GPU activity.
The totals exceed wall time because streams overlap. Rank 0's GPU activity union
is 4.1257 seconds; its summed durations are 4.4482 seconds. Name-based kernel
categories are attribution aids, not hardware utilization or bandwidth counters.
Operator and kernel tables describe the same work and must not be added together.

The explicit copy category excludes memory traffic inside arithmetic kernels.
It therefore does not measure total VRAM bandwidth consumption. This profile
does not establish a hardware bandwidth ceiling.

## Work outside matrix multiplication

The following rank-0 costs are subsets of the preceding categories, measured
over all eight accumulation microbatches and 28 cores:

| Candidate work | Summed milliseconds per update |
| --- | ---: |
| SwiGLU forward / backward | 41.156 / 87.166 |
| Core gradient BF16-to-FP32 conversion / FP32 addition | 39.749 / 89.327 |
| Repeated core weight casts | 54.006 |
| Normalization forward / backward | 12.087 / 50.377 |
| Two groups of backward attention/projection transforms | 82.969 / 32.730 |
| BF16 attention layout copy | 14.036 |
| DDP parameter/bucket scaling, 256 calls | Approximately 7.884 |

The conversion-plus-accumulation budget is **129.076 ms/update**. Its attribution
uses CUDA external IDs mapped to CPU operator input dimensions to distinguish
parameter gradients from activation gradients. Weight casts are the three
compiled transpose/copy families with durations 16.138, 14.107, and 23.761 ms.

NCCL overlaps substantial computation. Intersecting communication intervals with
the union of other GPU activity leaves approximately **100.05 ms on rank 0** and
**215.80 ms on rank 1** containing only communication. These exposed intervals
are not a guaranteed speedup: moving bucket boundaries can alter overlap and
contention elsewhere in the update.

The current compiler already fuses many operations. A custom normalization
kernel may save local kernel time while preventing a larger surrounding fusion.
The candidate budgets therefore cannot be summed into a projected speedup.

## Implemented execution options

The runner and disposable benchmark expose the following options. Defaults
retain the prior execution, and the runner's resume signature includes the new
settings. Legacy checkpoints normalize to these defaults; a selected change
requires explicit resume reconfiguration while preserving the objective, global
token batch, training cursor, optimizer, and schedule.

| Option | Default | Implementation and scope |
| --- | --- | --- |
| `--gradient-mode eager\|compiled-autograd\|compiled-autograd-overlap` | `eager` | Captures backward through leaf accumulation so Inductor can fuse BF16 gradient conversion with FP32 addition. |
| `--activation-mode baseline\|swiglu\|rmsnorm\|all` | `baseline` | Optional Triton activation/normalization forwards and registered autograd backwards. |
| `--cache-core-weight-casts` | Off | Reuses differentiable BF16 core linear-weight views for one accumulation update. |
| `--ddp-bucket-cap-mb` | `128` | Changes communication grouping while retaining FP32 reduction. |
| `--ddp-batched-grad-copy` | Off | Batches gradient copying/scaling in the C++ DDP reducer. |

Compiled autograd uses the Python DDP reducer and cannot be combined with the
C++ reducer's batched gradient copy option. The installed PyTorch can mishandle
a saved CPU scalar in compiled CUDA backward; `scale_loss()` materializes that
same scaling coefficient on the loss device for this mode. Unsupported requested
DDP options fail explicitly. The helper also supports capacity lists for
controlled programmatic experiments; the current CLI exposes a uniform capacity.

The weight cast cache creates fresh views at the start of each optimizer update
and clears them on normal or exceptional exit, before clipping/optimizer,
evaluation, or checkpoint operations. FP32 master identities and version counters
are checked. Its current scope requires one use of each core per forward,
activation checkpointing off, and unaliased trainable linear weights. It does not
cache the tied embedding/output weight. Extra BF16 views consume VRAM; a complete
model measurement must establish their memory cost and benefit.

These options retain BF16 tensor-core computation and BF16 activations. FP32
master parameters, parameter gradients, normalization reductions, DDP gradient
communication, and AdamW moments retain their original representation. No FP8,
FP16, quantized weights, or reduced-precision gradient communication is introduced.
Canonical checkpoint keys and the default CPU chat/evaluation paths are unchanged.

## Numerical checks and isolated measurements

The activation suite passed **22 tests**, including eager and compiled CUDA
forward/backward comparisons, BF16 residual integration, FP32 AdamW, canonical
state reload, and unchanged CPU/evaluation behavior. Its final log is
`work/validation/rtx-opt-20260912/activation-tests-final.log`.

The activation implementation distinguishes eager rounding from current compiled
arithmetic. Inductor can eliminate intermediate BF16 cast pairs; the compiled
variant follows that behavior instead of forcing additional eager truncations.
The comparison does not promise bitwise equality for every reduction shape or
future compiler configuration.

On GPU 1, five timing trials of 20 iterations after five warmup iterations used
the actual activation shapes, seed 951, BF16 random upstream gradients, and
compiled forward plus backward. Compilation used `fullgraph=True, dynamic=False`;
CUDA events timed each trial:

| Primitive | Compiled baseline | Custom | Numerical comparison |
| --- | ---: | ---: | --- |
| SwiGLU, packed `8192 x 14336` | 0.61335 ms | 0.59965 ms | Outputs and input gradients bitwise equal in this fixture. |
| RMSNorm, `8192 x 2048` | 0.43256 ms | 0.39248 ms | Relative RMS error: output `1.64e-6`, input gradient `1.17e-5`, FP32 weight gradient `0.001604`. |

These are approximately 2.23% and 9.27% primitive improvements, not complete
training gains. RMSNorm's large-shape baseline can retain a final BF16 rounding
at a compiler reduction boundary; the custom compiled reduction retains FP32.
That explains the approximately 0.16% weight-gradient discrepancy and requires
whole-model numerical screening before selection.

A separate large-linear gradient fixture verifies that compiled autograd can
replace repeated conversion/addition work with fused kernels while preserving
the accumulated gradient exactly in that fixture. Its measured cycle was
44.668 ms eager versus 41.013 ms compiled autograd. The fixture is not a complete
training update. Small-model comparisons show some changed BF16 rounding in
embedding, depth modulation, and final normalization, so exact linear gradients
do not establish bitwise equality for the whole model.

DDP tests exercise repeated eight-microbatch accumulation and optimizer updates,
including multiple buckets. They preserve FP32 reduction and optimizer state.
Full-model two-GPU tests remain the criterion for selecting an execution change.

## Whole-model candidate results

All screens start from the same step-40,081 full checkpoint and token cursor,
with three warmups and ten measured updates. The initial baseline measures 20
updates. The benchmark holds the checkpoint's learning rate fixed for every
candidate; production continues its existing schedule. The measurements below
exclude compilation, evaluation and saving.

| Candidate / artifact directory | Tokens/s | Peak reserved/GPU | Maximum absolute loss difference from baseline |
| --- | ---: | ---: | ---: |
| Baseline, 20 updates (`baseline`) | 30,799.17 | 66.44 GiB | 0 |
| Batched DDP copies, 64 MiB (`ddp-copy64`) | 30,896.39 | 67.42 GiB | 0 |
| Compiled autograd (`compiled-autograd`) | 30,498.20 | 66.45 GiB | 0.00009394 |
| Cached core casts (`cache`) | 29,956.94 | 66.44 GiB | 0.00008345 |
| Custom SwiGLU (`swiglu`) | 30,749.24 | 66.44 GiB | 0 |
| Custom RMSNorm (`rmsnorm`) | 30,737.22 | 66.14 GiB | 0.00012708 |
| Compiled autograd with wait sinking (`ca-overlap`) | **31,331.80** | 66.98 GiB | 0.00016069 |

A second pair of complete, unprofiled **20-update** measurements confirms the
communication fix: `ca-overlap-confirm` reaches **31,045.42 tokens/s**, versus
**30,578.17 tokens/s** for the subsequent `baseline-confirm` (**1.53% faster**).
The initial ten-update screen was 1.73% above its baseline. These are modest,
repeatable gains, not a claim of 50K throughput or long-run model quality.

Custom activation/normalization kernels and cached casts remain opt-in
experiments. No complete-model gain justifies selecting them in this preset.
The cache's BF16 views also stop DDP's compiler partitioner from counting those
inputs as `nn.Parameter`s. Generated layouts match baseline, but whole-core
backward regions can delay leaf-gradient readiness and change fusion boundaries.
Do not attribute that regression to a confirmed matrix-transpose problem.

### Why fusion alone lost, and the communication fix

Compiled autograd removes nearly all native gradient-add kernels. Rank 0's
other-kernel category falls from 206.17 to 51.36 ms while compiled elementwise
work rises from 512.12 to 585.32 ms, saving roughly 82 ms across those categories.
However, its default schedule puts each wait and gradient copy immediately after
the collective, before independent backward GEMMs. The 57 gradient collectives
carry the same 8,020,445,696 FP32 bytes.

| Communication time/update | Baseline | Compiled autograd | With wait sinking |
| --- | ---: | ---: | ---: |
| Overlapping compute, rank 0 | 322.51 ms | 0 ms | **190.74 ms** |
| Overlapping compute, rank 1 | 316.91 ms | 0 ms | **190.58 ms** |
| Communication only, rank 0 | 100.05 ms | 285.17 ms | **166.64 ms** |
| Communication only, rank 1 | 215.80 ms | 407.35 ms | **267.61 ms** |

`--gradient-mode compiled-autograd-overlap` enables Inductor's
`sink_waits_iterative` scheduler. It moves waits together with their dependent
copies past independent computation, retaining collective launch order and
requiring no increase in the scheduler's **estimated** peak memory. It does not
permit unsafe collective reordering or moves past subsequent collectives.
Actual allocator peaks remain measured separately.

The two CA profiles contain identical kernel counts: 5,400 matmuls, 59
communication events, 9,577 compiled pointwise/reduction kernels and 1,531 other
kernels per rank. Profiled update time changes from 4.25901 to 4.12683 seconds;
the gain is restored overlap, not less arithmetic. Compared with baseline,
52–67 ms more communication remains exposed. That is potential scheduling work,
not a guaranteed further saving.

Cold first-update compilation took approximately 195–196 seconds in these
full-model CA screens. A cached second process still incurs graph capture and
compiler setup; restarts can spend minutes before regular updates appear.

Whole-backward compilation can fuse a tied-head BF16 matrix product and gradient
addition into BF16 `addmm`, changing an intermediate rounding boundary. Original
FP32 master weights, gradients, AdamW state and communication remain FP32; GEMM
operands and outputs remain BF16. Short loss/gradient comparisons use explicit
numerical tolerances rather than claiming a bitwise baseline trajectory.

The CLI rejects weight cast caching with either compiled-autograd mode, and
rejects either feature with `--compile-mode reduce-overhead`. Cache plus
activation checkpointing is rejected before loading; it requires
`--no-activation-checkpointing`. Batched DDP copying uses the C++ reducer and is
incompatible with both compiled-autograd modes' Python reducer.

### Validation and selected preset

The completed CPU suite passed **159 tests** (28 CUDA tests skipped in that
CPU-only invocation). Separate activation, gradient and cast-cache CUDA checks
passed. The final two-GPU runner suite passed **both** continuation cases in
136.79 seconds: combined custom activations/cached casts/batched DDP copies, and
compiled autograd with communication overlap. Each compares uninterrupted
training against an asynchronous stop/resume, including an entirely masked
final rank. Model tensors, full FP32 optimizer state and both ranks' RNG states
match **bitwise** between continuous and resumed execution of the same mode.
This is separate from the bounded numerical differences versus baseline.

`train-local-resume.sh` selects `--gradient-mode compiled-autograd-overlap`.
Activation kernels, external weight-cast caching and batched DDP copies remain
at their baseline settings. BF16 compute, FP32 state, the global token batch,
context, data, optimizer schedule and save/evaluation intervals are preserved.
Production remains paused at step 40,081 until resumption is requested.

Follow-up scheduling controls and paired measurements are documented in
[COMMUNICATION_EXPLORATION_20260912.md](COMMUNICATION_EXPLORATION_20260912.md).
The controls are opt-in; the preset above keeps its conservative defaults.
The separate [backward layout experiment](BACKWARD_LAYOUT_EXPERIMENT.md)
measures a fused rotary-gradient and projection-packing region without changing
production model execution.

Test logs are `work/validation/rtx-opt-20260912/pytest-communication-final.log`
and `work/validation/rtx-opt-20260912/pytest-optimized-resume.log`.

## Reproducing and locating results

All paths above are relative to `rtx6000_40b/`. Source the workspace environment
before any test or benchmark. GPU experiments require training paused and an
exclusive device window. Run each candidate in a fresh process and a new output
directory because compiler/reducer settings are process-wide.

The baseline can be reproduced with:

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
.server-venv/bin/torchrun --standalone --nproc-per-node=2 \
  ops/benchmark_training.py \
  --micro-batch 4 --seq-len 2048 --global-tokens-per-update 131072 \
  --no-activation-checkpointing --autocast-cache \
  --compile-cores --compile-mode default --loss-mode compiled-full \
  --bf16-residuals --optimizer-state resume --vram-limit-gib 88 \
  --warmup-updates 3 --measured-updates 20 \
  --output-dir work/benchmark-opt-20260912/baseline-recheck
```

Use a new output directory and append the candidate flags for each comparison.
For the profile, use two warmup updates, one measured update, and `--profile`.
The original run artifacts are:

- `work/benchmark-opt-20260912/baseline/summary.json`: configuration, per-rank
  update measurements, source hashes, checkpoint identity, and memory peaks.
- `work/benchmark-opt-20260912/profile-baseline/`: both ranks' Chrome traces,
  operator tables, kernel TSV files, and summary/category JSON.
- `work/validation/rtx-opt-20260912/activation/actual-shape-compiled-arithmetic.json`:
  final primitive timings and random-gradient differences. The earlier
  `actual-shape-kernels.json` tested eager rounding against compiled math and is
  superseded for comparing the current implementation.
- `work/validation/rtx-opt-20260912/gradient/cuda-linear-proof.json`: isolated
  accumulation timing, gradient equality, and kernel trace counts.
- `work/validation/rtx-opt-20260912/gradient/cuda-gradient-deltas.json`: initial
  small-model per-parameter differences for compiled autograd.

The source implementations are `activation_execution.py`,
`gradient_execution.py`, `weight_cast_execution.py`, and `distributed_execution.py`
in this directory. Their focused tests live at the package root. Generated
benchmarks, traces, validation outputs, data, and checkpoints remain ignored in
Git; this document records methods and findings for source tracking.

The subsequent paired attention-backward integration and separate gradient
producer-scheduling experiment are documented in
[BACKWARD_LAYOUT_INTEGRATION.md](BACKWARD_LAYOUT_INTEGRATION.md), with gradient
and checkpoint-continuation checks in
[BACKWARD_LAYOUT_TRAINING_VALIDATION.md](BACKWARD_LAYOUT_TRAINING_VALIDATION.md).
The completed integration selects `--backward-layout paired` in
`train-local-resume.sh`: repeated update throughput improves from 31,001.57 to
31,199.03 tokens/second (+0.637%). The separate producer scheduler remains
disabled after a smaller, less consistent gain. Production remains paused.
