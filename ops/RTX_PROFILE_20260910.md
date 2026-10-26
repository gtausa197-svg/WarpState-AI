# RTX training profile, 2026-09-10

This report records the experiments before production promotion. The selected
BF16 configuration and subsequent checkpoint-overlap implementation are
documented in [RTX_ASYNC_TRAINING.md](RTX_ASYNC_TRAINING.md); use
[LOCAL_RESUME.md](LOCAL_RESUME.md) for the current launcher and status controls.

The production run was gracefully paused to measure training bottlenecks and
compare disposable execution experiments. Production remains paused while the
experiments run. Benchmark updates do not advance the saved training run.

## Preserved training state

The completed checkpoint was verified at 18:12:57 UTC:

| Field | Value |
| --- | --- |
| Checkpoint | `work/run/latest.pt` |
| Step | 5,618 |
| Tokens consumed | 736,362,496 |
| Validation loss at pause | 2.979250341653824 |
| Bytes | 24,061,700,237 (22.409 GiB) |
| Saved optimizer entries | 256 |
| Saved RNG ranks | 2 |
| SHA256 | `0f2e2a47f13b9d91d0049a0e99109c289399fe593c244bab046f120c1f7ea835` |

The benchmark loads this full checkpoint, including optimizer state, into
disposable processes. It does not write production checkpoints. Its configuration
and final summary record the checkpoint file identity before and after the run.
The verification record is generated at
`work/validation/rtx-profile-20260910/paused-checkpoint.json`.

## Configuration and measurement method

The hardware is two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
GPUs, each with approximately 96 GB VRAM. The 2,005,111,424-parameter model uses
BF16 autocast with FP32 master parameters, gradients, and full fused AdamW state.
No FP8 or reduced-precision optimizer state is used.

The production configuration uses sequence length 2,048, microbatch 4 per GPU,
and eight accumulation microsteps. The global batch is therefore 131,072 tokens
per optimizer update. It compiles the model cores in `default` mode, uses the
checkpointed vocabulary loss with 1,024-token chunks, enables the autocast cache,
and disables activation checkpointing for the model blocks. Production saves
every 50 updates and evaluates every 100 updates.

The unprofiled baseline restores the checkpoint's optimizer state and runs two
warmup updates followed by 20 measured updates. Its aggregate throughput was
**29,081.0 tokens/s**, with a median of **28,996.8 tokens/s**. Peak allocated VRAM
was **67.747 GiB** and reserved VRAM **68.643 GiB** per GPU, with a minimum of
**25.311 GiB** free. The short baseline excludes evaluation and checkpoint writes
and has a transient speed change as the GPUs warm up. Sustained production
measurements below provide the wall throughput reference.

Profiler runs identify operators and timing relationships. Their instrumented
update durations must not be presented as unprofiled training throughput.
Summed CUDA operator durations may overlap; they do not automatically partition
wall time. Candidate speed comparisons must use unprofiled measurements with
the same global token batch and full optimizer state.

Baseline artifacts are in `work/benchmark-rtx-20260910/baseline/`. Telemetry is
in `work/validation/rtx-profile-20260910/baseline-gpu-telemetry.json`.

## Production overhead

The last 200 completed production intervals end at step 5,618 and start just
after step 5,418. This interval stays within the restart segment and excludes
the shutdown gap. It includes four periodic checkpoints and two evaluations;
the final pause-time evaluation and save occur after its last update timestamp.

| Measurement | Result |
| --- | ---: |
| Tokens processed | 26,214,400 |
| Sum of measured optimizer-update time | 909.490 s |
| Elapsed time between metric timestamps | 1,000.906 s |
| Throughput within updates | **28,823.2 tokens/s** |
| Throughput including recurring overhead | **26,190.7 tokens/s** |
| Wall time outside updates | **9.133%** |
| Median ordinary inter-update overhead | **2.10 ms** |

For each consecutive metric pair, outside-update time is the difference between
record timestamps minus the later record's update duration. Metrics are emitted
before periodic evaluation and saving, so the gap following a save step captures
that save's stall.

| After step | Operations before next update | Outside-update time |
| --- | --- | ---: |
| 5,450 | Checkpoint | 22.730 s |
| 5,500 | Validation and checkpoint | 23.026 s |
| 5,550 | Checkpoint | 22.505 s |
| 5,600 | Validation and checkpoint | 22.743 s |

Checkpoint-only gaps average 22.618 s. The difference between grouped means
suggests approximately 0.27 s for validation in this window, but validation was
not separately timed and checkpoint duration also varies. The wider restart
segment gives approximately 22.49 s per checkpoint and a 0.35 s grouped
validation difference. Both windows show that synchronous checkpoint saving
dominates the approximately 9% operational overhead.

These gaps combine GPU-to-host staging, serialization, filesystem writes,
durability synchronization, and waiting. Dividing checkpoint bytes by the pause
gives approximately 1.06 GB/s for the combined path; this is not an isolated SSD
bandwidth measurement. Raw interval calculations are recorded in
`work/validation/rtx-profile-20260910/production-overhead.json`.

### Checkpoint cadence model

Holding compute time and evaluation cadence fixed, amortizing approximately
22.6 s per checkpoint gives the following estimates. These are predictions,
not measured candidate runs; production cadence has not been changed.

| Save every | Estimated wall throughput | Approximate gain | Maximum replay after a crash |
| --- | ---: | ---: | --- |
| 50 updates, current | 26.2K tokens/s | Reference | 6.55M tokens, 4.2 min |
| 100 updates | 27.4K tokens/s | 4.7% | 13.11M tokens, 8.0 min |
| 200 updates | 28.1K tokens/s | 7.3% | 26.21M tokens, 15.5 min |
| 500 updates | 28.5K tokens/s | 8.9% | 65.54M tokens, 38.4 min |

Eliminating all measured outside-update time would raise this production window
from 26.19K to 28.82K tokens/s: an approximately **10.1% upper bound** for
operational improvements alone. Substantially higher throughput requires faster
optimizer updates as well.

`run_40b.py` currently saves model and optimizer GPU tensors synchronously on
rank 0 and then waits at a distributed barrier. Its `--async-checkpoint` option
explicitly rejects a world size other than one. The existing
`ops/checkpoint_writer.py` stages an immutable CPU snapshot before overlapping
serialization and durable publication with training. Adapting this path to DDP
requires coordinated error handling and validation; the flag cannot simply be
enabled for the current two-GPU run. CPU snapshot staging would still block,
and approximately one full checkpoint of additional host memory is required.

## GPU profile

One complete update, after two warmup updates, took 4.4745 seconds with profiler
instrumentation. Actual CUDA activity occupied approximately 98.8% of that
interval. This is the union of kernel/copy intervals, not tensor-core utilization.

| GPU activity | Rank 0 summed time | Rank 1 summed time |
| --- | ---: | ---: |
| Matrix multiplication | 3.231 s | 3.122 s |
| Compiled pointwise functions and reductions | 0.561 s | 0.563 s |
| NCCL communication | 0.448 s | 0.545 s |
| Other kernels | 0.378 s | 0.378 s |
| Local attention | 0.100 s | 0.099 s |
| Fused AdamW | 0.0385 s | 0.0384 s |
| CUDA copies and fills | 0.0033 s | 0.0031 s |

Matrices represent 65.7–67.9% of summed GPU activity. Streams overlap, so this
table must not be added up as a partition of elapsed time. Rank 0 communication
overlaps other activity for 337 ms; its communication-only intervals total
111 ms. Rank 1 has 215 ms of communication-only activity. Those intervals are
not guaranteed recoverable by removing communication.

The eight data batches require only 5.65/5.12 ms of host time for construction,
pinning and transfer submission. Increasing loader workers does not address a
measured bottleneck. Gradient clipping's 280/380 ms host range mostly waits for
previously queued GPU work during the nonfinite check. Actual clipping-related
CUDA work is approximately 16 ms. Optimizer and attention replacements likewise
have small direct budgets compared with dense projections.

Raw traces and operator/kernel tables are under
`work/benchmark-rtx-20260910/profile/`. The generated interpretation is
`work/validation/rtx-profile-20260910/gpu-attribution.json`.

## What the matrix multiplication contains

For a linear projection with input X and weight W, training computes three
matrix products: `Y = X @ W.T`, `dX = dY @ W`, and `dW = dY.T @ X`. Each product
costs approximately `2*M*N*K` FLOPs. Thus backward contributes approximately
twice the dense arithmetic of forward, before other backward operations.

Each rank processes 65,536 tokens per update as eight 8,192-token microbatches.
Each core projection therefore executes 224 times: 28 cores times eight
microbatches. The following rank-0 times sum all calls within the profiled update.

| Projection | Forward input/output width | Forward | Input gradient | Weight gradient |
| --- | --- | ---: | ---: | ---: |
| FFN up and gate | 2,048 → 14,336 | 407.4 ms | 437.1 ms | 431.5 ms |
| FFN down | 7,168 → 2,048 | 222.2 ms | 210.5 ms | 195.5 ms |
| Q/K/V/gate/memory input | 2,048 → 10,240 | 291.0 ms | 314.9 ms | 309.1 ms |
| Core output | 2,048 → 2,048 | 57.8 ms | 57.4 ms | 57.5 ms |
| Vocabulary head | 2,048 → 32,768 | 71.2 ms | 39.8 ms | 31.2 ms |

Head forward includes recomputation: 64 original and 64 recomputed projections
of 1,024 tokens, with 64 calls for each derivative. Core activation checkpointing
is disabled; head checkpointing still repeats the head's forward projection.
State-memory batched matrix operations add another 96.5 ms on rank 0.

The dense linear products alone account for 797.146 trillion FLOPs per rank
per update, or 12.1635 billion FLOPs per training token. Their summed execution
time is 3.1342 s, approximately 254.3 TFLOPS during those operations. FFNs alone
take 1.905 s, and the five-way input projection takes 0.916 s. These are the
primary matrix optimization targets.

The large forward GEMMs have ideal one-pass BF16 operand arithmetic intensities
of roughly 910–1,470 FLOPs/byte. This is an operand-footprint calculation, not
measured device-memory traffic: tile reloads, caches, casts and accumulation
change actual traffic. It supports investigating compute efficiency before
assuming that the large GEMMs are starved for external memory bandwidth.

Shapes, strides, dtype, counts, FLOPs and matching kernel symbols are recorded
in `work/validation/rtx-profile-20260910/matmul-detail.json`. The dominant kernel
names begin `cutlass_80_tensorop_bf16`, with 128×256 or 256×128 tiles, K32 and
three pipeline stages. The symbol identifies an implementation family; it does
not establish that the GPU is executing in an obsolete architecture mode.

## Compute ceiling at the measured power limit

NVIDIA specifies **438.9 TFLOPS of dense BF16 with FP32 accumulation per Max-Q
GPU**, based on a **2,280 MHz boost clock**, with **300 W board power** and
**1,792 GB/s memory bandwidth**. The larger sparsity figure is not applicable
to this dense model. See Appendix A of the
[NVIDIA RTX Blackwell architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf#page=46).

During loaded baseline samples, median SM clocks were approximately 1,537 and
1,571 MHz. The software power cap was active; thermal throttling was not.
Scaling the advertised rate by those clocks gives roughly 296–302 TFLOPS per
GPU as an illustrative instantaneous ceiling, substantially below boost-rated
peak. This calculation assumes arithmetic capacity scales linearly with SM
clock; it is not a hardware-counter utilization measurement.

The isolated GEMM tests below sustain approximately 244–263 TFLOPS with
observed clocks of 1,417–1,537 MHz and approximately 300 W. Relative to the
clock-scaled specification, these correspond to approximately **88–94%**.
Telemetry is sampled after each timed case and clocks vary within a case, so
this percentage is approximate. Nevertheless, the evidence indicates that the
large library GEMMs already use much of the compute available at this power
limit. Whole-update efficiency is lower because normalization, activations,
gradient handling, memory operations and communication still consume time.

Nsight Compute 2025.1.1 is installed, but its hardware-counter query was rejected
with `ERR_NVGPUCTRPERM` by the host driver. No driver permissions or power settings
were changed. Profiler occupancy fields reporting zero for active kernels were
not treated as measured occupancy. The conclusion above uses CUDA timings,
shape-derived FLOPs and clock observations, with those limits made explicit.

## Isolated GEMM algorithm experiments

[`benchmark_gemm.py`](benchmark_gemm.py) generates deterministic BF16 operands
with the projection's actual transpose strides. It preserves the production
BF16 reduction flag, excludes compilation, warms each case for at least two
seconds, then measures five CUDA-event trials of 30 products. The inputs are
reused and no autograd, communication, optimizer or training checkpoint is
involved. These are kernel comparisons, not training throughput measurements.

| Product | cuBLAS | cuBLASLt preference | Default Triton-only autotune |
| --- | ---: | ---: | ---: |
| FFN up/gate forward | 1.828 ms | 1.838 ms | 1.981 ms |
| FFN up/gate dX | 1.885 ms | 1.885 ms | 2.057 ms |
| FFN up/gate dW | 1.871 ms | 1.777 ms | 1.869 ms |
| Input projection forward | 1.326 ms | 1.314 ms | 1.441 ms |
| Input projection dX | 1.389 ms | 1.376 ms | 1.474 ms |
| Input projection dW | 1.351 ms | 1.336 ms | 1.350 ms |
| FFN down forward | 0.984 ms | 0.965 ms | 1.076 ms |
| FFN down dX | 0.914 ms | 0.902 ms | 0.993 ms |
| FFN down dW | 0.935 ms | 0.955 ms | 0.993 ms |

ATEN-only compilation was essentially equal to eager cuBLAS. Default
Triton-only selection was usually slower; swapping backends globally is not an
established improvement. cuBLASLt gave mixed shape-specific results. Its 5.3%
up/gate dW gain coincided with a higher observed clock, so it must not be
attributed solely to a better algorithm. Some Triton input-gradient cases
differed from the BF16 cuBLAS reference by about 0.30% relative RMS; finite
outputs do not establish numerical equivalence.

Enabling `ENABLE_PERSISTENT_TMA_MATMUL=1` added persistent TMA candidates and
selected them for the three up/gate products. Measured times were 1.798 ms
forward, 1.890 ms dX and 1.810 ms dW: approximately 268, 255 and 266 TFLOPS.
This closes the default Triton gap and gives small shape-specific improvements
over cuBLAS, rather than a large untapped acceleration. Autotuner trial times
were faster than the subsequent warmed measurements; the latter are reported.

Evidence is under `work/benchmark-gemm-20260910/`,
`work/benchmark-gemm-cublaslt-20260910/`, and
`work/benchmark-gemm-tma-20260910/`.

## Fusion opportunities

The two FFN input projections already share one GEMM. Q/K/V/gate/memory likewise
share one projection. The compiled core already fuses residual addition,
RMSNorm and BF16 conversion, as well as SwiGLU split/SiLU/multiply and several
attention transforms. Generated wrappers still use `extern_kernels.mm` between
these fused regions.

The FFN computes `(X @ W_up * silu(X @ W_gate)) @ W_down`. The nonlinearity
prevents collapsing both projection stages into a single ordinary linear map.
A custom fused implementation still performs both matrix stages.

| Remaining target | Rank-0 measured kernel-time budget | Practical constraint |
| --- | ---: | --- |
| Weight-gradient conversion and FP32 accumulation | 130.5 ms | Preserve BF16 rounding and DDP gradient hooks |
| SwiGLU backward | 88.0 ms | Paired-element derivative can reuse loads and sigmoid |
| Repeated core weight casts | 56.4 ms | Cached BF16 copies need optimizer-step invalidation and correct autograd |
| SwiGLU forward | 41.1 ms | Up/gate columns are 7,168 apart; backward still needs saved values |
| Forward norms | 29.6 ms | Row reductions complicate fusion into tiled GEMMs |
| Forward Q/K/V/U transforms | 43.9 ms | Multiple attention and memory consumers need compatible layouts |

These are budgets, not promised savings, and are not additive end-to-end
speedups. A slower custom GEMM can erase the benefit of its fused epilogue.
The complete SwiGLU forward/backward pointwise cost is only 129 ms, about 2.9%
of the profiled update. Recomputing the up/gate projection to avoid saving its
activations would add approximately 407 ms, exceeding its forward fusion budget.

[`benchmark_gradient_accumulation.py`](benchmark_gradient_accumulation.py)
compares existing BF16 dW followed by FP32 accumulation, fused conversion/add,
and direct FP32-output `addmm`. The direct variant changes the original
BF16-round-before-add semantics. Neither helper implements production autograd
or DDP integration, and neither has been enabled in the training run.

The final accumulation test uses eight distinct deterministic input pairs. It
creates the first gradient and then performs seven additions, matching the
production accumulation order. Each case uses two seconds of burn-in and five
timing trials, each containing ten eight-gradient cycles.

| Weight-gradient family | Separate BF16 GEMM/cast/add | Fused cast/add | Direct FP32 GEMM accumulation |
| --- | ---: | ---: | ---: |
| FFN up/gate | 16.928 ms | 15.850 ms | 15.438 ms |
| Input projection | 11.804 ms | 11.308 ms | 11.620 ms |
| FFN down | 8.141 ms | 7.802 ms | 7.873 ms |

Fused conversion/add improves these complete gradient cycles by **4.3–6.8%**,
with zero output difference on all three tested fixtures. Scaling the measured
cycle differences over 28 blocks suggests approximately 54 ms per update,
around **1.2%** of profiled wall time. This is a hypothesis for integration,
not a measured whole-model improvement; autograd, DDP hooks and overlap remain
to be tested. Direct FP32 accumulation is not consistently faster than the
rounding-preserving fusion and differs by approximately 0.166–0.203% relative
RMS from the baseline on these synthetic gradients.

Final evidence is in `work/benchmark-gradient-distinct-20260910/`. The earlier
`work/benchmark-gradient-20260910/` experiment used eight identical inputs and
a zero-reset-plus-eight-add cycle; it is retained as exploratory evidence and
is not used for the table or integration estimate above.

## Complete training-update experiments

Each successful case uses the same checkpoint and token cursor, two warmup
updates and 20 measured updates, full FP32 optimizer state and BF16 computation.
Global batch and context are unchanged. The table excludes validation and saves.

| Execution | Tokens/s | Gain over baseline | Peak reserved/GPU |
| --- | ---: | ---: | ---: |
| Production baseline, microbatch 4 | 29,081 | Reference | 68.64 GiB |
| Compiled full vocabulary loss, microbatch 4 | 30,050 | 3.3% | 68.99 GiB |
| Compiled full loss + BF16 residuals, microbatch 4 | 30,707 | 5.6% | 66.42 GiB |
| Chunked loss + BF16 residuals, microbatch 8 | 30,394 | 4.5% | 89.49 GiB |
| Compiled full loss + BF16 residuals, microbatch 8 | Out of memory | — | Exceeded 91 GiB cap |
| Baseline repeated after the other experiments | 29,073 | -0.03% | 68.64 GiB |

The repeated baseline also measured 20 updates and returned within 0.03% of
the initial aggregate throughput, supporting the small measured candidate gains.

The full-loss microbatch-8 trial failed while allocating its 1 GiB logits
buffer, before completing a warmup update. Retaining chunked loss made that
microbatch fit, but it was slower than microbatch-4 compiled-full with BF16
residuals. Available VRAM did not translate into a larger useful speed gain.

Against baseline's corresponding 20 updates, maximum absolute loss differences
were 0.000210 for compiled-full, 0.000235 for compiled-full plus BF16 residuals,
and 0.000146 for microbatch-8 chunked residuals. Maximum relative gradient-norm
differences were 0.236%, 0.384% and 0.362%, respectively. All completed updates
had finite losses and norms. These short comparisons do not establish long-run
quality or distributed checkpoint/resume equivalence for an experimental path.

The immediate measured compute improvement is approximately 3–6%. A further
implementation effort should prioritize gradient conversion/accumulation,
selective GEMM tuning, and checkpoint staging/overlap. Preserving the current
save frequency is especially useful after the recent abrupt shutdown. Production
remains paused at step 5,618; no experiment has been promoted to its launcher.

## Reproduction and paused-state controls

These commands target the two-GPU workstation. The dedicated B200 host instead
uses `ops/b200ctl.sh` and its own handoff instructions.

Start at the repository root and inspect the production job. If it is running,
request a graceful stop and wait for `STOPPED` and the final checkpoint save
before starting a benchmark. A stopped job may make the status command return a
nonzero shell exit status; inspect its output.

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
rtx6000_40b/ops/datactl.sh status warpstate-train
# Only when the production job is running:
rtx6000_40b/ops/datactl.sh stop warpstate-train
rtx6000_40b/ops/datactl.sh status warpstate-train
tail -n 12 rtx6000_40b/work/logs/train.log
```

Use a fresh output directory for every disposable run. The benchmark restores
full optimizer state, preserves the global batch, and writes no checkpoint:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
rtx_profile_run=$(date -u +%Y%m%dT%H%M%S)
WARPSTATE_BENCH_DEVICES=0,1 WARPSTATE_BENCH_PROCESSES=2 TORCHINDUCTOR_COMPILE_THREADS=8 \
  rtx6000_40b/ops/benchmark.sh \
  --checkpoint /workspace/WarpState/rtx6000_40b/work/run/latest.pt \
  --data /workspace/WarpState/rtx6000_40b/work/data/train.bin \
  --output-dir "/workspace/WarpState/rtx6000_40b/work/benchmark-rtx-${rtx_profile_run}/baseline" \
  --micro-batch 4 --seq-len 2048 --global-tokens-per-update 131072 \
  --compile-cores --compile-mode default \
  --no-activation-checkpointing --autocast-cache \
  --loss-mode checkpointed --loss-chunk-size 1024 \
  --optimizer-state resume --warmup-updates 2 --measured-updates 20 \
  --vram-limit-gib 88 --threads 4 --ddp-bucket-cap-mb 128
```

For the operator trace, use another fresh output directory, add `--profile`, and
`--diagnose-stalls`, and use `--measured-updates 1`. Only the first measured update is traced. Inspect
the output summary and both ranks' profile artifacts, then compare throughput
using separate unprofiled runs. Leave production paused after the experiments.

Candidate overrides, each with a new output directory, are `--loss-mode
compiled-full`; then additionally `--bf16-residuals`; and, for the larger batch,
`--micro-batch 8 --vram-limit-gib 91`. The successful larger-batch case retains
`--loss-mode checkpointed`. Do not run these GPU experiments concurrently.

The isolated matrix and gradient comparisons use no training data or checkpoint:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_COMPILE_THREADS=8
rtx6000_40b/.server-venv/bin/python rtx6000_40b/ops/benchmark_gemm.py \
  --backends eager aten triton --warmup 10 --burnin-seconds 2 --repeats 30 --trials 5
rtx6000_40b/.server-venv/bin/python rtx6000_40b/ops/benchmark_gemm.py \
  --backends eager --blas-backend cublaslt
ENABLE_PERSISTENT_TMA_MATMUL=1 rtx6000_40b/.server-venv/bin/python \
  rtx6000_40b/ops/benchmark_gemm.py --families ffn_up_gate --backends triton
rtx6000_40b/.server-venv/bin/python \
  rtx6000_40b/ops/benchmark_gradient_accumulation.py
```

Each invocation defaults to a fresh timestamped output directory and refuses
to overwrite an existing directory. `--list-shapes` validates and lists the
selected layouts on CPU. GPU comparisons save source hashes, arguments,
precision settings, numerical errors and timing samples. The two new helpers
passed CPU syntax, layout/FLOP and accumulation-control checks, and actual BF16
GPU comparisons; they do not alter the production model or launcher.

When the user requests production resume, use the existing verified checkpoint
and `rtx6000_40b/ops/datactl.sh start warpstate-train`. The tmux session named
`warpstate-train` is the monitoring session; Supervisor controls the training
process. Never use a from-scratch launcher to resume this run.

Raw profiles, benchmark summaries, telemetry, and verification reports remain
ignored under `work/`. This source document records the recipe and interpretation
without adding generated artifacts or checkpoints to Git.
