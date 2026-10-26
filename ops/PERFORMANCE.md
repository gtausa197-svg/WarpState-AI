# WarpState training bottlenecks

For the current single-B200 measurements and selected preset, see
[B200_PERFORMANCE.md](B200_PERFORMANCE.md). This document records the original
workstation experiments.

**Paused at the user's request:** production and benchmarks are stopped,
with the full production checkpoint saved at step 448 / 58,720,256 tokens.
Another session will resume VORTEX. Do not start the reproduction commands
until the user requests further WarpState work and the GPUs are available.

Measurements and analysis from September 9, 2026, using the configuration in
[OPTIMIZED_TRAINING.md](OPTIMIZED_TRAINING.md): two RTX PRO 6000 Blackwell
Max-Q GPUs, 300 W each, compiled cores, BF16 matrix computation, FP32 AdamW,
microbatch 4 per GPU, accumulation 8, sequence length 2,048.

## Sustained throughput

Production updates 251–315 averaged 28,928 training tokens/second inside
the timed update loop. Their median update took 4.544 seconds for 131,072
global tokens. The interval between completed updates averaged 26,711
tokens/second including the validation/checkpoint pause after step 300.
That pause produced a 28.31-second gap between updates 300 and 301.

The GPUs had about 68.64 GiB reserved and 25.31 GiB physically free each.
They were busy at the 300 W power limit, with power-cap limiting observed
and no thermal slowdown. Available VRAM does not imply available compute.
The driver reported a maximum configurable power limit of 325 W; that is
only 8.3% above the current power allowance, and does not imply an equal
throughput gain. Hardware settings were left unchanged.

## GPU profile

One full update was profiled after two warmup updates on disposable copies
of checkpoint step 357. The update took 4.452 seconds. Rank 0 showed:

| GPU activity | Summed activity time |
| --- | ---: |
| Matrix multiplication | 3.215 seconds |
| Compiled pointwise functions and reductions | 0.561 seconds |
| NCCL communication | 0.439 seconds |
| Local attention | 0.100 seconds |
| Fused AdamW kernels | 0.039 seconds |
| Other kernels and copies | Approximately 0.382 seconds |

These are actual CUDA kernel/copy activities, excluding nested GPU annotation
ranges. They sum to more than elapsed time because streams overlap. Across
the two ranks, matrix multiplication was approximately 65–68% of summed
activity time and attention approximately 2.1%. Operator shapes attribute
1.888 seconds on rank 0 to FFN matrix multiplication, 0.919 seconds to the
input projection, and 0.141 seconds to the full-vocabulary head's matrix
multiplications. Normalization, gating, state operations, casts, gradient
accumulation, and other memory traffic account for much of the remaining
work. GPU utilization near 100% does not mean peak tensor arithmetic
throughput.

Communication overlaps computation, so its summed time is not the amount
recoverable by removing it. The previous two-GPU benchmark scaled 1.95 times
the equivalent single-GPU configuration. Neither more data-loader workers
nor a different attention implementation addresses the dominant matrix cost.

The raw evidence is under `work/benchmark-profile-20260909`. Profiler
instrumentation was confined to the benchmark; production resumed from
the unchanged step-357 checkpoint afterward.

## Compute required for 100K tokens/second

The following counts use two FLOPs per multiply-add. Let `d=2048`,
`f=7168`, `L=28`, and `V=32768`. Each core has an input projection with
`5*d*d` weights, an output projection with `d*d` weights, and SwiGLU
matrices with `3*d*f` weights. Forward and backward together require
approximately six FLOPs per dense matrix weight per token.

| Matrix computation | Billion FLOPs / training token |
| --- | ---: |
| FFNs: `6*L*3*d*f` | 7.398752 |
| Core input/output projections: `6*L*6*d*d` | 4.227858 |
| Tied output head with forward recomputation: `8*d*V` | 0.536871 |
| Parallel tensor-memory operations | 0.049545 |
| Local causal attention, including backward | Approximately 0.044–0.103 |
| **Total matrix computation** | **Approximately 12.26–12.32** |

The attention range reflects causal triangular arithmetic versus tiled work
and backward recomputation. These counts exclude normalization, nonlinear
functions, casts, optimizer work, and communication. Core activation
checkpointing is disabled; the chunked loss projection still recomputes
its forward during backward.

At 100,000 tokens/second, the matrix operations alone require approximately
1.23 PFLOPS. NVIDIA specifies 438.9 TFLOPS of dense BF16 with FP32
accumulation per Max-Q GPU, or 877.8 TFLOPS for this pair. Thus the
advertised ideal ceiling is about 71K tokens/second before other work.
Actual clocks, kernel efficiency, and memory traffic reduce that ceiling.
This rules out 100K through ordinary BF16 configuration tuning for the
unchanged dense computation. NVIDIA's 3,511 AI TOPS figure refers to sparse
FP4 and does not describe this BF16 training. See the
[NVIDIA architecture whitepaper, Appendix A, pages 45–46](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf#page=45).

## What to investigate next

Completed BF16 experiments used disposable copies of the same step-448
checkpoint, two warmup updates and three measured updates per case:

| Configuration | Tokens/second | Peak reserved/GPU |
| --- | ---: | ---: |
| Existing compiled-core baseline | 29,396 | 68.64 GiB |
| Full loss without recomputation | 29,576 | 72.03 GiB |
| Compiled full loss | 30,315 | 68.99 GiB |
| Compiled full loss + cuBLASLt / 32 MiB workspace | 30,329 | 68.99 GiB |
| Compiled full loss + matrix autotuning | 30,403 | 68.99 GiB |
| Compiled full loss + BF16 residual activations, microbatch 4 | 31,111 | 66.42 GiB |

The 50K target was not reached. None of these experimental configurations
was promoted to production. The microbatch-8 residual trial has no completed
summary and must not be treated as a measured result. All benchmark losses
and gradient norms in the completed cases were finite. The compiled-full
cases differed from baseline by up to 0.00078 loss and 2.47% gradient norm
after several updates; execution changes are not bitwise equivalent.

The added loss and residual helpers passed CPU numerical/parameter-identity
checks, and the full-model benchmarks exercised their GPU execution.
The new compiled-full two-GPU checkpoint/resume regression and a production
continuation/validation trial remain outstanding. Complete those before
promoting a new execution configuration. The production launcher retains
its previously validated BF16 settings. No FP8 GPU test or training ran.
At pause handoff, the combined CPU suite passed 84 tests; five CUDA cases
were skipped to leave the GPUs available for VORTEX.

The next requested target is 50K tokens/second, about 1.73 times the current
steady rate. It is below the advertised ideal BF16 ceiling, but attaining it
in BF16 would require unusually high sustained efficiency at this power cap.
As a rough illustration, halving the observed 3.215 seconds of matrix work
while leaving everything else fixed would reduce a 4.452-second update to
about 2.845 seconds, or 46K tokens/second. Overlap and power behavior make
this an illustration, not a measured prediction. Reaching 50K likely needs
both faster matrix computation and reductions in the remaining costs.

The requested constraint is BF16 matrix computation with FP32 master
weights and optimizer state. The experiments therefore focus on selecting
faster BF16 matrix kernels, using the spare memory to remove loss
recomputation, and reducing activation-memory traffic. Lower-precision
matrix training is outside this task's scope.

A fused linear cross-entropy implementation could reduce repeated output
projection and intermediate-memory traffic. The current 32,768-way head
is only 4.4% of the counted dense linear FLOPs, although its memory costs
can consume a larger fraction of execution time. An implementation such as
[Cut Cross Entropy](https://github.com/apple/ml-cross-entropy) needs a
numerical comparison and an end-to-end benchmark; its default gradient
filtering should not be silently introduced into this pretraining run.

Less frequent full checkpoints can recover some wall-clock throughput.
The current interval is 50 updates, about four minutes of compute. Raising
that interval increases the amount of work lost after an abrupt failure.
Checkpoint frequency does not change tokens/second inside the update loop.

Reaching 100K could instead use more aggregate GPU compute, or a smaller,
low-rank, or conditionally activated model. Those architecture changes
alter the experiment and require their own training/quality comparison.
VORTEX uses low-rank projections, conditional parameter pages, and a
hierarchical output head. Its total parameter count does not imply the
same computation per token as WarpState's 28 dense cores.

## Planning for one B200 GPU

The user specified one B200 GPU and BF16 training. NVIDIA lists 180 GB
HBM3e, approximately 7.7 TB/s memory bandwidth, and 2.25 PFLOPS dense
BF16 for one HGX B200 GPU. The larger 4.5 PFLOPS BF16 number uses
sparsity. These specifications differ from eight-GPU DGX totals. Sources:
[NVIDIA Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf),
[NVIDIA performance reference](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md#peak-theoretical-throughput).

For this custom model, 60–90K BF16 training tokens/second is an engineering
planning estimate, not a measured range or guarantee. The 75K midpoint
roughly preserves the current fraction of advertised compute throughput;
bandwidth, larger useful microbatches, B200 kernel selection, and removal
of two-GPU gradient synchronization can change the result. The theoretical
matrix-only ceiling is approximately 183K tokens/second. Reaching 100K
would need roughly 55% of advertised dense BF16 compute before accounting
for non-matrix work, so it is a plausible tuning target on B200 that still
requires an actual benchmark.

| Assumed update throughput | Training tokens/day | Compute time for 40B |
| ---: | ---: | ---: |
| 60K/s | 5.18 billion | 7.7 days |
| 75K/s | 6.48 billion | 6.2 days |
| 90K/s | 7.78 billion | 5.1 days |
| 100K/s | 8.64 billion | 4.6 days |

Checkpointing and validation add elapsed time to these compute estimates.
The current 50-update checkpoint interval becomes relatively more
expensive as updates get faster.

Storage is independent of GPU choice. In decimal GB, the prepared corpus
uses 80.002 GB, a complete model/Adam checkpoint uses 24.062 GB, and atomic
replacement needs another 24.062 GB temporarily. The current Python
environment is approximately 5.7 GB. Extra retained checkpoints each add
24.062 GB. A deployment with token data, one checkpoint, and its replacement
therefore needs roughly 134 GB before OS, cache, and reserve space. The
runner also checks for about 53 GiB free at startup for checkpoint writing.

Allocate **250 GB as a practical minimum**, or **500 GB NVMe SSD for
comfortable checkpoint retention and working space**. This assumes copying
the already prepared tokens, not retaining raw source downloads. A 1 TB
SSD is not required for this 40B-token run. SSD capacity does not cap the
lifetime number of training tokens: further data can be prepared or
streamed after old data is released.

## Reproducing the GPU profile

Stop production gracefully and wait for both GPUs to be free before running
this finite benchmark. It reads the latest checkpoint into disposable CUDA
copies and writes no training checkpoint. VORTEX must remain paused.

```bash
cd /workspace/WarpState/rtx6000_40b
ops/datactl.sh stop warpstate-train
nvidia-smi
WARPSTATE_BENCH_DEVICES=0,1 WARPSTATE_BENCH_PROCESSES=2 \
TORCHINDUCTOR_COMPILE_THREADS=8 ops/benchmark.sh \
  --checkpoint /workspace/WarpState/rtx6000_40b/work/run/latest.pt \
  --micro-batch 4 --seq-len 2048 --global-tokens-per-update 131072 \
  --no-activation-checkpointing --compile-cores --autocast-cache \
  --loss-chunk-size 1024 --vram-limit-gib 88 \
  --warmup-updates 2 --measured-updates 1 --profile
ops/datactl.sh start warpstate-train
```

Only the first measured update is profiled. Each rank exports a Chrome
trace, an operator table grouped by shape, a kernel table, and a JSON
summary under its fresh `work/benchmark-*` directory. Profiling collects
CPU and CUDA activity with shapes, without Python stacks or memory
tracking. Use unprofiled runs to measure throughput: recording adds
overhead. CUDA kernels on different streams can overlap, so summed kernel
time is not the same as elapsed time. Operator and kernel tables describe
the same work and must not be added together.
