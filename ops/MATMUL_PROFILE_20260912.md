# WarpState matrix multiplication profile, 2026-09-12

The largest dense BF16 products already sustain approximately **243–263 TFLOPS
per GPU** in the refreshed isolated test. Their observed clocks and 300 W board
power explain much of the gap from the boost-clock specification. The best next
experiments are selective algorithm/tile selection and small, measured GEMM
fusions. A global backend replacement has not demonstrated a useful improvement.

This analysis uses the refreshed production-configuration profile and new
isolated GEMM timings. Production remains paused at step 40,081 with its full
resume state preserved. All step-8,000, step-16,000, and step-40,000 archives are
unchanged. No matrix experiment in this report writes production checkpoints.

## Profile and attribution

The configuration is the two RTX PRO 6000 Blackwell Max-Q GPUs, microbatch 4 per
rank, sequence length 2,048, eight accumulation microbatches, 131,072 global
tokens/update, compiled cores in default mode, compiled full loss, BF16 residual
activations, and the existing FP32 fused AdamW state. The unprofiled 20-update
baseline is **30,799 tokens/s during optimizer updates**; the separate profiled
update took 4.1781 seconds after two warmups.

Each GPU kernel's `External id` is joined to its `aten::mm`, `aten::bmm`, or
`aten::addmm` CPU event. Dimensions, strides, and dtypes come from that event.
The square output projection's forward and input-gradient shapes are identical;
the RHS transpose stride distinguishes them. This avoids assigning the same
square GEMM to both directions.

For `A[M,K] @ B[K,N]`, arithmetic is counted as `2*M*N*K` FLOPs, multiplied by
batch and call count. Achieved TFLOPS divide these FLOPs by matched GPU kernel
time, not CPU launch time or the entire training update. CPU operator totals can
also include associated fills/copies; the tables below use kernel durations.

All **5,400 matrix kernels per rank** are accounted for:

| Profile | Rank 0 matrix time | Rank 1 matrix time | FLOPs per rank/update |
| --- | ---: | ---: | ---: |
| Current baseline | 3,166.449 ms | 3,049.219 ms | 791.597 trillion |
| Compiled-autograd candidate | 3,122.129 ms | 3,005.404 ms | 791.597 trillion |

Baseline matrix kernels represent 71.19% and 68.62% of summed GPU activity on
ranks 0 and 1. Streams overlap with communication and other work, so these
percentages are not partitions of elapsed update time. The matrix operations
themselves execute on the main compute stream in this trace. The compiled
autograd comparison changes surrounding execution and communication overlap;
its slightly shorter matrix times do not establish a faster full training run.

## Which products dominate

For a linear layer, forward computes `Y = X @ W.T`, the input derivative
computes `dX = dY @ W`, and the weight derivative computes `dW = dY.T @ X`.
Every core projection executes 224 times per direction: 28 cores times eight
microbatches. Backward therefore includes two dense products for each forward.

| Projection family | Rank 0 forward | Rank 0 dX | Rank 0 dW | Total | Share of matrix time |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFN up and gate, 2,048 → 14,336 | 405.939 ms | 433.643 ms | 427.218 ms | 1,266.800 ms | 40.01% |
| Q/K/V/gate/memory input, 2,048 → 10,240 | 284.852 ms | 312.126 ms | 305.255 ms | 902.232 ms | 28.49% |
| FFN down, 7,168 → 2,048 | 218.817 ms | 209.116 ms | 198.243 ms | 626.176 ms | 19.78% |
| Core output, 2,048 → 2,048 | 57.486 ms | 56.861 ms | 57.828 ms | 172.176 ms | 5.44% |
| Vocabulary head, 2,048 → 32,768 | 33.869 ms | 34.131 ms | 35.123 ms | 103.124 ms | 3.26% |
| State-memory batched products | — | — | — | 95.942 ms | 3.03% |

FFN up/gate, input projection, and FFN down together consume **88.28% of matrix
time**. Dense linear products perform 788.350 trillion FLOPs per rank/update;
state-memory batched products account for the remaining 3.247 trillion.

The vocabulary head now uses eight complete 8,192-token products per direction.
The September 10 profile used chunked head loss and repeated head forward
products during backward. Its larger forward-head count should not be reused
to estimate the current configuration.

The dominant shapes and their rank-0 kernel performance are:

| Product | M | N | K | Mean kernel time | Achieved TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFN up/gate forward | 8,192 | 14,336 | 2,048 | 1.8122 ms | 265.4 |
| FFN up/gate dX | 8,192 | 2,048 | 14,336 | 1.9359 ms | 248.5 |
| FFN up/gate dW | 14,336 | 2,048 | 8,192 | 1.9072 ms | 252.2 |
| Input projection forward | 8,192 | 10,240 | 2,048 | 1.2717 ms | 270.2 |
| Input projection dX | 8,192 | 2,048 | 10,240 | 1.3934 ms | 246.6 |
| Input projection dW | 10,240 | 2,048 | 8,192 | 1.3627 ms | 252.1 |
| FFN down forward | 8,192 | 2,048 | 7,168 | 0.9769 ms | 246.2 |
| FFN down dX | 8,192 | 7,168 | 2,048 | 0.9336 ms | 257.6 |
| FFN down dW | 2,048 | 7,168 | 8,192 | 0.8850 ms | 271.8 |

These are timings within the full-model profile. They can differ from isolated
tests because clocks, cache state, memory pressure, and communication differ.
The isolated tests below use longer warmup and repeated timing trials.

## What executes inside the dense kernels

Most dense kernels use symbols beginning
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16`. Forward commonly selects
`128x256_32x3_tn_align8`; dX uses `256x128_32x3_nn_align8`; dW uses
`128x256_32x3_nt_align8`. The head forward uses a `256x128` TN variant.

These are tiled tensor-core kernels: tiles load operands, reuse them while
advancing through K in pipelined stages, accumulate partial products, then store
the output. The symbols encode tile/pipeline/operand variants. The embedded
`80` and `relu` text does not establish obsolete hardware execution or an actual
ReLU in this model; semantic attribution comes from the originating operators.

Observed launches use 256 threads, approximately 218–232 registers per thread,
and 73,728 bytes of dynamic shared memory. Several have a grid Z dimension of
2 or 4. That is consistent with additional partitioned work, but launch metadata
alone does not prove the precise split-K/reduction implementation. A proposed
split-K experiment must first identify what the selected library algorithm
already does.

Profiler occupancy estimates report zero for some kernels that demonstrably
execute. They are not valid measured occupancy. The previous hardware-counter
attempt was denied by the driver's `ERR_NVGPUCTRPERM`; this analysis makes no
new occupancy, tensor-pipe-utilization, or bandwidth-counter claim.

The large dense shapes are multiples of tensor-core alignment and ordinary
tile sizes. Their transpose operands are views with unit stride in one
dimension, not evidence of a full physical transpose before every GEMM.
Blindly adding `.contiguous()` would create extra traffic; any alternative
layout must include its preparation cost in the benchmark.

## Where parallel execution already happens

Both GPUs already train concurrently using data parallelism. Each holds the
model and processes **65,536 of the 131,072 global tokens per update**: eight
8,192-token microbatches per rank. There is no intentional schedule that runs
GPU 0's training first and GPU 1's afterward. Gradient synchronization combines
their contributions to the same optimizer update.

The original compiled-autograd candidate lost overlap between computation and
communication **within each GPU's schedule**. That is different from serializing
the two GPUs: both ranks still compute concurrently, but local waits block
independent backward GEMMs while collectives finish. Recovering that overlap
is the immediate parallelization target identified by the trace.

The refreshed `nvidia-smi topo -m` reports the GPU-to-GPU path as `SYS`, meaning
PCIe plus an inter-NUMA CPU interconnect, with no NVLink entry. Reported CPU
affinities are `0-95,192-287` for GPU 0 and `96-191,288-383` for GPU 1. Topology
identifies the route; it does not measure achieved communication bandwidth.
No CPU/GPU affinity or power settings were changed for this analysis.

Some dX and dW products are mathematically independent and could use separate
CUDA streams. However, the large GEMMs already consume much of each GPU's
compute and power capacity; concurrent launches share those resources rather
than doubling them. A focused stream experiment could test thin or final-tail
products, including their synchronization cost, before changing the full backward
schedule. Model or pipeline parallelism would add activation transfers and
potential pipeline bubbles over this topology, with additional implementation
complexity. It is not an automatic improvement over the existing two-GPU data
parallel execution.

## Refreshed isolated performance and compute headroom

The existing `benchmark_gemm.py` tested all 12 core shape/direction pairs on
GPU 0 using deterministic BF16 operands with production transpose strides.
Each case had at least two seconds of burn-in and five CUDA-event timing trials
of 30 products. The existing BF16 intermediate-reduction allowance was preserved;
no lower-precision model or gradient representation was introduced.

NVIDIA specifies 438.9 dense BF16 TFLOPS with FP32 accumulation at a 2,280 MHz
boost clock, 300 W board power, and 1,792 GB/s memory bandwidth for this Max-Q
GPU. Source: Appendix A of the
[RTX Blackwell architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf#page=46).

The final column below divides achieved performance by
`438.9 * observed_SM_clock_MHz / 2280`. It is an illustrative clock-scaled
comparison, not a hardware-counter utilization measurement. Clocks are telemetry
snapshots after each timed case, not time-integrated clock measurements.

| Product | Median time | TFLOPS | Observed SM clock | Clock-scaled comparison |
| --- | ---: | ---: | ---: | ---: |
| FFN up/gate forward | 1.8473 ms | 260.4 | 1,492 MHz | 90.7% |
| FFN up/gate dX | 1.9148 ms | 251.2 | 1,402 MHz | 93.1% |
| FFN up/gate dW | 1.8925 ms | 254.2 | 1,425 MHz | 92.7% |
| Input projection forward | 1.3337 ms | 257.6 | 1,485 MHz | 90.1% |
| Input projection dX | 1.3958 ms | 246.2 | 1,417 MHz | 90.2% |
| Input projection dW | 1.3531 ms | 253.9 | 1,447 MHz | 91.2% |
| FFN down forward | 0.9879 ms | 243.5 | 1,440 MHz | 87.8% |
| FFN down dX | 0.9162 ms | 262.5 | 1,530 MHz | 89.1% |
| FFN down dW | 0.9399 ms | 255.9 | 1,485 MHz | 89.5% |
| Core output forward | 0.2656 ms | 258.8 | 1,612 MHz | 83.4% |
| Core output dX | 0.2625 ms | 261.8 | 1,620 MHz | 83.9% |
| Core output dW | 0.2578 ms | 266.5 | 1,725 MHz | 80.3% |

Every timed case's final power sample was approximately 300 W. The dominant
families remain close to the earlier measured range and approximately 88–93%
of this clock-scaled estimate. The smaller output projection has more apparent
relative headroom, but represents only 5.44% of matrix time. Neither comparison
implies all remaining theoretical capacity can be recovered by software.

## Compute versus operand traffic

For one idealized BF16 read of A/B and one write of C, arithmetic intensity is
`2*M*N*K / (2*(M*K + K*N + M*N))`. It is about 1,470 FLOPs/byte for the FFN
up/gate products, 1,412 for the input projection, 1,334 for FFN down, and 910
for the square output projection. Actual traffic includes reloads, cache effects,
and possible intermediate reductions; these are operand-footprint estimates.

This supports a compute-efficiency focus for the large dense GEMMs. It does not
prove that every tile load is perfectly hidden. A persistent/TMA implementation
can improve operand delivery or scheduling even when the overall dense product
is predominantly compute-limited.

State-memory products are different. Shapes with batch 128 and `M=16, N=4096,
K=16` have only about 8 FLOPs/byte under that estimate; the `128x64x64` and
`64x64x128` batched products have about 25.6. Observed rates are roughly
10–61 TFLOPS, with 16x16/32x32 WMMA kernel tiles. They merit smaller-shape or
producer/consumer fusion experiments, but their entire current kernel budget
is only 95.94 ms, approximately 2.3% of the profiled update.

## Prior experiments and specific next tests

The September 10 experiments already covered eager cuBLAS, cuBLASLt preference,
ATEN-only compilation, default Triton autotuning, and persistent TMA. See
[`RTX_PROFILE_20260910.md`](RTX_PROFILE_20260910.md).

| FFN up/gate product | Prior cuBLAS | Prior default Triton | Prior persistent TMA | Current cuBLAS |
| --- | ---: | ---: | ---: | ---: |
| Forward | 1.828 ms | 1.981 ms | 1.798 ms | 1.847 ms |
| dX | 1.885 ms | 2.057 ms | 1.890 ms | 1.915 ms |
| dW | 1.871 ms | 1.869 ms | 1.810 ms | 1.892 ms |

These are separate runs with different observed clocks, not a controlled
same-clock speedup claim. Default Triton usually lost to cuBLAS; persistent TMA
closed much of that gap and won some shapes modestly. cuBLASLt preference was
mixed, and its earlier up/gate dW result coincided with a higher clock. Repeating
a global backend toggle is therefore a weak next experiment.

The following targeted experiments have a better basis:

1. **Select algorithms per dominant shape/direction.** Retest persistent TMA
   against ATEN/cuBLAS for up/gate forward and dW first, with alternating warmed
   trials and clock samples. Keep cuBLAS for shapes where it wins. Measure an
   entire layer forward/backward and then the full two-GPU update so template
   fusion and DDP overlap are included. Preserve the BF16 dW rounding before
   FP32 accumulation.
2. **Test a GEMM epilogue that performs gradient conversion and accumulation.**
   The current conversion/addition budget is 129.076 ms/update. A fused kernel
   must retain the BF16 GEMM result's rounding before its FP32 addition and keep
   autograd/DDP ownership correct. Direct FP32-output `addmm` was already tested:
   it changes rounding and was not consistently better. The present compiled
   autograd candidate also demonstrates why local fusion needs full-update
   communication validation.
3. **Explore a paired up/gate epilogue only if its GEMM remains competitive.**
   The combined projection is already one GEMM, with up and gate columns 7,168
   apart. A tile/layout that places matching pairs together could compute SwiGLU
   before writing its activated output. Backward still needs the projected up
   and gate values, so their storage cannot simply disappear. The removable
   standalone forward budget is about 41.16 ms/update; recomputing the projection
   would cost about 405.94 ms and would defeat that purpose. Repacking must use
   temporary views/copies and retain canonical archive/checkpoint layout.
4. **Explore the small state-memory GEMMs separately.** Their low intensity and
   small dimensions make tiled producer/consumer fusion more plausible than
   expecting dense-projection TFLOPS. Account for both gradient products and
   layout preparation; a low TFLOPS number on its own does not establish waste.

The installed compiler already has epilogue/prologue fusion enabled for suitable
templates. Its `kernel/mm.py` adds persistent TMA choices only through the
max-autotune template route, gated by `ENABLE_PERSISTENT_TMA_MATMUL`. Enabling
that variable alone does not replace every existing library GEMM.

The distinction between GPU architectures matters: this workstation is SM120.
The installed `use_triton_blackwell_tma_template()` additionally requires
`is_datacenter_blackwell_arch()`, whose predicate is `100 <= arch < 110`.
The named datacenter-Blackwell warp-specialized template and the CUTEDSL route
therefore are not automatic options for this RTX GPU; generic persistent TMA
is the relevant already-tested path. The environment also lacks the `cutlass`
and `cutlass_library` packages and the configured CUTLASS source directory, so
CUTLASS/NVGEMM are not available simply by changing the backend flag.

As a conditional scale check, retaining the measured 3.1664 seconds of matrix
time and deleting every other cost would give about **41.4K tokens/s**. This is
not a hardware limit or an achievable configuration. Reaching 50K with all
other costs zero would still require about 17.2% less matrix time. The isolated
results give no evidence for a software-only reduction of that scale across all
products at the present power limit.

## Evidence and reproduction

Generated evidence remains ignored in Git:

- `work/validation/rtx-opt-20260912/matmul/matmul-attribution.json`: both ranks,
  both profiles, shapes/strides/dtypes, call counts, exact kernel names and launch
  metadata, FLOPs, operand intensities, and current clock-scaled comparisons.
- `work/benchmark-opt-20260912/profile-baseline/`: source traces, kernel TSVs,
  operator tables, and category summaries.
- `work/benchmark-opt-20260912/profile-compiled-autograd/`: corresponding
  candidate traces. Include its eight BF16 `aten::addmm` head-dW calls when
  counting arithmetic; otherwise the head's weight-gradient FLOPs are omitted.
- `work/benchmark-gemm-20260912-current/results.json`: refreshed isolated trials
  with telemetry for each product.
- `work/benchmark-gemm-20260910/`, `work/benchmark-gemm-cublaslt-20260910/`, and
  `work/benchmark-gemm-tma-20260910/`: earlier backend comparisons.

Installed compiler source evidence is under
`.server-venv/lib/python3.12/site-packages/torch/_inductor/`: `kernel/mm.py`,
`config.py`, `utils.py`, and `codegen/cuda/cuda_env.py`. These observations apply
to the installed `2.14.0+cu130` build and should be rechecked after an upgrade.

To repeat the isolated sweep in an exclusive GPU window, with a new output
directory:

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
CUDA_VISIBLE_DEVICES=0 .server-venv/bin/python -B ops/benchmark_gemm.py \
  --families ffn_up_gate in_proj ffn_down out_proj \
  --operations fwd dx dw --rows 8192 --backends eager \
  --burnin-seconds 2 --repeats 30 --trials 5 \
  --output-dir work/benchmark-gemm-20260912-recheck
```

The full-model profiling command and checkpoint protections are documented in
[`OPTIMIZATION_20260912.md`](OPTIMIZATION_20260912.md). Compilation and benchmark
outputs must stay under the workspace, and a new experiment must not overwrite
the retained reports or archived weights.
