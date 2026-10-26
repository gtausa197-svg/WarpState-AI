# Remaining pointwise work and GEMM saturation — September 13, 2026

This analysis uses the saved September 12 full-model profile of the selected
paired backward layout. It does not launch a new GPU profile, pause production,
or change the training recipe. At the initial status check, production was
running past step 53,854, at approximately 31.2K tokens/second during updates.

## Timing scope

The profile is one update after eight warmups, with two RTX PRO 6000 Blackwell
Max-Q GPUs, microbatch four per rank, sequence 2,048, eight accumulation
microbatches, and 131,072 global tokens. Model computation uses BF16; master
parameters, accumulated gradients, communication, and AdamW state use FP32.
The communication preset uses conservative wait sinking and 128 MiB buckets.

The profiled update lasts 4.16463 seconds. Repeated unprofiled measurements of
this preset averaged 4.20116 seconds, or 31,199.03 tokens/second. Profiles include
instrumentation; neither number includes periodic validation/checkpoint pauses.
Kernel durations from separate streams can overlap and must not be summed into
elapsed time without interval accounting. Individual operator totals and their
kernel totals describe the same work.

## Remaining pointwise and reduction work

The name-based compiled-pointwise/reduction category contains 475.795 ms on
rank zero and 479.331 ms on rank one. This includes arithmetic, conversions,
copies, and reductions fused into generated kernels. It excludes the selected
custom paired layout, whose two kernels take approximately 78.96 ms per rank.
Consequently, neither the small explicit memcpy category nor the 476 ms category
alone describes all memory traffic or all work outside matrix multiplication.

The largest individually attributed groups in that category are:

| Work | Rank 0 ms/update | Rank 1 ms/update | Current boundary |
| --- | ---: | ---: | --- |
| SwiGLU backward | 85.70 | 85.96 | Packed activation-gradient output |
| Parameter gradient conversion/accumulation | 69.55 | 70.29 | Includes already fused FP32 accumulation |
| Core normalization/residual backward | 68.42 | 68.10 | Several operations share each kernel |
| Attention/memory forward transforms | 66.84 | 69.35 | Rotary, gating, and layout work |
| Core FP32-to-BF16 weight casts | 49.59 | 49.54 | Four weight families, eight microbatches |
| SwiGLU forward | 49.20 | 49.13 | Already fused SiLU and multiplication |
| Core normalization forward | 11.20 | 11.18 | Includes fused surrounding work |

These are disjoint groups within the compiled category, not the full category.
Attention/memory backward transforms, residuals, bucket operations, loss, and small
parameter reductions account for the remainder. Attribution groups profiler
operators by complete input shapes and checked generated-code bodies. The same
generic kernel number can represent different operations in different graphs;
names alone do not establish the mapping. The generated report is
`work/validation/pointwise-headroom-20260913/semantic-families.json` and retains
the explicit per-operator mapping, input shapes, source paths, and provenance.

Cutting rank zero's 475.795 ms budget by 25% would remove about 119 ms. If every
saved millisecond reduced elapsed time and other execution stayed unchanged,
4.20116 seconds would become 4.08221 seconds: approximately 2.9% more throughput.
This is conditional arithmetic, not a measured candidate or a promised gain.

Prior experiments already tested standalone SwiGLU and RMSNorm replacements.
SwiGLU forward/backward improved from 0.61335 to 0.59965 ms in its isolated
production-shaped fixture; RMSNorm improved from 0.43256 to 0.39248 ms. Neither
standalone mode improved the full-model screen. External weight-cast caching
also regressed overall and changed compiler/DDP partition boundaries. Their
switches are not supported with the selected compiled-autograd configuration.
See [OPTIMIZATION_20260912.md](OPTIMIZATION_20260912.md).

The earlier fused SwiGLU-forward/TMA-weight-gradient recipe produced one +0.43%
full-model comparison after correcting duplicated weight storage. It lacks a
repeated gain and has not been combined with the selected paired layout.
The isolated epilogue's 45-microsecond gain predicts only about 10 ms per update
before interactions. See [MATMUL_TUNING.md](MATMUL_TUNING.md).

### Best focused next experiment

Start with the **85.7 ms SwiGLU backward region**, using a replacement inside
the compiled backward graph rather than the previously screened standalone
activation mode. Its generated kernel traverses a packed `8192 x 14336` output
and selects between the two derivative halves. The source has five masked
loads across those branches. A paired-output implementation can load the
up value, gate value, and upstream derivative once for each logical pair, then
write both derivative halves. This targets repeated operand access and branch
work while retaining native packed storage and the existing GEMMs.
The existing `activation_execution.py` already contains a paired backward
kernel. First compare that arithmetic and layout directly against this actual
compiled region; a new primitive implementation is not the starting requirement.
The integration route would preserve the surrounding compiled-autograd graph.

Arithmetic must follow the compiled reference. The up derivative uses
`g / (1 + libdevice.exp(-g))`, while the gate derivative uses `tl.sigmoid(g)`.
Sharing one sigmoid value by algebraic rewriting is not automatically equivalent
to those two implementations. Preserve their evaluation and BF16 output rounding
unless a separately declared numerical comparison justifies a change.

The relevant body is `triton_poi_fused_5` near line 436 of
`.cache/torchinductor/cw/ccwrp762vinzoemcnamsrl2wt54aoanpxjxxqobh7e7oxxutputc.py`.
Its synchronized variant is classified separately by matching operand shapes.
The source suggests an opportunity; it does not prove an instruction-count,
physical-bandwidth, or elapsed-time saving. A 25% reduction of this region
would save about 21.4 ms per update, approximately 0.5% more throughput if fully
realized. No new candidate kernel was timed or selected in this analysis.

The next candidate is tuning the **68.4 ms normalization/residual backward
group as a whole**. The existing kernels already fuse FP32 row reductions,
parameter-gradient partials, residual additions, and BF16 output stores.
Tune tiling and intermediate-buffer traffic while preserving those combinations;
the entire group is not removable normalization overhead.
One specific subregion accounts for approximately 24.5 ms and loops twice over
2,048 channels, rereading inputs for the normalization input derivative and
residual output. A persistent row kernel could retain loaded intermediates,
but increased register pressure could offset the saved reads. This is a narrower
test than replacing every normalization operation.

Gradient conversion/accumulation is already fused after many matrix products.
Further integration into a GEMM epilogue must preserve the BF16 product's
rounding before FP32 accumulation and keep gradient readiness/communication
ordering. Hoisting weight casts across accumulation microbatches could reduce
the 49.6 ms cast budget, but must solve the prior autograd/DDP-boundary regression.
These are more involved follow-ups than the isolated SwiGLU backward replacement.

Use the same sequence as the selected layout integration: exact-shape primitive
comparison, guarded graph matching and runtime dispatch evidence, full gradient/
optimizer/resume fixtures, then repeated complete-model timing. An isolated
kernel gain alone is insufficient to change the training preset.

## Where matrix time is spent

Every one of the 5,400 matrix kernels per rank is joined by its CUDA External ID
to the originating `aten::mm`, `aten::bmm`, or `aten::addmm` event. Exact operand
shapes and strides identify families and forward/input-gradient/weight-gradient
directions. This includes all eight vocabulary-head weight-gradient ADDMM calls.
The calculation counts 791.597 trillion matrix FLOPs per rank per update.
All matrix kernels use the same compute stream in these traces.

| Matrix family | Rank 0 ms | Rank 1 ms | Rank 0 share of matrix time |
| --- | ---: | ---: | ---: |
| FFN up/gate | 1,281.23 | 1,225.96 | 40.05% |
| Attention/memory input projection | 907.66 | 862.34 | 28.38% |
| FFN down | 632.02 | 602.74 | 19.76% |
| Core output projection | 174.37 | 171.46 | 5.45% |
| Vocabulary head | 103.37 | 98.64 | 3.23% |
| Small state-memory batched products | 100.09 | 96.61 | 3.13% |
| Total | 3,198.75 | 3,057.76 | 100% |

The first three families account for **88.19% / 88.01% of matrix time** and
**67.74% / 64.62% of the profiled update** on ranks zero/one. All matrix kernels
occupy 76.81% / 73.42% of that update. These are duration shares, not measures
of compute saturation. The generated attribution is
`work/validation/pointwise-headroom-20260913/paired-matmul-attribution.json`.
Its source profiles and reviewed shape/stride mapping are hashed in the report.

## What the saturation evidence establishes

The dominant dense families sustain approximately 252–256 TFLOPS on rank zero
and 264–268 TFLOPS on rank one in this stored full-model profile. Earlier warmed
isolated tests measured approximately 243–263 TFLOPS for their individual shapes.
Relative to an illustrative peak adjusted for their observed clocks, those
isolated results reached approximately **88–93%**. That estimate scales the
published 438.9 dense BF16 TFLOPS by observed MHz / 2,280 MHz. The specification
uses boost clocks; the denominator is not a measured sustained hardware limit.
[NVIDIA architecture specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf#page=46).

Those isolated clock samples are snapshots, not time-integrated clocks for the
new profile. The results support a compute-efficiency focus for large dense
GEMMs, while small state-memory products have much lower arithmetic intensity.
They do **not** mean that 88% of matrix time is fully saturated, or that each
dense kernel spends 88–93% of its lifetime at full utilization. Operand loads,
Tensor Core instructions, and output work overlap within a pipelined GEMM;
kernel duration alone cannot partition that execution into saturated and
unsaturated intervals.

The exact requested fraction remains unmeasured. A prior Nsight Compute attempt
failed with `ERR_NVGPUCTRPERM`; the read-only driver check still reports
`RmProfilingAdminOnly=1`. No counter-access policy was changed for this analysis.
The session's effective `CAP_SYS_ADMIN` and `CAP_PERFMON` capabilities are both
absent. The read-only policy/capability check does not collect GPU counters.
The prior failure is saved in
`work/validation/rtx-profile-20260910/ncu-metrics.txt`.
NVIDIA documents this as restricted access to GPU performance counters.
[Counter permission reference](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

After counter access is available, a bounded measurement should use disposable,
warmed copies of the dominant GEMM shapes, with the same BF16 operands, strides,
power limits, and relevant cache conditions. Collect Tensor-pipeline throughput,
SM activity, memory throughput, and scheduler statistics. Prefer elapsed-cycle
metrics when assessing whole-device utilization: active-cycle metrics exclude
idle cycles. Query the installed tool's supported metrics rather than assuming
a metric name from another architecture.

To answer a literal fraction-of-time question, collect supported PM-sampling
timelines and declare a near-saturation threshold, for example 90% of the
appropriate Tensor-pipeline peak. Correlate above-threshold sample intervals
with the GEMM execution intervals, and report sampling resolution and coverage.
Per-kernel average utilization alone cannot determine that time fraction.
Nsight Compute documents both active/elapsed normalization and time-resolved PM
sampling, including examples of operand-delivery stalls and underfilled tails.
[Profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#pm-sampling).
