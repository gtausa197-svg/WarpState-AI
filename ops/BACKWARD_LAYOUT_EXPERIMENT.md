# Attention backward layout experiment — September 12, 2026

The isolated paired-lane implementation reduces this region from **0.519935 ms
to 0.357944 ms** on one RTX PRO 6000 Blackwell Max-Q GPU: **31.16% less time**.
It matches the original profiled kernels byte for byte on the tested production
shape. This isolated experiment did not include a training adapter or establish
a full-model speedup. The subsequent opt-in graph integration and its training
measurements are documented in
[BACKWARD_LAYOUT_INTEGRATION.md](BACKWARD_LAYOUT_INTEGRATION.md).

## Boundary and provenance

The existing compiled-autograd overlap profile contains two adjacent kernels:

| Original kernel | Rank-0 time per update | Calls per update |
| --- | ---: | ---: |
| `triton_poi_fused_17`: rotary backward and Q/K scratch | 31.907 ms | 224 |
| `triton_poi_fused_18`: packed Q/K/V/gate/U gradient | 77.576 ms | 224 |

The evidence is
`work/benchmark-opt-20260912/profile-ca-overlap/profile-rank-0-kernels.tsv`.
Their combined **109.483 ms** is an existing kernel-time budget, not an amount
already removed from training. The isolated timing uses a different execution
context; do not scale it into a claimed training throughput improvement.

`ops/backward_layout_baseline.py` retains the exact two profiled arithmetic and
indexing bodies as source, with their original launch dimensions. It records
the original generated source identifiers
`cq665jkyfy6uac7wrzvwa5aigwids6rzt4n7vzip6xvufai4ajjw` and
`cbr6gfx3vjhnzjbykfch3pg5peunfivhgdrfvl5hnzolhvfjfgdy` for provenance.
Running the experiment recompiles these tracked bodies; it does not load
ignored compiler-cache files or depend on preserving them.

The shape is `[B,N,H,C,D] = [4,16,32,128,64]`: microbatch four, 2,048-token
sequences, 32 heads, and a 64-element head dimension. Saved activations and
gradients are BF16. The original saved rotary sine/cosine tables are FP32.
K-write gradients have `[B,N,H,D,C]` storage; the output gradient is token-major.
The final packed tensor has shape `[4,2048,10240]`, ordered Q/K/V/gate/U.

## Arithmetic and implementation

`ops/backward_layout.py` provides an independent tensor reference, deterministic
synthetic saved inputs, the validated `launch_layout` interface, and
`prepare_layout`, which returns an allocation-free callable for timing.

The new rotary kernel processes even/odd pairs together. It reads the separate
local and memory Q gradients and adds them in FP32, applies the rotary backward
transform, and rounds the final Q result to BF16. It preserves the original
interleave's inactive-lane `+0.0`, including signed-zero behavior.

K first combines the transposed write gradient through the tanh derivative with
the local gradient in FP32. Its sine/rotary contribution is **rounded to BF16
before** adding the cosine contribution, as in the original two-kernel path.
That rounded intermediate stays in registers. K then rounds to BF16 on output.

A second kernel copies V, computes the sigmoid gate derivative in the original
operation order, and applies the U tanh derivative. Both kernels write directly
to their disjoint slices of the packed output. The `single` variant launches
both sets of programs in one grid using a uniform per-program branch. Keeping
two launches was slightly faster in the final comparison.

There are no reductions, GEMMs, parameter-gradient updates, collectives, or
optimizer operations in this boundary. Model parameters, FP32 gradient
accumulation, FP32 communication, and FP32 optimizer state are untouched.
Inputs, output, and profile scratch have explicit dtype/shape/device/stride
checks; output and scratch storage may not overwrite saved inputs or each other.

## Measurement and correctness

The initial screen covered two launch algorithms, blocks 256/512/1024, and
4/8 warps. All twelve configurations matched the original outputs numerically.
The largest single-grid configuration was slightly slower than the baseline,
so fewer launches alone did not establish a win. The initial report's equality
field compared values; a later stricter byte comparison detected a few signed
zeros, prompting the explicit inactive-lane zero addition in the final source.

The final confirmation used a different seed, block 256, eight warps, nine
trials, and 100 calls per timing interval. Each candidate interval was bracketed
by original-kernel intervals. Inputs, output, and original scratch were
preallocated; both paths were warmed before measurements. CUDA events measure
the queued execution, including possible launch scheduling gaps.

| Final algorithm | Median original | Median candidate | Time reduction |
| --- | ---: | ---: | ---: |
| Paired, two launches | 0.519935 ms | 0.357944 ms | 31.16% |
| Single grid | 0.520078 ms | 0.358521 ms | 31.06% |

Both final candidates had **zero byte differences** across every Q/K/V/gate/U
projection: 83,886,080 BF16 output elements per comparison. Production-shape
tests use another seed and strict integer-view byte equality. Ragged-shape
tests compare each projection independently against tensor expressions, use
unequal even/odd rotary coefficients, and verify saved inputs remain unchanged.
CPU checks include the model's actual rotary VJP, gate autograd, packing order,
observable intermediate BF16 rounding, and malformed input rejection.

The independent eager FP32 tensor expression and original compiled kernels can
have different FMA contraction. In the final run their maximum relative RMS
error was `8.72e-6`; V/gate/U were exactly equal. Therefore the independent
reference establishes mathematical indexing/arithmetic, while the original
compiled bodies supply the exact GPU-rounding oracle. These finite synthetic
fixtures do not prove equivalence for every possible activation distribution.

The final GPU suite passed **18 tests**. The focused suite also supports CPU-only
execution. Reports remain ignored:

- `work/benchmark-backward-layout-20260912/results.json`: initial twelve-way screen.
- `work/benchmark-backward-layout-confirm-20260912/results.json`: signed-zero diagnostic.
- `work/benchmark-backward-layout-final-20260912/results.json`: final source, strict byte comparison, and confirmation timings.
- `work/validation/rtx-opt-20260912/backward-layout-tests.log`: final GPU tests.

The final timing report records source hashes, arguments, Torch `2.14.0+cu130`,
CUDA 13.0, device identity, and per-projection errors. Later test/documentation
additions do not change kernel arithmetic. After measurement, one extra EOF
blank line was removed from `backward_layout_baseline.py` for the Git whitespace
check. Its parsed AST is identical; the original and final source hashes are
recorded in `work/validation/communication-exploration/backward-source-normalization.json`.

## Traffic and integration limits

The rewrite avoids the logical stores and subsequent reads of two
16,777,216-element BF16 scratch tensors: **134,217,728 bytes (128 MiB) per
region invocation**. This is a source-level traffic calculation, not a measured
DRAM counter. Cache behavior and other reads also affect the measured timing.
The production compiler reuses a dead buffer for one scratch tensor, so this
does **not** establish two fewer allocations or a corresponding peak-VRAM gain.

The prototype accepts separate local and memory gradients at the same backward
boundary as the profiled graph. A naive custom forward that returns one shared
BF16 Q tensor can cause autograd to round the local-plus-memory gradient sum
before rotary backward; the current graph adds those contributions in FP32.
Integrating the kernel requires preserving that join and the K rounding point,
then validating real model gradients, optimizer updates, checkpoint resumption,
and full-model timing. No such integration is selected by this experiment.

## Recommended next integration experiment

The narrowest route is a **post-gradient graph replacement**, leaving the
forward and autograd ownership intact. The installed PyTorch exposes
`torch._inductor.config.post_grad_custom_post_pass`; its
`fx_passes/post_grad.py` applies this hook to normalized, functionalized graphs
before final topological sorting and fake-tensor metadata updates. Implement a
`CustomGraphPass` that replaces only the existing packed-gradient region with a
pure opaque `torch.library.custom_op` and a matching fake implementation.
`ops/matmul_execution.py` already uses that opaque backward-kernel boundary to
avoid this PyTorch version's Compiled Autograd Triton higher-order tracing issue.
These are version-dependent compiler interfaces; their availability is a
proposed route, not evidence that this replacement has been integrated.

The pass should match the actual backward dataflow and arithmetic, not generated
kernel names or an inference flag. Require the exact supported shapes, dtypes,
strides, projection ordering, separate Q gradient branches, K rounding boundary,
and gate scale/expression. Verify that removed intermediates have no consumers
outside the replacement. On any mismatch, leave the original graph intact and
report the missed replacement; an opt-in validation run must fail if expected
dispatches are absent. Initially support only the measured parallel-memory,
dropout-zero production shape. Pass existing native-layout tensors into the
opaque operator and reshape views inside it, avoiding new contiguous copies
merely to satisfy a custom-operator input layout.

Use `CustomGraphPass.uuid()` with hashes of the matcher, kernel, and wrapper
sources plus all specialization settings, so a changed replacement cannot reuse
an incompatible compiled artifact. The installed `custom_graph_pass.py`
documents `get_hash_for_files` for this purpose. Record the same execution
identity in benchmark and checkpoint metadata, with explicit reconfiguration
for resumption changes. A new forward custom-autograd splitter is less suitable:
separate Q outputs address its gradient join, but the complete gate/packing
fusion also needs the later blend's raw output gradient and saved local/memory
values. Capturing that region through a forward boundary would require broader
ownership of the attention/memory computation or extra materialized gradients.

Begin with a match-only graph inspection, then replace one supported core.
Require generated-kernel/profile evidence that the expected regions actually
dispatch the paired implementation, with no added layout copies. Compare
forward outputs, all parameter gradients, accumulated FP32 gradients and Adam
updates against the same compiled baseline; cover first and later accumulation
microbatches, the final DDP synchronization, and checkpoint continuation. Then
measure repeated full-model runs from the same checkpoint/token cursor,
including peak VRAM and communication overlap. Select the replacement only
after those correctness and full-model performance checks pass.

## Reproduction

Reserve one idle GPU and keep training paused before running CUDA commands.
Use a new output directory each time. All source, caches, and artifacts stay
inside `/workspace/WarpState`.

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
CUDA_VISIBLE_DEVICES='' .server-venv/bin/python -B -m pytest -q test_backward_layout.py
CUDA_VISIBLE_DEVICES=1 WARPSTATE_TEST_CUDA=1 \
  .server-venv/bin/python -B -m pytest -q test_backward_layout.py
CUDA_VISIBLE_DEVICES=1 .server-venv/bin/python -B \
  ops/backward_layout_benchmark.py \
  --output-dir work/benchmark-backward-layout-screen-new
CUDA_VISIBLE_DEVICES=1 .server-venv/bin/python -B \
  ops/backward_layout_benchmark.py --algorithms paired single \
  --blocks 256 --warps 8 --seed 9929 --trials 9 --repeats 100 \
  --output-dir work/benchmark-backward-layout-confirm-new
```

The benchmark needs no dataset or model checkpoint. It reports candidates
even when their bytes differ, so inspect correctness fields before interpreting
timings. It does not generate a training recipe or modify the production run.
