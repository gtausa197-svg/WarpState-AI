WarpState's matrix experiments select BF16 kernels by operation, exact matrix
dimensions, and operand strides. They are opt-in through `--matmul-plan` in
`run_40b.py` and `ops/benchmark_training.py`. The workstation preset selects
compiled autograd with communication overlap. Matrix kernels remain explicit
experiments; isolated GEMM improvements alone do not select them for production.
The complete two-GPU comparisons and validation are recorded below.

The implementation is split across these source files:

| File | Responsibility |
| --- | --- |
| [matmul_kernels.py](matmul_kernels.py) | BF16 ATen, tiled, persistent, and persistent TMA matrix launches; tile validation and shape/stride signatures. |
| [matmul_epilogue.py](matmul_epilogue.py) | Paired up/gate matrix multiplication with a SwiGLU epilogue; returns both packed projection and activation. |
| [matmul_execution.py](matmul_execution.py) | Strict plan loading, registered autograd, and selected core-linear/SwiGLU training adapters. |
| [tune_matmul.py](tune_matmul.py) | Disposable ordinary/fused FFN matrix tuning with warmed controls and generated candidate recipes. |
| [activation_execution.py](activation_execution.py) | Existing BF16 SwiGLU derivative used by the fused epilogue's backward. |

The adapter is applied before core compilation and DDP construction. Original
FP32 parameters remain the autograd leaves. Their differentiable BF16 casts stay
inside the visible forward graph; the adapter does not retain external casts
across updates or write leaf gradients through hooks. Matrix operands, returned
matrix gradients, projections, and activations are BF16. Custom dot products
accumulate in FP32. Parameter-gradient accumulation, clipping, DDP reduction,
FP32 master parameters, and AdamW state remain under the existing training
framework. The token cursor, objective, optimizer schedule, and checkpoint keys
retain their existing meanings. CPU and evaluation model execution use the
original module paths.

Selected kernel launches use opaque registered custom operations with fake
output metadata, surrounded by the adapter's registered autograd. This is
necessary because installed PyTorch 2.14 compiled autograd cannot retrace AOT's
functionalized Triton launch operator. Unselected ATen products remain visible
to the compiler. The CUDA suite exercises this second backward capture as well
as ordinary eager/AOT execution; a plain forward compilation test would miss it.

Each JSON plan has format `warpstate-matmul-plan-v1`, an optional description,
and an `entries` array. For `A[M,K] @ B[K,N]`, `m`, `n`, and `k` describe that
operation; strides are in elements. They are essential: a transposed weight and
a contiguous matrix with the same dimensions can require different kernels.
The complete dispatch key is `(kind, m, n, k, a_stride, b_stride)`. Missing entries
use ATen. An empty plan selects no custom kernels. Duplicate JSON keys, duplicate
dispatch entries, unknown fields, invalid tiles, and non-BF16 entries are rejected.

This is a **candidate example**, not a selected production recipe:

```json
{
  "format": "warpstate-matmul-plan-v1",
  "description": "Candidate paired FFN epilogue; requires complete-model validation.",
  "entries": [
    {
      "kind": "swiglu",
      "name": "ffn_up_gate.fwd",
      "m": 8192,
      "n": 14336,
      "k": 2048,
      "a_stride": [2048, 1],
      "b_stride": [1, 2048],
      "dtype": "bfloat16",
      "algorithm": "tiled",
      "tile": {
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "num_warps": 8,
        "num_stages": 3
      }
    }
  ]
}
```

`kind: "mm"` selects an ordinary matrix product. Forward and backward products
have independent entries. The FFN up/gate forward uses `(M,N,K) =
(8192,14336,2048)` with strides `[2048,1]` and `[1,2048]`. Its weight-gradient
product uses `(14336,2048,8192)` with strides `[1,14336]` and `[2048,1]`, matching
the original `dY.T @ X` orientation. A forward winner therefore does not select
that weight-gradient kernel automatically.

`kind: "swiglu"` selects the fused forward. Here `N=2H` is the complete packed
projection width, while `block_n` counts columns in **each** half. Backward uses
the saved BF16 packed projection, the existing SwiGLU derivative, and independently
selected `mm` entries for `dX` and `dW`.

| Algorithm | Execution | Supported plan kinds |
| --- | --- | --- |
| `aten` | Calls the installed PyTorch matrix implementation with its existing process settings. | `mm`, `swiglu` |
| `tiled` | Launches one program for each output tile, with grouped row tiles for weight reuse. | `mm`, `swiglu` |
| `persistent` | Launches at most one program per SM; each program strides over several output tiles using ordinary loads. | `mm`, `swiglu` |
| `persistent_tma` | Persistent scheduling with Triton tensor descriptors and hardware TMA operand loads. | `mm` |

Persistent scheduling alone does not imply TMA or warp specialization. The TMA
implementation uses an opaque registered custom-op boundary because the installed
PyTorch 2.14 AOT path does not preserve the host descriptor metadata through a
plain `triton_op` launch. It does not secretly substitute another algorithm.
TMA operands need an inner stride of one and 16-byte-aligned rows and storage
offsets. The training selector falls back to ATen for an incompatible layout;
the explicit kernel launcher reports unsupported requests. The paired SwiGLU
kernel does not currently support TMA.

Tiles specify `(block_m, block_n, block_k, num_warps, num_stages)`. Block sizes
must be powers of two from 16 through 256, with 4 or 8 warps and 1 through 4
pipeline stages. A conservative SM120 operand-pipeline estimate rejects ordinary
tiles above `(BM+BN)*BK*2*stages = 96 KiB` and paired tiles above
`(BM+2*BN)*BK*2*stages = 96 KiB`. This is an estimate, not a guarantee about
compiler register/shared-memory allocation. The tuner records compilation failures
without promoting those candidates.

The paired epilogue arranges adjacent logical columns as up/gate pairs, using
their physical packed offsets `j` and `j+H`. One wider tensor-core dot reuses the
input tile, then splits the accumulator in registers. Both FP32 projection halves
are rounded to BF16 **before** the activation. The kernel stores that rounded
packed projection for backward and also writes the BF16 activation. It saves a
subsequent packed-projection read and activation launch; it does not remove the
projection storage required by backward.

The `eager_rounding` argument additionally rounds the SiLU result to BF16 before
multiplication when following eager execution. Compiled execution follows the
existing compiler's fused SiLU/product arithmetic, while still rounding the GEMM
projection first. No FP8, FP16, quantization, split-K atomics, or reduced-precision
gradient communication is introduced. Exact equality on measured aligned shapes
does not promise bitwise equality for every GEMM layout or compiler version.
For example, the tested ragged `K=193` cuBLAS reference used a different reduction
result: its packed relative-L2 difference was 0.269%, while the custom result
differed by 0.000394% from an FP32-dot reference. Tests separately check projection
accuracy and that activation consumes the saved rounded BF16 projection.

On GPU 1 of the RTX PRO 6000 Blackwell Max-Q workstation, the final paired kernel
was compared with `torch.compile(fullgraph=True)` around the complete BF16
`mm -> split -> SiLU -> multiply` forward, returning both packed and activation
outputs. Inputs had shape `[8192,2048]` and transposed weights `[2048,14336]`.
With seed 563, 400 warmup calls per variant, and 15 alternating-order batches of
40 CUDA-event-timed calls, the selected microbenchmark tile measured:

| Forward implementation | Median milliseconds |
| --- | ---: |
| Compiled ATen matrix multiplication plus SwiGLU | 1.950789 |
| Paired single-dot epilogue, tiled `(128,128,32,8,3)` | 1.905561 |

This is **2.37% greater primitive throughput**, or approximately 45 microseconds
less per call; paired batch gains ranged from 2.08% to 2.75%. Packed projections
and activations matched the compiled reference bitwise for two production-size
random input seeds. The final epilogue suite passed 33 tests, covering both
schedules, eager/compiled rounding, ragged dimensions, and the default/best tiles.
A subsequent CPU-only invocation passed 9 and skipped 24 CUDA fixtures.

Applying the measured 45-microsecond difference to 28 cores and eight microbatches
gives about 10 milliseconds per update before interaction with backward,
communication, compiler scheduling, and allocator behavior. That arithmetic is
not a measured complete-training gain. The earlier two-dot implementation lost
to the control; it is superseded by the single-dot source.

Generated evidence remains under
`work/validation/rtx-opt-20260912/epilogue/`:

- `paired-swiglu-confirm.json`: final steady-state comparison and second input seed.
- `paired-swiglu-single-dot.json`: final single-dot implementation's tile/schedule screen.
- `paired-swiglu.json` and `paired-swiglu-more-tiles.json`: superseded two-dot screens.

The ordinary GEMM screen tested three algorithms and five tiles for each FFN
up/gate forward and weight-gradient shape: 30 candidates total. Every candidate
matched the ATen BF16 output exactly. All forward candidates lost. The selected
weight-gradient candidate used persistent TMA with tile `(256,128,32,8,3)`:
median candidate time 1.811495 ms and median paired throughput gain 3.39%.
Its weakest of three paired trials gained 1.30%. The first trial's two controls
drifted from 1.589741 to 1.765839 ms, so this short screen alone is insufficient
to select a production configuration. The complete generated sweep and candidate
recipe are in `work/benchmark-matmul-tiles-20260912/`.

To reproduce the ordinary FFN forward/weight-gradient tuning, reserve an idle
GPU first. From the repository root, the following command uses physical GPU 0
and keeps outputs and compiler caches inside the workspace. The output directory
must be a new `work/benchmark-*` directory; existing directories are rejected.

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
CUDA_VISIBLE_DEVICES=0 .server-venv/bin/python -B ops/tune_matmul.py \
  --kind mm --operations fwd dw --algorithms tiled persistent persistent_tma \
  --rows 8192 --seed 1337 --warmup 10 --burnin-seconds 1 \
  --repeats 30 --trials 3 --minimum-gain 0.01 \
  --maximum-relative-rms 0.003 \
  --output-dir work/benchmark-matmul-tuning-reproduction
```

To recheck the measured fused tile with the same dimensions, seed, and operand
scale, use the fused mode on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=1 .server-venv/bin/python -B ops/tune_matmul.py \
  --kind swiglu --operations fwd --algorithms tiled \
  --tile 128,128,32,8,3 --rows 8192 --seed 563 --input-scale 0.1 \
  --warmup 400 --burnin-seconds 1 --repeats 40 --trials 15 \
  --minimum-gain 0.01 --maximum-relative-rms 0.003 \
  --output-dir work/benchmark-matmul-swiglu-reproduction
```

This uses the tuner's bracketed control/candidate/control protocol. The historical
15-batch report used alternating pairs, so compare the resulting measurements
as independent checks rather than expecting identical timing samples. Fused
mode compiles the complete matrix-plus-SwiGLU reference and validates both packed
projection and activation separately. It supports only `--operations fwd` with
`tiled` or `persistent`. Omit `--tile` to use that mode's bounded tile list, or
repeat it to specify several candidates. `--input-scale` defaults to 1.0 for both
operands; the recorded epilogue comparisons used 0.1.

`--list-cases` prints the requested dimensions and strides without using CUDA.
The tuner generates deterministic BF16 random operands from the supplied seed;
it does not open a model, checkpoint, or token database. In this disposable process
it disables BF16 reduced-precision reduction and TF32 for the ATen reference.
Every timed candidate has preceding and following warmed controls. Eligibility
requires finite results, the configured relative RMS tolerance, and at least
the configured gain in **every** trial. Compilation failures remain in
`results.json`. The best eligible entry for each tested operation is written to
`candidate-policy.json`; an empty recipe means no candidate passed selection.
Results include source hashes, software/device metadata, input signatures,
timing triples, per-output numerical error, and GPU telemetry. Generated reports and
candidate recipes remain ignored under `work/`; a recipe selected after complete
validation should be copied into tracked `configs/` with its provenance.

Use the same checkpoint, input offsets, microbatch, loss/gradient modes, and
warmup/measurement counts to compare a candidate with a control. For example,
after reserving both workstation GPUs, this disposable command tests the generated
recipe against the retained full checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 .server-venv/bin/torchrun --standalone --nproc-per-node=2 \
  ops/benchmark_training.py --checkpoint work/run/latest.pt \
  --data work/data/train.bin --micro-batch 4 --seq-len 2048 \
  --global-tokens-per-update 131072 --no-activation-checkpointing \
  --compile-cores --compile-mode default --loss-mode compiled-full \
  --bf16-residuals --gradient-mode compiled-autograd-overlap \
  --optimizer-state resume --warmup-updates 8 --measured-updates 20 \
  --matmul-plan work/benchmark-matmul-tuning-reproduction/candidate-policy.json \
  --output-dir work/benchmark-matmul-training-candidate
```

Run a matching control with `--matmul-plan` omitted and a different new output
directory. Neither benchmark advances the production checkpoint. Run numerical
backward/optimizer and checkpoint/resume tests before selecting a training recipe.
The production runner accepts the same `--matmul-plan PATH` option alongside its
existing resume arguments; it is not an instruction to start a new run or change
the established global token batch. External core-weight cast caching and the
`reduce-overhead` CUDA-graph mode are explicitly incompatible with this adapter.
Standalone `--activation-mode swiglu|rmsnorm|all` kernels require eager gradient
execution; their transparent backward launches have the same compiled-autograd
limitation described above. Both CLIs reject that unsupported combination.
Use `--activation-mode baseline` with compiled autograd; matrix-plan fused
epilogues still work through their separate opaque boundaries.
Combining a matrix plan and compiled autograd also requires activation
checkpointing off. The installed compiler cannot trace the saved-tensor unpack
hook when the runner wraps an already compiled core in an eager checkpoint.
Recomputation captured inside AOT passes the small kernel tests, but does not
establish support for that different runner nesting.

Plan loading happens once, before execution. It computes a canonical recipe
SHA256 and implementation SHA256 values for `matmul_execution.py` and
`matmul_kernels.py`; a fused plan also includes `matmul_epilogue.py` and
`activation_execution.py`. The runner records this metadata in settings and the
checkpoint training signature, and the benchmark records it with its source
provenance. A changed recipe or implementation therefore changes the execution
signature. Legacy checkpoints normalize to no matrix plan. Resuming with an
execution change requires the existing explicit `--reconfigure` mechanism,
which still protects the data, context, global token batch, and optimizer schedule.
Reformatting JSON without changing its parsed canonical content leaves the recipe
identity unchanged; changing a description or an entry changes it.

The complete-model comparisons resume the same step-40,081 checkpoint into
disposable GPU model/AdamW copies. Each uses eight warmup updates and 20 timed
updates, BF16 residuals, compiled full loss, the same 131,072-token global batch,
and compiled autograd with delayed communication waits. They exclude compilation,
evaluation and checkpoint pauses. The generated rank logs and summaries are
under `work/benchmark-opt-20260912/`.

| Configuration | Timed tokens/s | Change from control | Reserved GiB/rank |
| --- | ---: | ---: | ---: |
| Communication-overlap control (`matmul-control`) | 31,022.34 | — | 66.98 |
| Persistent TMA dW only (`matmul-dw`) | 31,063.57 | +0.13% | 66.99 |
| Fused forward + TMA dW, initial boundary (`matmul-combined`) | 31,002.83 | -0.06% | 66.54 |
| Fused forward + TMA dW, shared weight buffer (`matmul-combined-native`) | 31,156.20 | +0.43% | 66.98 |

These first comparisons are effectively tied. Over all 28 updates, the initial
combined candidate's maximum loss difference from the fresh control was
0.00008535, and the maximum relative gradient-norm difference was 0.124%.
Small BF16 reduction/fusion differences do not imply training progress.

After the weight-layout correction, the candidate averaged 4.20693 seconds per
update versus 4.22508 for the control. Its maximum loss difference was 0.00010061
and its maximum relative gradient-norm difference was below 0.114%; all observed losses
and norms were finite. This single comparison shows a small +0.43% gain, which
has not been confirmed across repeated timing windows. The matrix recipe remains
opt-in. The resume launcher retains the repeatedly verified communication-overlap
configuration from [OPTIMIZATION_20260912.md](OPTIMIZATION_20260912.md), which
measured a 1.53% improvement over eager gradients in the repeated comparison.
These percentages come from separate experiments and must not be added into a
claimed combined speedup. The generated numerical/timing comparison is
`work/validation/rtx-opt-20260912/matmul/full-model-comparison.json`.

Generated-code inspection verified the selected forward and dW routes, with
224 uses each per rank/update inferred from the compiled graphs and loop counts.
It also found an avoidable cost in the initial fused boundary: passing `W.T`
into an opaque operation made Inductor store two identical BF16 weight buffers,
while the control shared one between forward and backward. The final adapter
passes native-layout BF16 `W` and forms its transpose inside the launch. The
actual-size compiled core now shares one allocation and one cast with backward,
removing 58,720,256 extra bytes of stores per core call, or 13,153,337,344 bytes
per rank/update. This is generated-code evidence, not a measured bandwidth
counter or a guaranteed wall-time saving. The TMA dW route already reuses its
gradient buffer and required no corresponding change.

The tracked
[`matmul_rtx6000_bf16_experimental.json`](../configs/matmul_rtx6000_bf16_experimental.json)
contains both exact candidate shapes and tiles. The final native-weight boundary
comparison uses this recipe directly. Static dispatch and buffer evidence is
recorded in `work/validation/rtx-opt-20260912/matmul/compiled-dispatch-evidence.json`
and `dw-dispatch-evidence.json`. Production remains paused; these experiments
leave the step-40,081 checkpoint and archived weights unchanged.

The integration validation passed 208 CPU tests, with 83 CUDA fixtures skipped
and three existing compiler warnings. Separate reserved-GPU runs passed the
33-test epilogue suite and 35 adapter tests, including tiled, persistent, and
TMA forward/dX/dW, FP32 gradient accumulation and AdamW, ordinary AOT, and compiled
autograd. The adapter suite includes two alignment regressions: shifted inputs
use exact ATen fallback in eager mode and match a compiled ATen control when
compiled. Inductor can realign an input before the opaque boundary, making
TMA valid; the original caller's offset need not survive compilation.

A two-GPU runner test combined the fused persistent epilogue, persistent TMA
weight gradients, and delayed communication waits. Continuous execution matched
asynchronous save/stop/resume exactly for model weights, full FP32 AdamW state,
both ranks' RNG state, losses, and token cursors. Its final partial update also
covered a rank with entirely masked targets. The final implementation passed in
73.63 seconds; its log is
`work/validation/rtx-opt-20260912/pytest-matmul-resume-final.log`.
The CPU and alignment regression logs are `pytest-matmul-final-cpu.log` and
`pytest-offset-guard.log` in the same validation directory. These checks prove
continuation consistency within the selected execution mode, not bitwise
equivalence to a different BF16 reduction/fusion strategy.
The final CLI compatibility/signature checks passed 103 tests with eight GPU
fixtures skipped (`pytest-matmul-cli-final.log`). The complete package manifest
was verified after updating the runner's SHA256/byte count. After the final
benchmark, the production checkpoint's SHA256 still matched the paused record.
