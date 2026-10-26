# BF16 backward-layout integration experiment

The paired attention-backward layout prototype is integrated into the compiled
training graph. Repeated complete-model measurements improve update throughput
by 0.637%, so `train-local-resume.sh` selects `--backward-layout paired` for the
next resumption. The CLI default remains `baseline`. The conservative 128 MiB
communication preset remains selected; the separate producer-scheduling trial
is available but disabled. Production training remains paused at step 40,081.

## Execution contract

Both `run_40b.py` and `ops/benchmark_training.py` accept
`--backward-layout baseline|inspect|paired`. `baseline` preserves existing
execution. `inspect` checks the graph without replacing it. `paired` replaces
only the recognized backward region and requires actual kernel dispatches.

The initial supported envelope is microbatch four, sequence length 2,048,
32 heads of width 64, chunk length 128, parallel memory, and zero dropout.
Use BF16 residuals, compiled cores in default mode, baseline activation and
normalization implementations, no activation checkpointing, and
`compiled-autograd-overlap`. Separate matmul recipes are excluded from this
initial experiment. Master parameters, gradient accumulation, collectives, and
AdamW state retain their existing FP32 representation; model compute uses BF16.

The graph pass matches normalized ATen operations, constants, tensor metadata,
branch relationships, and external consumers. Generated kernel names are not
match keys. Separate local and memory Q gradients and the K intermediate
rounding boundary are preserved. Unsupported regions retain the original graph;
an opted-in training or benchmark update fails if expected dispatches are absent.

Checkpoint execution signatures and benchmark metadata contain the implementation
identity and source hashes. A changed identity requires `--reconfigure` on
resumption. Compiler cache identity also includes the pass implementation.
Runtime dispatch counters establish execution even when a compiler cache hit
bypasses the graph matcher. Benchmarks require exactly one replacement per
logical layer per accumulation microbatch in each measured update.

The opaque operator follows PyTorch's [functional custom-operator contract](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html):
it mutates no input, returns a fresh output, and registers fake output metadata.
The hook and arithmetic specialization are tied to the installed PyTorch
version. This is a backward graph transformation; it does not replace the
model's forward autograd ownership.

## Validation and timing protocol

The isolated prototype's 31% reduction applies only to its two-kernel region.
Roughly 34 ms saved per training update, or 0.8% overall, was the pre-integration
estimate. The measured complete-model result appears below.

Validation compares the compiled baseline and candidate at production attention
geometry using a smaller number of layers. It covers forward outputs, every
parameter gradient after initial and accumulated backwards, optimizer updates,
and two-rank continuation through a final partly masked update. Continuation
within an execution mode is a separate check from numerical agreement between
different execution modes.

Complete-model timings use the retained checkpoint and data cursor, microbatch
four per rank, eight accumulation microbatches, and 131,072 global tokens per
update. Both GPUs retain their 300 W power limits. Full resumed AdamW state and
an 88 GiB allocator cap remain enabled. No production checkpoint is written.

Repeated runs use fresh processes and interleave baseline and candidate order.
Warmup and measured counts are matched. GPU telemetry is sampled outside the
timed Python region. Compare total tokens divided by total update time, timing
drift, matched losses/gradient norms, and allocated/reserved VRAM. Profiles are
separate runs; their instrumented throughput is excluded from timing results.

## Reproduction

Keep production training paused and reserve both GPUs before running:

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,ENV \
.server-venv/bin/torchrun --standalone --nproc-per-node=2 \
  ops/benchmark_training.py \
  --checkpoint work/run/latest.pt --data work/data/train.bin \
  --micro-batch 4 --seq-len 2048 --global-tokens-per-update 131072 \
  --no-activation-checkpointing --autocast-cache \
  --compile-cores --compile-mode default --loss-mode compiled-full \
  --bf16-residuals --gradient-mode compiled-autograd-overlap \
  --ddp-bucket-cap-mb 128 --optimizer-state resume --vram-limit-gib 88 \
  --warmup-updates 8 --measured-updates 40 --backward-layout paired \
  --output-dir work/benchmark-backward-layout-reproduction/paired-1
```

Use `--backward-layout baseline` for the control and a fresh output directory for
every process. For dispatch and overlap profiles, use `--profile` with one
measured update after the same warmup. Existing comparison and communication
profile tools consume the saved JSON reports without opening model weights.

The underlying kernel arithmetic and original isolated measurements are in
[BACKWARD_LAYOUT_EXPERIMENT.md](BACKWARD_LAYOUT_EXPERIMENT.md).

## Training validation results

One production-shaped core matched the compiled baseline exactly for forward
output, input gradients, and all nine core parameter gradients. The opaque
operator's schema-purity and fake-metadata checks passed on CUDA.

The two-rank training fixture uses production attention/FFN dimensions, one
logical layer, three accumulation microbatches, two full updates, and a final
nine-token update with an entirely masked rank. It compares every parameter
gradient after the first, middle, and final synchronization backward, then
checks each parameter and Adam-state tensor after every optimizer update.

An initial 512-vocabulary fixture with repeated input tokens passed numerical
comparison but failed byte-exact continuation: 33 embedding-weight elements
differed by at most `3.7253e-9`. Differences in optimizer state were also confined
to the embedding. Other model weights, RNG states, token counters, and execution
identity matched. Its report is retained as a diagnostic of the existing
embedding accumulation's nondeterminism, not counted as an exact-resume pass.

The follow-up `--unique-inputs` fixture uses vocabulary 8,192 and a deterministic
permutation so each embedding row appears once per backward. It retains the
same attention geometry and precision without relaxing comparison thresholds.
Both ranks passed exact continuation for model, full optimizer, RNG, and cursor
state. All Adam parameters and moments also matched the baseline byte for byte
at every update. The largest gradient relative L2 difference was `1.22e-7`,
with maximum absolute difference `2.73e-12`, confined to the small
`memory_mix_logit` reduction. Both ranks agreed exactly on final model and Adam
state within each run. Every expected layout dispatch occurred.

Reproduce the strict continuation fixture with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 .server-venv/bin/python -B \
  ops/validate_backward_layout_training.py --world-size 2 --unique-inputs \
  --output-dir work/validation/backward-layout-training-reproduction
```

The fixture initializes deterministic synthetic weights, tokens, and nonzero
FP32 Adam moments. This checks integration and continuation at production
attention geometry; it is not a byte-exact continuation claim for the complete
28-layer model on repeated-token corpus data. The reports are
`work/validation/backward-layout-training-20260912/` and
`work/validation/backward-layout-training-unique-20260912/`.

A two-core follow-up covers the compiler's sharing of an identical zero constant
across layers. The matcher retains that strictly matched constant while another
region still uses it, then removes it after its final replacement. Shared
computed arithmetic still causes rejection. The installed compiler's actual
common-subexpression pass is exercised by a focused structural test.

The final two-core, two-GPU fixture also passed exact model/optimizer/RNG/cursor
continuation. It required two dispatches per backward and observed every one.
All updated parameters and Adam states matched baseline byte for byte; maximum
gradient errors were `3.64e-12` absolute and `1.27e-7` relative L2. Add `--cores 2`
to the command above to reproduce it. Results are in
`work/validation/backward-layout-training-two-core-20260912/results.json`.
See [BACKWARD_LAYOUT_TRAINING_VALIDATION.md](BACKWARD_LAYOUT_TRAINING_VALIDATION.md)
for the full validation scope and source identities.

The separate producer-scheduling fixture also passed the existing two-GPU
asynchronous-save/stop/resume test, including full model/optimizer/RNG equality
and a final masked rank. Its log is
`work/validation/backward-layout-training-producer-resume-20260912.log`.

## Complete-model timing results

Four fresh-process runs used the order control, paired, paired, control, each
with eight warmup and 40 measured updates. All started from step 40,081 and
token cursor 5,253,496,832 with the same retained checkpoint, data, optimizer,
and source identities. Profiling and evaluation/checkpoint pauses are
excluded; these numbers describe training-update throughput.

| Run | Layout | Tokens/second |
| --- | --- | ---: |
| `control-1` | Baseline | 31,010.48 |
| `paired-2` | Paired | 31,197.36 |
| `paired-3` | Paired | 31,200.70 |
| `control-2` | Baseline | 30,992.67 |

Aggregating total tokens over total measured time gives **31,001.57 to
31,199.03 tokens/second**, a **0.637%** gain. Mean update time falls from
4.22791 to 4.20116 seconds, saving **26.76 ms per update**. The two controls
differ by 0.0575%; the two candidates differ by 0.011%. Both candidates beat
both controls. This is a small, repeatable improvement on this hardware and
preset; the isolated region's 31% gain does not apply to whole-model training.

Peak live allocator usage is unchanged at **65.7119 GiB per GPU**. Peak reserved
allocator memory falls from **66.9785 to 66.4785 GiB per GPU**. The 0.5 GiB
reduction is reservation, not a reduction in peak live tensors. Power limits
remain 300 W and measured-window temperatures and clock rates are comparable.

Every measured candidate update dispatches exactly 224 paired operations per
rank: 28 logical layers times eight accumulation microbatches. The second
candidate run loads compiler caches and still records every runtime dispatch,
despite the graph matcher itself not being invoked on those cache hits.

The first update's loss and gradient norm are exactly equal across all four
runs. Later trajectories are not byte-identical, including the repeated control.
Against the first control, the second control's maximum loss difference is
`1.28e-4` and relative gradient-norm difference is 0.219%. Candidate maxima are
`1.48e-4` / 0.179% and `1.35e-4` / 0.307%. These small scalar differences are
consistent with nondeterministic training reductions; they do not establish
byte equality of the complete model's gradients. The fixture checks above
provide the separate tensor-level evidence.

Run artifacts are under `work/benchmark-backward-layout-integration-20260912/`.
The comparison and telemetry summary is
`work/validation/backward-layout-integration/abba-summary.json`.
The manually stopped `paired-1` compilation probe has no completed update and
is excluded from all timing results.

## Profile verification

Separate baseline and paired runs each use eight warmup updates and one profiled
update. Each candidate rank records 224 custom-operator calls and 448 actual
`_paired_kernel` launches: 224 rotary and 224 value/gate/U kernels. The original
rotary and projection-packing launch signatures are absent. Their combined
region time falls from **114.95 to 78.96 ms** on rank zero, and from **114.65 to
78.91 ms** on rank one, about 31% less CUDA time for this region. This saves
roughly 36 ms of kernel time; the repeated unprofiled whole-update improvement
is the smaller 26.76 ms reported above.

The total CUDA kernel count stays at 17,477 per rank, and every copy/fill count
and byte total is unchanged. There are no added layout-materialization kernels.
The cached synchronized backward graph retains 45 identical generated Triton
bodies and the same 225 MM, 224 BMM, and one ADDMM calls; only the two recognized
layout bodies are removed. The replacement consumes native buffers directly.
These checks are recorded in
`work/validation/backward-layout-integration/profile-kernel-copy-comparison.json`
and `generated-sync-graph-comparison.json` in the same directory.

The complete captured update retains all 59 collectives and
8,020,445,708 logical input bytes per rank, with identical collective signatures.
The following values are local-device timeline unions, not sums across GPUs:

| Profile metric (ms/update) | Baseline rank 0 | Paired rank 0 | Baseline rank 1 | Paired rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Noncommunication CUDA activity | 3,957.68 | 3,935.16 | 3,834.21 | 3,796.91 |
| Compute/communication overlap | 192.63 | 192.56 | 191.92 | 191.57 |
| Communication with no overlapping compute | 149.87 | 154.23 | 281.09 | 290.27 |

Overlap is preserved. The exposed communication duration varies slightly upward
in this single profile pair; the measured layout gain is not a communication
speedup. Removing computation can also reduce work available to hide communication.
Instrumented profile durations are excluded from the repeated timing result.
The timeline accounting is in
`work/validation/backward-layout-integration/profile-communication.json`.

## Separate producer scheduling experiment

`--overlap-producer-schedule` is an opt-in communication experiment. Its
two separate runs use `--backward-layout baseline` and 128 MiB buckets, with
the same checkpoint, cursor, geometry, warmup, timing, and telemetry protocol
above. `producer-1` reached **31,133.27 tokens/second** (+0.425% against the
aggregate control); `producer-2` reached **31,090.54** (+0.287%). The repeat is
0.137% slower than the first screen, and its gain falls just below the approximate
0.3% threshold used to justify testing a combined preset. Peak allocated and
reserved VRAM match the control's 65.7119 and 66.9785 GiB per GPU.

Both runs show a small positive average difference, but the evidence is weaker
than the layout's repeated gain. The producer option remains available for
experimentation and is **not selected** in the production launcher. No combined
layout/producer run was started; additive gains have not been established.
This is a conservative selection decision, not proof that the scheduler cannot
help. The run summaries remain under the same benchmark directory as the ABBA
measurements.

The installed scheduler already moves ready collective launches earlier through
`simple_overlap`. This experiment additionally lets
`reorder_communication_preserving_peak_memory` move a collective with its
packing/casting producers, followed by the existing `sink_waits_iterative` pass.
Without runtime estimates, producer grouping stops at GEMM-producing nodes.
It therefore targets earlier bucket readiness rather than promising to reorder
all independent backward GEMMs.

The installed `comms.py` docstring names a `sink_comms_and_waits` prerequisite
that has no callable implementation in this version. The installed `config.py`
instead recommends producer reordering and iterative wait sinking; ordinary
peak-memory scheduling runs before these passes. The initial experiment uses one
producer pass followed by one wait sink, retaining the existing final wait
scheduling behavior.

Both passes explicitly prohibit changing collective launch order and allow
zero increase in their estimated peak memory. Producer groups cannot include
another collective. These settings override the producer pass's installed
unsafe-order and 20% memory defaults. The flag requires
`compiled-autograd-overlap` and initially rejects wait swapping, runtime
estimation, and a nonzero memory allowance. Its immutable execution setting is
recorded in benchmark metadata and checkpoint signatures; changing it on resume
requires `--reconfigure`. The timing and continuation checks above measure this
option separately from the paired layout.

## Final checks and resume state

The final CPU suite passed **356 tests**, with 24 CUDA tests skipped in that
CPU-only invocation and three existing Dynamo cache-wrapper warnings. Separate
CUDA operator checks, production-shaped one-/two-core gradient/optimizer and
exact-continuation fixtures, and the producer scheduling async-resume test are
documented above. The final CPU log is
`work/validation/backward-layout-integration/pytest-final-cpu.log`.

All ten package-manifest entries validate. The launcher passes `bash -n`.
The retained full checkpoint at `work/run/latest.pt` passes the before/after
SHA256 check, retaining
`9745d93345dff7c389ac6fa65948dc78ec021c90f56b40e9caed8ddcd84e8c54`.
The verification is saved in `checkpoint-after.log` beside the CPU log.
It remains at step **40,081**, **5,253,496,832 tokens**, with recorded validation
loss **2.4545078724622726**. Production and preparation jobs remain stopped,
both GPUs are idle, and their power limits remain 300 W. Archived weights are
untouched. The selected launcher will use the paired layout when training is
next resumed; this experiment does not resume production.
