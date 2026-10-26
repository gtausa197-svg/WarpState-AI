# Communication scheduling exploration — September 12, 2026

These experiments use disposable copies of the step-40,081 full checkpoint.
Production training remains paused. The model, corpus cursor, 2,048-token
context, 131,072-token update, BF16 computation, FP32 gradient communication,
and full FP32 AdamW state are held fixed. The production resume preset retains
the previously selected conservative communication scheduler.

## Explicit scheduler controls

`run_40b.py` and `ops/benchmark_training.py` expose three opt-in controls for
`--gradient-mode compiled-autograd-overlap`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--overlap-swap-collectives` | Off | Move a wait and its dependent copies past later collective launches while preserving collective launch order. |
| `--overlap-runtime-estimates` | Off | Use measured GEMM estimates, aligned across distributed ranks, to guide wait placement. |
| `--overlap-memory-budget FRACTION` | `0.0` | Permit a fractional increase in the scheduler's estimated peak memory, bounded to 0–0.25. This does not increase the CUDA allocator cap. |

The helper always disables unsafe collective reordering. Settings are immutable,
validated before compiler configuration, and recorded in execution metadata.
The runner records all three controls in checkpoint signatures and requires
explicit `--reconfigure` for changes. Legacy signatures normalize to the same
False/False/0 defaults, preserving existing resume behavior.

Runtime estimation uses installed experimental PyTorch compiler APIs. It adds
GEMM benchmarking during compilation and distributed estimate alignment; this
is separate from the timed training updates. Configurations use fresh processes
because compiler/reducer settings are process-wide. Both ranks must compile the
same collective-bearing graphs.

The installed FX cache explicitly includes `config_comms` in its key, and the
AOT cache uses the same payload. Actual cache-key checks distinguish these
options; they cannot silently reuse a differently configured scheduler merely
because the forward graph is unchanged. The runtime-estimate screen also
records an additional pre-update AllGather consistent with rank alignment.
`TORCH_LOGS=overlap` alone does not print the detailed swap decisions: the
installed pass additionally requires `reorder_sink_verbose_logging=True`.

## Complete-model screens

All rows use the same source versions and checkpoint. Changes are relative to
the fresh 128 MiB conservative control, not the earlier eager-gradient baseline.

| Configuration / artifact directory | Tokens/s | Change | Reserved GiB/rank |
| --- | ---: | ---: | ---: |
| Conservative, 128 MiB (`control-128`) | 31,174.13 | — | 66.98 |
| Waits past launches, 128 MiB (`swap-128`) | 31,201.81 | +0.09% | 66.98 |
| Waits past launches, 64 MiB (`swap-64`) | 30,824.54 | -1.12% | 66.98 |
| Waits past launches, 256 MiB (`swap-256`) | 30,385.00 | -2.53% | 66.98 |
| Waits past launches + runtime estimates, 128 MiB (`swap-runtime-128`) | 31,117.39 | -0.18% | 66.46 / 66.96 |
| Same conservative control repeated (`control-repeat`) | 31,070.03 | -0.33% | 66.98 |

The wait-only and runtime-estimate results are effectively tied with control;
neither establishes a throughput gain. The runtime-estimate choice reduces
allocator reservations by approximately 0.5 GiB on rank zero; rank one is
essentially unchanged. Peak allocated memory remains 65.71188 GiB on both
ranks, so this does not demonstrate a smaller model working set.
Smaller and larger buckets both lose throughput here. None of these results
justifies changing the production preset.

Repeating the unchanged control moves throughput by -0.33%, larger than either
apparent wait-only or runtime-estimate effect. Its maximum loss difference is
0.00007796 and maximum relative gradient-norm difference is 0.0836%. Thus even
the unchanged full model does not give a bitwise scalar trajectory in these
screens. The other configurations' norm differences can be larger; a pair of
scalar traces is not proof of per-parameter equivalence or long-run quality.

Control and wait-only runs both slow by about 1.14% from their first five to
last five measured updates. Telemetry shows similar temperatures and clocks
across the control, wait-only, and 64 MiB runs; their mean GPU temperatures are
approximately 52 C and 55 C by rank. One-second clock samples mix GEMM and
non-GEMM phases and are not measurements of sustained GEMM clock alone.

`ops/compare_communication_benchmarks.py` validates checkpoint/data identities,
batch geometry, update cursors, recorded precision, and rank consistency before
comparing complete unprofiled windows. Its report preserves paired loss/norm
differences, timing variation, per-rank memory peaks, and source/input hashes.
These screens' largest loss difference is 0.00018311 and largest relative
gradient-norm difference is 0.366%; those are observed differences, not general
acceptance thresholds or evidence of improved model quality.

Generated conservative and wait-only synchronization graphs have identical
Triton arithmetic bodies, GEMM/attention order, collective launch order, and
statement multisets; only wait/copy placement changes. Existing embedding
backward uses relaxed FP32 atomic additions to repeated-token rows. Small
cross-run differences are therefore consistent with existing nondeterministic
reductions amplified by subsequent BF16 rounding, although this observation
alone does not prove their cause. Both ranks agree on gradient norms in each
run. Same-mode repetition and explicit continuation tests are separate checks.

The paired report is
`work/validation/communication-exploration/comparison-final.json`. It can be
rebuilt from the saved benchmark summaries and rank logs with:

```bash
.server-venv/bin/python ops/compare_communication_benchmarks.py \
  --control work/benchmark-communication-20260912/control-128 \
  --candidate work/benchmark-communication-20260912/swap-128 \
    work/benchmark-communication-20260912/swap-64 \
    work/benchmark-communication-20260912/swap-256 \
    work/benchmark-communication-20260912/swap-runtime-128 \
    work/benchmark-communication-20260912/control-repeat \
  --output work/validation/communication-exploration/comparison-rebuilt.json
```

## Method

### Fresh trace comparison

Separate profiles use eight warmup updates and one instrumented update, with
identical checkpoint/corpus offsets. Their throughput is not included in the
unprofiled comparison above.

| CUDA interval budget | Conservative rank 0 / rank 1 | Wait swap rank 0 / rank 1 |
| --- | ---: | ---: |
| NCCL overlapping other CUDA work | 191.805 / 191.069 ms | 193.185 / 191.983 ms |
| NCCL with no other CUDA work | 153.136 / 266.595 ms | 152.460 / 274.337 ms |

Profiled update wall time is 4.17505 seconds for control and 4.18611 for swap.
Both contain 59 collective calls including scalar metrics, with identical CPU
operator/shape/type submission fingerprints. The structural wait movement does
not create a material increase in observed overlap on this model.

Rank one's larger exposed interval includes waiting for its peer: its
noncommunication CUDA union is approximately 3.833 seconds, versus 3.946 on
rank zero in the control. Exposed intervals must not be added across ranks or
treated as wholly recoverable communication overhead.

The report is `work/validation/communication-exploration/profile-comparison.json`.
Rebuild it with `ops/analyze_communication_profile.py`, the four
`profile-{control,swap}/profile-rank-{0,1}-trace.json` files under the benchmark
directory, and a fresh `--output` JSON path under `work/`.

### Next scheduling work

The installed compiler already enables `simple_overlap` after the iterative
wait pass. Enabling it again is not a new optimization. Its adjacent launch
moves stop at a gradient producer, an earlier collective, or estimated memory
growth. The next useful experiment would move the producer work with the
collective, making a bucket ready sooner rather than only delaying its wait.

`reorder_communication_preserving_peak_memory` can group packing/casting
producers with collectives. A future screen must explicitly set
`reorder_iterative_unsafe_collectives_reorder=False` and
`reorder_iterative_peak_memory_budget=0.0`; the installed defaults are True and
0.2. Its non-estimated mode stops at GEMM producers; aligned runtime estimates
allow considering larger producer groups. Another option, `raise_comms`,
prioritizes the next collective's producer ancestors but has no peak-memory
guard. Neither pass was implemented or benchmarked by this exploration.

The nonzero memory-allowance option is exercised by the continuation fixture;
nonzero allowances were not included in the complete-model throughput sweep.
Bucket readiness still depends on every member gradient, and the final
reductions must complete before clipping and AdamW. These are real limits on
how much overlap any scheduler can obtain.

### Timing procedure

The unprofiled comparisons use eight complete warmup updates followed by twenty
timed updates, with microbatch four per GPU and eight accumulation microbatches.
Each starts at the same checkpoint and token offset. The benchmark neither
modifies nor writes production checkpoints. Both GPUs retain their 300 W limits
and an 88 GiB PyTorch allocator cap. Matrix recipes, external weight caching,
and standalone activation replacements are off.

NVIDIA-SMI samples clocks, power, temperature, and memory once per second outside
the timed Python regions. Initial diagnostic comparisons use identical NCCL
logging settings. A single sub-percent timing difference is insufficient to
select a default, particularly when sustained clock or thermal drift is similar
in size. Loss and gradient-norm sequences must be compared at identical update
indices, including warmups.

Profiles are separate diagnostics. CUDA intervals are unioned per rank/device;
overlapping streams and annotation ranges are not summed as elapsed time.
Communication-only time is the union of NCCL intervals outside other CUDA
activity. It is an exposed-work budget, not a guarantee that every millisecond
can be hidden. Ranks' times must not be added together.

`ops/analyze_communication_profile.py` produces reproducible overlap reports
from the benchmark's Chrome traces. Collective payload metadata describes
logical input bytes, not measured physical link traffic. Its CPU collective
sequence fingerprint covers operation names and shapes; it is supporting
evidence and does not establish tensor identity or numerical correctness.

## Transport evidence

The fresh control's NCCL initialization confirms direct `P2P/CUMEM` transport
in both directions and two collective channels. Its tuning messages identify
large all-reduces as RING/SIMPLE. Network/socket initialization messages are not
evidence that gradient traffic uses sockets; kernel symbol suffixes alone also
do not establish the selected protocol. The GPUs have a SYS path across CPU
NUMA nodes, so direct peer access still traverses the system interconnect.

The benchmark records NCCL version, relevant environment overrides, and CPU
affinity. Raw logs and telemetry are under
`work/benchmark-communication-20260912/` and remain ignored.

## Reproduction

Reserve both GPUs with production training paused, then use a fresh output
directory for each process:

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
  --optimizer-state resume --vram-limit-gib 88 \
  --warmup-updates 8 --measured-updates 20 \
  --output-dir work/benchmark-communication-reproduction/control-128
```

Append `--overlap-swap-collectives` for wait movement. Vary
`--ddp-bucket-cap-mb 64|128|256` in separate runs. The Python reducer groups
whole parameter gradients; a nominal bucket capacity does not split a larger
individual parameter. Runtime estimates and an estimated memory allowance
must be identified separately in results.

The backward memory-traffic prototype is a separate experiment; see
[BACKWARD_LAYOUT_EXPERIMENT.md](BACKWARD_LAYOUT_EXPERIMENT.md). Isolated kernel
speedups cannot be added to complete-model throughput results.

## Validation and retained state

The final CPU suite passed **214 tests** with 17 CUDA checks skipped in that
CPU-only invocation. It covers option validation, legacy signatures, explicit
execution reconfiguration, gradient accumulation, rank-aligned estimates,
trace interval accounting, paired-run consistency, and backward-layout math.
The separate backward-layout CUDA comparisons are documented in its report.

Both selected two-GPU continuation fixtures passed in 136.90 seconds:
wait movement with a 2% estimated memory allowance and 64 MiB buckets, and
wait movement with aligned runtime estimates and 128 MiB buckets. Each compares
continuous training with an asynchronous save/stop/resume, including a final
update with an entirely masked rank. Model tensors, complete FP32 optimizer
state, and both ranks' RNG state match bitwise within the same execution mode.
These small-model continuation checks are distinct from the full-model
cross-configuration numerical differences above.

Validation logs and reports are under
`work/validation/communication-exploration/`, including `pytest-final-cpu.log`,
`pytest-resume.log`, `comparison-final.json`, and `profile-comparison.json`.
The retained step-40,081 checkpoint passes its before/after SHA256 check
(`checkpoint-before.sha256`, `checkpoint-after.log`). All package manifest
entries also verify. Production remains paused, and no new scheduler setting
or backward-layout adapter is selected in the production preset.
