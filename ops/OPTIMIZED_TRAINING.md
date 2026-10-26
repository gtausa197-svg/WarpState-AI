# Training with both GPUs available

**Currently paused at the user's request.** Training and benchmarks are
stopped so another session can resume VORTEX. The latest full checkpoint is
step 448 / 58,720,256 tokens. Start commands below are for a future
user-requested resume after checking GPU availability.

The selected configuration measured **29,490 training tokens/second** on two
NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, with approximately **68.8 GiB reserved
and 25 GiB free per GPU**. The measurements below were taken on September 9,
2026, after VORTEX and the previous WarpState training process had stopped.

See [PERFORMANCE.md](PERFORMANCE.md) for the subsequent bottleneck analysis,
GPU profiling method, and the compute requirements of a 100K-token/second
target.

## Selected configuration

[`train-fast.sh`](train-fast.sh) launches two workers through
[`launch_training.py`](launch_training.py). Each worker holds the complete
2,005,111,424-parameter model and AdamW state; DDP synchronizes gradients.

| Setting | Value |
| --- | --- |
| GPUs / workers | 2, one worker per GPU |
| Sequence length | 2,048 tokens |
| Microbatch per GPU | 4 sequences |
| Gradient accumulation | 8 microbatches per GPU |
| Global tokens per optimizer update | **2 × 4 × 2,048 × 8 = 131,072** |
| Parameters, gradients, AdamW moments | FP32 on each GPU |
| Matrix computation | BF16 autocast, weight cache enabled |
| Compilation | Each core's training forward, full graph, fixed shapes |
| Core activation checkpointing | Disabled |
| Loss projection chunk | 1,024 tokens; loss projection recomputation remains enabled |
| DDP gradient buckets | 128 MiB target, gradients use bucket views |
| PyTorch allocator cap | 88 GiB per GPU |
| Additional free memory required at startup | 3 GiB per GPU |
| Validation | 8 batches of 1 sequence, initially and every 100 updates |
| Checkpoint | Every 50 updates, on graceful stop, and at completion |

The global batch, sequence length, data cursor, optimizer state, and learning
rate schedule are preserved when moving from the previous shared-GPU run.
The launcher's `--reconfigure` allows execution changes while the runner
checks those invariants and records the transition in `execution_history`.
Compilation and changed batch grouping can change floating-point rounding;
this is not a promise of bitwise equivalence across configurations.

The allocator cap excludes CUDA context and library allocations. The startup
headroom check does not reserve memory for another process. This configuration
assumes VORTEX remains paused and both GPUs are available.

## Measured configuration grid

All rows use sequence length 2,048, 131,072 global tokens per update, BF16
autocast with its cache enabled, FP32 parameters and full fused AdamW state.
Memory is the largest steady reserved allocation on any participating GPU.
The single-GPU trials used a 90 GiB allocator cap; DDP trials used 88 GiB.

| GPUs | Microbatch/GPU | Core execution | Core checkpointing | Loss chunk | Tokens/s | Reserved/GPU |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 2 | Eager | Off | 1,024 | 12,180 | 51.72 GiB |
| 1 | 4 | Eager | Off | 1,024 | 12,252 | 68.83 GiB |
| 1 | 8 | Eager | Off | 1,024 | Out of memory | Exceeded 90 GiB cap |
| 1 | 8 | Eager | On | 1,024 | 9,249 | 45.13 GiB |
| 1 | 4 | Compiled cores | Off | 1,024 | 15,142 | 60.44 GiB |
| 1 | 8 | Compiled cores | Off | 1,024 | 15,014 | 86.08 GiB |
| 1 | 4 | Compiled cores | Off | 8,192 | 15,321 | 62.89 GiB |
| **2** | **4** | **Compiled cores** | **Off** | **1,024** | **29,490** | **68.76 GiB** |
| 2 | 2 | Compiled cores | Off | 1,024 | 28,140 | 55.33 GiB |

Compiled cores improved single-GPU throughput by about 24% over eager
microbatch 4 and reduced reserved memory by about 8.4 GiB. Increasing the
compiled microbatch to 8 consumed another 25.6 GiB without improving speed.
Activation checkpointing saved memory but repeated enough computation to
make the tested eager microbatch-8 configuration slower.

The 8,192-token loss chunk improved single-GPU throughput by only 1.2%, used
another 2.5 GiB, and produced slightly different gradient rounding. Retaining
1,024 keeps the existing loss computation closer without a material measured
speed penalty. Among the two-GPU trials, microbatch 4 was 4.8% faster than
microbatch 2 and still left substantial memory available.

`nvidia-smi topo -m` reported a **SYS** connection between these GPUs: traffic
crosses PCIe and the CPU interconnect between NUMA nodes. There is no NVLink
connection between this pair. Despite that topology, the measured compiled
microbatch-4 DDP run achieved **1.95×** the corresponding single-GPU throughput.

## Measurement method and limits

[`benchmark_training.py`](benchmark_training.py) loaded the trusted local
checkpoint `work/run/before-optimization-step-237.pt`, an immutable hardlink
to the checkpoint at step 237 and 31,064,064 training tokens. It mmap-loaded
the checkpoint on the CPU, copied model parameters and every optimizer
tensor into separate CUDA storage, and optimized those disposable copies.
It read the prepared corpus in place and never saved a training checkpoint.

Each case performed complete forward/backward, gradient clipping, and fused
AdamW updates over 131,072 global tokens. Timings include data batching and
device transfer. DDP timings include gradient communication and use the
slowest rank's elapsed time. Compilation and the first update were excluded
as warmup; DDP excluded two warmup updates to allow gradient bucket rebuilding.
Every successful result aggregates **three measured updates**. The full
14.94 GiB of FP32 Adam moments was present on each GPU, and all measured
losses and gradient norms were finite.

These are short steady-compute samples, not long-run throughput guarantees.
They exclude checkpoint loading/saving, validation, process startup, and
initial compilation. At an unchanged 29,490 tokens/second, 40B tokens take
about 15.7 days of training compute; operational overhead adds time.

Generated evidence remains ignored by Git:

- `work/benchmark-grid-summary.json`: all nine cases and throughput ranges.
- `work/benchmark-*/summary.json`: exact arguments, source hashes, checkpoint
  identity, per-update times, and first-update/steady memory peaks.
- `work/benchmark-*/rank-*.jsonl`: individual worker measurements.
- `work/benchmark-eager-mb8/failure.json`: the bounded allocation failure.

The source hashes in each result identify the files present for that run.
Runner source evolved during this optimization task; the full-model benchmark
used its own update loop and the same batching/model computation throughout.

## Live continuation and validation

The production run resumed at step 237 with 31,064,064 tokens consumed and
completed twelve updates using the selected configuration. Steady updates
ran at approximately 29.0–29.7K tokens/second, with 67.75 GiB allocated,
68.64 GiB reserved, and 25.31 GiB physically free per GPU. The previous
shared settings measured 8.5K tokens/second after VORTEX stopped and
5.5–6K while VORTEX was training.

A real `datactl.sh stop warpstate-train` completed the current update and
saved step 249 at 32,636,928 tokens. Both workers exited and released GPU
memory. Inspection of the 22.41 GiB checkpoint confirmed all 256 canonical
model tensors, full FP32 Adam moments and step counters, both rank random
states, the original initialization metadata, and the execution transition
at step 237. Restart restored validation loss 5.56880 from that checkpoint.
The original step-237 checkpoint remains at
`work/run/before-optimization-step-237.pt`.

Verification also passed 64 CPU tests and four CUDA integration tests
(two original single-GPU cases and two distributed cases). These cover
exact resume within one configuration, one-to-two-GPU continuation,
the final partial update with an empty rank, and graceful/forced worker
cleanup. A full-model eager-versus-compiled comparison had a maximum
loss difference of 0.000012 and relative gradient-norm difference below
0.072% across three updates. Initial validation across the two execution
configurations was 5.63291 versus 5.63279, consistent with small rounding
differences.

The generated live checkpoint verification is recorded in
`work/optimization-resume-validation.json`.

## Production controls

In optimized mode, the `warpstate-train` program in
[`supervisord.conf`](supervisord.conf) must point to `ops/train-fast.sh`.
Use the project-private Supervisor through `datactl.sh`:

```bash
cd /workspace/WarpState
rtx6000_40b/ops/datactl.sh status
rtx6000_40b/ops/datactl.sh start warpstate-train
tail -n 10 rtx6000_40b/work/logs/train.log
tail -n 3 rtx6000_40b/work/run/metrics.jsonl
```

To stop and preserve progress:

```bash
rtx6000_40b/ops/datactl.sh stop warpstate-train
```

The launcher receives the stop signal and creates
`work/control/training-stop-request`. Both workers finish the current update;
the runner saves the model, full optimizer state, progress, and rank random
states before exiting. The launcher allows up to 820 seconds for this,
followed by 45 seconds for termination if workers fail to exit. Supervisor's
stop timeout is 900 seconds. Use the launcher/Supervisor control rather than
signalling an individual DDP worker. Starting the same program resumes
`work/run/latest.pt`; do not delete the checkpoint to change execution options.

Keep at least 53 GiB free for a full checkpoint and atomic replacement.
Training artifacts, caches, and runtime control files remain inside
`/workspace/WarpState` and are excluded from Git.

## Reproducing benchmarks

**Stop training first and wait until both GPUs are free.** A benchmark uses
disposable model state, but it still competes for GPU memory and computation.
VORTEX must also remain paused. From the repository root:

```bash
rtx6000_40b/ops/datactl.sh stop warpstate-train
rtx6000_40b/ops/datactl.sh status
nvidia-smi
rtx6000_40b/ops/datactl.sh start warpstate-benchmark
tail -n 10 rtx6000_40b/work/logs/benchmark.log
```

[`benchmark-supervisord.conf`](benchmark-supervisord.conf) specifies the
winning two-GPU configuration and sets `autostart=false` and
`autorestart=false`. Its command, expressed with environment variables, is:

```bash
cd /workspace/WarpState/rtx6000_40b
WARPSTATE_BENCH_DEVICES=0,1 WARPSTATE_BENCH_PROCESSES=2 \
TORCHINDUCTOR_COMPILE_THREADS=8 ops/benchmark.sh \
  --checkpoint /workspace/WarpState/rtx6000_40b/work/run/before-optimization-step-237.pt \
  --micro-batch 4 --seq-len 2048 --global-tokens-per-update 131072 \
  --no-activation-checkpointing --compile-cores --autocast-cache \
  --loss-chunk-size 1024 --vram-limit-gib 88 \
  --warmup-updates 2 --measured-updates 3
```

Use Supervisor for routine runs; the command above documents its effective
arguments. `benchmark.sh` sources `training-env.sh` and launches `torchrun`.
Each invocation automatically receives a fresh
`work/benchmark-<TORCHELASTIC_RUN_ID>` directory. An explicit `--output-dir`
must name a new directory under `work/benchmark-*`; existing rank log files
are not overwritten. If reproducing against another trusted checkpoint,
change the configured checkpoint path and record which checkpoint was used.

For a different candidate, edit only the benchmark program's command or
environment, then apply it with `ops/datactl.sh reread` and
`ops/datactl.sh update warpstate-benchmark` from the package directory.
Starting that program runs the selected finite benchmark. Once it has exited,
check `nvidia-smi` and restart `warpstate-train` to resume production.
