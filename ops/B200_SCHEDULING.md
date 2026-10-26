# B200 scheduling investigation

On September 10, 2026, a BF16 benchmark reproduced the recurring training
stalls while production remained paused at step 488 / 63,963,136 tokens.
The evidence strongly implicates the provider's HAIShare GPU scheduler.
The current compute entitlement and pool configuration have not been verified.

## Measured behavior

The diagnostic benchmark restored the full paused checkpoint, including AdamW
state, and used microbatch 4, accumulation 16, sequence length 2,048, compiled
cores, and checkpointed 1,024-token loss chunks. It performed two warmup and
20 measured updates of 131,072 tokens each, without checkpoint writes.
Generated evidence remains in
`work/benchmark-b200-baseline-diagnostic/rank-0.jsonl` and
`work/logs/b200-optimization.log`, relative to `rtx6000_40b/`.

Sustained measured throughput was **13,807.94 tokens/s**, versus a median
per-update rate of **61,853.22 tokens/s**. Most updates took about 2.12 seconds.
Five slow updates account for the large gap:

| Benchmark update, including warmup | Total seconds | Forward wall seconds |
| --- | ---: | ---: |
| 3 | 29.1877 | 27.8748 |
| 8 | 32.1939 | 30.7726 |
| 12 | 32.2163 | 30.8157 |
| 17 | 32.1830 | 30.8250 |
| 22 | 32.1977 | 30.8020 |

These are host wall times, including implicit synchronization waits, not GPU
kernel durations. Batch preparation and transfer took only 4.6-6.5 milliseconds
on those slow updates. During observed stalls, the main process slept in
`futex_wait_queue` without accumulating CPU work and GPU utilization dropped
to zero. Diagnostic Python stacks stopped inside the existing Triton launcher:
compiled model forward -> Inductor generated call -> Triton runtime `run`
-> generated `launcher`, line 5. This was the steady kernel launch path;
the captured stack did not show active compilation.
The final slow update also produced a stack in ordinary eager depth-modulation
tensor arithmetic, without Triton or Inductor frames. The wait therefore
occurs across CUDA entrypoints rather than only one compiled kernel path.

## Runtime evidence and interpretation

Read-only inspection found `/usr/lib/haishare/libhaishare.so` in
`/etc/ld.so.preload` and in both benchmark process maps. `nvidia-smi -q` also
identified a HAIShare wrapper before returning its report. The library exports
`cuLaunchKernel`, `continue_with_lock`, `own_lock`, `own_lock_cv`, and scheduler
state symbols. Disassembly of `continue_with_lock` showed a GPU ownership
check, a lock request, and `pthread_cond_timedwait` on `own_lock_cv`, retried
at five-second intervals until ownership is granted.

Together, CUDA launch interception and the scheduler's condition-variable
wait provide a concrete mechanism for the observed futex sleep beneath the
Python launcher. A native stack tracing every intermediate frame was not
captured, so the association remains an inference rather than a complete
dynamic call trace. Static symbols alone do not establish the active quota.

The GPU report showed zero ECC errors, no active thermal or power slowdown,
healthy fabric, and no recovery action. MIG and driver virtualization were
disabled; those settings do not exclude user-space GPU scheduling.

The provider's [GPUaaS FAQ](https://docs.hosted.ai/user-panel/faqs/gpuaas-faqs)
documents time sharing with a default **30-second time quantum**. Guaranteed
compute depends on the pool's sharing ratio, while idle tenants can allow
others to use more of the GPU. This closely matches the recurring extra
30-second waits. The provider's
[pool configuration documentation](https://docs.hosted.ai/admin-panel/gpuaas/creating-a-gpuaas-pool)
describes sharing ratio, scheduling mode, and time-quantum controls.

## Follow-up

Ask the provider to confirm this instance's guaranteed compute percentage,
pool sharing ratio, time quantum, scheduling mode, and exclusive-allocation
status. Correlating scheduler grants and releases with the benchmark timestamps
would confirm the cause and distinguish expected allocation from a scheduler
fault. A full-memory B200 view alone does not establish exclusive compute.

No provider configuration or interposition settings were changed. Local BF16
optimization continues within the assigned allocation; benchmark comparisons
must report sustained throughput including waits as well as typical update
speed. See [B200_OPTIMIZATION_PLAN.md](B200_OPTIMIZATION_PLAN.md) for the
performance experiments and [B200_HANDOFF.md](B200_HANDOFF.md) for checkpoint
and production controls.
