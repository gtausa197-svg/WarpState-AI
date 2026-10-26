# B200 optimization research, September 10, 2026

This records the initial research before implementation. The subsequent
measurements, selected launcher settings, CUDA graph validation, backend
comparison and asynchronous checkpoint implementation are in
[B200_PERFORMANCE.md](B200_PERFORMANCE.md). See
[B200_SCHEDULING.md](B200_SCHEDULING.md) for the completed stall investigation.

At the time of this initial research, training was paused at step **488**, with
**63,963,136** tokens consumed and validation loss **4.7938699126**.
`work/run/latest.pt` and `work/run/paused-step-488.pt` then referenced
the same verified 24,061,671,905 bytes. Subsequent atomic replacements of
`latest.pt` preserved the step-488 hardlink. Consult [B200_HANDOFF.md](B200_HANDOFF.md)
for the current paused checkpoint.

Checkpoint SHA256:
`69d37e902c41f541d55a07b3d10dcb7643721240cc6aae032455cd8904c64608`.
Inspection verified 2,005,111,424 FP32 parameters, all 256 AdamW state entries
and their step-488 counters, FP32 moments, rank-zero RNG, token cursor, and
learning rate 0.0000732. Details are in the ignored generated report
`work/validation/b200/paused-checkpoint-step-488.json`.

The dataset and production execution settings are preserved. The constraints
for all candidates remain BF16 computation, FP32 master parameters and AdamW,
sequence length 2,048, and **131,072 global tokens per optimizer update**.
The original learning-rate schedule, objective, tokenizer, and data order
remain unchanged. No optimization benchmark or production restart was run
for this research request.

## Measured baseline and first priority

Production updates 450–488 processed 5,111,808 tokens in 351.161 seconds
inside the update timer: **14,557 sustained tokens/s**. The median individual
update ran at **61,888 tokens/s**. Nine updates took approximately 30–32
seconds; ordinary updates took approximately 2.12 seconds. The timer excludes
checkpoint writes and evaluation. Source: `work/run/metrics.jsonl`.

Peak allocation was **59.40 GiB**, with **60.48 GiB reserved**. The B200 reports
182,631 MiB total memory; the current allocator cap is 150 GiB. Raising that
cap alone does not increase useful work.

During sampled stalls, GPU utilization fell to zero and the trainer's main
thread waited on a futex without accumulating CPU execution time. CPU quota
throttling did not increase; cgroup CPU/I/O pressure averages were zero.
There is no matching periodic branch in the training loop. This does not
identify the cause or eliminate data/pinning, compiler, autograd, driver,
or provider waits. Removing the waits is the largest observed opportunity;
the median rate is evidence of active compute speed, not a promised sustained
rate after a fix.

Use the benchmark's optional `--diagnose-stalls` to attribute host time to
batch transfer, forward, backward, clipping, AdamW, and final synchronization.
It adds no CUDA synchronization and emits Python stacks every ten seconds
during measured updates. Its timings include implicit waits and are not GPU
kernel timings. Then correlate the waiting call with CUDA and OS-runtime
traces across several updates. The existing `--profile` captures only one
measured update and can miss these stalls. Installed Nsight Systems is
2024.3.2.0, so verify B200/tool compatibility before using options documented
for newer releases. Sources: [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#os-runtime-libraries-trace),
[PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html).

## Candidates, in order

| Candidate | Concrete experiment | Why it may help and what remains unverified |
| --- | --- | --- |
| Larger microbatch | Compare microbatch/accumulation **4/16**, **8/8**, **16/4** at the same sequence length | Larger GEMMs and fewer microbatch dispatches can improve compute efficiency. Prior workstation memory was about 86 GiB at microbatch 8; roughly 137 GiB at 16 is an extrapolation, not a B200 measurement. Measure startup and steady peaks below the existing cap. [NVIDIA GEMM guidance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) |
| Keep loss activations instead of recomputing | Existing `--loss-mode compiled-full` at the best fitting microbatch | Uses more memory to avoid repeating the vocabulary projection during backward. At microbatch 16, BF16 logits alone are 2 GiB, before FP32 loss intermediates. Earlier workstation gains were about 3%; B200 gains are unmeasured. [PyTorch recomputation tradeoff](https://pytorch.org/blog/activation-checkpointing-techniques/), [local prior results](PERFORMANCE.md) |
| Tune matrix kernels for the chosen shape | Existing `--compile-mode max-autotune-no-cudagraphs` | Profiles candidate implementations while preserving BF16. Compile time increases; exclude it from timing and compare numerical results. [PyTorch compile modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html) |
| Compare cuBLASLt and workspace choices | Compare `TORCH_BLAS_PREFER_CUBLASLT=1` against the current backend, recording effective workspace | NVIDIA recommends 32 MiB workspace for Blackwell sm10x. Current `CUBLAS_WORKSPACE_CONFIG=:4096:8` already provides 32 MiB. PyTorch's unified workspace can cap cuBLASLt at the cuBLAS allocation; changing only `CUBLASLT_WORKSPACE_SIZE` may have no effect. Library preference may fall back. [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublassetworkspace), [PyTorch BLAS controls](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cuda.preferred_blas_library) |
| CUDA graphs when traces show dispatch overhead | Existing `--compile-mode reduce-overhead` | Can trade additional persistent GPU memory for lower host launch overhead. Static training shapes help, but accumulation, evaluation, and partial final updates require validation. Graphs are not an established remedy for the observed waits. [PyTorch compile modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html), [graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints) |

The model already uses compiled cores, BF16 autocast caching, fused AdamW,
and disabled core activation checkpointing. Recommending those as new changes
would not explain further gains. The prior workstation profile attributes
most matrix time to FFN and input projections; attention and the output head
are smaller fractions. Those proportions need confirmation on B200 before
investing in custom attention or loss kernels.

As a later wall-clock improvement, investigate checkpoint staging and
asynchronous serialization. PyTorch documents copying a stable model/optimizer
snapshot into CPU buffers before background writing, with one outstanding
save at a time. This needs approximately another checkpoint's worth of host
memory and careful completion/error handling. It requires integration work
to retain this runner's full-state format, atomic publication and graceful
stop guarantee. It would address save pauses, not the recurring waits inside
the measured updates. Keep the existing save cadence during initial compute
comparisons. [PyTorch asynchronous checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html).

## Comparison and promotion requirements

Use the preserved step-488 checkpoint for every disposable benchmark. The
existing benchmark copies model and optimizer tensors to separate CUDA
storage, reads prepared data in place, and writes no training checkpoints.
Its learning rate stays fixed at the loaded value and RNG is reseeded; its
updates are timing experiments rather than production continuation.

Start with two warmup updates and at least twenty measured updates, extending
the window if needed to capture several stalls. Report total tokens divided
by total update time, median update speed, slow-update count, VRAM peaks,
loss and gradient norm. Keep profiling runs separate from throughput runs.
Use fresh `work/benchmark-*` output directories and identical checkpoint/data
inputs for each candidate. Benchmark the baseline again beside the winner
to distinguish configuration effects from changing host conditions.

Before promoting a candidate, compare loss and gradients with the baseline
on identical tokens and validate full-state continuation. The existing CPU
loss tests preserve objective and parameter identity, but their compiled
case uses an eager backend. The compiled-full resume test currently requires
two GPUs. Add and run a single-GPU CUDA regression for compiled-full before
promoting it: continuous versus interrupted/resumed updates, canonical model
keys, full optimizer and RNG state, token cursor, and the final partial batch.
Microbatch-only changes still require the existing resume/signature checks
and a short production continuation with finite loss and a verified saved
checkpoint. Preserve schedule and data invariants through `--reconfigure`.

The currently committed diagnostics have only CPU validation: syntax and CLI
parsing, accumulated stage timing including exception cleanup, and watchdog
cancellation. GPU profiling and the candidate grid remain pending. Resume
optimization work when the user asks to continue; keep the B200 job stopped
until then.
