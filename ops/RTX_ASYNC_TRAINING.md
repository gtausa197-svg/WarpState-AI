# RTX BF16 execution and overlapping checkpoints

The local launcher selects the fastest complete-update experiment from
[RTX_PROFILE_20260910.md](RTX_PROFILE_20260910.md): compiled cores in default
mode, compiled full vocabulary loss, BF16 residual activations, microbatch 4
per GPU, and accumulation 8 across two RTX PRO 6000 GPUs. The measured update
rate was 30,707 tokens/second versus 29,081 for the previous configuration,
a 5.59% increase. These measurements exclude checkpoint and evaluation pauses.
The larger microbatch experiments and standalone GEMM/gradient prototypes
are not enabled in the production launcher.

The model, tokenizer, token corpus, 131,072-token global batch, context length,
optimizer and learning-rate schedule are preserved. Forward/backward compute
uses BF16. Master parameters, gradients and Adam moments remain FP32 on the
GPU. BF16 residual execution changes activation rounding; the first validation
at an unchanged checkpoint measures that numerical difference, not learning.

## Checkpoint lifecycle

`--async-checkpoint` works with one GPU or DDP. Every save captures random
states from every rank at a completed optimizer update. Only rank zero
constructs the checkpoint and owns the background writer.

1. Finish the previous write, if it is still pending.
2. Copy all checkpoint tensors and mutable metadata into an immutable CPU
   snapshot. Both ranks wait; later updates cannot mutate this snapshot.
3. Submit serialization to one CPU thread and let both GPUs resume training.
4. Write `latest.pt.tmp`, synchronize the file, atomically replace `latest.pt`,
   and synchronize its directory. The previous published checkpoint stays
   available until replacement.

One full snapshot costs approximately 22.4 GiB of host RAM for this model and
optimizer. There is no additional full checkpoint clone on either GPU, and
at most one snapshot/write is outstanding. The local launcher now saves and
evaluates every 100 updates. The initial promotion used a 50-update save
interval; increasing it to 100 reduces time spent staging checkpoints.

Step lines display validation loss only when that step has a fresh result,
together with its evaluation step. Metrics and checkpoints still record
`val_step` and `val_tokens` alongside `val_loss`, retaining the latest measured
values between evaluations. This avoids repeated validation values in the
terminal and adds no GPU work. Startup evaluation and final evaluation and
saving on a graceful stop remain enabled.

The main training threads coordinate initialization, construction, staging,
completed background errors and final close. The writer thread performs no
CUDA or distributed operations. Every update first checks the prior writer's
status on both ranks. A failure observed there is raised on both ranks before
the next update; a failure occurring during an update is observed at the next
check or save. Normal completion and graceful stop join the final write.
Exceptional exit joins only the local writer, avoiding extra collectives
when another worker may already have failed.

`Staged step ...` identifies a frozen snapshot, not durable progress. A later
`Saved ... (step ..., tokens=...; background write ...s)` identifies completed
publication. GPU-to-CPU staging and any wait for a previous save remain
synchronous; only disk serialization and publication overlap training.

## Validation and reproduction

The promotion checks passed 112 CPU tests and two CUDA continuation tests
(one GPU and two GPUs). The full-model canary started from the verified
production checkpoint at step 5,618 / 736,362,496 tokens. Its initial
validation remained 2.97925 to five decimal places. After startup, the first
three measured updates ran at 31,637, 31,482 and 31,450 tokens/second with
66.57 GiB reserved per GPU. This short canary validates operation; the
20-update comparison above remains the controlled performance experiment.

The canary staged step 5,620 in 11.41 seconds, then completed updates 5,621
and 5,622 before that checkpoint's 10.62-second background write finished.
It published the final step-5,622 full checkpoint before process exit.
This confirms real overlap and also shows that the CPU snapshot remains a
material pause. The initial production save interval was 50 updates; the
canary used two updates to exercise overlap and queue backpressure quickly.

Reloading that full checkpoint completed two further updates and published
step 5,624 / 737,148,928 tokens. These six canary updates belong only to the
disposable run; production resumes its original step-5,618 checkpoint. The
canary's logs, exact launcher arguments and integrity report remain in
`work/validation/async-promotion/`.

The distributed writer tests inject failures during state construction, CPU
snapshot allocation, previous-save waits, background checks, flush and close.
Two CPU/Gloo processes must both report each failure without hanging. Tests
also verify immutable snapshots, rank-zero-only writing, file and directory
synchronization, and cleanup without collectives on exceptional exit.

The CUDA execution regression compares continuous synchronous training with
an asynchronous split/resume run using compiled full loss and BF16 residuals.
Model tensors, Adam state, RNG state, validation loss and progress must match
exactly. It exercises periodic saves, final flush and a partially filled last
update where one rank has no valid tokens. This verifies exact continuation
for the test model and selected execution configuration; it does not establish
bitwise equivalence to the previous FP32-residual configuration.

Run from `rtx6000_40b` with both GPUs available:

```bash
source ops/training-env.sh
mkdir -p work/validation/async-promotion
.server-venv/bin/python -m pytest -q \
  test_checkpoint_writer.py test_distributed_checkpoint_writer.py \
  --basetemp="$PWD/work/validation/async-promotion/pytest-writer"
.server-venv/bin/python -m pytest -q test_training_execution.py \
  -k 'two_gpu_bf16_residual_async_full_state_resume_and_empty_final_rank or max-autotune-no-cudagraphs-async' \
  --basetemp="$PWD/work/validation/async-promotion/pytest-gpu"
```

Full-model validation uses a separate work directory with read-only access to
the prepared data and an initial hardlink to the paused checkpoint. Saving
atomically replaces only the disposable checkpoint path. Source checkpoint
integrity, canary logs and generated verification reports are retained under
`work/validation/async-promotion/`, outside Git. Production controls and status
commands are in [LOCAL_RESUME.md](LOCAL_RESUME.md).
