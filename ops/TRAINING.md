# Training environment and shared GPU configuration

This document records the original two-GPU workstation configuration. Its
Supervisor and launchers are not the controls for the single-B200 host.
Current B200 settings, checkpoint and pause/resume controls are in
[B200_HANDOFF.md](B200_HANDOFF.md). For workstation history, see
[OPTIMIZED_TRAINING.md](OPTIMIZED_TRAINING.md); the shared GPU settings below
preceded that workstation preset.

The completed corpus in `work/data` contains exactly 40,000,000,000 training
tokens and 1,048,576 validation tokens. Its generated manifest records the
SHA256 checksums. The model has 2,005,111,424 parameters; 40B is the token
budget, not the parameter count.

No initial WarpState checkpoint was supplied. The shared-GPU launcher uses
`--from-scratch` with seed 1337 for its first launch. Once `work/run/latest.pt`
exists, subsequent launches restore its model, full AdamW state, token cursor,
learning-rate schedule, and random states. `--from-scratch` never overwrites
an existing run. To initialize another run from trusted weights, invoke
`run_40b.py --weights ...` with a separate work directory instead.

## Shared GPU settings

`train-shared.sh` selects physical GPU 1 by its UUID. VORTEX uses both GPUs,
so concurrent training shares GPU compute and can slow both jobs.
On another machine, set `WARPSTATE_CUDA_DEVICE` to the intended GPU UUID or
index before launching.

| Setting | Value |
| --- | --- |
| Sequence length | 2,048 tokens |
| Microbatch | 1 sequence |
| Gradient accumulation | 64 microbatches |
| Tokens per optimizer update | 131,072 |
| Parameters, gradients, AdamW moments | FP32 on the GPU |
| Forward/backward matrix operations | BF16 autocast |
| Activation checkpointing | Enabled |
| Autocast weight cache | Disabled to reduce retained weight copies |
| Loss projection chunks | 256 tokens |
| PyTorch allocator limit | 33.5 GiB |
| Required extra free memory at startup | 3 GiB |
| Validation | 8 batches initially and every 100 updates |
| Checkpoint | Every 50 updates, on a graceful stop, and at completion |

The September 9, 2026 live trial completed two full updates, saved a
22.41 GiB checkpoint, and resumed through step 5. Peak memory was 31.54 GiB
allocated and 32.02 GiB reserved, leaving about 4.7 GiB physically free.
Observed training throughput was approximately 5,500–6,000 tokens/second.
At that unchanged rate, 40B tokens would take roughly 77–84 days before
checkpoint/evaluation overhead; this is an early measurement, not an ETA
that accounts for future GPU availability. VORTEX continued advancing,
with overlapping updates around 68–69K tokens/second versus roughly 100K
before the trial.

The fixed FP32 parameters, gradients, and two AdamW moments alone occupy
about 29.88 GiB. Activations and allocator overhead must also fit the cap.
The launcher refuses to start if the cap plus startup headroom exceeds
available memory. This is a check, not a reservation for VORTEX.

PyTorch's [allocator memory limit](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
excludes CUDA context and library allocations. Compare both the allocated
and reserved peaks in `work/run/metrics.jsonl` with physical memory in
`nvidia-smi`. A different VORTEX workload may need different headroom.

Each full checkpoint is approximately 22.4 GiB. Keep at least 53 GiB free
for checkpoint writing and atomic replacement. Checkpoints, metrics, data,
and environments stay inside this project and are ignored by Git.

## Environment

The CUDA training environment is separate from the CPU data-preparation
environment and from VORTEX's environment. Recreate it with Python 3.12:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
python3 -m venv rtx6000_40b/.server-venv
uv pip install --python rtx6000_40b/.server-venv/bin/python \
  -r rtx6000_40b/ops/training-requirements-resolved.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
uv pip check --python rtx6000_40b/.server-venv/bin/python
```

The lock pins PyTorch 2.14.0+cu130 and its installed dependencies. Always
source `training-env.sh` when testing or provisioning so caches and temporary
files remain under `/workspace/WarpState`. The launcher sets these paths
itself and uses `--no-bootstrap`; the minimal GPU environment does not
contain the separate data-preparation dependencies.

## Operation

The existing project-private Supervisor manages both preparation and training.
Its training command is now `train-fast.sh`, which needs both GPUs available.
`datactl.sh` is its control wrapper, including for the training job:

```bash
cd /workspace/WarpState
rtx6000_40b/ops/datactl.sh status
rtx6000_40b/ops/datactl.sh start warpstate-train
tail -n 10 rtx6000_40b/work/logs/train.log
tail -n 3 rtx6000_40b/work/run/metrics.jsonl
nvidia-smi
```

To stop, use `rtx6000_40b/ops/datactl.sh stop warpstate-train`. The runner
finishes its current optimizer update and saves before exiting. Supervisor
allows up to 15 minutes for that. Starting the same job resumes its latest
checkpoint. `--reconfigure` permits changes to execution settings while keeping
the sequence length, global tokens per update, data, and optimizer schedule
unchanged. Changes to accumulation or GPU count can change floating-point
rounding; the model and full optimizer state are retained.

Training is started explicitly and does not automatically restart after a
failure, avoiding repeated allocation attempts on a shared GPU. Inspect the
log and available memory before restarting a failed job. A container restart
also requires starting the project Supervisor and the training job again.

For a bounded trial of the current configuration, run
`ops/train-fast.sh --max-steps 2` from the package directory while the
supervised training job is stopped. This saves after two
additional updates while preserving the full 40B schedule. Only one runner
can hold `work/pipeline.lock` at a time.

To return to the original shared GPU configuration, stop the supervised job
and run `ops/train-shared.sh --reconfigure` with enough free memory on GPU 1.
For persistent supervision, update the program command to that launcher plus
`--reconfigure` and use `stopsignal=TERM` before `datactl.sh reread` and
`datactl.sh update warpstate-train`. The shared launcher runs the worker
directly; the optimized launcher relays stop requests to distributed workers.
Do not run the benchmark or another launcher alongside the production job.

## Checks

The tests exercise interrupted data preparation, exact final partial updates,
the memory-headroom refusal, and exact CUDA checkpoint/resume behavior for
both imported weights and fresh initialization. The CUDA tests use a tiny
model and a 0.5 GiB allocator cap:

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
CUDA_VISIBLE_DEVICES=1 rtx6000_40b/.server-venv/bin/python \
  -m pytest rtx6000_40b/test_server.py -q
```

Use `CUDA_VISIBLE_DEVICES=''` to skip CUDA tests when there is insufficient
free memory. Test artifacts use the project-local `TMPDIR`.
