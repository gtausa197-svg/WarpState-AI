# WarpState training data preparation

The managed job prepares 40,000,000,000 training tokens and 1,048,576
validation tokens from `HuggingFaceFW/fineweb-edu`, configuration
`sample-100BT`, using the package's unchanged SentencePiece tokenizer.
The upstream commit is pinned in the preparation state and final manifest.

All job files, dependencies, caches, logs, and temporary files are below
`/workspace/WarpState`. This job uses CPU and network resources, with 16
tokenizer threads and a lower CPU scheduling priority. It does not train a
model. The package's original Python files are unchanged.

The launcher uses `ops/prepare_data.py` to avoid a reproduced PyArrow/Python
shutdown crash on this host. It waits for the original pipeline to return
(after token/state files are flushed and committed), flushes console output,
then exits without interpreter finalization. Exceptions are logged and keep
a nonzero exit code, so Supervisor can restart from committed progress.
Garbage collection alone did not eliminate the crash. This follows the
same process-exit approach documented in NVIDIA's
[Cosmos launcher](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/scripts/train.py).

The launcher was checked on a real FineWeb-Edu stream with 1,000,000 train
tokens and 64 validation tokens. A SIGTERM stop committed its partial batch
and exited successfully. Resuming produced byte-identical files to an
uninterrupted run, and both completion manifests matched the computed
SHA256 checksums. Changed data settings were rejected with a nonzero exit
code. The results are in
`work/validation/live-resume-exitguard/validation.json`.

## Paths

- Python environment: `rtx6000_40b/.data-venv`
- Resolved Python dependencies: `rtx6000_40b/ops/data-requirements-resolved.txt`
- Training tokens: `rtx6000_40b/work/data/train.bin`
- Validation tokens: `rtx6000_40b/work/data/val.bin`
- Resumable state: `rtx6000_40b/work/data/prepare-state.pkl`
- Completion manifest and SHA256 checksums: `rtx6000_40b/work/data/manifest.json`
- Preparation log: `rtx6000_40b/work/logs/prepare.log`
- Hugging Face cache: `rtx6000_40b/work/hf-cache`
- Package download cache: `.cache/uv`
- Temporary files: `tmp`

The final token files contain little-endian uint16 IDs and occupy
80,002,097,152 bytes together. An incomplete `train.bin` is not a completed
dataset. Completion requires `manifest.json` with `complete: true`; the job
computes both file checksums before writing that manifest.

## Monitor and control

Run from `/workspace/WarpState/rtx6000_40b`:

```bash
ops/datactl.sh status
tail -f work/logs/prepare.log

# Gracefully stop, committing the current preparation batch:
ops/datactl.sh stop warpstate-prepare

# Resume from the saved iterator and committed token counts:
ops/datactl.sh start warpstate-prepare
```

Supervisor runs independently of the terminal, restarts the job after an
unexpected exit, and leaves it stopped after successful completion. Its
configuration, Unix socket, PID, and logs are local to this project. It does
not change the instance's system Supervisor configuration or expose a port.

If this project's Supervisor itself is no longer running, start it with:

```bash
TMPDIR=/workspace/WarpState/tmp PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/bin/supervisord -c ops/supervisord.conf
```

The prepared data can later be used with the existing trainer's
`--stage train --work-dir /workspace/WarpState/rtx6000_40b/work` and its
default dataset, revision, tokenizer, and token-count arguments. Training
also requires a CUDA PyTorch environment and compatible initial weights.
