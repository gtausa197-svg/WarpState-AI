# Step 76,294 archive and supervised fine-tuning

The SFT stage runs from
[`../configs/posttrain_10b.json`](../configs/posttrain_10b.json). The user later
authorized **archiving the full completed SFT state, then automatically running
DPO**. A separate `warpstate-dpo` Supervisor job implements that continuation;
see [DPO_HANDOFF.md](DPO_HANDOFF.md). The active SFT plan/source bytes stay fixed
to preserve their checkpoint/resume signatures. Its original guardian still
finishes and exits cleanly before the DPO guardian can take over.

`prepare_posttrain_data.py` prepares conversations and preference pairs. The
separate `posttrain.py` runner implements assistant-only supervised training.

## Automatic sequence

1. `warpstate-posttrain-data` prepares the pinned dataset recipe on CPU now.
2. `warpstate-posttrain` waits for `warpstate-train` to exit successfully. It
   never changes the base run or resumes an earlier stopped base checkpoint.
3. Under the base launch/pipeline locks, require the full checkpoint to contain
   step **76,294**, **10,000,007,168 tokens**, and final validation at that same
   boundary. A checkpoint before the boundary remains waiting; an overshoot,
   stale validation, or mismatched tokenizer fails without starting SFT.
4. Export the original FP32 model using the same `chat.capture_latest` method as
   the 8K, 16K and 40K archives. Compare every exported tensor exactly against the
   stopped checkpoint; copy the exact tokenizer; write provenance/checksums;
   seal files read-only; publish the complete archive atomically without
   replacing an existing directory.
5. Make a physical, independent copy of the archive as SFT initialization. The
   full pretraining checkpoint, corpus, historical archives and chat selection
   are retained. SFT starts with a fresh AdamW optimizer on its first launch.
6. Verify prepared SFT data, run a tiny two-GPU BF16 training/resume preflight,
   then start the full SFT model. Failed preflight or data checks stop the
   handoff and retain the raw archive for diagnosis.
7. Save final SFT state and evaluation results, verify the raw archive and
   initialization remain unchanged, then exit. Neither program automatically
   restarts training after completion or failure. The separately armed DPO job
   then archives this full SFT state and continues according to DPO_HANDOFF.md.

The boundary triggers archival, checksum verification, compilation/preflight
and initialization before the first SFT update; allow several minutes for this
handoff. GPU preflight is deliberately deferred until base training releases
the GPUs. Its result is not claimed by the CPU-only setup tests.

## Paths

All paths below are relative to `/workspace/WarpState/rtx6000_40b`.

| Purpose | Path |
| --- | --- |
| Untouched full base resume checkpoint | `work/run/latest.pt` |
| Raw historical model archive | `work/archives/step-00076294/` |
| Independent SFT initialization | `work/posttrain/step-00076294/initialization/` |
| Prepared SFT/DPO data and final manifest | `work/posttrain/data/` |
| SFT checkpoints, settings and metrics | `work/posttrain/step-00076294/run/` |
| Final evaluation / completion | `run/evaluation.json`, `run/completion.json` under the SFT directory |
| Handoff state and frozen execution provenance | `work/posttrain/step-00076294/control/` |
| GPU preflight artifacts | `work/posttrain/step-00076294/preflight/` |
| Handoff and training log | `work/logs/posttrain.log` |
| CPU preparation log | `work/logs/posttrain-data.log` |

Archives contain model weights only, without Adam or training RNG state. SFT's
`latest.pt` is a separate full `warpstate-posttrain-v1` checkpoint with optimizer,
per-rank RNG, deterministic sample cursor, source/data identities and base
provenance. It is not interchangeable with the base runner's checkpoint.

## SFT recipe and validation

This first run uses one epoch, two GPUs, sequence length 2,048, microbatch 4 per
rank and four accumulation microbatches: **32 conversations per update**. There
is no cross-conversation packing. All parameters train with BF16 compute,
original FP32 parameters/gradients and fresh FP32 AdamW state; no FP8 is used.
The learning rate peaks at `5e-6`, with 3% warmup and cosine decay to zero,
zero weight decay and gradient clipping at 1. Evaluation/checkpoints occur
every 100 SFT updates and at the end. Initial and final validation use the full
SFT holdout; intermediate checks use the same fixed 128-example subset.
Settings and effective epoch length are
recorded in the SFT run's `settings.json` once data is loaded.

The learning rate is a conservative experimental starting point, not a
validated optimum for this 2B model. This is not a reproduction of Tülu 3's
SFT schedule: its published learning rate was calibrated using summed token
loss, and its models/context/batch sizes differ. See the
[Tülu 3 paper, section 4.3.2](https://arxiv.org/html/2411.15124v3).

Role headers use ordinary tokens from the existing SentencePiece vocabulary.
Only assistant content and the actual EOS ID contribute to the loss. Prompt,
role-header and padding labels are masked; DDP gradients are normalized by the
global count of supervised tokens, including uneven final batches. The shared
renderer is `ops/posttrain_data.py`; its hash is pinned in the data manifest.

SFT metrics report actual nonpadding input tokens and supervised assistant
tokens separately. SFT validation loss/perplexity uses the assistant-token
objective on the prepared holdout; it is **not directly comparable** with the
base model's all-token pretraining validation loss. Evaluation reports include
the exact evaluated sample/token counts, dataset and base-weight identities.
Behavioral instruction-following and tool-use evaluation is the next manual
decision after the full authorized SFT/DPO sequence pauses; a lower SFT loss
alone does not establish those capabilities. The final SFT archive preserves
the exact model and its evaluation for comparison with DPO.

## Monitor, pause and resume

From the repository root:

```bash
rtx6000_40b/ops/datactl.sh status warpstate-train warpstate-posttrain-data warpstate-posttrain
rtx6000_40b/ops/posttrain-handoff.sh --check
tmux attach -t warpstate-posttrain
tail -f rtx6000_40b/work/logs/posttrain.log
```

The tmux session displays the durable Supervisor log. Detaching tmux does not
stop the handoff or training. `RUNNING` on the posttraining program can mean
waiting for the base boundary; inspect `--check` or its phase log.

To pause the handoff or gracefully checkpoint an active SFT run:

```bash
rtx6000_40b/ops/datactl.sh stop warpstate-posttrain
```

To explicitly re-enable the unchanged handoff or resume an interrupted SFT run:

```bash
rtx6000_40b/ops/datactl.sh start warpstate-posttrain
```

The saved plan/source identity must still match. A completed run stays paused
even if the program is started again. Do not remove its completion state to
repeat SFT. Use a separately reviewed plan and output directory for a new
experiment. Do not clear lock files or launch the base trainer manually while
SFT workers are alive: every SFT rank holds a shared base pipeline lock, and
the base/handoff require its exclusive lock. The launcher also retains its
exclusion locks across child launch and unexpected watcher death.

SFT Supervisor includes are in `ops/posttrain-supervisord.conf`; both programs
have `autostart=false` and `autorestart=false`. Setup explicitly starts the
authorized jobs. A whole-machine/Supervisor restart requires an explicit
restart of the appropriate jobs; there is no hidden reboot training launch.

To cancel automatic continuation into DPO while allowing SFT to finish, stop
the separate waiting job with `rtx6000_40b/ops/datactl.sh stop warpstate-dpo`.
Stopping SFT itself leaves the DPO guardian waiting for verified completion.

## Rebuild the prepared inputs

The recipe in `configs/posttrain_data.json` pins immutable dataset revisions,
the tokenizer identity, sample counts/seed, split rule, renderer and required
preparation packages. Preparation records downloaded file hashes, schemas,
source/licensing metadata, row/token counts, rejections and output hashes.
Prompt groups use the same deterministic partition across SFT and DPO; exact
duplicate conversations are excluded. Overlength examples and DPO branches
are rejected rather than truncated. Gated xLAM data is recorded as skipped
without automatically accepting terms or requiring interactive input.

To rebuild the isolated CPU environment, source `ops/training-env.sh`, create
`work/posttrain-data-venv` with Python 3.12, then install
`ops/requirements-posttrain-data.lock` into that environment. Do not change the
active training environment. `ops/prepare-posttrain-data.sh` runs the recipe;
do not regenerate/replace data while SFT is running. The completed manifest is
published only after every selected output has been written and hashed.

Source/configuration/tests/dependency metadata belong in Git. Downloaded data,
token caches, generated manifests/reports, checkpoints, archives, environments
and logs under `work/` remain ignored.
