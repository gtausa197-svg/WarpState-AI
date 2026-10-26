# Archive completed SFT and continue into DPO

The user authorized this sequence on 2026-09-14:

**Finish SFT → archive its full state → independent initialization → DPO → pause.**

The separate `warpstate-dpo` Supervisor job waits while the existing SFT job
continues unchanged. The original SFT plan still says pause because it is part
of that run's immutable resume signature; its guardian must complete and exit
before this newly authorized continuation can start. The active DPO plan is
[`../configs/dpo_10b.json`](../configs/dpo_10b.json).

## Completion and archival gate

The DPO guardian requires all of the following before any GPU work:

- SFT Supervisor state `EXITED`, exit status 0, and its guardian's durable state
  `complete`.
- Exclusive SFT guardian, base launcher and pipeline locks, then recheck.
- Exactly 10,312 SFT updates, 329,973 samples, 212,321,372 shifted input tokens,
  and 143,695,039 supervised assistant tokens.
- A completed full SFT checkpoint with Adam and both ranks' RNG state, and its
  matching final evaluation of all 1,603 held-out conversations.
- Unchanged parent SFT plan, tokenizer, prepared-data manifest and source pins.

The archive is `rtx6000_40b/work/archives/step-00076294-sft-final/`. It contains:

| File | Purpose |
| --- | --- |
| `checkpoint.pt` | Physical byte-for-byte copy of the full SFT checkpoint, including optimizer/RNG/cursor/signatures |
| `model.pt`, `snapshot.json` | Original FP32 model-only export, with every tensor compared exactly to the full checkpoint |
| `tokenizer.model` | Exact original SentencePiece tokenizer |
| `evaluation.json`, `completion.json`, `settings.json`, `metrics.jsonl` | Original final evaluation, completion, execution settings and complete SFT history |
| `sft-execution.json`, `sft-handoff.json`, `PROVENANCE.json` | Source/data identities and both raw-pretraining and SFT stage counters |
| `ARCHIVE.md`, `SHA256SUMS` | Retention instructions and integrity checks |

All files are sealed read-only before the complete directory is atomically
published without replacement. Existing archives are validated and reused, or
rejected without alteration. Interrupted hidden staging directories remain for
diagnosis. The original SFT resume checkpoint and all raw historical archives
remain in place.

DPO reads a physically independent copy of `model.pt`, `snapshot.json` and the
tokenizer under `work/posttrain/step-00076294-dpo/initialization/`. It does not
inherit SFT Adam state or write into either SFT archive or SFT run directory.
The full SFT archive is roughly 32 GB decimal; its counters describe SFT after
the raw step-76,294 model, not an additional 10B-token pretraining run.

## Preference objective and execution

This first DPO experiment uses:

| Setting | Value |
| --- | --- |
| Data | 74,661 training pairs and 339 held-out pairs already prepared |
| Objective | Length-normalized DPO, beta 5 |
| Epochs | 1; 2,334 optimizer updates |
| Global batch | 32 preference pairs; 2 pairs per rank × 2 ranks × 8 accumulation microbatches |
| Context | 2,048 tokens per chosen/rejected branch; no truncation or cross-conversation packing |
| Learning rate | Peak `5e-7`, 10% warmup, linear decay to zero |
| Optimizer | Fresh FP32 AdamW, betas 0.9/0.95, zero weight decay, clipping at 1 |
| Precision | BF16 compute, original FP32 parameters/gradients/Adam; no FP8 |
| Evaluation/save | Every 100 DPO updates; full holdout at start and completion |

For each branch, sum the log probabilities of **only the final assistant
response and its actual EOS token**. System/user/tool messages, earlier
assistant turns, role headers and padding are conditioning or masked. Divide
each policy-minus-reference log-probability sum by that branch's response-token
count, subtract rejected from chosen, multiply by beta, and apply negative
log-sigmoid. Average over actual preference pairs across ranks; short final
batches and empty rank slots do not change the normalization.

This uses the length-normalized objective and conservative hyperparameters
described in the [Tülu 3 paper, equation 6 and table 20](https://arxiv.org/html/2411.15124v3).
It is a pilot for this 2B/10B base, not a reproduced or tuned Tülu result.
Ordinary summed-log-probability DPO is a separate supported objective and its
usual beta cannot be substituted for beta 5 here without changing the recipe.

Before policy updates, score the immutable SFT model in BF16 on all selected
chosen/rejected branches. Each GPU saves resumable reference-score shards; the
final cache is bound to SFT weights, data, tokenizer, masks, execution settings,
library version and source hashes. Reusing or rebuilding it must preserve that
identity. The reference remains the completed SFT model after every policy
checkpoint/resume. Once scores are cached, release the reference model before
allocating policy optimizer state, avoiding a second resident full model.

The output head uses checkpointed token chunks to avoid retaining full-sequence
vocabulary logits for both responses. A tiny two-GPU BF16 preflight checks the
actual DPO/reference/continuation path after SFT releases the GPUs. CPU setup
tests do not claim that this deferred GPU gate or production-size startup has
already run. Any failed gate stops automatic progression with the archive kept.

## State, monitoring and controls

Paths below are relative to `rtx6000_40b/work/posttrain/step-00076294-dpo/`.

| Path | Purpose |
| --- | --- |
| `control/handoff.json`, `control/execution.json` | Guardian phase and frozen execution provenance |
| `initialization/` | Independent immutable SFT model copy |
| `preflight/` | Tiny GPU validation and its acceptance report |
| `reference-cache/` | Fixed SFT reference log probabilities and resumable scoring shards |
| `run/latest.pt` | Independent full DPO resume checkpoint, format `warpstate-dpo-v1` |
| `run/metrics.jsonl`, `run/evaluation.json`, `run/completion.json` | Training history, evaluation and durable completion |

From `/workspace/WarpState`:

```bash
rtx6000_40b/ops/datactl.sh status warpstate-posttrain warpstate-dpo
rtx6000_40b/ops/dpo-handoff.sh --check
tmux attach -t warpstate-dpo
tail -f rtx6000_40b/work/logs/dpo.log
```

To cancel automatic DPO while SFT continues, or gracefully stop active DPO:

```bash
rtx6000_40b/ops/datactl.sh stop warpstate-dpo
```

To explicitly resume the unchanged guardian/reference preparation/DPO state:

```bash
rtx6000_40b/ops/datactl.sh start warpstate-dpo
```

No restart of SFT is needed. The new guardian preserves the SFT guardian and
base-launch exclusion locks throughout DPO. Every worker holds a shared base
pipeline lock, so surviving workers block accidental base/SFT/DPO relaunches
even if their launcher dies. Stop files and lock files are never shared with
the live SFT stop request. The program has `autostart=false` and
`autorestart=false`; a whole-machine/Supervisor restart requires explicitly
restarting the appropriate jobs. A completed DPO run stays paused when its
guardian is started again.

On normal completion, validate final counters, all 339 held-out pairs, fixed
reference-cache identity and full archive/init hashes before recording success.
No additional training stage starts automatically. Next compare the archived
SFT model with DPO on instruction following and tool use. DPO loss, reward
margin and preference accuracy describe this objective; exponentiating DPO
loss does not produce language-model perplexity.

All generated data, scores, archives, checkpoints, logs and environments remain
ignored under `work/`. Only source/configuration/tests/documentation are tracked.
