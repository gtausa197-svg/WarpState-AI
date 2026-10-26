# Final DPO archive and chat

DPO finished on 2026-09-15 at 01:05:06.882 UTC, with the guardian's final
verification complete at 01:05:42 UTC. It completed 2,334 updates over 74,661
preference pairs. Training remains paused.

The historical archive is `work/archives/step-00076294-dpo-final/` relative to
`rtx6000_40b`. It preserves the full FP32 model/Adam/RNG checkpoint and a
tensor-exact inference export, plus the tokenizer, evaluation, recipes,
reference scores, and provenance. Files are sealed read-only and checksummed.
Existing raw and SFT archives are preserved.

From `/workspace/WarpState`, create or verify the archive and prepare the
independent chat copy:

```bash
source rtx6000_40b/ops/training-env.sh
rtx6000_40b/.server-venv/bin/python -B rtx6000_40b/ops/archive_dpo.py
rtx6000_40b/.server-venv/bin/python -B rtx6000_40b/ops/prepare_dpo_chat.py
```

The chat copy is `rtx6000_40b/work/chat/dpo-final/`: separate physical files,
with hashes equal to the archive. The chat launcher reads this copy by default.
Neither chat nor preparation modifies archived weights.

```bash
rtx6000_40b/ops/chat-dpo.sh
# Or attach to the already-open interface:
tmux attach -t warpstate-chat
```

Type a message at `You:`. Ctrl-C discards unfinished input or cancels the current
reply and returns to `You:`, preserving completed conversation turns.
`/reset` clears conversation history; `/quit` exits.
Detach tmux with Ctrl-B followed by D. The interface uses GPU 0, BF16 compute,
original FP32 master weights, a 2,048-token context, and greedy decoding.
Responses print when complete. Alternate CUDA devices can be selected with
`--device cuda:1`; `--temperature 0.7` enables sampling.

Prompt serialization matches the segmented role headers, BOS and assistant EOS
used during posttraining. Cached inference follows the model's parallel chunk
memory; shape-dependent BF16 rounding can differ from full-sequence execution.

Run reproducible sample prompts, choosing a new output filename each time:

```bash
rtx6000_40b/ops/chat-dpo.sh --prompts-json rtx6000_40b/configs/posttrain_smoke_prompts.json --max-new-tokens 128 --output-json rtx6000_40b/work/validation/dpo-smoke.json
```

These few prompts are sanity checks, not a capability benchmark. DPO's final
held-out preference loss was 0.5790; 245 of 339 pairs had positive relative
reward margins. This 72.27% is not a generation win rate or answer accuracy.
Exponentiating DPO loss does not produce language-model perplexity.

For a matched full SFT holdout evaluation, `ops/evaluate_sft_snapshot.py`
accepts `--snapshot`, `--device`, and a new `--output` path. Reports and generated
responses stay under ignored `work/`; source scripts and prompts are tracked.
