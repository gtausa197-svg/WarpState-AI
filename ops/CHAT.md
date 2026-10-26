# CPU chat with the latest completed checkpoint

This is a text-generation probe of the pretrained **2.005B WarpState** model,
not a chat-finetuned assistant. The training corpus is English text plus EOS,
without role tokens. The default `Question: ... / Answer:` wrapper is only a
prompting convention. Use `/raw` to test ordinary text completion instead.
Fluency or one good answer does not by itself establish model quality.

Run from `/workspace/WarpState`:

```bash
bash rtx6000_40b/ops/chat-cpu.sh
```

The launcher uses CPU FP32 inference and 16 intra-op threads; it never
initializes CUDA or signals the training job. This allows both GPUs to keep
training. CPU responses are slower, and CPU/disk contention is still possible.
No training/model implementation, checkpoint writer, optimizer or data setting
is modified. No quantization, compilation or external service is introduced.

## Snapshot safety

At startup the tool opens the completed `work/run/latest.pt` once and pins that
inode using a read-only descriptor. The existing training writer publishes by
atomic rename, so a later save does not change the opened version. The tool
memory-maps that descriptor on CPU and exports only original model weights,
configuration, step/token boundary and provenance into a fresh directory under
`work/chat/`. Adam/RNG/dataset state are not exported: this is **inference-only**,
not a training-resume checkpoint. The training checkpoint is not overwritten,
hardlinked, chmodded, or otherwise mutated.

An export uses about 7.47 GiB. Its new weight file is read-only, and
`snapshot.json` records its SHA256, tokenizer hash and source inode/boundary.
The hash is verified before strict model loading. Only load trusted local torch
checkpoints. A snapshot is frozen at its printed step, not refreshed silently
while chatting. To test a later checkpoint, quit and launch again without
`--snapshot`; this creates another independent export. Reuse an existing export
without consuming another export's storage:

```bash
bash rtx6000_40b/ops/chat-cpu.sh --snapshot /absolute/path/to/work/chat/step-N-suffix/model.pt
```

`--snapshot` also accepts a full training checkpoint such as `work/run/latest.pt`.
When no `snapshot.json` is present, the tool validates the embedded training
format and creates an independent model-only export under `work/chat/` before
loading it. This creates a fresh export on each launch and prints its path:

```bash
bash rtx6000_40b/ops/chat-cpu.sh --snapshot /workspace/WarpState/rtx6000_40b/work/run/latest.pt \
  --max-new-tokens 128 --repetition-penalty 1.2 --no-repeat-ngram-size 4
```

`--checkpoint /path/to/latest.pt` remains the explicit way to capture training
weights. A chat export whose original `snapshot.json` is missing or invalid is
rejected; its integrity check is never bypassed or replaced with new metadata.
Restore the original metadata or create a fresh export from a training checkpoint.

No generated snapshots, metadata or transcripts are committed to Git.

## Controls and one-shot probes

- `/new`: clear conversation context.
- `/raw`: clear context and switch to plain text completion.
- `/chat`: clear context and restore the Q/A wrapper.
- `/info`: show the exact snapshot boundary and metadata.
- `/help`, `/exit` (or Ctrl-D): help / quit.
- Ctrl-C during generation cancels that reply, not training.

Each turn replays the retained prompt with one independent generation cache per
logical layer. A completed tile updates fast/slow memory only after producing
its own outputs. The total context budget defaults to 2048 tokens, including
the requested new tokens; an oversized prompt drops its oldest tokens. Long
histories therefore have substantial CPU prefill latency.

```bash
# A short reproducible greedy probe; no interactive session.
bash rtx6000_40b/ops/chat-cpu.sh --snapshot /absolute/path/to/model.pt \
  --prompt 'What is Facebook?' --temperature 0 --max-new-tokens 32

# Sampling settings are explicit and do not affect training RNG.
bash rtx6000_40b/ops/chat-cpu.sh --snapshot /absolute/path/to/model.pt \
  --threads 16 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 1337 \
  --max-new-tokens 128
```

For a separate terminal, attach to the prepared session using
`tmux attach -t warpstate-chat`. Detach with Ctrl-b, then d. Chatting in that
session does not attach to or control the training console.

## Repetition protection

Chat and raw completion now default to `--repetition-penalty 1.2` and
`--no-repeat-ngram-size 4`. Already generated tokens receive a lower score,
and a token cannot complete a four-token sequence already emitted in the
same reply. These controls run before greedy selection, top-k and top-p;
EOS can still end a reply normally. The prompt and earlier turns are excluded
from the restrictions, so the model can reuse names and phrases from a question.
Protection resets for each reply. The active settings are printed at startup.

This prevents exact token loops of that size and discourages shorter repeats;
it does not guarantee distinct meanings or turn a pretrained model into an
instruction-following assistant. Strict n-gram blocking can also restrict
intentional repetition in code, quotations or lists. For an unfiltered baseline,
pass `--repetition-penalty 1 --no-repeat-ngram-size 0`. Both options affect only
inference; no weights or training settings change.

After updating a running chat, use `/exit` and relaunch it with the same
`--snapshot` path. This loads the new defaults and clears old repetitive
conversation context. Use `/new` to clear context later without reloading weights.

## Verification

The CPU test suite covers strict state loading, tokenizer identity, independent
depth caches, greedy/seeded sampling, repetition penalties for both logit signs,
phrase-loop blocking, per-reply reset, EOS, nonfinite logits, and snapshot
independence when `latest.pt` is atomically replaced during loading. It also
checks the CLI with `--snapshot latest.pt`, sealed snapshot reuse and rejection
of missing or invalid export metadata. It compares
incremental logits against full forward across three tile boundaries, with
both sequential and parallel memory. Agreement is tolerance-based: their
floating-point reduction orders are different, and CPU FP32 inference is not
claimed bitwise-equivalent to BF16 training.

```bash
cd /workspace/WarpState
source rtx6000_40b/ops/training-env.sh
CUDA_VISIBLE_DEVICES='' rtx6000_40b/.server-venv/bin/python -B -m pytest \
  rtx6000_40b/test_chat.py -q --basetemp=/workspace/WarpState/tmp/pytest-chat
```

The initial real-model smoke probe loaded an independent step-3250 snapshot
(425,984,000 training tokens) and produced finite greedy output on CPU. This is
a functionality check, not a new validation benchmark or a quality guarantee.
