# RTX workstation restart handoff

Prepared on 2026-09-15. All paths below are relative to `/workspace/WarpState`.
Training, posttraining, and benchmarks have finished; leave them stopped.
Use the existing environments and prepared data. BF16 compute remains required.

## Saved model state

| Stage | Final step | Full resumable checkpoint | Immutable model archive |
| --- | ---: | --- | --- |
| Raw pretraining | 76,294 | `rtx6000_40b/work/run/latest.pt` | `rtx6000_40b/work/archives/step-00076294/` |
| SFT | 10,312 | `rtx6000_40b/work/archives/step-00076294-sft-final/checkpoint.pt` | `rtx6000_40b/work/archives/step-00076294-sft-final/` |
| DPO | 2,334 | `rtx6000_40b/work/archives/step-00076294-dpo-final/checkpoint.pt` | `rtx6000_40b/work/archives/step-00076294-dpo-final/` |

Raw training reached 10,000,007,168 tokens with validation loss 2.3496886492.
SFT and DPO checkpoints both record `completed=true`. Their validation losses
are 1.1157395900 and 0.5790014537 respectively, for different objectives.
The earlier step 8,000, 16,000, and 40,000 archives are also retained.
Do not modify archived files. Chat uses an independent copy in
`rtx6000_40b/work/chat/dpo-final/`.

With user approval, five byte-identical working weight files were removed
after comparison with their retained archives, reclaiming 72,184,948,679 bytes.
These are the SFT and DPO working `run/latest.pt` and `initialization/model.pt`
files, plus `work/chat/step-8000-s536yem8/model.pt` (under `rtx6000_40b/`).
All archives and the independent current chat copy remain intact.

Before any future posttraining continuation or rerun of completion checks,
restore independent working files from the following sources. Paths in this
table are relative to `rtx6000_40b/work/`:

| Archived source | Working destination |
| --- | --- |
| `archives/step-00076294/model.pt` | `posttrain/step-00076294/initialization/model.pt` |
| `archives/step-00076294-sft-final/checkpoint.pt` | `posttrain/step-00076294/run/latest.pt` |
| `archives/step-00076294-sft-final/model.pt` | `posttrain/step-00076294-dpo/initialization/model.pt` |
| `archives/step-00076294-dpo-final/checkpoint.pt` | `posttrain/step-00076294-dpo/run/latest.pt` |

Verify the archive's `SHA256SUMS`, copy only to an absent destination using
`cp --reflink=auto --no-clobber`, and compare the copy with its source before
use. Never train against an archive path or a hardlink to archived weights.
Restoring these files does not authorize starting training.

Prepared pretraining tokens remain in `rtx6000_40b/work/data/`; posttraining
data remain in `rtx6000_40b/work/posttrain/data/`. Benchmark results are in
`rtx6000_40b/work/validation/nord-dpo/` and
`rtx6000_40b/work/validation/nord-stage-comparison/COMPARISON.md`.

## After reboot

Start in `/workspace/WarpState` and read `AGENTS.md` and this handoff.
Check `nvidia-smi` and `git status --short --branch`.
To restore post-DPO chat in its own tmux session:

```bash
cd /workspace/WarpState
tmux has-session -t warpstate-chat 2>/dev/null || tmux new-session -d -s warpstate-chat -n dpo-chat -c /workspace/WarpState 'rtx6000_40b/ops/chat-dpo.sh'
tmux attach -t warpstate-chat
```

Ctrl-C cancels typing or generation and returns to the next input. `/reset`
clears conversation history, `/quit` exits, and Ctrl-B then D detaches tmux.
Tmux sessions and in-memory chat history do not survive reboot; captured
terminal history is available in the restart record below.

All workspace Supervisor jobs have `autostart=false` and `autorestart=false`.
If job management is needed, check `rtx6000_40b/ops/datactl.sh status` first.
Initialize the workspace Supervisor only when it is absent:

```bash
/usr/local/bin/supervisord -c /workspace/WarpState/rtx6000_40b/ops/supervisord.conf
```

This starts no training. The raw launcher still stops at step 76,294, and
posttraining has completed. Further training requires a new user request and
an explicitly chosen continuation configuration; do not simply restart jobs.
Do not use the B200 launcher on this RTX workstation.

Jupyter currently uses HTTPS port 8080 with the existing authentication.
After reboot, check the platform-managed Jupyter service before launching
another server. Runtime files and logs from the last standalone launch are
under `rtx6000_40b/work/control/jupyter/`; treat them as private because they
can contain authentication data. Never commit them or expose their contents.
The raw model download path on the Jupyter host is
`/files/workspace/WarpState/rtx6000_40b/work/archives/step-00076294/model.pt?download=1`.
For post-DPO weights, substitute `step-00076294-dpo-final`.

## Restart record

`rtx6000_40b/work/restart/LATEST` identifies the private local restart record.
It contains available tmux history, checkpoint metadata, file sizes, archive
checksum results, job state, and a Git bundle of committed sources.
The earlier checkpoint inventory is a pre-cleanup snapshot; the current paths
are listed above. `rtx6000_40b/work/restart/duplicate-weight-deletion.json`
records the approved removals and their retained counterparts.
Artifacts and this bundle are on the same machine, not an off-machine backup.
Keep datasets, weights, runtime files, and this generated record out of Git.
