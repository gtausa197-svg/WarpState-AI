# Backward-layout training validation — September 12, 2026

The paired graph replacement was validated with two-GPU training and checkpoint
continuation fixtures with production attention and FFN dimensions. The fixtures
use one or two physical cores/logical layers, not the complete 28-layer model.
They verify complete parameter-gradient and optimizer behavior for those fixtures;
they are not throughput measurements or proof about every trained activation.

`ops/validate_backward_layout_training.py` retains microbatch four, sequence
length 2,048, 32 heads, head dimension 64, chunk size 128, and FFN width 7,168.
It uses BF16 autocast/residuals, compiled cores and full loss, conservative
compiled-autograd communication overlap, 128 MiB DDP buckets, and FP32 master
parameters, gradients, all-reduces, and fused AdamW state. Weights and tokens are
deterministic synthetic fixtures. Adam starts with deterministic nonzero FP32
moments, rather than fresh first-update moments or imported production state.

Three accumulation microbatches exercise separate backward variants: first
`no_sync()` with absent gradients, later `no_sync()` accumulation, and final
accumulation with DDP synchronization. There are two full updates and a final
nine-token update, during which rank one has entirely masked targets. Each
candidate backward must execute the paired operation; an earlier dispatch cannot
hide fallback in a later graph. Expected totals per rank are seven dispatches
for continuous execution, three for the partial run, and four after resumption.

The harness runs baseline, continuous candidate, partial candidate, and resumed
candidate in separate processes. It compares initial forward logits, every
parameter gradient after every backward, and every parameter/Adam state after
every update. It also requires byte-identical final model, optimizer, RNG, token
cursor, configuration, and execution/source identity across candidate stop/resume.
Both ranks' final model and optimizer states must agree exactly in each run.
Source hashes are checked between phases, and outputs require a fresh directory
under `work/validation`. No production checkpoint or dataset is opened or changed.

The later `--cores` option selects a positive number of distinct physical cores
and the same number of logical layers. It scales both the per-backward dispatch
requirement and exact final dispatch count with logical depth. This covers
cross-core compilation that a single-core fixture cannot test.

The numerical limits were declared before CUDA measurements. Gradient and Adam
moment errors must satisfy both L2 error at most 0.5% of reference L2 plus
`1e-8`, and maximum error at most 1% of reference maximum plus `1e-8`. Updated
parameters have a `2e-5` maximum absolute limit; losses have a `2e-4` absolute
limit. Adam step counters, initial forward logits, and complete candidate resume
state require byte equality. Every comparison separately records exact bytes,
finite values, and numerical error; a tolerance pass is not labeled byte equality.

The initial fixture used a 512-token vocabulary with repeated input indices.
Its candidate gradient/optimizer comparisons passed, but strict continuation
failed. Differences were confined to the tied embedding: 33 parameter elements
with maximum difference `3.725290298461914e-9`, 50 first-moment elements with
maximum `9.516952559351921e-8`, and 28 second-moment elements with maximum
`4.3655745685100555e-11`. All other parameter/state tensors, RNG states, and
execution/cursor metadata matched exactly. Every run's two ranks agreed. This
is consistent with the existing relaxed embedding atomic accumulation for
repeated tokens. The original failure and numerical diagnostics remain recorded;
no comparison limit was relaxed to turn it into a pass.

The successful `--unique-inputs` fixture uses an 8,192-token vocabulary and a
deterministically shuffled permutation of all input indices in every microbatch.
Each embedding row is therefore visited once per backward. Targets can still
repeat. This removes that repeated-index accumulation from the exact-continuation
fixture without changing the model's training implementation. The guarded pass
also gained a rejection for unsupported `disable_ftz=True` before this fresh run;
the measured default setting and kernel arithmetic were unchanged.

The single-core run passed on both RTX PRO 6000 Blackwell Max-Q GPUs:

| Check | Result |
| --- | --- |
| Initial forward logits | Byte-identical |
| Every required backward dispatch | Present; one per microbatch |
| Gradient comparisons across candidate/partial/resumed ranks | 28 complete parameter sets |
| Maximum gradient absolute / relative L2 error | `2.7284841053187847e-12` / `1.2185588970205807e-7` |
| Remaining gradient differences | Only the 32-element memory-mix parameter |
| Parameters and Adam moments after all 12 compared rank-updates | Byte-identical to baseline |
| Full candidate checkpoint continuation on both ranks | Byte-identical for every saved field |
| Model/optimizer agreement between ranks in every run | Byte-identical |
| Maximum allocated / reserved CUDA memory | 3.064 / 3.445 GiB |

For that single-core run, the harness source hash was
`a2d24d27b81db2f24daac76fb23096a67116e86feed2b43621a866ec5b7d1656`;
and the guarded graph-pass hash was
`4029da453f54b470f8c091f3faf825913583d357d0285450fccf82082c4557c3`.
The JSON reports contain all other source hashes, per-tensor results, and runtime
dispatch/match counts. That run found four matching compiled backward
variants per rank with no missed candidates. The initial and final vocabulary
fixtures are different correctness experiments and must not be compared for speed.

While the full-model probe was compiling, a CPU reproduction exposed a limitation
that a single-core fixture cannot detect: installed AOT common-subexpression
elimination shares the same zero constant between cores. The original matcher's
external-user guard rejected both regions in that reproduction. The disposable
full-model probe was stopped before its first update completed; it supplies no
timing result. The matcher was corrected to retain only that strictly
matched, immutable zero constant while another region still uses it; external
users of all other matched arithmetic remain rejected. The arithmetic kernels
were unchanged. A separate CPU test applies the installed CSE pass to two regions
and checks that both replacements succeed.

The fresh two-core run used `--cores 2 --unique-inputs` and passed all the same
checks with unchanged numerical limits. It found eight matched regions per rank
across four backward variants, with no misses. Each backward dispatched twice;
the continuous/partial/resumed totals were exactly 14/6/8 per rank. Across 28
complete gradient comparisons, the largest difference was
`3.637978807091713e-12` absolute and `1.2654258902165433e-7` relative L2. Every
parameter and Adam tensor after all 12 compared rank-updates remained
byte-identical to baseline. Complete candidate stop/resume state and model/Adam
agreement between ranks were byte-identical for every saved field. Maximum
allocated/reserved CUDA memory was 5.366/5.736 GiB.

For the two-core run, the final harness source hash is
`5f2d161aa624b670b07205345c63177b3b0f46baa67eaef7e9d4d15482f42f6d`
and the graph-pass hash is
`174e97d06602471c258c70b7c0cef9efba7d14e7b78d71251ea012136cbc5ff9`.
These supersede the single-core hashes above; all older reports are preserved.
The CUDA kernel source stayed unchanged across both successful fixtures.

The expanded CPU oracle suite passed 18 tests. Independently, the producer-scheduling
option passed the existing two-GPU asynchronous checkpoint/resumption test:
one passed, five deselected, in 71.06 seconds. That smaller fixture validates the
producer scheduler's continuation behavior; it does not exercise the paired
kernel or establish a combined full-model performance gain.

Ignored evidence is retained at:

- `work/validation/backward-layout-training-20260912/results.json`: repeated-index fixture, including strict-resume failure.
- `work/validation/backward-layout-training-20260912/resume-differences.json`: embedding-only continuation differences.
- `work/validation/backward-layout-training-unique-20260912/results.json`: successful single-core fixture and all source hashes.
- `work/validation/backward-layout-training-two-core-20260912/results.json`: successful two-core fixture after the shared-constant fix.
- `work/validation/backward-layout-training-producer-resume-20260912.log`: producer-scheduling continuation test.

To reproduce, reserve both GPUs, keep production training paused, and use a new
output directory. Use `--unique-inputs` for the strict continuation fixture;
omitting it retains the repeated-index diagnostic, which can fail exact resume
because of the existing embedding accumulation nondeterminism.

```bash
cd /workspace/WarpState/rtx6000_40b
source ops/training-env.sh
CUDA_VISIBLE_DEVICES='' .server-venv/bin/python -B -m pytest -q test_backward_layout_training.py
CUDA_VISIBLE_DEVICES=0,1 .server-venv/bin/python -B \
  ops/validate_backward_layout_training.py --world-size 2 --unique-inputs --cores 2 \
  --output-dir work/validation/backward-layout-training-two-core-new
CUDA_VISIBLE_DEVICES=0,1 WARPSTATE_TEST_CUDA=1 \
  .server-venv/bin/python -B -m pytest -q test_training_optimization.py \
  -k producer-scheduled-overlap
```
