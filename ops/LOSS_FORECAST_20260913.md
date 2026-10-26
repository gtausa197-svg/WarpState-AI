# WarpState validation-loss forecasting, 2026-09-13

**Neither proposed token budget currently guarantees validation loss below 2.0.**
The observed curve supports continued improvement, but the crossing estimate is
extremely sensitive to the assumed eventual loss floor. A power-law fit to the
1–7B-token history crosses near 1.07T tokens; excluding the first billion tokens
of that fit moves its extrapolated floor above 2.0. This is uncertainty about the
forecast, not evidence of a demonstrated architecture limit.

This research did not change training, checkpoints, the dataset, or the schedule.
Generated snapshots, numerical results, and figures belong under
`work/validation/loss-forecast-20260913/` and remain excluded from Git.

## Current measurements and interpretation of the budgets

The frozen validation history ends at step 54,300: **7,117,209,600 tokens,
validation loss 2.42521**, with 555 evaluation records. Recent values fluctuate
around 2.41–2.43. The complete 6.75–7.00B bin averages 2.41219. Loss 2.0 corresponds
to perplexity `exp(2) = 7.389` on this tokenizer and evaluation distribution.

Interpret `8000*256` and `8000*1024` as **total optimizer steps**, not additional
steps or individual tokens. The active configuration processes
`2 GPUs * 4 sequences * 2048 tokens * 8 accumulation microsteps = 131072`
tokens per optimizer update.

| Budget | Optimizer steps | Total training tokens | Tokens per model parameter | Passes over the existing 40B-token corpus | Remaining wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8000*256` | 2,048,000 | 268,435,456,000 | 133.88 | 6.711 | 100.5 days |
| `8000*1024` | 8,192,000 | 1,073,741,824,000 | 535.50 | 26.844 | 410.3 days |

The model has 2,005,111,424 parameters. Wall-time arithmetic uses the live metrics
at step 54,360 on 2026-09-13 11:41:14 UTC: the preceding 3,000 optimizer steps
averaged **4.35589249 seconds/update**, or **30,090.7 effective tokens/second**,
including periodic evaluation and checkpoint overhead. Durations assume the
current two-GPU workstation, unchanged batch size and throughput, and uninterrupted
operation. They estimate the time to consume a budget, not the time to attain a
particular loss. From zero, the budgets would take 103.3 and 413.0 days.

The configured objective and prepared token file are still **40B tokens**.
The larger budgets require an explicit continuation schedule and data plan.
If the question meant additional steps, add the current consumed tokens and
about 2.74 days to the corresponding remaining-duration arithmetic.

## What the primary research establishes

Language-model learning curves often admit power-law approximations after an
initial transient. These describe diminishing returns: each additional billion
tokens generally produces a smaller gain. Published coefficients depend on the
model, tokenization, data, and experimental setup.
[Kaplan et al., Sections 4–5](https://arxiv.org/pdf/2001.08361).

Chinchilla fits endpoint loss as
`L(N,D) = E + A/N^alpha + B/D^beta`. At fixed model size this has the form
`L(D) = F + B/D^beta`, where `F` includes the model-size contribution as well as
irreducible uncertainty. Its compute-optimal parameter/data allocation does not
define a stopping point for a fixed model. Appendix B also shows that the cosine
schedule duration must be calibrated to the intended training duration.
[Hoffmann et al., Section 3.3 and Appendix B](https://arxiv.org/pdf/2203.15556).

Schedule-aware work separates continuing optimization from the extra improvement
caused by lowering the learning rate. Tissue et al. model validation loss using
both accumulated learning rate and an annealing-history term; their Appendix C.3
uses shorter constant and cosine runs to forecast a 1.7B model trained on 1.4T
tokens. Such calibrated experiments support forecasting, but do not establish
that one early training trace can identify a 150-fold extension.
[Scaling Law with Learning Rate Annealing](https://arxiv.org/html/2408.11029).

Constant-rate training followed by a terminal cooldown provides a way to compare
different training budgets while reusing checkpoints. A loss reached after
cooldown need not have the same subsequent slope as a loss reached at a high
learning rate.
[Hägele et al., Sections 2–3](https://arxiv.org/html/2405.18392v3).

Long training of small models can remain useful. Sardana et al. study a 2.46B
model up to 500 tokens/parameter, close to the larger proposed budget here. Their
study also warns that fits from shorter training can overstate the benefits of
much longer training. Its reported final training loss and unique-data setup
cannot be converted into WarpState validation loss 2.0.
[Sardana et al., Sections 3–4 and Appendix C](https://arxiv.org/html/2401.00448).
Gadre et al. successfully forecast a 1.4B/900B-token model using calibrated
experiments, further supporting the feasibility of forecasting with the right
measurements.
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a91869936a63d814971b6423990ecf6e-Abstract-Conference.html).

Repeated tokens have diminishing value compared with fresh tokens. Muennighoff
et al. find small penalties through roughly four epochs in their experiments,
with progressively smaller returns from further repetition. That finding is not
a universal epoch cutoff. Repeating our corpus 6.7 or 26.8 times is a different
experiment from acquiring 268B or 1.07T comparable fresh tokens.
[Scaling Data-Constrained Language Models](https://arxiv.org/html/2305.16264v3).

## Fits to WarpState: scenarios, not confidence intervals

Use actual evaluations from `work/run/validation.jsonl`, rather than the latest
validation value repeated in every training-metric row. Verify strictly increasing,
unique token counts, start at 1B tokens to exclude early execution changes, and
average complete 0.25B-token bins. Each bin receives equal weight, at its actual
mean token position. The frozen snapshot contains 24 complete bins from 1–7B;
the partial final bin is excluded from fitting.

Fit in loss space, using `t = tokens / 1e9`:

- Logarithmic: `L(t) = a + b*ln(t)`.
- Power law with a floor: `L(t) = F + A*t^(-beta)`, with `A >= 0`, `F >= 0`,
  and positive `beta`.
- Fixed-floor sensitivity: choose an illustrative `F` and fit `A` and `beta`.
  These chosen floors are assumptions, not measured properties of WarpState.

For the power-law scenario, a finite crossing exists only if `F < 2`, and its
location is `t = (A/(2-F))^(1/beta)`. A small uncertainty in `F` close to 2 thus
creates a very large uncertainty in the crossing time.

| Fit | Loss at 268.435B | Loss at 1.073742T | Crossing below 2.0 |
| --- | ---: | ---: | --- |
| Logarithmic, starts at 3/4/5B and ends at 7B | 1.68–1.85 | 1.40–1.64 | 53–103B |
| Power law, free floor, 1–7B history | 2.052 | approximately 2.000 | approximately 1.07T |
| Power law, free floor, 2–7B history | 2.096 | 2.058 | No finite crossing in this fitted curve |
| Power law, **assumed floor 1.5**, 1–7B history | 1.905 | 1.799 | 103B |
| Power law, **assumed floor 1.8**, 1–7B history | 2.001 | 1.932 | 272B |
| Power law, **assumed floor 1.9**, 1–7B history | 2.043 | 1.989 | 762B |

The 1–7B free-floor fit is approximately
`L(t) = 1.918124 + 0.975783*t^(-0.355255)`.
The 2–7B fit has floor 2.009366 and exponent 0.419261. Neither fitted floor is a
measured asymptotic limit. Shifting the start to 3, 4, or 5B yields floors around
2.19–2.33, showing how poorly the eventual floor is identified by this history.
The numerical 1.07T crossing should not be interpreted as a precise prediction
or as evidence that the larger budget was chosen correctly.

The logarithmic fit eventually predicts unboundedly negative loss, so it is a
local trend approximation. It is included because it illustrates an optimistic
extrapolation, not because it is a suitable global asymptotic law. Small residuals
on fitted data also do not establish long-range predictive accuracy.

Historical forward checks fit an earlier interval and predict the next complete
billion tokens. The table shows mean predicted minus observed loss; negative
values mean the forecast was too optimistic.

| Fit interval → held-out interval | Logarithmic bias | Free-floor power-law bias |
| --- | ---: | ---: |
| 2–4B → 4–5B | -0.00197 | +0.00695 |
| 2–5B → 5–6B | -0.00604 | -0.00194 |
| 2–6B → 6–7B | -0.01665 | -0.01023 |

Neither family wins every historical comparison. Both overpredicted the latest
interval's improvement, particularly the logarithmic fit. These checks use
0.25B-bin means, so their residuals should not be confused with errors on raw
individual validation points.

## Why there is no dependable crossing date yet

The observations extend to about 7.1B tokens, whereas the requested budgets are
roughly **38 and 151 times** that amount. Historical one-billion-token holdouts
can reject some overly optimistic local trends; they cannot validate such a
large extrapolation. Successive measurements are correlated observations of one
training run, not independent experimental replications.

The run still uses a learning rate near `1.40e-4`, about 93% of its `1.50e-4`
peak, with a planned 40B-token cosine decay toward `1.50e-5`. The simple fits
above do not model that future annealing. Cooldown may improve loss beyond their
predictions; a changed continuation schedule or repeated data may produce a
different curve. There is no fitted magnitude for those effects in WarpState.

Current evaluation samples eight fixed sequences of length 2,048: **16,384
target tokens**, drawn at fixed spaced positions from the 1,048,576-token
held-out file. A single reported value below 2 would be weaker evidence than a
sustained crossing and confirmation on more held-out tokens. Absolute loss 2
also cannot be compared directly with another paper's tokenizer or validation
distribution, and does not by itself determine downstream task performance.

The useful next forecast experiment is a matched control and cooldown branch
from the same full checkpoint, retaining optimizer state, data order, batch
geometry, BF16 computation, and the same evaluation metric. Evaluate additional
held-out tokens separately, preserving the historical metric for comparison.
Combining those measurements with later 10–15B-token checkpoints would help
separate continued-learning gains from annealing gains and allow an actual
out-of-sample test of a schedule-aware model. This document recommends that
experiment; it does not launch it.

For planning, treat sub-2.0 loss as an unresolved target. **268B tokens might be
enough, and 1.07T provides much more training, but the present evidence does not
justify promising either budget will cross the threshold.**

## Reproduction

The input snapshot SHA256 is
`13f7fa766a59b884d376515132e5f41fe07b771438e7e66ed294dce030e9f38e`.
The analysis program is `ops/loss_curve_forecast.py`. It requires NumPy;
Matplotlib is optional for the figure. The generated JSON records dependency
versions and hashes of both the snapshot and analysis source. The isolated
`work/analysis-venv` keeps analysis dependencies separate from training.
This analysis used Python 3.12.3, NumPy 2.5.3, and Matplotlib 3.11.2.

From `/workspace/WarpState/rtx6000_40b`:

```bash
source ops/training-env.sh
python3.12 -m venv work/analysis-venv  # Only when rebuilding an absent environment.
work/analysis-venv/bin/python -m pip install 'numpy==2.5.3' 'matplotlib==3.11.2'
work/analysis-venv/bin/python -B ops/loss_curve_forecast.py \
  --input work/validation/loss-forecast-20260913/validation-snapshot.jsonl \
  --output work/validation/loss-forecast-20260913/forecast.json \
  --plot work/validation/loss-forecast-20260913/loss-scenarios.png
```

The same command with an `.svg` plot extension generates a vector figure.
Use a new output directory when analyzing a later frozen snapshot so that the
original forecast remains available for comparison with subsequent observations.
