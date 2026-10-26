#!/usr/bin/env python3
"""Illustrative loss-curve sensitivity; CPU only, no training/checkpoint access.

Use a frozen validation JSONL snapshot. NumPy is required; --plot also requires
Matplotlib. Fits describe the observed schedule and do not establish a loss
floor, confidence interval, or ETA for a different training schedule.
"""

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np


BIN_WIDTH = 0.25  # Billions of tokens.
WINDOW_STARTS = (1, 2, 3, 4, 5)
FIXED_FLOORS = (1.5, 1.8, 1.9, 2.0, 2.1)
BUDGETS = {"268.435456B": 268.435456, "1073.741824B": 1073.741824}
MODELS = ("log", "power_free") + tuple(f"power_fixed:{f}" for f in FIXED_FLOORS)


def power_parameters(x, y, alpha, fixed_floor):
    z = x ** (-alpha)
    if fixed_floor is None:
        centered = z - z.mean()
        amplitude = float(centered @ (y - y.mean()) / (centered @ centered))
        floor = float(np.clip(y.mean() - amplitude * z.mean(), 0, y.min()))
    else:
        floor = fixed_floor
    amplitude = max(0.0, float(z @ (y - floor) / (z @ z)))
    mse = float(np.mean((floor + amplitude * z - y) ** 2))
    return mse, floor, amplitude


def fit(x, y, model):
    if len(x) < 3:
        raise ValueError("Each fitting window needs at least three complete bins")
    if model == "log":
        z = np.log(x)
        centered = z - z.mean()
        slope = float(centered @ (y - y.mean()) / (centered @ centered))
        intercept = float(y.mean() - slope * z.mean())
        mse = float(np.mean((intercept + slope * z - y) ** 2))
        return {"model": model, "intercept": intercept, "slope": slope,
                "rmse": math.sqrt(mse)}
    fixed = None if model == "power_free" else float(model.split(":")[1])
    grid = np.geomspace(0.0001, 3, 1000)
    errors = [power_parameters(x, y, a, fixed)[0] for a in grid]
    index = int(np.argmin(errors))
    low, high = grid[max(index - 1, 0)], grid[min(index + 1, len(grid) - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(55):
        p, q = high - ratio * (high - low), low + ratio * (high - low)
        if power_parameters(x, y, p, fixed)[0] < power_parameters(x, y, q, fixed)[0]:
            high = q
        else:
            low = p
    alpha = float((low + high) / 2)
    mse, floor, amplitude = power_parameters(x, y, alpha, fixed)
    return {"model": model, "floor": floor, "amplitude": amplitude,
            "alpha": alpha, "rmse": math.sqrt(mse),
            "alpha_at_search_boundary": index in (0, len(grid) - 1)}


def predict(parameters, tokens_b):
    x = np.asarray(tokens_b)
    if parameters["model"] == "log":
        return parameters["intercept"] + parameters["slope"] * np.log(x)
    return parameters["floor"] + parameters["amplitude"] * x ** (-parameters["alpha"])


def crossing(parameters, target=2.0):
    if parameters["model"] == "log":
        if parameters["slope"] >= 0:
            return None
        exponent = (target - parameters["intercept"]) / parameters["slope"]
    else:
        if parameters["floor"] >= target or parameters["amplitude"] <= 0:
            return None
        exponent = math.log(parameters["amplitude"] / (target - parameters["floor"]))
        exponent /= parameters["alpha"]
    return math.exp(exponent) if exponent < 700 else None


def analyze(raw):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(b["tokens"] <= a["tokens"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Validation snapshot must have strictly increasing token counts")
    right = math.floor(rows[-1]["tokens"] / 1e9 / BIN_WIDTH) * BIN_WIDTH
    bins = []
    for left in np.arange(1, right, BIN_WIDTH):
        selected = [r for r in rows if left <= r["tokens"] / 1e9 < left + BIN_WIDTH]
        if not selected:
            raise ValueError(f"Missing validation data in bin starting at {left}B")
        bins.append({"left_B": float(left), "right_B": float(left + BIN_WIDTH),
                     "tokens_B": float(np.mean([r["tokens"] / 1e9 for r in selected])),
                     "loss": float(np.mean([r["loss"] for r in selected])),
                     "count": len(selected)})
    x = np.array([b["tokens_B"] for b in bins])
    y = np.array([b["loss"] for b in bins])
    fits = []
    for start in WINDOW_STARTS:
        selected = x >= start
        for model in MODELS:
            parameters = fit(x[selected], y[selected], model)
            fits.append({"start_B": start, "end_B": right, "bins": int(selected.sum()),
                         "parameters": parameters, "cross_2_tokens_B": crossing(parameters),
                         "endpoint_losses": {k: float(predict(parameters, v))
                                             for k, v in {"40B": 40, **BUDGETS}.items()}})
    holdouts = []
    for end in (4, 5, 6):
        train, test = (x >= 2) & (x < end), (x >= end) & (x < end + 1)
        if not test.any() or right < end + 1:
            raise ValueError("Requested holdouts require complete validation bins through 7B")
        for model in MODELS:
            parameters = fit(x[train], y[train], model)
            difference = predict(parameters, x[test]) - y[test]
            holdouts.append({"train_start_B": 2, "train_end_B": end, "test_end_B": end + 1,
                             "parameters": parameters, "actual_mean_loss": float(y[test].mean()),
                             "predicted_mean_loss": float(predict(parameters, x[test]).mean()),
                             "mean_bias": float(difference.mean()),
                             "rmse": float(np.sqrt(np.mean(difference ** 2)))})
    return {"method": "Equal-weight least squares in loss space on completed 0.25B token bins; "
            "x is each bin's actual mean token count. Log: L=a+b*ln(t). Power: L=f+A*t^-alpha. "
            "Tokens t are in billions. Free floor constrained to [0, minimum binned loss], A>=0; "
            "alpha searched on [0.0001,3] with 1000 logarithmic points and 55 golden refinements. "
            "Fixed floors are sensitivity assumptions, not measured irreducible losses. "
            "Holdouts fit 2B to 4/5/6B and predict the next complete 1B.",
            "caveats": ["Illustrative extrapolations, not confidence intervals or reliable ETAs.",
                        "Short in-distribution holdouts do not validate a 38x or 151x token extrapolation.",
                        "Current schedule, changing LR, data repetition, and distribution changes are not modeled.",
                        "A fitted floor at or above two does not prove an architectural limit.",
                        "Null crossing means this fitted curve has no finite crossing or exceeds numeric range."],
            "source_sha256": hashlib.sha256(raw).hexdigest(), "source_rows": len(rows),
            "latest_validation": rows[-1], "complete_bins_end_B": right,
            "observed_validation": [{"tokens_B": r["tokens"] / 1e9, "loss": r["loss"]}
                                    for r in rows if r["tokens"] >= 1e9],
            "budget_steps": {"268.435456B": 8000 * 256, "1073.741824B": 8000 * 1024},
            "tokens_per_update": 131072, "bins": bins, "fits": fits, "holdouts": holdouts}


def plot(report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([b["tokens_B"] for b in report["bins"]])
    y = np.array([b["loss"] for b in report["bins"]])
    end = report["complete_bins_end_B"]
    maximum = max(1100, end * 1.25)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    scenarios = (("power_free", 1, "#2469aa", "Power, fitted floor"),
                 ("power_free", 2, "#8051a5", "Power, fitted floor"),
                 ("log", 3, "#ca6d12", "Log-token"),
                 ("log", 5, "#cfa323", "Log-token"),
                 ("power_fixed:1.8", 1, "#238456", "Power, ASSUMED floor 1.8"))
    raw_x = [r["tokens_B"] for r in report["observed_validation"]]
    raw_y = [r["loss"] for r in report["observed_validation"]]
    axes[0].scatter(raw_x, raw_y, color="gray", alpha=.35, s=6, label="Individual validations")
    for axis in axes:
        axis.plot(x, y, ".-", color="black", lw=1.1, ms=4, label="Complete 0.25B-bin means")
        axis.set_xlabel("Tokens trained (billions)")
        axis.set_ylabel("Validation cross-entropy loss")
        axis.grid(alpha=.2)
    for model, start, color, label in scenarios:
        label += f"; fit {start}–{end:g}B"
        parameters = next(r["parameters"] for r in report["fits"]
                          if r["start_B"] == start and r["parameters"]["model"] == model)
        observed = np.linspace(start, end, 150)
        future = np.geomspace(end, maximum, 350)
        if model == "power_free":
            axes[0].plot(observed, predict(parameters, observed), color=color, lw=1, label=label)
        axes[1].plot(observed, predict(parameters, observed), color=color, lw=1, alpha=.6)
        axes[1].plot(future, predict(parameters, future), color=color, ls="--", lw=1.5, label=label)
    axes[0].set_title("Observed losses and fitted power curves")
    axes[0].set_xlim(1, max(raw_x) + BIN_WIDTH * .15)
    axes[0].set_ylim(min(raw_y) - .05, max(raw_y) + .05)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1, maximum)
    axes[1].set_ylim(1.1, 3)
    axes[1].axhline(2, color="#8f3333", ls="--", lw=1, label="Loss 2.0")
    axes[1].axvspan(end, maximum, color="gray", alpha=.05)
    for value, label in ((40, "40B schedule"), (268.435456, "268B budget"),
                         (1073.741824, "1.074T budget")):
        axes[1].axvline(value, color="gray", ls=":", lw=1)
        axes[1].text(value, 2.9, label, rotation=90, ha="right", va="top", fontsize=8)
    axes[1].set_title("Long extrapolation: strongly dependent on assumed curve")
    axes[1].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("WarpState: illustrative loss scenarios, NOT confidence intervals or reliable ETAs",
                 fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frozen validation JSONL snapshot")
    parser.add_argument("--output", required=True, type=Path, help="Generated JSON analysis")
    parser.add_argument("--plot", type=Path, help="Optional generated PNG/PDF/SVG chart")
    args = parser.parse_args()
    paths = [args.input.resolve(), args.output.resolve()]
    if args.plot:
        paths.append(args.plot.resolve())
    if len(paths) != len(set(paths)):
        parser.error("Input snapshot, JSON output, and plot must use distinct file paths")
    report = analyze(args.input.read_bytes())
    report["source_path"] = str(args.input)
    report["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__}
    if args.plot:
        report["versions"]["matplotlib"] = plot(report, args.plot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Analyzed {report['source_rows']} validation rows through "
          f"{report['complete_bins_end_B']:g}B complete bins; wrote {args.output}")


if __name__ == "__main__":
    main()
