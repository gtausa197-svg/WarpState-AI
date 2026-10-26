#!/usr/bin/env python3
"""Reproduce isolated BF16 rotary/packed-gradient traffic measurements on one GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=["paired", "single"], default=["paired", "single"])
    parser.add_argument("--blocks", nargs="+", type=int, choices=[128, 256, 512, 1024], default=[256, 512, 1024])
    parser.add_argument("--warps", nargs="+", type=int, choices=[4, 8], default=[4, 8])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--burnin-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("trials", "repeats", "warmup"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if not math.isfinite(args.burnin_seconds) or args.burnin_seconds < 0:
        parser.error("burnin-seconds must be finite and nonnegative")
    args.output_dir = args.output_dir.expanduser().resolve()
    work = PACKAGE / "work"
    if (not args.output_dir.is_relative_to(work) or args.output_dir == work
            or not args.output_dir.relative_to(work).parts[0].startswith("benchmark-")):
        parser.error("output-dir must be below rtx6000_40b/work/benchmark-*")
    for name in ("TMPDIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(WORKSPACE):
            parser.error("Source ops/training-env.sh; " + name + " must stay inside the workspace")
    return args


def output_errors(actual, expected):
    """Check each projection independently so a bad derivative cannot hide."""
    import torch
    result = {}
    for name, value, wanted in zip(("q", "k", "v", "gate", "u"),
                                   actual.chunk(5, -1), expected.chunk(5, -1), strict=True):
        delta = value.float() - wanted.float()
        finite = bool(torch.isfinite(value).all())
        result[name] = {"finite": finite,
                        "unequal": int((value != wanted).sum()),
                        "bitwise_unequal": int((value.view(torch.int16) != wanted.view(torch.int16)).sum()),
                        "relative_rms": (float(delta.norm() / wanted.float().norm().clamp_min(1e-30))
                                         if finite else None),
                        "max_abs": float(delta.abs().max()) if finite else None}
    return result


def main(argv=None):
    args = arguments(argv)
    import torch
    from ops.backward_layout import PROFILE_SHAPE, make_inputs, prepare_layout, reference_layout
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Reserve exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    inputs = make_inputs(PROFILE_SHAPE, device="cuda", seed=args.seed)
    baseline_output = torch.empty((4, 2048, 10240), device="cuda", dtype=torch.bfloat16)
    candidate_output = torch.empty_like(baseline_output)
    scratch = tuple(torch.empty_like(inputs.q_memory) for _ in range(2))
    baseline = prepare_layout(inputs, "profile", output=baseline_output, scratch=scratch)
    baseline()
    independent = reference_layout(inputs)
    baseline_errors = output_errors(baseline_output, independent)
    del independent
    if any(not v["finite"] or v["relative_rms"] > 0.002 for v in baseline_errors.values()):
        raise RuntimeError("Tracked profile arithmetic disagrees with independent tensor reference")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = ["ops/backward_layout.py", "ops/backward_layout_baseline.py",
               "ops/backward_layout_benchmark.py", "warpstate/model.py"]
    report = {"format": "warpstate-backward-layout-v1", "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "shape_b_n_h_c_d": PROFILE_SHAPE, "dtype": "bfloat16", "rope_dtype": "float32",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_sha256": {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() for name in sources},
        "baseline_reference_errors": baseline_errors,
        "static_removed_scratch_store_and_read_bytes_per_call": inputs.q_memory.numel() * 2 * 4,
        "limitations": ["Isolated backward region, not a measured training speedup.",
            "Preallocated outputs and scratch; CUDA event times include launch scheduling.",
            "Synthetic saved inputs are reused; no model/checkpoint/token files are read.",
            "Static removed scratch traffic is not a measured bandwidth counter.",
            "Production reuses one dead buffer; this is not a claim of two saved allocations.",
            "Integration must preserve separate local/memory gradients and rounding joins."], "results": []}

    def save():
        path = args.output_dir / "results.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def elapsed_ms(operation):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.repeats):
            operation()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / args.repeats

    for algorithm in args.algorithms:
        for block in args.blocks:
            for warps in args.warps:
                row = {"algorithm": algorithm, "block": block, "warps": warps}
                try:
                    candidate = prepare_layout(inputs, algorithm, block=block, warps=warps,
                                               output=candidate_output)
                    candidate()
                    row["errors"] = output_errors(candidate_output, baseline_output)
                    if any(not v["finite"] for v in row["errors"].values()):
                        raise RuntimeError("Nonfinite candidate output")
                    row["bitwise_equal"] = all(v["bitwise_unequal"] == 0 for v in row["errors"].values())
                    for _ in range(args.warmup):
                        baseline()
                        candidate()
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    while time.monotonic() - begin < args.burnin_seconds:
                        for _ in range(10):
                            baseline()
                            candidate()
                        torch.cuda.synchronize()
                    trials = []
                    for _ in range(args.trials):
                        trials.append({"baseline_before_ms": elapsed_ms(baseline),
                                       "candidate_ms": elapsed_ms(candidate),
                                       "baseline_after_ms": elapsed_ms(baseline)})
                    gains = [((t["baseline_before_ms"] + t["baseline_after_ms"]) / 2
                              / t["candidate_ms"] - 1) * 100 for t in trials]
                    row.update(status="ok", trials=trials,
                               median_candidate_ms=statistics.median(t["candidate_ms"] for t in trials),
                               median_baseline_ms=statistics.median((t["baseline_before_ms"]
                                    + t["baseline_after_ms"]) / 2 for t in trials),
                               median_gain_percent=statistics.median(gains), minimum_gain_percent=min(gains))
                except Exception as error:
                    row.update(status="error", error=type(error).__name__ + ": " + str(error)[:4000])
                report["results"].append(row)
                save()
                print(json.dumps(row), flush=True)
    report["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    save()
    if not any(row["status"] == "ok" for row in report["results"]):
        raise RuntimeError("No candidate completed")


if __name__ == "__main__":
    main()
