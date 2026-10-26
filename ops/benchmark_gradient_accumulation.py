#!/usr/bin/env python3
"""Disposable BF16 weight-gradient GEMM/FP32 accumulation comparisons.

Source ops/training-env.sh first. Uses deterministic random matrices only; no
checkpoint or dataset access. None of these experiments is promoted to training.
Direct FP32-output addmm changes rounding versus BF16 GEMM then FP32 accumulation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from benchmark_gemm import FAMILIES, GIB, PACKAGE, WORKSPACE, gpu_telemetry, projection_cases

VARIANTS = ("baseline", "fused_cast_add", "direct_fp32_addmm")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        default=["ffn_up_gate", "in_proj", "ffn_down"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2, help="Complete accumulation cycles")
    parser.add_argument("--repeats", type=int, default=10, help="Cycles per timed trial")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--burnin-seconds", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--blas-backend", choices=["default", "cublas", "cublaslt"], default="default")
    parser.add_argument("--allow-reduced-precision-reduction", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--list-shapes", action="store_true", help="CPU-only validation and shape listing")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "work" /
                        ("benchmark-gradient-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")))
    args = parser.parse_args(argv)
    for key in ("rows", "accumulation_steps", "warmup", "repeats", "trials"):
        if getattr(args, key) < 1:
            parser.error(key + " must be positive")
    if args.device < 0 or not math.isfinite(args.burnin_seconds) or args.burnin_seconds < 0:
        parser.error("device must be nonnegative; burnin-seconds must be finite and nonnegative")
    for key in ("families", "variants"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(key + " must not contain duplicates")
    args.output_dir = args.output_dir.expanduser().resolve()
    work = PACKAGE / "work"
    if (not args.output_dir.is_relative_to(work) or not args.output_dir.relative_to(work).parts or
            not args.output_dir.relative_to(work).parts[0].startswith("benchmark-")):
        parser.error("output-dir must be below rtx6000_40b/work/benchmark-*")
    if not args.list_shapes:
        for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(WORKSPACE):
                parser.error("Source ops/training-env.sh first; " + key + " must stay in the workspace")
    return args


def accumulation_cycle(pairs, create, add):
    """Adopt the first newly allocated gradient, then add subsequent gradients."""
    if not pairs:
        raise ValueError("At least one gradient fixture is required")
    accumulator = create(*pairs[0])
    for a, b in pairs[1:]:
        add(accumulator, a, b)
    return accumulator


def main():
    args = arguments()
    cases = projection_cases(args.families, args.rows, ["dw"])
    if args.list_shapes:
        print(json.dumps(cases, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for timings")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.allow_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library(args.blas_backend)
    report = {"started_utc": dt.datetime.now(dt.UTC).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_source_sha256": hashlib.sha256(Path(__file__).with_name("benchmark_gemm.py").read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "device": torch.cuda.get_device_name(device), "input_dtype": "bfloat16",
              "accumulator_dtype": "float32", "results": [],
              "notes": ["Eight deterministic, distinct BF16 gradient input pairs are used by default; every variant uses the same pairs.",
                        "Each cycle adopts its first newly allocated FP32 gradient, then performs seven in-place additions by default; there is no zero reset.",
                        "Timings include first-gradient creation and subsequent accumulation; each cycle output is released before the next cycle.",
                        "Effective TFLOPS counts only 2MNK GEMM FLOPs over this complete cycle time.",
                        "Direct FP32 addmm skips BF16 output rounding and is not numerically equivalent to baseline.",
                        "Experiments do not implement autograd/DDP integration and are not promoted to training."]}

    def save():
        partial = args.output_dir / "results.json.partial"
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        partial.replace(args.output_dir / "results.json")

    def baseline(acc, a, b):
        acc.add_(torch.mm(a, b).float())

    def baseline_create(a, b):
        return torch.mm(a, b).float()

    save()
    with torch.inference_mode():
        for index, case in enumerate(cases):
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            pairs = [(torch.randn((case["k"], case["m"]), device=device, dtype=torch.bfloat16, generator=generator).T,
                      torch.randn(case["b_shape"], device=device, dtype=torch.bfloat16, generator=generator))
                     for _ in range(args.accumulation_steps)]
            assert all(list(a.stride()) == case["a_stride"] and list(b.stride()) == case["b_stride"] for a, b in pairs)
            reference = accumulation_cycle(pairs, baseline_create, baseline)
            for variant in args.variants:
                row = {**case, "variant": variant, "status": "running", "telemetry_before": gpu_telemetry(),
                       "gemm_output_dtype": "float32" if variant == "direct_fp32_addmm" else "bfloat16",
                       "preserves_baseline_bf16_rounding": variant != "direct_fp32_addmm"}
                print(json.dumps({"case": case["name"], "variant": variant, "status": "starting"}), flush=True)
                try:
                    create = baseline_create
                    if variant == "baseline":
                        operation = baseline
                    elif variant == "fused_cast_add":
                        torch._dynamo.reset()
                        cast_add = torch.compile(lambda dest, grad: dest.add_(grad.float()),
                                                 fullgraph=True, dynamic=False)
                        operation = lambda dest, x, y: cast_add(dest, torch.mm(x, y))
                    else:
                        create = lambda x, y: torch.mm(x, y, out_dtype=torch.float32)
                        operation = lambda dest, x, y: torch.addmm(
                            dest, x, y, beta=1, out_dtype=torch.float32, out=dest)

                    def cycle():
                        return accumulation_cycle(pairs, create, operation)

                    first_start = time.perf_counter()
                    output = cycle()
                    torch.cuda.synchronize(device)
                    row["first_cycle_seconds"] = time.perf_counter() - first_start
                    difference = output.sub(reference)
                    row["max_abs_error"] = difference.abs().max().item()
                    row["rms_error"] = difference.square_().mean().sqrt().item()
                    row["relative_rms_error"] = row["rms_error"] / max(reference.square().mean().sqrt().item(), 1e-30)
                    if not all(math.isfinite(row[key]) for key in ("max_abs_error", "rms_error", "relative_rms_error")):
                        raise RuntimeError("Nonfinite output difference")
                    del difference, output
                    for _ in range(args.warmup):
                        output = cycle()
                        del output
                    torch.cuda.synchronize(device)
                    burnin_start = time.perf_counter()
                    burnin_cycles = 0
                    while time.perf_counter() - burnin_start < args.burnin_seconds:
                        output = cycle()
                        del output
                        torch.cuda.synchronize(device)
                        burnin_cycles += 1
                    row["burnin_elapsed_seconds"] = time.perf_counter() - burnin_start
                    row["burnin_cycles"] = burnin_cycles
                    torch.cuda.reset_peak_memory_stats(device)
                    row["baseline_allocated_gib"] = torch.cuda.memory_allocated(device) / GIB
                    times = []
                    for _ in range(args.trials):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.repeats):
                            output = cycle()
                            del output
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end) / args.repeats)
                    row["trial_ms_per_cycle"] = times
                    row["median_ms_per_cycle"] = statistics.median(times)
                    row["median_ms_per_accumulation"] = row["median_ms_per_cycle"] / args.accumulation_steps
                    row["effective_tflops"] = case["flops"] * args.accumulation_steps / (row["median_ms_per_cycle"] * 1e9)
                    row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / GIB
                    row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / GIB
                    row["status"] = "ok"
                except Exception as error:
                    row["status"] = "error"
                    row["error"] = type(error).__name__ + ": " + str(error)[:4000]
                row["telemetry_after"] = gpu_telemetry()
                report["results"].append(row)
                save()
                print(json.dumps({key: row[key] for key in ("name", "variant", "status", "median_ms_per_cycle",
                                  "effective_tflops", "max_abs_error", "relative_rms_error", "error") if key in row}), flush=True)
            del pairs, reference
            torch.cuda.empty_cache()
    report["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    save()
    print("Results: " + str(args.output_dir / "results.json"), flush=True)
    if any(row["status"] != "ok" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
