#!/usr/bin/env python3
"""Time isolated BF16 GEMMs with the dominant WarpState projection layouts.

Source ops/training-env.sh first. Only deterministic random fixtures are used;
this program never opens a checkpoint or token dataset. Compilation/autotuning
are excluded from CUDA-event timings. These timings exclude communication,
optimizer updates and most Python overhead, so are not training throughput.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
GIB = 1024**3
FAMILIES = {"ffn_up_gate": (2048, 14336), "in_proj": (2048, 10240),
            "ffn_down": (7168, 2048), "out_proj": (2048, 2048)}
BACKENDS = {"eager": None, "aten": "ATEN", "triton": "TRITON", "mixed": "ATEN,TRITON"}


def projection_cases(families, rows, operations):
    """Return M,N,K and storage layouts for Y=X@W.T, dX=dY@W, dW=dY.T@X."""
    cases = []
    for family in families:
        input_dim, output_dim = FAMILIES[family]
        specs = {
            "fwd": (rows, output_dim, input_dim, False, True),
            "dx": (rows, input_dim, output_dim, False, False),
            "dw": (output_dim, input_dim, rows, True, False),
        }
        for operation in operations:
            m, n, k, at, bt = specs[operation]
            cases.append({"name": family + "." + operation, "family": family,
                          "operation": operation, "m": m, "n": n, "k": k,
                          "a_transposed": at, "b_transposed": bt,
                          "a_shape": [m, k], "b_shape": [k, n],
                          "a_stride": [1, m] if at else [k, 1],
                          "b_stride": [1, k] if bt else [n, 1],
                          "flops": 2 * m * n * k})
    return cases


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        default=["ffn_up_gate", "in_proj", "ffn_down"])
    parser.add_argument("--operations", nargs="+", choices=["fwd", "dx", "dw"],
                        default=["fwd", "dx", "dw"])
    parser.add_argument("--rows", type=int, default=8192,
                        help="Micro-batch times sequence length; production uses 8192")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        default=["eager", "aten", "triton"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--burnin-seconds", type=float, default=2.0,
                        help="Sustain GEMM load before timing to reduce cold boost effects")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index after CUDA_VISIBLE_DEVICES filtering")
    parser.add_argument("--blas-backend", choices=["default", "cublas", "cublaslt"],
                        default="default")
    parser.add_argument("--allow-reduced-precision-reduction", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Allow BF16 intermediate reductions, matching the current production setting")
    parser.add_argument("--list-shapes", action="store_true",
                        help="Print selected shapes/strides and exit without importing torch")
    parser.add_argument("--output-dir", type=Path,
                        default=PACKAGE / "work" / ("benchmark-gemm-" +
                                dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")))
    args = parser.parse_args(argv)
    for key in ("rows", "warmup", "repeats", "trials"):
        if getattr(args, key) < 1:
            parser.error("--" + key + " must be positive")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if not math.isfinite(args.burnin_seconds) or args.burnin_seconds < 0:
        parser.error("--burnin-seconds must be finite and nonnegative")
    for key in ("families", "operations", "backends"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error("--" + key + " must not contain duplicates")
    args.output_dir = args.output_dir.expanduser().resolve()
    work = PACKAGE / "work"
    if (not args.output_dir.is_relative_to(work) or
            not args.output_dir.relative_to(work).parts or
            not args.output_dir.relative_to(work).parts[0].startswith("benchmark-")):
        parser.error("--output-dir must be below rtx6000_40b/work/benchmark-*")
    if not args.list_shapes:
        for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
                    "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(WORKSPACE):
                parser.error("Source ops/training-env.sh first; " + key + " must stay in the workspace")
    return args


def gpu_telemetry():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,power.draw,power.limit,clocks.sm,"
             "clocks.mem,temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        return {"csv": result.stdout.strip(), "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    args = arguments()
    cases = projection_cases(args.families, args.rows, args.operations)
    if args.list_shapes:
        print(json.dumps(cases, indent=2))
        return
    # Reject an existing directory to preserve previous measurements.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from torch._inductor import config

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for timings; --list-shapes is CPU-only")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.allow_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library(args.blas_backend)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "dtype": "bfloat16", "output_dtype": "bfloat16",
        "allow_bf16_reduced_precision_reduction": args.allow_reduced_precision_reduction,
        "compile_mode": "max-autotune-no-cudagraphs",
        "autotune_fallback_to_aten": False,
        "device": {"name": properties.name, "capability": list(torch.cuda.get_device_capability(device)),
                   "multiprocessors": properties.multi_processor_count,
                   "total_memory_gib": properties.total_memory / GIB},
        "environment": {key: os.environ.get(key) for key in
                        ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                         "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "PYTORCH_ALLOC_CONF",
                         "TORCHINDUCTOR_COMPILE_THREADS", "ENABLE_PERSISTENT_TMA_MATMUL",
                         "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
                         "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "CUTLASS_EPILOGUE_FUSION")},
        "telemetry_before": gpu_telemetry(), "results": [],
        "notes": ["Shapes represent isolated linear algebra, not an end-to-end backward graph.",
                  "Each repeat allocates a BF16 output, as torch.mm normally does.",
                  "Timings reuse inputs and omit compilation/autotuning; compare to full training separately.",
                  "Error is against eager cuBLAS BF16 output, not an exact FP32 reference.",
                  "Telemetry is a snapshot, not a synchronized power integral."],
    }

    def save():
        target = args.output_dir / "results.json"
        temporary = args.output_dir / "results.json.partial"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    save()
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            generator = torch.Generator(device=device).manual_seed(args.seed + case_index)

            def fixture(rows, columns, transposed):
                shape = (columns, rows) if transposed else (rows, columns)
                tensor = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
                return tensor.T if transposed else tensor

            a = fixture(case["m"], case["k"], case["a_transposed"])
            b = fixture(case["k"], case["n"], case["b_transposed"])
            assert list(a.stride()) == case["a_stride"]
            assert list(b.stride()) == case["b_stride"]
            reference = torch.mm(a, b)
            torch.cuda.synchronize(device)
            for backend in args.backends:
                row = {**case, "backend": backend, "status": "running",
                       "max_autotune_gemm_backends": BACKENDS[backend],
                       "telemetry_before": gpu_telemetry()}
                print(json.dumps({"case": case["name"], "backend": backend, "status": "starting"}), flush=True)
                try:
                    torch._dynamo.reset()
                    settings = ({"max_autotune_gemm_backends": BACKENDS[backend],
                                 "autotune_fallback_to_aten": False}
                                if BACKENDS[backend] else {})
                    with config.patch(settings) if settings else contextlib.nullcontext():
                        operation = torch.mm if backend == "eager" else torch.compile(
                            lambda x, y: torch.mm(x, y), fullgraph=True, dynamic=False,
                            mode="max-autotune-no-cudagraphs")
                        compile_start = time.perf_counter()
                        output = operation(a, b)
                        torch.cuda.synchronize(device)
                        row["first_call_seconds"] = time.perf_counter() - compile_start
                        difference = output.float().sub_(reference.float())
                        reference_rms = reference.float().square_().mean().sqrt().item()
                        row["max_abs_error"] = difference.abs().max().item()
                        row["rms_error"] = difference.square_().mean().sqrt().item()
                        row["relative_rms_error"] = row["rms_error"] / max(reference_rms, 1e-30)
                        if not all(math.isfinite(row[key]) for key in
                                   ("max_abs_error", "rms_error", "relative_rms_error")):
                            raise RuntimeError("Nonfinite output error")
                        del difference, output
                        for _ in range(args.warmup):
                            output = operation(a, b)
                        del output
                        torch.cuda.synchronize(device)
                        burnin_start = time.perf_counter()
                        burnin_iterations = 0
                        while time.perf_counter() - burnin_start < args.burnin_seconds:
                            for _ in range(10):
                                output = operation(a, b)
                            del output
                            torch.cuda.synchronize(device)
                            burnin_iterations += 10
                        row["burnin_elapsed_seconds"] = time.perf_counter() - burnin_start
                        row["burnin_iterations"] = burnin_iterations
                        torch.cuda.reset_peak_memory_stats(device)
                        row["baseline_allocated_gib"] = torch.cuda.memory_allocated(device) / GIB
                        trial_ms = []
                        for _ in range(args.trials):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(args.repeats):
                                output = operation(a, b)
                            end.record()
                            end.synchronize()
                            trial_ms.append(start.elapsed_time(end) / args.repeats)
                            del output
                        row["trial_ms_per_gemm"] = trial_ms
                        row["median_ms"] = statistics.median(trial_ms)
                        row["min_ms"] = min(trial_ms)
                        row["tflops"] = case["flops"] / (row["median_ms"] * 1e9)
                        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / GIB
                        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / GIB
                        row["telemetry_after"] = gpu_telemetry()
                        row["status"] = "ok"
                except Exception as error:
                    row["status"] = "error"
                    row["error"] = type(error).__name__ + ": " + str(error)[:4000]
                report["results"].append(row)
                save()
                print(json.dumps({key: row[key] for key in
                                  ("name", "backend", "status", "median_ms", "tflops",
                                   "max_abs_error", "rms_error", "error") if key in row}), flush=True)
            del a, b, reference
            torch.cuda.empty_cache()
    report["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    report["telemetry_after"] = gpu_telemetry()
    save()
    print("Results: " + str(args.output_dir / "results.json"), flush=True)
    if any(row["status"] != "ok" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
