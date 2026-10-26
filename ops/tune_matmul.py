#!/usr/bin/env python3
"""Screen BF16 FFN GEMM/epilogue tiles with alternating warmed ATen controls.

No model/checkpoint/token files are opened. Generated policy entries are
isolated-kernel candidates and require full-model validation before promotion.
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
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mm", "swiglu"], default="mm")
    parser.add_argument("--operations", nargs="+", choices=["fwd", "dw"])
    parser.add_argument("--algorithms", nargs="+", choices=["tiled", "persistent", "persistent_tma"],
                        help="defaults to all supported algorithms for the selected kind")
    parser.add_argument("--tile", action="append", metavar="BM,BN,BK,WARPS,STAGES",
                        help="repeat to select explicit tiles instead of the bounded default sweep")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--burnin-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--maximum-relative-rms", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--input-scale", type=float, default=1.0,
                        help="multiply both BF16 random operands by this positive scalar")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "work" /
                        ("benchmark-matmul-tiles-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")))
    args = parser.parse_args(argv)
    if args.operations is None:
        args.operations = ["fwd"] if args.kind == "swiglu" else ["fwd", "dw"]
    if args.kind == "swiglu" and args.operations != ["fwd"]:
        parser.error("swiglu supports only --operations fwd")
    if args.algorithms is None:
        args.algorithms = ["tiled", "persistent"]
        if args.kind == "mm":
            args.algorithms.append("persistent_tma")
    if args.kind == "swiglu" and "persistent_tma" in args.algorithms:
        parser.error("persistent_tma is implemented for mm only")
    for key in ("operations", "algorithms"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(key + " must not contain duplicate selections")
    if args.tile:
        from ops.matmul_kernels import normalize_tile
        try:
            args.tile = [normalize_tile(tuple(map(int, tile.split(",")))) for tile in args.tile]
            if len(args.tile) != len(set(args.tile)):
                raise ValueError("tile must not contain duplicate selections; use --trials for repeats")
            if args.kind == "swiglu" and any((bm + 2 * bn) * bk * 2 * stages > 96 * 1024
                                             for bm, bn, bk, _, stages in args.tile):
                raise ValueError("Paired SwiGLU pipeline exceeds the conservative 96 KiB budget")
        except ValueError as error:
            parser.error(str(error))
    for key in ["rows", "repeats", "trials", "warmup"]:
        if getattr(args, key) < 1:
            parser.error(key + " must be positive")
    for key in ["burnin_seconds", "minimum_gain", "maximum_relative_rms"]:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(key + " must be finite and nonnegative")
    if args.device < 0:
        parser.error("device must be nonnegative")
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("input-scale must be finite and positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    work = PACKAGE / "work"
    if (not args.output_dir.is_relative_to(work) or args.output_dir == work
            or not args.output_dir.relative_to(work).parts[0].startswith("benchmark-")):
        parser.error("output-dir must be below rtx6000_40b/work/benchmark-*")
    if not args.list_cases:
        for key in ["TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
            if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(WORKSPACE):
                parser.error("Source ops/training-env.sh; " + key + " must stay inside the workspace")
    return args


def cases(rows, operations):
    result = []
    for operation in operations:
        m, n, k, at, bt = ((rows, 14336, 2048, False, True) if operation == "fwd"
                           else (14336, 2048, rows, True, False))
        result.append({"operation": "ffn_up_gate." + operation, "m": m, "n": n, "k": k,
                       "a_transposed": at, "b_transposed": bt,
                       "a_stride": [1, m] if at else [k, 1],
                       "b_stride": [1, k] if bt else [n, 1], "dtype": "bfloat16"})
    return result


def telemetry():
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,power.draw,power.limit,clocks.sm,"
            "clocks.mem,temperature.gpu,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return {"timestamp": time.time(), "csv": result.stdout.strip(), "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def output_metrics(output, reference):
    """Validate every returned BF16 output; one bad epilogue cannot hide in RMS."""
    import torch

    outputs = output if isinstance(output, tuple) else (output,)
    references = reference if isinstance(reference, tuple) else (reference,)
    if len(outputs) != len(references):
        raise ValueError("Candidate and reference must return the same number of tensors")
    metrics = []
    for result, expected in zip(outputs, references, strict=True):
        if result.shape != expected.shape or result.dtype != torch.bfloat16:
            raise ValueError("Candidate output must preserve the reference shape and BF16 dtype")
        difference = result.float() - expected.float()
        relative_rms = (difference.square().mean().sqrt()
                        / expected.float().square().mean().sqrt().clamp_min(1e-30)).item()
        if not bool(torch.isfinite(result).all()) or not math.isfinite(relative_rms):
            raise ValueError("Nonfinite candidate output or error metric")
        metrics.append({"max_abs_error": difference.abs().max().item(),
                        "relative_rms_error": relative_rms})
    return {"output_metrics": metrics, "finite": True,
            "max_abs_error": max(metric["max_abs_error"] for metric in metrics),
            "relative_rms_error": max(metric["relative_rms_error"] for metric in metrics)}


def paired_timing_metrics(trials, minimum_gain):
    """Compare each candidate with its adjacent controls, retaining the worst trial."""
    if not trials or any(not math.isfinite(value) or value <= 0
                         for trial in trials for value in trial.values()):
        raise ValueError("Paired CUDA timings must be finite and positive")
    controls = [(trial["aten_before_ms"] + trial["aten_after_ms"]) / 2 for trial in trials]
    gains = [control / trial["candidate_ms"] - 1
             for control, trial in zip(controls, trials, strict=True)]
    return {"median_ms": statistics.median(trial["candidate_ms"] for trial in trials),
            "median_control_ms": statistics.median(controls),
            "paired_gain": statistics.median(gains), "minimum_trial_gain": min(gains),
            "timing_eligible": min(gains) >= minimum_gain,
            "maximum_control_drift_fraction": max(
                abs(trial["aten_after_ms"] - trial["aten_before_ms"]) / control
                for control, trial in zip(controls, trials, strict=True))}


def main(argv=None):
    args = arguments(argv)
    selected_cases = cases(args.rows, args.operations)
    if args.list_cases:
        print(json.dumps(selected_cases, indent=2))
        return
    import torch
    from ops.matmul_kernels import TUNING_TILES, launch_mm, matrix_signature, policy_key, tile_dict

    tiles = args.tile or TUNING_TILES
    source_names = ["ops/tune_matmul.py", "ops/matmul_kernels.py"]
    if args.kind == "swiglu":
        from ops.matmul_epilogue import SUPPORTED_TILES, launch_swiglu_mm
        tiles = args.tile or SUPPORTED_TILES
        source_names.append("ops/matmul_epilogue.py")

        def swiglu_reference(a, b):
            packed = torch.mm(a, b)
            up, gate = packed.chunk(2, dim=-1)
            # Matches the existing compiled FFN: the packed projection is
            # rounded to BF16; the SiLU/multiply has no intermediate BF16 store.
            output = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
            return packed, output

        compiled_swiglu_reference = torch.compile(swiglu_reference, fullgraph=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; --list-cases is available without CUDA")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This bounded policy tuning run targets SM120")
    # A process-local reference setting: all tested dot accumulators stay FP32.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "format": "warpstate-matmul-tuning-v1", "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": {"name": properties.name, "capability": [12, 0],
                   "multiprocessors": properties.multi_processor_count,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "input_dtype": "bfloat16", "output_dtype": "bfloat16", "accumulation_dtype": "float32",
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "source_sha256": {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                          for name in source_names},
        "control": "torch.mm" if args.kind == "mm" else "compiled torch.mm + SwiGLU",
        "output_names": ["matrix"] if args.kind == "mm" else ["packed_projection", "activation"],
        "results": [],
        "limitations": ["Isolated kernels, not measured training speedups.",
                        "Inputs are reused; timing includes output allocation and launch scheduling.",
                        "Every candidate trial has a preceding and following warmed ATen control.",
                        "Correctness metrics compare BF16 outputs, not exact real arithmetic.",
                        "Pipeline footprint checks cannot predict every compiler resource requirement."],
    }
    policy = {"format": "warpstate-matmul-plan-v1",
              "description": "Isolated SM120 BF16 winners; require full-model validation before promotion. "
                             "Unmatched shapes retain ATen.", "entries": []}

    def save():
        for name, value in [("results.json", report), ("candidate-policy.json", policy)]:
            temporary = args.output_dir / (name + ".partial")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output_dir / name)

    def elapsed_ms(operation):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            output = operation()
        end.record()
        end.synchronize()
        del output
        return start.elapsed_time(end) / args.repeats

    def warm(operation):
        for _ in range(args.warmup):
            output = operation()
        torch.cuda.synchronize(device)
        start = time.monotonic()
        while time.monotonic() - start < args.burnin_seconds:
            for _ in range(10):
                output = operation()
            torch.cuda.synchronize(device)
        del output

    save()
    with torch.inference_mode():
        for case_index, case in enumerate(selected_cases):
            generator = torch.Generator(device=device).manual_seed(args.seed + case_index)

            def operand(rows, cols, transposed):
                value = torch.randn((cols, rows) if transposed else (rows, cols),
                                    device=device, dtype=torch.bfloat16, generator=generator)
                if args.input_scale != 1.0:
                    value.mul_(args.input_scale)
                return value.T if transposed else value

            a = operand(case["m"], case["k"], case["a_transposed"])
            b = operand(case["k"], case["n"], case["b_transposed"])
            control = (lambda: launch_mm(a, b, "aten")) if args.kind == "mm" else (
                lambda: compiled_swiglu_reference(a, b))
            reference = control()
            candidates = []
            for tile in tiles:
                for algorithm in args.algorithms:
                    row = {"case": case, "signature": matrix_signature(a, b), "key": policy_key(a, b),
                           "algorithm": algorithm, "tile": tile_dict(tile), "status": "running"}
                    print(json.dumps({"operation": case["operation"], "algorithm": algorithm,
                                      "tile": tile, "status": "starting"}), flush=True)
                    operation = (lambda: launch_mm(a, b, algorithm, tile)) if args.kind == "mm" else (
                        lambda: launch_swiglu_mm(a, b, algorithm, tile, eager_rounding=False))
                    try:
                        start = time.monotonic()
                        output = operation()
                        torch.cuda.synchronize(device)
                        row["first_call_seconds"] = time.monotonic() - start
                        row.update(output_metrics(output, reference))
                        del output
                        warm(control)
                        warm(operation)
                        row["telemetry_before"] = telemetry()
                        triples = []
                        for _ in range(args.trials):
                            triples.append({"aten_before_ms": elapsed_ms(control),
                                            "candidate_ms": elapsed_ms(operation),
                                            "aten_after_ms": elapsed_ms(control)})
                        row["trials"] = triples
                        row["telemetry_after"] = telemetry()
                        row.update(paired_timing_metrics(triples, args.minimum_gain))
                        row["tflops"] = 2 * case["m"] * case["n"] * case["k"] / (row["median_ms"] * 1e9)
                        row["status"] = "ok"
                        row["eligible"] = (row["relative_rms_error"] <= args.maximum_relative_rms
                                           and row["timing_eligible"])
                        if row["eligible"]:
                            candidates.append(row)
                    except Exception as error:
                        row["status"] = "error"
                        row["error"] = type(error).__name__ + ": " + str(error)[:4000]
                    report["results"].append(row)
                    save()
                    print(json.dumps({k: row[k] for k in ["algorithm", "tile", "status", "median_ms",
                        "paired_gain", "relative_rms_error", "error"] if k in row}), flush=True)
            if candidates:
                winner = max(candidates, key=lambda row: row["paired_gain"])
                policy["entries"].append({"kind": args.kind, "name": case["operation"],
                    **winner["signature"], "algorithm": winner["algorithm"], "tile": winner["tile"]})
            save()
            del a, b, reference
    report["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
