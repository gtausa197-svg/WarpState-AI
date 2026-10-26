#!/usr/bin/env python3
"""Compare the opt-in backward graph replacement in real WarpState training.

The fixture retains production attention and FFN dimensions, with one physical
core and one logical layer by default; --cores selects additional distinct cores
and layers to exercise cross-core graph optimization. Its default 512-token vocabulary exercises repeated
embedding indices. --unique-inputs uses an 8192-token vocabulary and each input
index once per microbatch, removing the existing embedding-atomic nondeterminism
for strict continuation checks. It is a correctness test, not a throughput
benchmark or a claim about all 28 trained layers. Separate
processes compare compiled baseline/candidate execution and candidate checkpoint
continuation. Every artifact stays in a fresh directory below work/validation.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))

# Declared independently of measured results. The graph pass preserves BF16
# arithmetic, but the existing full graph contains relaxed embedding atomics.
# Byte identity is reported separately and is mandatory for resumed full state.
TOLERANCES = {
    "gradient_relative_l2": 0.005,
    "gradient_relative_max": 0.01,
    "gradient_absolute": 1e-8,
    "parameter_absolute": 2e-5,
    "moment_relative_l2": 0.005,
    "moment_relative_max": 0.01,
    "moment_absolute": 1e-8,
    "loss_absolute": 2e-4,
}
MICRO_BATCH = 4
SEQ_LEN = 2048
MICRO_TOKENS = MICRO_BATCH * SEQ_LEN
# Three exercises three distinct CA graphs: first no-sync with None gradients,
# later no-sync accumulation, and final accumulation with DDP synchronization.
ACCUMULATION = 3


def tensor_comparison(actual, expected):
    """Finite, byte-sensitive error metrics with bounded CPU scratch space."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Tensor shape/dtype mismatch")
    shape = list(actual.shape)
    actual = actual.detach().cpu().contiguous().reshape(-1)
    expected = expected.detach().cpu().contiguous().reshape(-1)
    squared_error = squared_reference = maximum_error = maximum_reference = 0.0
    unequal = bitwise_unequal = 0
    finite = True
    for offset in range(0, actual.numel(), 1 << 20):
        observed = actual[offset:offset + (1 << 20)]
        wanted = expected[offset:offset + (1 << 20)]
        bitwise_unequal += int((observed.view(torch.uint8).reshape(-1, observed.element_size())
                              != wanted.view(torch.uint8).reshape(-1, wanted.element_size()))
                             .any(dim=1).sum())
        unequal += int((observed != wanted).sum())
        observed, wanted = observed.double(), wanted.double()
        if not (torch.isfinite(observed).all() and torch.isfinite(wanted).all()):
            finite = False
            continue
        difference = observed - wanted
        squared_error += float(difference.square().sum())
        squared_reference += float(wanted.square().sum())
        maximum_error = max(maximum_error, float(difference.abs().max()))
        maximum_reference = max(maximum_reference, float(wanted.abs().max()))
    return {
        "shape": shape, "dtype": str(actual.dtype),
        "elements": actual.numel(), "finite": bool(finite),
        "unequal": unequal, "bitwise_unequal": bitwise_unequal,
        "error_l2": squared_error ** 0.5 if finite else None,
        "reference_l2": squared_reference ** 0.5 if finite else None,
        "relative_l2": ((squared_error / squared_reference) ** 0.5
                        if finite and squared_reference else (0.0 if finite and not squared_error else None)),
        "max_absolute": maximum_error if finite else None,
        "reference_max_absolute": maximum_reference if finite else None,
    }


def within_tolerance(metrics, kind):
    if not metrics["finite"]:
        return False
    if kind == "exact":
        return metrics["bitwise_unequal"] == 0
    if kind == "parameter":
        return metrics["max_absolute"] <= TOLERANCES["parameter_absolute"]
    if kind not in {"gradient", "moment"}:
        raise ValueError("Unknown comparison kind: " + kind)
    absolute = TOLERANCES[kind + "_absolute"]
    return (metrics["error_l2"] <= metrics["reference_l2"] * TOLERANCES[kind + "_relative_l2"] + absolute
            and metrics["max_absolute"] <= metrics["reference_max_absolute"]
            * TOLERANCES[kind + "_relative_max"] + absolute)


def compare_named_tensors(actual, expected, kind):
    if actual.keys() != expected.keys():
        raise ValueError("Tensor names differ")
    results = {name: tensor_comparison(value, expected[name]) for name, value in actual.items()}
    for name, metrics in results.items():
        # Adam step counters have discrete semantics despite their FP32 dtype.
        selected_kind = "exact" if kind == "moment" and name.endswith("/step") else kind
        metrics["accepted"] = within_tolerance(metrics, selected_kind)
    return {"accepted": all(value["accepted"] for value in results.values()),
            "bitwise_equal": all(value["bitwise_unequal"] == 0 for value in results.values()),
            "tensors": results}


def exact_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(exact_tree_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(exact_tree_equal(a, b) for a, b in zip(actual, expected, strict=True)))
    return type(actual) is type(expected) and actual == expected


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def update_plan(world_size, update):
    """Two full accumulated updates, then nine valid global tokens."""
    if world_size not in {1, 2} or update not in {0, 1, 2}:
        raise ValueError("The fixture supports one/two ranks and three updates")
    return MICRO_TOKENS * ACCUMULATION * world_size if update < 2 else 9


def fixture_config(unique_inputs=False, cores=1):
    from warpstate import WarpStateConfig
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("cores must be a positive integer")
    return WarpStateConfig(vocab_size=MICRO_TOKENS if unique_inputs else 512, d_model=2048, n_heads=32,
                           n_cores=cores, logical_depth=cores, ffn_hidden=7168,
                           chunk_size=128, gradient_checkpointing=False,
                           parallel_memory=True, loss_chunk_size=512)


def fixture_batch_cpu(seed, rank, update, micro, unique_inputs=False):
    generator = torch.Generator().manual_seed(seed + 100000 * rank + 100 * update + micro)
    vocab_size = MICRO_TOKENS if unique_inputs else 512
    if unique_inputs:
        inputs = torch.randperm(MICRO_TOKENS, generator=generator).reshape(MICRO_BATCH, SEQ_LEN)
    else:
        inputs = torch.randint(vocab_size, (MICRO_BATCH, SEQ_LEN), generator=generator)
    targets = torch.randint(vocab_size, inputs.shape, generator=generator)
    return inputs, targets


def save_json(path, document):
    with path.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_identity():
    names = ["ops/validate_backward_layout_training.py", "ops/gradient_execution.py",
             "ops/distributed_execution.py", "ops/loss_experiment.py", "ops/residual_experiment.py",
             "ops/backward_layout_execution.py", "ops/backward_layout.py",
             "warpstate/model.py", "warpstate/config.py"]
    return {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() for name in names}


def check_workspace_environment():
    for name in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR"):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(WORKSPACE):
            raise ValueError("Source ops/training-env.sh first: " + name)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per subprocess")
    parser.add_argument("--seed", type=int, default=41619)
    parser.add_argument("--cores", type=int, default=1,
                        help="Number of distinct physical cores and logical layers")
    parser.add_argument("--unique-inputs", action="store_true",
                        help="Use each input embedding row once per microbatch for strict resume checks")
    parser.add_argument("--worker", choices=("baseline", "candidate", "partial", "resumed"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.output_dir.is_relative_to(PACKAGE / "work/validation"):
        parser.error("--output-dir must be below rtx6000_40b/work/validation")
    if args.output_dir == PACKAGE / "work/validation":
        parser.error("--output-dir must name a new child directory")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.cores < 1:
        parser.error("--cores must be positive")
    return args


def worker(args):
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist
    from ops.backward_layout_execution import (backward_layout_stats,
        configure_backward_layout, require_backward_layout_dispatch)
    from ops.distributed_execution import ddp_kwargs, ddp_runtime_settings
    from ops.gradient_execution import make_backward, scale_loss
    from ops.loss_experiment import apply_loss_execution
    from ops.residual_experiment import apply_bf16_residuals
    from warpstate import WarpStateLM

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != args.world_size:
        raise ValueError("Worker world size differs from its fixture")
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    selected = "baseline" if args.worker == "baseline" else "paired"
    layout_metadata = configure_backward_layout(selected)
    backward = make_backward("compiled-autograd-overlap", distributed=world_size > 1)
    initial = torch.load(args.output_dir / "initial.pt", map_location="cpu", weights_only=False)
    model = WarpStateLM(fixture_config(args.unique_inputs, args.cores)).to(device=device, dtype=torch.float32)
    model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1, fused=True)
    # Nonzero FP32 Adam state avoids first-update sign amplification near zero;
    # this is a deterministic synthetic fixture, not imported production state.
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(40081.0, device=device),
            "exp_avg": torch.full_like(parameter, 0.001),
            "exp_avg_sq": torch.full_like(parameter, 0.0001),
        }
    del initial
    start_update = 0
    if args.worker == "resumed":
        saved = torch.load(args.output_dir / f"partial/rank-{rank}-state.pt",
                           map_location="cpu", weights_only=False)
        if (saved["layout_execution"] != layout_metadata
                or saved["source_sha256"] != source_identity()
                or saved["config"] != asdict(model.cfg)
                or saved["world_size"] != world_size):
            raise ValueError("Checkpoint execution identity/configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_update = saved["update"]
        resume_rng = saved["rng"]
        del saved
    for core in model.cores:
        core.forward = torch.compile(core.forward, fullgraph=True, dynamic=False, mode="default")
    apply_loss_execution(model, "compiled-full")
    apply_bf16_residuals(model)
    wrapped = model
    if world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, **ddp_kwargs(ddp_runtime_settings(bucket_cap_mb=128)))
    model.train()
    directory = args.output_dir / args.worker
    directory.mkdir(exist_ok=True)
    records = []
    accepted = True

    def fixture_batch(update, micro):
        inputs, targets = fixture_batch_cpu(args.seed, rank, update, micro, args.unique_inputs)
        return inputs.to(device), targets.to(device)

    if args.worker != "resumed":
        inputs, _ = fixture_batch(0, 0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapped(inputs)
        logits = logits.detach().cpu()
        if args.worker == "baseline":
            torch.save(logits, directory / f"rank-{rank}-forward.pt")
        else:
            expected = torch.load(args.output_dir / f"baseline/rank-{rank}-forward.pt",
                                  map_location="cpu", weights_only=True)
            metrics = tensor_comparison(logits, expected)
            metrics["accepted"] = within_tolerance(metrics, "exact")
            accepted &= metrics["accepted"]
            records.append({"stage": "initial_forward_logits", **metrics})
        del logits, inputs
    # Startup/cold compilation need not consume the same RNG as a warm process.
    # Set the training RNG after initialization, or restore it after resume setup.
    if args.worker == "resumed":
        torch.set_rng_state(resume_rng["cpu"])
        torch.cuda.set_rng_state(resume_rng["cuda"], device)
    else:
        torch.manual_seed(args.seed + rank + 81)
    stop_update = 1 if args.worker == "partial" else 3
    for update in range(start_update, stop_update):
        optimizer.zero_grad(set_to_none=True)
        target_tokens = update_plan(world_size, update)
        microbatches = (target_tokens + MICRO_TOKENS * world_size - 1) // (MICRO_TOKENS * world_size)
        for micro in range(microbatches):
            offset = (micro * world_size + rank) * MICRO_TOKENS
            valid = max(0, min(MICRO_TOKENS, target_tokens - offset))
            inputs, targets = fixture_batch(update, micro)
            targets.reshape(-1)[valid:] = -100
            synchronize = micro == microbatches - 1
            synchronization = contextlib.nullcontext() if synchronize or world_size == 1 else wrapped.no_sync()
            previous_calls = backward_layout_stats()["runtime_calls"]
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                    _, loss = wrapped(inputs, targets, return_logits=False)
                backward(scale_loss(loss, world_size * valid / target_tokens, "compiled-autograd-overlap"))
            torch.cuda.synchronize(device)
            if selected == "paired":
                # Every backward variant must dispatch, including later
                # accumulation and the masked final rank; an earlier match
                # must not conceal fallback in a later graph.
                require_backward_layout_dispatch(min_calls=previous_calls + model.cfg.logical_depth)
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.dtype != torch.float32 or parameter.grad is None or parameter.grad.dtype != torch.float32:
                    raise RuntimeError("Missing FP32 master/gradient: " + name)
                gradients[name] = parameter.grad.detach().cpu()
            reference_path = args.output_dir / f"baseline/rank-{rank}-u{update}-m{micro}.pt"
            record = {"stage": "gradient", "update": update, "microbatch": micro,
                      "synchronize": synchronize, "valid_tokens": valid,
                      "loss": float(loss.detach()), "layout_stats": backward_layout_stats(),
                      "runtime_calls_this_backward": backward_layout_stats()["runtime_calls"] - previous_calls}
            if args.worker == "baseline":
                if not all(torch.isfinite(value).all() for value in gradients.values()):
                    raise RuntimeError("Baseline produced nonfinite gradients")
                torch.save({"gradients": gradients, "loss": record["loss"]}, reference_path)
            else:
                expected = torch.load(reference_path, map_location="cpu", weights_only=True, mmap=True)
                record["comparison"] = compare_named_tensors(gradients, expected["gradients"], "gradient")
                record["loss_difference"] = abs(record["loss"] - expected["loss"])
                record["accepted"] = (record["comparison"]["accepted"]
                                      and record["loss_difference"] <= TOLERANCES["loss_absolute"])
                accepted &= record["accepted"]
                del expected
            records.append(record)
            del gradients, inputs, targets, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        actual_parameters = cpu_tree(model.state_dict())
        actual_moments = {f"{name}/{key}": cpu_tree(value)
                          for name, parameter in model.named_parameters()
                          for key, value in optimizer.state[parameter].items()}
        reference_path = args.output_dir / f"baseline/rank-{rank}-update-{update}.pt"
        record = {"stage": "adam_update", "update": update, "grad_norm": float(norm)}
        if args.worker == "baseline":
            torch.save({"model": actual_parameters, "moments": actual_moments}, reference_path)
        else:
            expected = torch.load(reference_path, map_location="cpu", weights_only=True, mmap=True)
            record["parameters"] = compare_named_tensors(actual_parameters, expected["model"], "parameter")
            record["moments"] = compare_named_tensors(actual_moments, expected["moments"], "moment")
            record["accepted"] = record["parameters"]["accepted"] and record["moments"]["accepted"]
            accepted &= record["accepted"]
            del expected
        records.append(record)
        del actual_parameters, actual_moments
    state = {"config": asdict(model.cfg), "world_size": world_size, "update": stop_update,
             "tokens": sum(update_plan(world_size, update) for update in range(stop_update)),
             "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
             "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)},
             "layout_execution": layout_metadata, "source_sha256": source_identity()}
    expected_dispatches = model.cfg.logical_depth * sum(
        (update_plan(world_size, update) + MICRO_TOKENS * world_size - 1)
        // (MICRO_TOKENS * world_size)
        for update in range(start_update, stop_update)) if selected == "paired" else 0
    dispatch_exact = backward_layout_stats()["runtime_calls"] == expected_dispatches
    accepted &= dispatch_exact
    records.append({"stage": "dispatch_count", "expected": expected_dispatches,
                    "actual": backward_layout_stats()["runtime_calls"], "accepted": dispatch_exact})
    torch.save(state, directory / f"rank-{rank}-state.pt")
    if args.worker == "resumed":
        expected = torch.load(args.output_dir / f"candidate/rank-{rank}-state.pt",
                              map_location="cpu", weights_only=False, mmap=True)
        exact = {key: exact_tree_equal(state[key], expected[key]) for key in state}
        accepted &= all(exact.values())
        records.append({"stage": "checkpoint_continuation", "accepted": all(exact.values()),
                        "exact_fields": exact})
    save_json(directory / f"rank-{rank}.json", {
        "rank": rank, "worker": args.worker, "accepted": accepted,
        "tolerances": TOLERANCES, "records": records, "layout_execution": layout_metadata,
        "layout_stats": backward_layout_stats(), "source_sha256": source_identity(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    })
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    print(json.dumps({"worker": args.worker, "rank": rank, "accepted": accepted}), flush=True)


def main(argv=None):
    args = arguments(argv)
    check_workspace_environment()
    if args.worker:
        worker(args)
        return
    from warpstate import WarpStateLM
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cfg = fixture_config(args.unique_inputs, args.cores)
    initial = WarpStateLM(cfg)
    torch.save({"config": asdict(cfg), "model": initial.state_dict()}, args.output_dir / "initial.pt")
    del initial
    source_hashes = source_identity()
    save_json(args.output_dir / "fixture.json", {
        "config": asdict(cfg), "seed": args.seed, "world_size": args.world_size,
        "micro_batch": MICRO_BATCH, "seq_len": SEQ_LEN, "accumulation": ACCUMULATION,
        "unique_inputs": args.unique_inputs,
        "global_tokens": [update_plan(args.world_size, update) for update in range(3)],
        "tolerances": TOLERANCES, "source_sha256": source_hashes,
        "scope": f"{args.cores} production-shaped cores/layers; synthetic weights/tokens and nonzero FP32 Adam moments",
    })
    environment = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                 "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    reports = []
    for mode in ("baseline", "candidate", "partial", "resumed"):
        command = [sys.executable]
        if args.world_size > 1:
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes", "1",
                        "--nproc-per-node", str(args.world_size)]
        command += [str(Path(__file__).resolve()), "--output-dir", str(args.output_dir),
                    "--world-size", str(args.world_size), "--seed", str(args.seed),
                    "--cores", str(args.cores), "--worker", mode]
        if args.unique_inputs:
            command.append("--unique-inputs")
        with (args.output_dir / f"{mode}.log").open("x") as log:
            print("Running " + mode, flush=True)
            result = subprocess.run(command, cwd=PACKAGE, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode:
            raise RuntimeError(f"{mode} exited {result.returncode}; inspect {mode}.log")
        if source_identity() != source_hashes:
            raise RuntimeError("Validation source changed during the experiment")
        reports.extend(json.loads((args.output_dir / f"{mode}/rank-{rank}.json").read_text())
                       for rank in range(args.world_size))
    rank_consistency = {}
    if args.world_size == 2:
        for mode in ("baseline", "candidate", "partial", "resumed"):
            states = [torch.load(args.output_dir / f"{mode}/rank-{rank}-state.pt", map_location="cpu",
                                 weights_only=False, mmap=True) for rank in range(args.world_size)]
            rank_consistency[mode] = {key: exact_tree_equal(states[0][key], states[1][key])
                                      for key in ("model", "optimizer", "update", "tokens")}
            del states
    summary = {"accepted": (all(report["accepted"] for report in reports)
                            and all(all(fields.values()) for fields in rank_consistency.values())),
               "rank_consistency": rank_consistency, "runs": reports,
               "source_sha256": source_hashes, "tolerances": TOLERANCES}
    save_json(args.output_dir / "results.json", summary)
    print(json.dumps({"accepted": summary["accepted"], "report": str(args.output_dir / "results.json")}), flush=True)
    if not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
