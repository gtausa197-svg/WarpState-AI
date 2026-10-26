#!/usr/bin/env python3
"""Compare explicitly paired, unprofiled training benchmarks using saved JSON only.

This reads no checkpoints, tokens, or devices. It checks recorded identities and
batch geometry, reports differences, and does not declare a statistical winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
REQUIRED_CONFIG = ("checkpoint", "checkpoint_step", "checkpoint_tokens", "data", "world_size",
                   "grad_accum", "parameters", "parameter_dtype", "autocast_dtype", "optimizer",
                   "torch", "cuda", "gpu", "update_timing_includes_gradient_zeroing")
REQUIRED_ARGS = ("global_tokens_per_update", "seq_len", "micro_batch", "warmup_updates",
                 "measured_updates", "seed", "optimizer_state", "bf16_residuals")
PRECISION_CONTROLS = ("precision", "precision_flags", "float32_matmul_precision", "allow_tf32",
                      "allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                      "optimizer_hyperparameters")


def finite_number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return value


def require_fields(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} lacks required fields: {', '.join(missing)}")


def differences(left, right):
    return {key: {"control": left.get(key), "candidate": right.get(key),
                  "control_recorded": key in left, "candidate_recorded": key in right}
            for key in sorted(set(left) | set(right))
            if key not in left or key not in right or left[key] != right[key]}


def validate_run(summary):
    config = summary["configuration"]
    args = config["args"]
    require_fields(config, REQUIRED_CONFIG, "configuration")
    require_fields(args, REQUIRED_ARGS, "arguments")
    if args.get("profile") or config.get("checkpoint_writes"):
        raise ValueError("Comparisons require unprofiled benchmarks without checkpoint writes")
    for identity in ("checkpoint", "data"):
        require_fields(config[identity], ("path", "size", "inode", "mtime_ns"), identity)
    for key, value in (("world_size", config["world_size"]), ("grad_accum", config["grad_accum"]),
                       *[(key, args[key]) for key in ("global_tokens_per_update", "seq_len", "micro_batch",
                                                     "measured_updates")]):
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(args["warmup_updates"]) is not int or args["warmup_updates"] < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    for name in ("checkpoint_step", "checkpoint_tokens"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if args["global_tokens_per_update"] != (config["world_size"] * config["grad_accum"]
                                               * args["seq_len"] * args["micro_batch"]):
        raise ValueError("Global batch does not match world size, accumulation, microbatch, and sequence")
    ranks = summary.get("ranks") or [summary]
    if len(ranks) != config["world_size"]:
        raise ValueError("Missing per-rank benchmark records")
    by_rank = {}
    for rank_summary in ranks:
        rc = rank_summary["configuration"]
        rank = rc.get("rank")
        if type(rank) is not int or rank < 0 or rank >= config["world_size"] or rank in by_rank:
            raise ValueError("Rank identities must be complete and unique")
        for key in REQUIRED_CONFIG:
            if key not in rc or rc[key] != config[key]:
                raise ValueError(f"Rank {rank} configuration mismatch: {key}")
        for key in REQUIRED_ARGS:
            if rc["args"].get(key) != args[key]:
                raise ValueError(f"Rank {rank} argument mismatch: {key}")
        if rc["args"] != args:
            changed = ", ".join(differences(args, rc["args"]))
            raise ValueError(f"Rank {rank} execution argument mismatch: {changed}")
        for key in PRECISION_CONTROLS:
            if (key in rc) != (key in config) or rc.get(key) != config.get(key):
                raise ValueError(f"Rank {rank} precision/optimizer control mismatch: {key}")
        if rc["args"].get("profile"):
            raise ValueError("Profiled rank configuration cannot be compared")
        if rank_summary.get("checkpoint_path_identity_after") != config["checkpoint"]:
            raise ValueError("Checkpoint identity changed during a benchmark or was not verified")
        updates = rank_summary["updates"]
        if len(updates) != args["warmup_updates"] + args["measured_updates"]:
            raise ValueError("Recorded update count does not match warmup/measured counts")
        for index, update in enumerate(updates):
            phase = "warmup" if index < args["warmup_updates"] else "measured"
            if update.get("update") != index + 1 or update.get("phase") != phase:
                raise ValueError("Update indices/phases must describe one continuous benchmark")
            if update.get("rank") != rank or update.get("global_tokens") != args["global_tokens_per_update"]:
                raise ValueError("Update rank/global batch mismatch")
            if update.get("profiled") is not False:
                raise ValueError("Every update must explicitly be unprofiled")
            finite_number(update["seconds"], "update seconds", positive=True)
            for name in ("loss", "grad_norm", "peak_allocated_gib", "peak_reserved_gib"):
                finite_number(update[name], name)
            expected_tps = update["global_tokens"] / update["seconds"]
            if not math.isclose(update.get("tokens_per_second", 0), expected_tps, rel_tol=1e-9):
                raise ValueError("Recorded throughput disagrees with tokens/update and elapsed time")
            cursor = config["checkpoint_tokens"] + index * args["global_tokens_per_update"]
            if "token_cursor" in update and update["token_cursor"] != cursor:
                raise ValueError("Recorded token cursor disagrees with checkpoint/batch progression")
        by_rank[rank] = updates
    if summary["updates"] != by_rank[config.get("rank", 0)]:
        raise ValueError("Top-level and per-rank update records disagree")
    return config, by_rank


def load_run(directory):
    directory = directory.expanduser().resolve()
    if not directory.is_relative_to(WORKSPACE):
        raise ValueError("Benchmark directories must stay inside the WarpState workspace")
    inputs = []

    def read_input(path):
        payload = path.read_bytes()
        inputs.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        return payload

    payload = read_input(directory / "summary.json")
    summary = json.loads(payload)
    config = summary["configuration"]
    # Older summaries may omit embedded ranks. Reconstruct from the canonical
    # per-rank JSONL files, without opening model, data, or environment files.
    if not summary.get("ranks") and config["world_size"] > 1:
        ranks = []
        for rank in range(config["world_size"]):
            records = [json.loads(line) for line in read_input(directory / f"rank-{rank}.jsonl").splitlines()]
            configurations = [row for row in records if row.get("event") == "configuration"]
            if len(configurations) != 1:
                raise ValueError("Expected one configuration per rank JSONL")
            ranks.append({"configuration": configurations[0],
                          "updates": [row for row in records if row.get("event") == "update"],
                          "checkpoint_path_identity_after": summary.get("checkpoint_path_identity_after")})
        summary["ranks"] = ranks
    validate_run(summary)
    return {"directory": str(directory), "summary_sha256": hashlib.sha256(payload).hexdigest(),
            "input_files": inputs, "summary": summary}


def timing_statistics(seconds, tokens_per_update):
    early, late = statistics.mean(seconds[:5]), statistics.mean(seconds[-5:])
    return {"updates": len(seconds), "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds),
            "std_seconds": statistics.pstdev(seconds), "std_definition": "population",
            "tokens_per_second": len(seconds) * tokens_per_update / sum(seconds),
            "first_five_mean_seconds": early, "last_five_mean_seconds": late,
            "first_to_last_seconds_drift_fraction": late / early - 1,
            "drift_windows_overlap": len(seconds) < 10}


def compare_summaries(control, candidate):
    cc, cr = validate_run(control)
    dc, dr = validate_run(candidate)
    for key in REQUIRED_CONFIG:
        if cc[key] != dc[key]:
            raise ValueError(f"Incompatible configuration: {key}")
    for key in REQUIRED_ARGS:
        if cc["args"][key] != dc["args"][key]:
            raise ValueError(f"Incompatible arguments: {key}")
    for key in PRECISION_CONTROLS:
        if (key in cc) != (key in dc) or cc.get(key) != dc.get(key):
            raise ValueError(f"Incompatible or incompletely recorded precision/optimizer control: {key}")
    start = cc["args"]["warmup_updates"]
    total = cc["args"]["global_tokens_per_update"]
    micro_tokens = cc["args"]["seq_len"] * cc["args"]["micro_batch"]
    paired = []
    for index in range(start, len(cr[0])):
        control_seconds = max(updates[index]["seconds"] for updates in cr.values())
        candidate_seconds = max(updates[index]["seconds"] for updates in dr.values())
        cursor = cc["checkpoint_tokens"] + index * total
        per_rank = []
        for rank in sorted(cr):
            before, after = cr[rank][index], dr[rank][index]
            per_rank.append({"rank": rank,
                "microbatch_cursors": [cursor + (micro * cc["world_size"] + rank) * micro_tokens
                                       for micro in range(cc["grad_accum"])],
                "control_loss": before["loss"], "candidate_loss": after["loss"],
                "loss_difference": after["loss"] - before["loss"],
                "control_grad_norm": before["grad_norm"], "candidate_grad_norm": after["grad_norm"],
                "grad_norm_difference": after["grad_norm"] - before["grad_norm"],
                "grad_norm_relative_difference": (abs(after["grad_norm"] - before["grad_norm"])
                                                   / max(abs(before["grad_norm"]), 1e-30))})
        paired.append({"update": index + 1, "token_cursor": cursor,
                       "control_seconds": control_seconds, "candidate_seconds": candidate_seconds,
                       "throughput_ratio": control_seconds / candidate_seconds, "ranks": per_rank})
    control_timing = timing_statistics([row["control_seconds"] for row in paired], total)
    candidate_timing = timing_statistics([row["candidate_seconds"] for row in paired], total)
    memory = []
    for rank in sorted(cr):
        row = {"rank": rank}
        for label, ranks in (("control", cr), ("candidate", dr)):
            row[label] = {phase: {name: max(update[name] for update in updates)
                for name in ("peak_allocated_gib", "peak_reserved_gib")}
                for phase, updates in (("all_updates", ranks[rank]), ("measured_updates", ranks[rank][start:]))}
        memory.append(row)
    ignore = {"timestamp", "rank", "args", "source_sha256", "initial_device_free_gib"}
    control_rank_configs = {row["configuration"]["rank"]: row["configuration"]
                            for row in (control.get("ranks") or [control])}
    candidate_rank_configs = {row["configuration"]["rank"]: row["configuration"]
                              for row in (candidate.get("ranks") or [candidate])}
    return {"control_timing": control_timing, "candidate_timing": candidate_timing,
            "throughput_ratio": candidate_timing["tokens_per_second"] / control_timing["tokens_per_second"],
            "maximum_absolute_loss_difference": max(abs(rank["loss_difference"]) for row in paired for rank in row["ranks"]),
            "maximum_relative_grad_norm_difference": max(rank["grad_norm_relative_difference"] for row in paired for rank in row["ranks"]),
            "paired_updates": paired, "rank_peak_vram": memory,
            "argument_differences": differences(cc["args"], dc["args"]),
            "source_differences": differences(cc.get("source_sha256", {}), dc.get("source_sha256", {})),
            "configuration_differences": differences({key: value for key, value in cc.items() if key not in ignore},
                                                      {key: value for key, value in dc.items() if key not in ignore}),
            "rank_configuration_differences": [{"rank": rank,
                "source_differences": differences(control_rank_configs[rank].get("source_sha256", {}),
                                                   candidate_rank_configs[rank].get("source_sha256", {})),
                "configuration_differences": differences(
                    {key: value for key, value in control_rank_configs[rank].items() if key not in ignore},
                    {key: value for key, value in candidate_rank_configs[rank].items() if key not in ignore})}
                for rank in sorted(cr)],
            "unrecorded_precision_optimizer_controls": [key for key in PRECISION_CONTROLS
                if key not in ("precision", "precision_flags") and key not in cc
                and not any(key in (cc.get(container) or {}) for container in ("precision", "precision_flags"))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, nargs="+", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PACKAGE / "work") or output.suffix != ".json" or output.exists():
        parser.error("Select a new JSON output file under rtx6000_40b/work")
    control = load_run(args.control)
    comparisons = []
    for directory in (path for group in args.candidate for path in group):
        candidate = load_run(directory)
        comparisons.append({"candidate_directory": candidate["directory"],
                            "candidate_summary_sha256": candidate["summary_sha256"],
                            "candidate_input_files": candidate["input_files"],
                            **compare_summaries(control["summary"], candidate["summary"])})
    report = {"format": "warpstate-communication-comparison-v1",
              "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "control_directory": control["directory"], "control_summary_sha256": control["summary_sha256"],
              "control_input_files": control["input_files"],
              "comparisons": comparisons, "limitations": [
                  "Identities are recorded path/stat metadata, not independent checkpoint/data content hashes.",
                  "Token cursors are derived from recorded checkpoint tokens and batch geometry.",
                  "Global elapsed time uses the slowest reported rank; ranks are never summed.",
                  "Sequential update pairs are not randomized trials or independent confidence estimates.",
                  "Source and execution differences are disclosed; changed sources can alter unrecorded behavior.",
                  "No threshold declares a winner; inspect drift, numerical differences, and peak VRAM."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()
