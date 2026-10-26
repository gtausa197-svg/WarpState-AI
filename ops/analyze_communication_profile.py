#!/usr/bin/env python3
"""Summarize compute/communication overlap from saved Chrome traces, without CUDA.

Each rank and device is reported separately. All time values use microseconds.
Noncommunication activity includes kernels, copies, and fills. Logical NCCL
input payload bytes are distinct from bytes physically transported over a link.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
COLLECTIVE_OPERATORS = {
    "c10d::allreduce_", "c10d::allreduce_coalesced_", "c10d::allgather_",
    "c10d::_allgather_base_", "c10d::allgather_into_tensor_coalesced_",
    "c10d::reduce_scatter_", "c10d::_reduce_scatter_base_",
    "c10d::reduce_scatter_tensor_coalesced_", "c10d::broadcast_", "c10d::reduce_",
    "c10d::alltoall_", "c10d::alltoall_base_", "c10d::gather_", "c10d::scatter_",
    "c10d::barrier",
}
DTYPE_BYTES = {"float": 4, "float32": 4, "double": 8, "float64": 8,
               "half": 2, "float16": 2, "bfloat16": 2, "long": 8,
               "int64": 8, "int": 4, "int32": 4, "short": 2,
               "int16": 2, "char": 1, "int8": 1, "byte": 1, "uint8": 1,
               "bool": 1}


def merge_intervals(intervals):
    """Return disjoint intervals, counting overlapping streams only once."""
    merged = []
    for begin, end in sorted(intervals):
        if not math.isfinite(begin) or not math.isfinite(end) or end < begin:
            raise ValueError("Activity intervals must have finite, ordered endpoints")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def interval_duration(intervals):
    return sum(end - begin for begin, end in intervals)


def intersection_duration(left, right):
    """Measure intersection of two disjoint, ordered interval lists."""
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def collective_metadata(kernels):
    """Deduplicate channel kernels using process-group name and sequence number.

    Unknown/inconsistent metadata yields null totals instead of guessed bytes or
    a kernel count mislabeled as a collective count. Scalar metric collectives
    are included; the returned sequence permits separating them from gradients.
    """
    calls = {}
    missing_identity = 0
    for event in sorted(kernels, key=lambda item: item["ts"]):
        args = event.get("args", {})
        sequence_id = args.get("Seq")
        if (args.get("Process Group Name") is None or not args.get("Collective name")
                or type(sequence_id) not in (int, str) or not str(sequence_id).isdigit()):
            missing_identity += 1
            continue
        key = (str(args["Process Group Name"]), str(args["Seq"]))
        count = args.get("In msg nelems")
        width = DTYPE_BYTES.get(str(args.get("dtype", "")).lower())
        byte_count = count * width if type(count) is int and count >= 0 and width else None
        row = {"process_group": key[0], "sequence": key[1],
               "collective": args["Collective name"], "dtype": args.get("dtype"),
               "input_elements": count, "input_payload_bytes": byte_count}
        previous = calls.get(key)
        if previous is None:
            calls[key] = {**row, "kernel_count": 1, "metadata_consistent": True}
        else:
            previous["kernel_count"] += 1
            if any(previous[name] != value for name, value in row.items()):
                previous["metadata_consistent"] = False
                previous["input_payload_bytes"] = None
    sequence = list(calls.values())
    known_bytes = sum(row["input_payload_bytes"] or 0 for row in sequence)
    complete = not missing_identity and all(row["metadata_consistent"] and
                                            row["input_payload_bytes"] is not None
                                            for row in sequence)
    return {"nccl_kernel_count": len(kernels),
            "collective_call_count": len(sequence) if not missing_identity else None,
            "known_collective_call_count": len(sequence),
            "kernels_missing_collective_identity": missing_identity,
            "logical_input_payload_bytes": known_bytes if complete else None,
            "known_logical_input_payload_bytes": known_bytes,
            "collectives": sequence}


def cpu_collective_fingerprint(events):
    """Structural CPU submission order, never double-counting NCCL annotations.

    Shapes/types do not identify tensor contents. This can detect many bucket
    order changes but cannot prove equal tensor order for equal-shaped buckets.
    """
    selected = sorted((event for event in events if event.get("ph") == "X"
                       and event.get("cat") == "cpu_op"
                       and event.get("name") in COLLECTIVE_OPERATORS),
                      key=lambda event: event["ts"])
    sequence = [{"operator": event["name"],
                 "input_dims": event.get("args", {}).get("Input Dims"),
                 "input_types": event.get("args", {}).get("Input type")}
                for event in selected]
    payload = json.dumps(sequence, sort_keys=True, separators=(",", ":"))
    return {"call_count": len(sequence), "sequence": sequence,
            "sha256": hashlib.sha256(payload.encode()).hexdigest() if sequence else None,
            "all_input_shapes_recorded": all(row["input_dims"] is not None for row in sequence)}


def summarize_trace(trace):
    events = trace.get("traceEvents", [])
    devices = {}
    excluded = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category == "gpu_user_annotation":
            excluded += 1
        if category not in ACTIVITY_CATEGORIES:
            continue
        begin, duration = event.get("ts"), event.get("dur")
        if (not isinstance(begin, (int, float)) or not isinstance(duration, (int, float))
                or not math.isfinite(begin) or not math.isfinite(duration) or duration < 0):
            raise ValueError("CUDA activities require finite timestamps and nonnegative durations")
        device = str(event.get("args", {}).get("device", event.get("pid", "unknown")))
        group = devices.setdefault(device, {"communication": [], "compute": [], "kernels": []})
        communication = category == "kernel" and "nccl" in event.get("name", "").lower()
        group["communication" if communication else "compute"].append((begin, begin + duration))
        if communication:
            group["kernels"].append(event)
    summaries = []
    for device, group in sorted(devices.items()):
        compute = merge_intervals(group["compute"])
        communication = merge_intervals(group["communication"])
        active = merge_intervals([*compute, *communication])
        overlap = intersection_duration(compute, communication)
        compute_us = interval_duration(compute)
        communication_us = interval_duration(communication)
        span = active[-1][1] - active[0][0] if active else 0.0
        active_us = interval_duration(active)
        summaries.append({"device": device, "activity_span_us": span,
                          "active_union_us": active_us, "idle_within_activity_span_us": span - active_us,
                          "noncommunication_union_us": compute_us,
                          "communication_union_us": communication_us,
                          "compute_only_us": max(0.0, compute_us - overlap),
                          "compute_communication_overlap_us": overlap,
                          "communication_only_us": max(0.0, communication_us - overlap),
                          "communication_overlap_fraction": overlap / communication_us if communication_us else None,
                          "summed_noncommunication_us": interval_duration(group["compute"]),
                          "summed_communication_us": interval_duration(group["communication"]),
                          "nccl_kernel_names": dict(Counter(event["name"] for event in group["kernels"])),
                          **collective_metadata(group["kernels"])})
    return {"rank": trace.get("distributedInfo", {}).get("rank"),
            "distributed_info": trace.get("distributedInfo"), "devices": summaries,
            "excluded_gpu_annotation_ranges": excluded,
            "cpu_collective_fingerprint": cpu_collective_fingerprint(events)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.traces = [path.expanduser().resolve() for path in args.traces]
    args.output = args.output.expanduser().resolve()
    if any(not path.is_relative_to(WORKSPACE) for path in args.traces):
        parser.error("Trace inputs must stay inside the WarpState workspace")
    if not args.output.is_relative_to(PACKAGE / "work") or args.output.suffix != ".json":
        parser.error("Output must be a JSON file under rtx6000_40b/work")
    if args.output.exists():
        parser.error("Output already exists; select a new report path")
    report = {"format": "warpstate-communication-profile-v1",
              "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ranks": [], "limitations": [
                  "Per-rank/device budgets; never sum ranks or devices into a wall-time estimate.",
                  "Compute includes all non-NCCL kernels, memory copies, and fills.",
                  "GPU annotation ranges are excluded; real CUDA activities alone define intervals.",
                  "Communication-only time is an exposed interval, not a guaranteed removable delay.",
                  "Payload bytes describe logical collective inputs, not physical transport traffic.",
                  "Scalar benchmark metric collectives remain included in counts and bytes.",
                  "CPU fingerprints compare operator/shape/type submission sequences, not tensor identities.",
                  "Activity spans exclude untraced startup/tail time; profiling adds instrumentation overhead."]}
    for path in args.traces:
        payload = path.read_bytes()
        summary = summarize_trace(json.loads(payload))
        if summary["rank"] is None:
            match = re.search(r"profile-rank-(\d+)", path.name)
            if match:
                summary["rank"] = int(match.group(1))
        report["ranks"].append({"trace": str(path), "trace_sha256": hashlib.sha256(payload).hexdigest(),
                                **summary})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    with temporary.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "traces": len(report["ranks"])}))


if __name__ == "__main__":
    main()
