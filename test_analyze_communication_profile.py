"""CPU fixtures for stream unions, exposed communication, and collective metadata."""

import copy

import pytest

from ops.analyze_communication_profile import merge_intervals, summarize_trace


def activity(name, begin, duration, *, device=0, category="kernel", **metadata):
    return {"ph": "X", "cat": category, "name": name, "ts": begin, "dur": duration,
            "pid": device, "args": {"device": device, **metadata}}


def nccl(begin, duration, sequence=1, **metadata):
    return activity("nccl_AllReduce", begin, duration,
                    **{"Process Group Name": "0", "Seq": sequence, "Collective name": "allreduce",
                       "In msg nelems": 32, "dtype": "Float", **metadata})


def test_interval_union_excludes_duplicate_streams_and_gpu_annotations():
    events = [activity("gemm", 0, 10), activity("pointwise", 5, 10),
              nccl(8, 12), nccl(9, 2), activity("copy", 25, 5, category="gpu_memcpy"),
              activity("nccl:annotation", 0, 100, category="gpu_user_annotation")]
    result = summarize_trace({"traceEvents": events, "distributedInfo": {"rank": 1}})
    device = result["devices"][0]
    assert result["rank"] == 1 and result["excluded_gpu_annotation_ranges"] == 1
    assert device["noncommunication_union_us"] == 20
    assert device["communication_union_us"] == 12
    assert device["compute_communication_overlap_us"] == 7
    assert device["compute_only_us"] == 13
    assert device["communication_only_us"] == 5
    assert device["active_union_us"] == 25
    assert device["activity_span_us"] == 30
    assert device["idle_within_activity_span_us"] == 5
    assert device["summed_communication_us"] == 14
    assert device["collective_call_count"] == 1
    assert device["nccl_kernel_count"] == 2
    assert device["logical_input_payload_bytes"] == 128


def test_devices_are_separate_and_unknown_transport_bytes_remain_unknown():
    trace = {"traceEvents": [activity("gemm", 0, 10, device=0),
                             activity("gemm", 0, 10, device=1),
                             activity("nccl_kernel", 8, 5, device=1)]}
    first, second = summarize_trace(trace)["devices"]
    assert first["active_union_us"] == 10
    assert second["active_union_us"] == 13
    assert second["compute_communication_overlap_us"] == 2
    assert second["collective_call_count"] is None
    assert second["logical_input_payload_bytes"] is None
    assert second["nccl_kernel_count"] == 1


@pytest.mark.parametrize("metadata", [{"dtype": "unknown"}, {"In msg nelems": "32"}])
def test_unknown_payload_metadata_cannot_generate_plausible_byte_totals(metadata):
    device = summarize_trace({"traceEvents": [nccl(0, 1, **metadata)]})["devices"][0]
    assert device["collective_call_count"] == 1
    assert device["logical_input_payload_bytes"] is None


@pytest.mark.parametrize("sequence", [-1, None, True])
def test_unavailable_sequence_numbers_are_not_deduplicated_into_fake_calls(sequence):
    device = summarize_trace({"traceEvents": [nccl(0, 1, sequence), nccl(2, 1, sequence)]})["devices"][0]
    assert device["nccl_kernel_count"] == 2
    assert device["collective_call_count"] is None
    assert device["logical_input_payload_bytes"] is None


def test_inconsistent_duplicate_collective_metadata_is_reported():
    device = summarize_trace({"traceEvents": [nccl(0, 1), nccl(1, 1, **{"In msg nelems": 64})]})["devices"][0]
    assert device["collective_call_count"] == 1
    assert device["logical_input_payload_bytes"] is None
    assert device["collectives"][0]["metadata_consistent"] is False


def test_cpu_fingerprint_uses_submission_order_not_annotations_or_timestamps():
    def collective(begin, size):
        return {"ph": "X", "cat": "cpu_op", "name": "c10d::allreduce_", "ts": begin, "dur": 2,
                "args": {"Input Dims": [[[size]]], "Input type": ["TensorList"]}}

    trace = {"traceEvents": [collective(3, 64), collective(1, 32),
                             activity("nccl:all_reduce", 2, 1, category="user_annotation")]}
    expected = summarize_trace(trace)["cpu_collective_fingerprint"]
    assert expected["call_count"] == 2
    shifted = copy.deepcopy(trace)
    for event in shifted["traceEvents"]:
        event["ts"] += 100
    assert summarize_trace(shifted)["cpu_collective_fingerprint"]["sha256"] == expected["sha256"]
    shifted["traceEvents"][0]["ts"] = 0
    assert summarize_trace(shifted)["cpu_collective_fingerprint"]["sha256"] != expected["sha256"]


def test_empty_trace_and_touching_intervals():
    assert merge_intervals([(2, 3), (0, 2), (1, 1)]) == [(0, 3)]
    assert summarize_trace({"traceEvents": []})["devices"] == []


@pytest.mark.parametrize("interval", [(1, 0), (0, float("inf")), (float("nan"), 2)])
def test_invalid_intervals_fail_explicitly(interval):
    with pytest.raises(ValueError):
        merge_intervals([interval])
