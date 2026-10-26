"""CPU fixtures reject misleading comparisons and preserve per-rank evidence."""

import copy
import hashlib
import json

import pytest

from ops.compare_communication_benchmarks import compare_summaries, load_run


def fixture():
    identity = {"path": "/workspace/WarpState/fixture", "size": 100, "inode": 1, "mtime_ns": 2}
    config = {"checkpoint": identity, "checkpoint_step": 40, "checkpoint_tokens": 1280,
              "data": {**identity, "path": "/workspace/WarpState/data"},
              "world_size": 2, "grad_accum": 2, "parameters": 32,
              "parameter_dtype": "float32", "autocast_dtype": "bfloat16", "optimizer": "AdamW(fused=True)",
              "torch": "fixture", "cuda": "fixture", "gpu": "fixture",
              "update_timing_includes_gradient_zeroing": True, "checkpoint_writes": False,
              "source_sha256": {"kernel.py": "old"}, "allow_bf16_reduced_precision_reduction": False,
              "args": {"global_tokens_per_update": 32, "seq_len": 8, "micro_batch": 1,
                       "warmup_updates": 1, "measured_updates": 3, "seed": 1337,
                       "optimizer_state": "resume", "bf16_residuals": True, "profile": False}}
    ranks = []
    for rank in range(2):
        ranks.append({"configuration": {**copy.deepcopy(config), "rank": rank},
                      "checkpoint_path_identity_after": copy.deepcopy(identity), "updates": [
            {"rank": rank, "update": index + 1, "phase": "warmup" if index == 0 else "measured",
             "profiled": False, "global_tokens": 32, "seconds": 2.0 + rank,
             "tokens_per_second": 32 / (2.0 + rank), "loss": 2.5, "grad_norm": 0.25,
             "peak_allocated_gib": 60.0 + rank, "peak_reserved_gib": 70.0 + rank}
            for index in range(4)]})
    return {**copy.deepcopy(ranks[0]), "ranks": ranks}


def alter_config(run, key, value, *, argument=False):
    for summary in [run, *run["ranks"]]:
        config = summary["configuration"]
        (config["args"] if argument else config)[key] = value


def test_comparison_uses_slowest_rank_and_records_intended_source_changes():
    control, candidate = fixture(), fixture()
    alter_config(candidate, "source_sha256", {"kernel.py": "new", "scheduler.py": "added"})
    for summary in [candidate, *candidate["ranks"]]:
        for update in summary["updates"]:
            update["seconds"] /= 2
            update["tokens_per_second"] *= 2
            update["loss"] += 0.125
            update["grad_norm"] += 0.025
    result = compare_summaries(control, candidate)
    assert result["throughput_ratio"] == 2
    assert result["control_timing"]["mean_seconds"] == 3
    assert result["maximum_absolute_loss_difference"] == 0.125
    assert result["maximum_relative_grad_norm_difference"] == pytest.approx(0.1)
    assert result["rank_peak_vram"][1]["candidate"]["measured_updates"]["peak_reserved_gib"] == 71
    assert result["paired_updates"][0]["token_cursor"] == 1312
    assert result["paired_updates"][0]["ranks"][1]["microbatch_cursors"] == [1320, 1336]
    assert result["source_differences"]["kernel.py"]["candidate"] == "new"
    assert result["candidate_timing"]["drift_windows_overlap"]


@pytest.mark.parametrize(("key", "value", "argument"), [
    ("checkpoint_tokens", 1281, False), ("checkpoint_step", 41, False),
    ("parameter_dtype", "bfloat16", False), ("autocast_dtype", "float16", False),
    ("allow_bf16_reduced_precision_reduction", True, False),
    ("optimizer", "SGD", False), ("optimizer_state", "fresh", True),
    ("seed", 1338, True), ("bf16_residuals", False, True),
    ("warmup_updates", 0, True), ("micro_batch", 2, True),
])
def test_incompatible_inputs_and_numerical_controls_fail(key, value, argument):
    control, candidate = fixture(), fixture()
    alter_config(candidate, key, value, argument=argument)
    with pytest.raises(ValueError):
        compare_summaries(control, candidate)


@pytest.mark.parametrize("kind", ["profile", "missing_rank", "missing_update", "cursor", "phase", "identity", "rank_identity"])
def test_incomplete_or_inconsistent_recordings_fail(kind):
    candidate = fixture()
    if kind == "profile":
        candidate["ranks"][1]["updates"][2]["profiled"] = True
    elif kind == "missing_rank":
        candidate["ranks"].pop()
    elif kind == "missing_update":
        candidate["ranks"][1]["updates"].pop()
    elif kind == "cursor":
        candidate["ranks"][1]["updates"][2]["token_cursor"] = 0
    elif kind == "phase":
        candidate["ranks"][1]["updates"][2]["phase"] = "warmup"
    elif kind == "identity":
        candidate["ranks"][1]["checkpoint_path_identity_after"]["inode"] += 1
    elif kind == "rank_identity":
        candidate["ranks"][1]["configuration"]["rank"] = 0
    with pytest.raises(ValueError):
        compare_summaries(fixture(), candidate)


def test_missing_precision_controls_are_explicit_and_partial_recording_fails():
    control, candidate = fixture(), fixture()
    result = compare_summaries(control, candidate)
    assert "allow_tf32" in result["unrecorded_precision_optimizer_controls"]
    alter_config(candidate, "allow_tf32", True)
    with pytest.raises(ValueError, match="incompletely recorded"):
        compare_summaries(control, candidate)


def test_precision_mismatch_on_only_the_second_rank_is_rejected():
    candidate = fixture()
    candidate["ranks"][1]["configuration"]["allow_bf16_reduced_precision_reduction"] = True
    with pytest.raises(ValueError, match="Rank 1 precision"):
        compare_summaries(fixture(), candidate)


def test_data_identity_difference_is_rejected_even_with_equal_shapes():
    candidate = fixture()
    changed = {**candidate["configuration"]["data"], "mtime_ns": 3}
    alter_config(candidate, "data", changed)
    with pytest.raises(ValueError, match="configuration: data"):
        compare_summaries(fixture(), candidate)


@pytest.mark.parametrize(("argument", "value"), [
    ("overlap_swap_collectives", True), ("gradient_mode", "eager"), ("compile_mode", "default"),
])
def test_execution_argument_changes_on_only_one_rank_are_rejected(argument, value):
    candidate = fixture()
    candidate["ranks"][1]["configuration"]["args"][argument] = value
    with pytest.raises(ValueError, match="Rank 1 execution argument mismatch"):
        compare_summaries(fixture(), candidate)


def test_jsonl_fallback_hashes_every_consumed_input(tmp_path):
    run = fixture()
    for rank_summary in run["ranks"]:
        rank_summary["configuration"]["event"] = "configuration"
        for update in rank_summary["updates"]:
            update["event"] = "update"
    summary = copy.deepcopy(run["ranks"][0])
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for rank, rank_summary in enumerate(run["ranks"]):
        records = [rank_summary["configuration"], *rank_summary["updates"]]
        (tmp_path / f"rank-{rank}.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    loaded = load_run(tmp_path)
    paths = [tmp_path / name for name in ("summary.json", "rank-0.jsonl", "rank-1.jsonl")]
    assert loaded["input_files"] == [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                     for path in paths]
    assert len(loaded["summary"]["ranks"]) == 2
    before = loaded["input_files"][2]["sha256"]
    # Semantically irrelevant trailing whitespace changes the exact consumed
    # input hash, while preserving successful reconstruction and validation.
    paths[2].write_text(paths[2].read_text().rstrip() + " \n")
    after = load_run(tmp_path)
    assert after["summary"] == loaded["summary"]
    assert after["input_files"][2]["sha256"] != before
