"""Absolute training stops preserve the objective, checkpoint, and continuation."""

import json
import math
import sys

import numpy as np
import pytest
import torch

from test_training_execution import (
    TWO_GPUS,
    assert_tensors_equal,
    fixture_runs,
    pipeline,
    run_training,
)
from warpstate import WarpStateConfig, WarpStateLM


def test_stop_at_step_cli_defaults_to_no_absolute_limit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_40b.py"])
    args = pipeline.arguments()
    assert args.stop_at_step is None
    assert args.max_steps is None
    assert args.train_tokens == 40_000_000_000


@pytest.mark.parametrize("target", [1, 64_000, 2_048_000])
def test_stop_at_step_cli_accepts_positive_absolute_target(monkeypatch, target):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--stop-at-step", str(target)])
    args = pipeline.arguments()
    assert args.stop_at_step == target
    assert args.max_steps is None


@pytest.mark.parametrize("target", ["0", "-1", "2.5", "nan"])
def test_stop_at_step_cli_rejects_nonpositive_or_noninteger_target(monkeypatch, target):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--stop-at-step", target])
    with pytest.raises(SystemExit) as error:
        pipeline.arguments()
    assert error.value.code == 2


def test_stop_at_step_and_additional_limit_preserve_resume_signature(monkeypatch):
    manifest = {"complete": True, "signature": {"source": "unchanged-fixture"}}
    monkeypatch.setattr(sys, "argv", ["run_40b.py"])
    expected = pipeline.train_signature(pipeline.arguments(), manifest)
    for flags in (["--stop-at-step", "64000"],
                  ["--stop-at-step", "64000", "--max-steps", "1"],
                  ["--stop-at-step", "96000"], []):
        monkeypatch.setattr(sys, "argv", ["run_40b.py", *flags])
        args = pipeline.arguments()
        actual = pipeline.train_signature(args, manifest)
        assert actual == expected
        assert actual["settings"]["train_tokens"] == 40_000_000_000
        assert "stop_at_step" not in actual["settings"]
        assert "max_steps" not in actual["settings"]
        assert pipeline.check_resume_signature(expected, actual) is False
        if "--max-steps" in flags:
            assert args.max_steps == 1 and args.stop_at_step == 64000


def _records(directory, name):
    return [json.loads(line) for line in (directory / "run" / name).read_text().splitlines()]


@pytest.mark.parametrize(("world_size", "async_checkpoint"), [
    pytest.param(1, False, id="single-gpu-sync",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    pytest.param(2, True, id="two-gpu-async", marks=TWO_GPUS),
])
def test_absolute_stop_checkpoint_and_unlimited_resume_preserve_full_state(
        tmp_path, monkeypatch, world_size, async_checkpoint):
    (continuous, resumed), common = fixture_runs(tmp_path, ["continuous", "resumed"])
    # The original two-head tiny fixture showed a one-ULP step-two grad-norm
    # difference after eager DDP resume, reproduced exactly with --max-steps
    # alone. Its two-element parameter offsets may change foreach-norm alignment
    # when DDP rebuilds buckets. Four heads make every parameter size divisible
    # by four and retain strict uninterrupted-reference equality below, without
    # changing training code or weakening tolerances.
    config_path = tmp_path / "config.json"
    config = WarpStateConfig(**json.loads(config_path.read_text()))
    config.n_heads = 4
    config.validate()
    config.to_json(config_path)
    torch.manual_seed(133)
    initial = WarpStateLM(config)
    assert all(parameter.numel() % 4 == 0 for parameter in initial.parameters())
    torch.save({"config": vars(config), "model": initial.state_dict(), "tokens_seen": 100},
               tmp_path / "initial.pt")
    # Four optimizer updates consume 32, 32, 32, then 9 tokens. Target three
    # falls outside the ten-step evaluation/save cadence, after cosine decay
    # begins. Replacing the full LR horizon with the pause target must fail.
    # Two GPUs also exercise an entirely masked final rank on the last update.
    common += ["--loss-mode", "full", "--bf16-residuals", "--no-activation-checkpointing",
               "--eval-every", "10", "--save-every", "10",
               "--train-tokens", "105", "--warmup-steps", "1"]
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", common)
        data_signature = pipeline.data_signature(pipeline.arguments())
    for directory in (continuous, resumed):
        np.arange(105, dtype="<u2").tofile(directory / "data/train.bin")
        manifest = directory / "data/manifest.json"
        document = json.loads(manifest.read_text())
        document["signature"] = data_signature
        manifest.write_text(json.dumps(document))
    expected = run_training(continuous, common, world_size, expect_validation_display=True)
    resumed_common = common + (["--async-checkpoint"] if async_checkpoint else [])
    stop_flags = ["--stop-at-step", "3"]

    partial = run_training(resumed, resumed_common, world_size,
                           [*stop_flags, "--max-steps", "1"], expect_async=async_checkpoint,
                           expect_validation_display=True)
    assert partial["step"] == partial["val_step"] == 1
    assert partial["tokens_seen"] == partial["val_tokens"] == 32
    runtime = json.loads((resumed / "run/settings.json").read_text())["runtime"]
    assert (runtime["stop_at_step"], runtime["max_steps"], runtime["end_step"]) == (3, 1, 1)

    # The same absolute target survives an earlier --max-steps pause. Treating
    # it as three additional updates would incorrectly finish all four here.
    stopped = run_training(resumed, resumed_common, world_size, stop_flags,
                           expect_async=async_checkpoint, expect_validation_display=True)
    assert stopped["format"] == "warpstate-server-v1"
    assert stopped["step"] == stopped["val_step"] == 3
    assert stopped["tokens_seen"] == stopped["val_tokens"] == 96
    assert stopped["optimizer"]["state"]
    assert all(state["step"].item() == 3 for state in stopped["optimizer"]["state"].values())
    assert len(stopped["rng_by_rank"]) == world_size
    assert stopped["signature"] == partial["signature"] == expected["signature"]
    assert stopped["execution_history"] == partial["execution_history"] == expected["execution_history"]
    assert "stop_at_step" not in stopped["signature"]["settings"]
    validations = _records(resumed, "validation.jsonl")
    assert [(r["step"], r["tokens"]) for r in validations] == [(1, 32), (3, 96)]
    assert stopped["val_loss"] == validations[-1]["loss"]
    assert math.isfinite(stopped["val_loss"])
    assert not (resumed / "run/latest.pt.tmp").exists()
    runtime = json.loads((resumed / "run/settings.json").read_text())["runtime"]
    assert (runtime["stop_at_step"], runtime["max_steps"], runtime["end_step"]) == (3, None, 3)

    # Starting at or already beyond the requested target must not execute a
    # speculative optimizer update. Compare all checkpoint fields, including
    # model, Adam, RNG, counters, validation metadata, and the objective.
    metrics_before = (resumed / "run/metrics.jsonl").read_bytes()
    for target in (3, 2):
        unchanged = run_training(resumed, resumed_common, world_size,
                                 ["--stop-at-step", str(target)])
        assert_tensors_equal(unchanged, stopped)
        assert (resumed / "run/metrics.jsonl").read_bytes() == metrics_before
        assert not (resumed / "run/latest.pt.tmp").exists()

    # Removing a stopping policy needs neither --reconfigure nor a changed LR
    # horizon; it must continue exactly as the uninterrupted reference did.
    actual = run_training(resumed, resumed_common, world_size,
                          expect_async=async_checkpoint, expect_validation_display=True)
    assert actual["step"] == actual["val_step"] == 4
    assert actual["tokens_seen"] == actual["val_tokens"] == 105
    assert_tensors_equal(actual, expected)
    runtime = json.loads((resumed / "run/settings.json").read_text())["runtime"]
    assert (runtime["stop_at_step"], runtime["max_steps"], runtime["end_step"]) == (None, None, 4)
    for directory in (continuous, resumed):
        records = _records(directory, "metrics.jsonl")
        assert [r["step"] for r in records] == [1, 2, 3, 4]
        assert [r["tokens"] for r in records] == [32, 64, 96, 105]
        assert [r["lr"] for r in records] == pytest.approx([0.002, 0.002, 0.00155, 0.00065],
                                                         rel=0, abs=1e-15)
        assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in records)
    assert [(r["step"], r["tokens"]) for r in _records(resumed, "validation.jsonl")] == [
        (1, 32), (3, 96), (4, 105)]
