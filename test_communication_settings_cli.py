"""Both entry points validate and record communication experiment controls."""

import sys
from pathlib import Path

import pytest

import run_40b
from ops import benchmark_training


@pytest.fixture(params=[run_40b, benchmark_training], ids=["runner", "benchmark"])
def cli(request, monkeypatch):
    workspace = Path(__file__).resolve().parents[1]
    for name in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(name, str(workspace / "tmp"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    def parse(flags):
        monkeypatch.setattr(sys, "argv", [request.param.__file__, "--micro-batch", "4", *flags])
        return request.param.arguments()

    return parse


def test_current_overlap_behavior_remains_the_default(cli):
    args = cli(["--gradient-mode", "compiled-autograd-overlap"])
    assert not args.overlap_swap_collectives
    assert not args.overlap_runtime_estimates
    assert not args.overlap_producer_schedule
    assert args.overlap_memory_budget == 0


def test_overlap_experiment_is_explicit_and_in_resume_signature(cli):
    args = cli(["--gradient-mode", "compiled-autograd-overlap",
                "--overlap-swap-collectives", "--overlap-runtime-estimates",
                "--overlap-memory-budget", "0.02"])
    assert args.overlap_swap_collectives and args.overlap_runtime_estimates
    assert args.overlap_memory_budget == 0.02
    if hasattr(args, "grad_accum"):
        settings = run_40b.train_signature(args, {})["settings"]
        assert settings["overlap_swap_collectives"] is True
        assert settings["overlap_runtime_estimates"] is True
        assert settings["overlap_memory_budget"] == 0.02


def test_producer_scheduling_is_explicit_and_recorded(cli):
    args = cli(["--gradient-mode", "compiled-autograd-overlap", "--overlap-producer-schedule"])
    assert args.overlap_producer_schedule is True
    assert not args.overlap_swap_collectives and not args.overlap_runtime_estimates
    assert args.overlap_memory_budget == 0.0
    if hasattr(args, "grad_accum"):
        assert run_40b.train_signature(args, {})["settings"]["overlap_producer_schedule"] is True


@pytest.mark.parametrize("flags", [
    ["--overlap-swap-collectives"],
    ["--gradient-mode", "compiled-autograd", "--overlap-runtime-estimates"],
    ["--overlap-memory-budget", "0.02"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "nan"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "-0.02"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "0.26"],
    ["--overlap-producer-schedule"],
    ["--gradient-mode", "compiled-autograd", "--overlap-producer-schedule"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-producer-schedule",
     "--overlap-swap-collectives"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-producer-schedule",
     "--overlap-runtime-estimates"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-producer-schedule",
     "--overlap-memory-budget", "0.02"],
])
def test_reject_silent_noops_and_unbounded_memory_before_cuda(cli, flags):
    with pytest.raises(SystemExit) as error:
        cli(flags)
    assert error.value.code == 2
