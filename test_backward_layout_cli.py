"""Backward layout selection must be explicit, applicable, and resume-visible."""

from copy import deepcopy
from pathlib import Path
import sys

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
    compile_flag = "--compile-core" if request.param is run_40b else "--compile-cores"

    def parse(flags=(), *, supported=False, omit=()):
        common = ["--micro-batch", "4"]
        if supported:
            common += ["--gradient-mode", "compiled-autograd-overlap", compile_flag,
                       "--no-activation-checkpointing", "--bf16-residuals"]
        common = [flag for flag in common if flag not in omit]
        monkeypatch.setattr(sys, "argv", [request.param.__file__, *common, *flags])
        return request.param.arguments()

    return parse, compile_flag


def test_baseline_keeps_existing_cli_usable(cli):
    parse, _ = cli
    args = parse()
    assert args.backward_layout == "baseline"
    assert args.gradient_mode == "eager"


@pytest.mark.parametrize("mode", ["inspect", "paired"])
def test_supported_selection_requires_an_explicit_flag(cli, mode):
    parse, _ = cli
    args = parse(["--backward-layout", mode], supported=True)
    assert args.backward_layout == mode
    assert args.gradient_mode == "compiled-autograd-overlap"
    assert args.bf16_residuals
    assert not args.activation_checkpointing


@pytest.mark.parametrize("flags", [
    ["--gradient-mode", "eager"],
    ["--gradient-mode", "compiled-autograd"],
    ["--activation-checkpointing"],
    ["--micro-batch", "2"],
    ["--seq-len", "1024"],
    ["--activation-mode", "swiglu"],
    ["--matmul-plan", "/workspace/WarpState/rtx6000_40b/configs/not-loaded.json"],
    ["--cache-core-weight-casts"],
    ["--compile-mode", "reduce-overhead"],
    ["--compile-mode", "max-autotune-no-cudagraphs"],
    ["--ddp-batched-grad-copy"],
])
@pytest.mark.parametrize("mode", ["inspect", "paired"])
def test_reject_unsupported_selection_before_loading_cuda(cli, flags, mode):
    parse, _ = cli
    with pytest.raises(SystemExit) as error:
        parse(["--backward-layout", mode, *flags], supported=True)
    assert error.value.code == 2


@pytest.mark.parametrize("missing", ["compile", "bf16"])
@pytest.mark.parametrize("mode", ["inspect", "paired"])
def test_reject_missing_required_execution_setting(cli, missing, mode):
    parse, compile_flag = cli
    omit = (compile_flag,) if missing == "compile" else ("--bf16-residuals",)
    with pytest.raises(SystemExit) as error:
        parse(["--backward-layout", mode], supported=True, omit=omit)
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["inspect", "paired"])
def test_benchmark_rejects_unvalidated_fused_normalization(monkeypatch, capsys, mode):
    monkeypatch.setattr(sys, "argv", [benchmark_training.__file__, "--micro-batch", "4",
                       "--gradient-mode", "compiled-autograd-overlap", "--compile-cores",
                       "--bf16-residuals", "--backward-layout", mode, "--fused-rms-norm"])
    with pytest.raises(SystemExit) as error:
        benchmark_training.arguments()
    assert error.value.code == 2
    assert "--backward-layout requires" in capsys.readouterr().err


def runner_signature(monkeypatch, metadata):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", [run_40b.__file__, "--micro-batch", "4"])
    args = run_40b.arguments()
    args.backward_layout_metadata = metadata
    return run_40b.train_signature(args, {"recipe": "fixed-corpus"})


def test_signature_records_layout_implementation_identity(monkeypatch):
    metadata = {"mode": "paired", "identity": "implementation-a",
                "source_sha256": {"ops/backward_layout.py": "kernel-a"},
                "block": 256, "warps": 8}
    expected = deepcopy(metadata)
    signature = runner_signature(monkeypatch, metadata)
    assert signature["settings"]["backward_layout_execution"] == expected
    assert metadata == expected


def test_legacy_checkpoint_remains_compatible_with_baseline(monkeypatch):
    current = runner_signature(monkeypatch, None)
    legacy = deepcopy(current)
    del legacy["settings"]["backward_layout_execution"]
    before = deepcopy(legacy)
    assert run_40b.check_resume_signature(legacy, current) is False
    assert legacy == before


@pytest.mark.parametrize("change", [
    {"mode": "inspect", "identity": "implementation-a"},
    {"mode": "paired", "identity": "implementation-a"},
    {"mode": "paired", "identity": "implementation-b"},
    {"mode": "paired", "identity": "implementation-a", "source_sha256": {"kernel": "changed"}},
    None,
])
def test_execution_identity_changes_require_reconfiguration(monkeypatch, change):
    original = runner_signature(monkeypatch, {"mode": "paired", "identity": "implementation-a",
                                             "source_sha256": {"kernel": "original"}})
    candidate = runner_signature(monkeypatch, change)
    before = deepcopy((original, candidate))
    with pytest.raises(ValueError, match="Resume settings/data differ"):
        run_40b.check_resume_signature(original, candidate)
    assert run_40b.check_resume_signature(original, candidate, reconfigure=True) is True
    assert (original, candidate) == before


def test_same_layout_identity_does_not_require_reconfiguration(monkeypatch):
    original = runner_signature(monkeypatch, {"mode": "paired", "identity": "implementation-a"})
    assert run_40b.check_resume_signature(original, deepcopy(original)) is False


@pytest.mark.parametrize("change", ["data", "tokens", "batch"])
def test_layout_reconfiguration_cannot_change_training_objective(monkeypatch, change):
    original = runner_signature(monkeypatch, None)
    candidate = runner_signature(monkeypatch, {"mode": "paired", "identity": "implementation-a"})
    if change == "data":
        candidate["data"]["recipe"] = "another-corpus"
    elif change == "tokens":
        candidate["settings"]["train_tokens"] += 1
    else:
        candidate["settings"]["grad_accum"] += 1
    with pytest.raises(ValueError):
        run_40b.check_resume_signature(original, candidate, reconfigure=True)
