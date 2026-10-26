"""Execution reconfiguration and distributed full-state continuation checks."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("execution_pipeline", ROOT / "run_40b.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
sys.path.insert(0, str(ROOT))
from warpstate import WarpStateConfig, WarpStateLM


def signature():
    return {
        "settings": {
            "seq_len": 8, "micro_batch": 2, "grad_accum": 2,
            "world_size": 1, "train_tokens": 73, "lr": 0.002,
            "min_lr": 0.0002, "warmup_steps": 2, "weight_decay": 0.1,
            "seed": 1337, "activation_checkpointing": True,
            "loss_chunk_size": 5, "compile": False, "compile_core": False,
            "loss_mode": "checkpointed", "compile_mode": "default",
            "bf16_residuals": False,
            "activation_mode": "baseline", "gradient_mode": "eager",
            "overlap_swap_collectives": False, "overlap_runtime_estimates": False,
            "overlap_memory_budget": 0.0,
            "overlap_producer_schedule": False,
            "cache_core_weight_casts": False, "matmul_execution": None,
            "backward_layout_execution": None,
            "ddp_bucket_cap_mb": 128, "ddp_batched_grad_copy": False,
        },
        "data": {"complete": True, "signature": {"recipe": 1, "source": "fixture"}},
    }


@pytest.mark.parametrize("missing", [
    ("world_size", "compile_core", "loss_mode", "compile_mode", "bf16_residuals"),
    ("loss_mode", "compile_mode"),
    ("bf16_residuals",),
    ("matmul_execution",),
    ("backward_layout_execution",),
    ("overlap_swap_collectives", "overlap_runtime_estimates", "overlap_memory_budget"),
    ("overlap_producer_schedule",),
    ("activation_mode", "gradient_mode", "ddp_bucket_cap_mb", "ddp_batched_grad_copy", "cache_core_weight_casts"),
])
def test_legacy_execution_defaults_do_not_require_reconfiguration(missing):
    old = signature()
    for key in missing:
        del old["settings"][key]
    original = deepcopy(old)
    assert pipeline.check_resume_signature(old, signature()) is False
    assert old == original


@pytest.mark.parametrize("execution", [
    {"micro_batch": 4, "grad_accum": 1},
    {"micro_batch": 1, "grad_accum": 4},
    {"world_size": 2, "grad_accum": 1},
    {"activation_checkpointing": False},
    {"loss_chunk_size": 16},
    {"compile": True},
    {"compile_core": True},
    {"loss_mode": "full"},
    {"loss_mode": "compiled-full"},
    {"bf16_residuals": True},
    {"activation_mode": "swiglu"},
    {"activation_mode": "rmsnorm"},
    {"activation_mode": "all"},
    {"gradient_mode": "compiled-autograd"},
    {"gradient_mode": "compiled-autograd-overlap"},
    {"gradient_mode": "compiled-autograd-overlap", "overlap_swap_collectives": True},
    {"gradient_mode": "compiled-autograd-overlap", "overlap_runtime_estimates": True},
    {"gradient_mode": "compiled-autograd-overlap", "overlap_memory_budget": 0.02},
    {"gradient_mode": "compiled-autograd-overlap", "overlap_producer_schedule": True},
    {"cache_core_weight_casts": True},
    {"matmul_execution": {"sha256": "fixture", "entries": []}},
    {"backward_layout_execution": {"mode": "paired", "identity": "fixture"}},
    {"ddp_bucket_cap_mb": 64},
    {"ddp_batched_grad_copy": True},
    {"compile_core": True, "compile_mode": "reduce-overhead"},
    {"compile_core": True, "compile_mode": "max-autotune-no-cudagraphs"},
])
def test_execution_changes_require_explicit_reconfiguration(execution):
    old = signature()
    new = deepcopy(old)
    new["settings"].update(execution)
    with pytest.raises(ValueError):
        pipeline.check_resume_signature(old, new)
    assert pipeline.check_resume_signature(old, new, reconfigure=True) is True


@pytest.mark.parametrize("change", [
    {"seq_len": 16, "micro_batch": 1},  # Same effective batch, changed context.
    {"train_tokens": 105},
    {"lr": 0.003},
    {"min_lr": 0.0003},
    {"warmup_steps": 3},
    {"weight_decay": 0.2},
    {"seed": 1338},
    {"world_size": 2},  # Effective batch doubles.
    {"grad_accum": 3},
])
def test_reconfiguration_rejects_objective_or_global_batch_changes(change):
    old = signature()
    new = deepcopy(old)
    new["settings"]["loss_mode"] = "full"
    new["settings"].update(change)
    with pytest.raises(ValueError):
        pipeline.check_resume_signature(old, new, reconfigure=True)


def test_reconfiguration_rejects_changed_data():
    old = signature()
    new = deepcopy(old)
    new["data"]["signature"]["source"] = "another-corpus"
    with pytest.raises(ValueError):
        pipeline.check_resume_signature(old, new, reconfigure=True)


def test_train_signature_records_selected_loss_compile_and_residual_modes():
    expected = signature()
    expected["settings"].update(loss_mode="compiled-full", compile_core=True,
                                 compile_mode="max-autotune-no-cudagraphs", bf16_residuals=True)
    args = types.SimpleNamespace(**expected["settings"])
    assert pipeline.train_signature(args, expected["data"]) == expected


def test_train_signature_records_matmul_recipe_and_implementation_identity():
    expected = signature()
    metadata = {"format": "warpstate-matmul-plan-v1", "sha256": "recipe-sha",
                "source_sha256": {"ops/matmul_kernels.py": "implementation-sha"}, "entries": []}
    args = types.SimpleNamespace(**expected["settings"], matmul_metadata=metadata)
    expected["settings"]["matmul_execution"] = metadata
    assert pipeline.train_signature(args, expected["data"]) == expected
    changed = deepcopy(expected)
    changed["settings"]["matmul_execution"]["source_sha256"]["ops/matmul_kernels.py"] = "changed"
    with pytest.raises(ValueError):
        pipeline.check_resume_signature(expected, changed)
    assert pipeline.check_resume_signature(expected, changed, reconfigure=True)


def test_train_signature_defaults_legacy_arguments_to_fp32_residuals():
    expected = signature()
    legacy = dict(expected["settings"])
    del legacy["bf16_residuals"]
    assert pipeline.train_signature(types.SimpleNamespace(**legacy), expected["data"]) == expected


def test_legacy_checkpoint_requires_reconfiguration_to_enable_bf16_residuals():
    old = signature()
    del old["settings"]["bf16_residuals"]
    original = deepcopy(old)
    new = signature()
    new["settings"]["bf16_residuals"] = True
    with pytest.raises(ValueError):
        pipeline.check_resume_signature(old, new)
    assert pipeline.check_resume_signature(old, new, reconfigure=True) is True
    assert old == original


@pytest.mark.parametrize(("flags", "enabled"), [
    ([], False),
    (["--bf16-residuals"], True),
    (["--bf16-residuals", "--no-bf16-residuals"], False),
])
def test_bf16_residual_cli_is_explicit_and_reversible(monkeypatch, flags, enabled):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", *flags])
    assert pipeline.arguments().bf16_residuals is enabled


def test_nondefault_compile_mode_requires_compilation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--compile-mode", "reduce-overhead"])
    with pytest.raises(SystemExit) as error:
        pipeline.arguments()
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [
    ["--gradient-mode", "compiled-autograd", "--ddp-batched-grad-copy"],
    ["--gradient-mode", "compiled-autograd-overlap", "--ddp-batched-grad-copy"],
    ["--ddp-bucket-cap-mb", "0"],
    ["--ddp-bucket-cap-mb", "nan"],
    ["--ddp-bucket-cap-mb", "inf"],
    ["--overlap-swap-collectives"],
    ["--gradient-mode", "compiled-autograd", "--overlap-runtime-estimates"],
    ["--overlap-memory-budget", "0.02"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "nan"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "inf"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "-0.01"],
    ["--gradient-mode", "compiled-autograd-overlap", "--overlap-memory-budget", "0.26"],
    ["--activation-mode", "invalid"],
    ["--activation-mode", "swiglu", "--gradient-mode", "compiled-autograd"],
    ["--activation-mode", "rmsnorm", "--gradient-mode", "compiled-autograd-overlap"],
    ["--activation-mode", "all", "--gradient-mode", "compiled-autograd-overlap"],
    ["--matmul-plan", "configs/example.json", "--cache-core-weight-casts", "--no-activation-checkpointing"],
    ["--matmul-plan", "configs/example.json", "--compile-core", "--compile-mode", "reduce-overhead"],
    ["--matmul-plan", "configs/example.json", "--gradient-mode", "compiled-autograd-overlap", "--activation-checkpointing"],
    ["--cache-core-weight-casts"],
    ["--cache-core-weight-casts", "--no-activation-checkpointing", "--gradient-mode", "compiled-autograd"],
    ["--cache-core-weight-casts", "--no-activation-checkpointing", "--compile-core", "--compile-mode", "reduce-overhead"],
    ["--gradient-mode", "compiled-autograd", "--compile-core", "--compile-mode", "reduce-overhead"],
])
def test_optimization_cli_rejects_invalid_or_incompatible_settings(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", *flags])
    with pytest.raises(SystemExit) as error:
        pipeline.arguments()
    assert error.value.code == 2


def test_optimization_cli_defaults_and_selected_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_40b.py"])
    defaults = pipeline.arguments()
    assert (defaults.activation_mode, defaults.gradient_mode) == ("baseline", "eager")
    assert defaults.ddp_bucket_cap_mb == 128 and not defaults.ddp_batched_grad_copy
    assert not defaults.cache_core_weight_casts
    assert defaults.matmul_plan is None
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--activation-mode", "all",
                                    "--ddp-bucket-cap-mb", "64", "--ddp-batched-grad-copy",
                                    "--cache-core-weight-casts", "--no-activation-checkpointing"])
    selected = pipeline.arguments()
    assert selected.activation_mode == "all"
    assert selected.cache_core_weight_casts
    assert selected.ddp_bucket_cap_mb == 64 and selected.ddp_batched_grad_copy


@pytest.mark.parametrize("compile_flag", ["--compile-core", "--compile"])
def test_loss_and_compile_modes_can_be_selected_together(monkeypatch, compile_flag):
    monkeypatch.setattr(sys, "argv", ["run_40b.py", compile_flag,
                                     "--compile-mode", "max-autotune-no-cudagraphs",
                                     "--loss-mode", "compiled-full"])
    args = pipeline.arguments()
    assert args.compile_mode == "max-autotune-no-cudagraphs"
    assert args.loss_mode == "compiled-full"


@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("target", [1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 63, 64, 65])
def test_update_partition_covers_tokens_once_and_keeps_ranks_in_step(world_size, target):
    micro_tokens = 8
    covered = []
    for rank in range(world_size):
        slices = list(pipeline.update_slices(target, micro_tokens, world_size, rank))
        assert len(slices) == math.ceil(target / (micro_tokens * world_size))
        for iteration, (offset, valid) in enumerate(slices):
            assert offset == (iteration * world_size + rank) * micro_tokens
            assert valid == max(0, min(micro_tokens, target - offset))
            covered.extend(range(offset, offset + valid))
    assert sorted(covered) == list(range(target))


def test_all_masked_rank_loss_is_finite_and_contributes_zero_gradients():
    cfg = WarpStateConfig(vocab_size=32, d_model=16, n_heads=2, n_cores=1,
                          logical_depth=1, ffn_hidden=32, chunk_size=8,
                          parallel_memory=True, loss_chunk_size=5)
    model = WarpStateLM(cfg)
    inputs = torch.arange(8).reshape(1, 8)
    labels = torch.full_like(inputs, -100)
    _, loss = model(inputs, labels, return_logits=False)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() == 0


def fixture_runs(tmp_path, names):
    cfg = WarpStateConfig(vocab_size=32768, d_model=16, n_heads=2, n_cores=1,
                          logical_depth=1, ffn_hidden=32, chunk_size=8,
                          parallel_memory=True, loss_chunk_size=5)
    config = tmp_path / "config.json"
    cfg.to_json(config)
    torch.manual_seed(133)
    weights = tmp_path / "initial.pt"
    torch.save({"config": vars(cfg), "model": WarpStateLM(cfg).state_dict(),
                "tokens_seen": 100}, weights)
    directories = []
    for name in names:
        directory = tmp_path / name
        data = directory / "data"
        data.mkdir(parents=True)
        np.arange(73, dtype="<u2").tofile(data / "train.bin")
        np.arange(256, dtype="<u2").tofile(data / "val.bin")
        args = types.SimpleNamespace(
            tokenizer=ROOT / "tokenizer.model", dataset="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-100BT", revision="main", train_tokens=73,
            val_tokens=256,
        )
        (data / "manifest.json").write_text(json.dumps({
            "complete": True, "signature": pipeline.data_signature(args),
        }))
        directories.append(directory)
    common = [
        str(ROOT / "run_40b.py"), "--no-bootstrap", "--test-small-gpu",
        "--stage", "train", "--config", str(config), "--weights", str(weights),
        "--train-tokens", "73", "--val-tokens", "256", "--seq-len", "8",
        "--micro-batch", "2", "--loss-chunk-size", "5", "--workers", "2",
        "--eval-batches", "2", "--eval-micro-batch", "1", "--eval-every", "1",
        "--lr", "0.002", "--min-lr", "0.0002", "--warmup-steps", "2",
        "--no-autocast-cache", "--vram-limit-gib", "1", "--vram-headroom-gib", "1",
    ]
    return directories, common


def run_training(directory, common, world_size, extra=(), *, timeout=180, expect_async=False,
                 expect_validation_display=False):
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes", "1",
                    "--nproc-per-node", str(world_size)]
    command += common + ["--work-dir", str(directory), "--grad-accum",
                         str(2 // world_size), *extra]
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=timeout)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    if expect_validation_display:
        records = {record["step"]: record for record in map(
            json.loads, (directory / "run/metrics.jsonl").read_text().splitlines())}
        step_lines = [line for line in result.stdout.splitlines() if line.startswith("step ")]
        assert step_lines, result.stdout
        for line in step_lines:
            step = int(line.split()[1].split("/")[0])
            record = records[step]
            if record["val_step"] == step:
                assert f"val {record['val_loss']:.5f} (step {record['val_step']})" in line, line
            else:
                assert " | val " not in line, line
    if expect_async:
        assert "Staging full CPU checkpoint" in result.stdout, result.stdout
        assert "for asynchronous checkpoint" in result.stdout, result.stdout
        assert "background write" in result.stdout, result.stdout
        runtime = json.loads((directory / "run/settings.json").read_text())["runtime"]
        assert runtime["async_checkpoint"] is True
    checkpoint = torch.load(directory / "run/latest.pt", map_location="cpu", weights_only=False)
    if expect_async:
        # The final full-state checkpoint must already be published when the
        # trainer exits; a queued write or leftover temporary file is insufficient.
        final_progress = f'step {checkpoint["step"]}, tokens={checkpoint["tokens_seen"]:,}; '
        assert final_progress + "background write" in result.stdout, result.stdout
        assert not (directory / "run/latest.pt.tmp").exists()
    return checkpoint


def assert_tensors_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tensors_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for value, other in zip(first, second, strict=True):
            assert_tensors_equal(value, other)
    else:
        assert first == second


TWO_GPUS = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two visible CUDA GPUs required")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("compile_mode", "async_checkpoint"), [
    pytest.param("default", False, id="default"),
    pytest.param("reduce-overhead", False, id="reduce-overhead"),
    pytest.param("max-autotune-no-cudagraphs", False, id="max-autotune-no-cudagraphs"),
    pytest.param("max-autotune-no-cudagraphs", True, id="max-autotune-no-cudagraphs-async"),
])
def test_single_gpu_compiled_full_state_resume(tmp_path, compile_mode, async_checkpoint):
    (continuous, resumed), common = fixture_runs(tmp_path, ["continuous", "resumed"])
    common += ["--loss-mode", "compiled-full", "--compile-core",
               "--compile-mode", compile_mode, "--no-activation-checkpointing", "--autocast-cache"]
    if async_checkpoint:
        # Exercise overlapping periodic saves, then flush at max-steps and final exit.
        common += ["--save-every", "1"]
    initial = torch.load(tmp_path / "initial.pt", map_location="cpu", weights_only=False)
    # Allow cold Inductor/autotuning compilation while keeping each subprocess bounded.
    # The continuous reference always saves synchronously, including the async case.
    expected = run_training(continuous, common, 1, timeout=300)
    resumed_common = common + (["--async-checkpoint"] if async_checkpoint else [])
    partial = run_training(resumed, resumed_common, 1, ["--max-steps", "1"],
                           timeout=300, expect_async=async_checkpoint)
    assert partial["tokens_seen"] == 32 and partial["step"] == 1
    actual = run_training(resumed, resumed_common, 1, timeout=300, expect_async=async_checkpoint)

    assert actual["format"] == expected["format"] == "warpstate-server-v1"
    assert actual["tokens_seen"] == expected["tokens_seen"] == 73
    assert actual["step"] == expected["step"] == 3
    assert actual["origin"] == partial["origin"] == expected["origin"]
    assert actual["signature"] == partial["signature"] == expected["signature"]
    settings = actual["signature"]["settings"]
    assert settings["world_size"] == 1
    assert settings["loss_mode"] == "compiled-full"
    assert settings["compile_core"] is True
    assert settings["compile_mode"] == compile_mode
    assert settings["activation_checkpointing"] is False
    assert settings["seq_len"] * settings["micro_batch"] * settings["grad_accum"] == 32
    assert len(actual["rng_by_rank"]) == 1
    for key in ("model", "optimizer", "rng_by_rank"):
        assert_tensors_equal(actual[key], expected[key])
    assert actual["model"].keys() == initial["model"].keys()
    assert all(not key.startswith(("module.", "_orig_mod.")) for key in actual["model"])
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
               for value in actual["model"].values())
    assert actual["optimizer"]["state"]
    for state in actual["optimizer"]["state"].values():
        assert state["step"].item() == 3
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
    assert math.isfinite(actual["val_loss"])
    assert actual["val_loss"] == expected["val_loss"]

    records = [json.loads(line) for line in (resumed / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3]
    assert [record["tokens"] for record in records] == [32, 64, 73]
    assert [record["lr"] for record in records] == [0.001, 0.002, 0.002]
    assert all(math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"])
               for record in records)
    validations = [json.loads(line) for line in (resumed / "run/validation.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in validations] == [32, 64, 73]
    assert all(math.isfinite(record["loss"]) for record in validations)


@TWO_GPUS
@pytest.mark.parametrize("loss_mode", ["checkpointed", "compiled-full"])
def test_two_gpu_full_state_resume_and_empty_final_rank(tmp_path, loss_mode):
    (continuous, resumed), common = fixture_runs(tmp_path, ["continuous", "resumed"])
    common += ["--loss-mode", loss_mode]
    if loss_mode == "compiled-full":
        common += ["--compile-core", "--compile-mode", "default", "--no-activation-checkpointing"]
    expected = run_training(continuous, common, 2)
    partial = run_training(resumed, common, 2, ["--max-steps", "1"])
    assert partial["tokens_seen"] == 32 and partial["step"] == 1
    actual = run_training(resumed, common, 2)
    assert actual["tokens_seen"] == expected["tokens_seen"] == 73
    assert actual["step"] == expected["step"] == 3
    assert actual["origin"] == expected["origin"]
    assert actual["signature"]["settings"]["world_size"] == 2
    assert actual["signature"]["settings"]["loss_mode"] == loss_mode
    if loss_mode == "compiled-full":
        assert actual["signature"]["settings"]["compile_core"] is True
        assert actual["signature"]["settings"]["activation_checkpointing"] is False
    assert len(actual["rng_by_rank"]) == 2
    for key in ("model", "optimizer", "rng_by_rank"):
        assert_tensors_equal(actual[key], expected[key])
    assert all(not key.startswith(("module.", "_orig_mod.")) for key in actual["model"])
    records = [json.loads(line) for line in (resumed / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in records] == [32, 64, 73]
    assert all(math.isfinite(record["loss"]) for record in records)


@TWO_GPUS
def test_two_gpu_bf16_residual_async_full_state_resume_and_empty_final_rank(tmp_path):
    (continuous, resumed), common = fixture_runs(tmp_path, ["continuous", "resumed"])
    common += ["--loss-mode", "compiled-full", "--compile-core", "--compile-mode", "default",
               "--no-activation-checkpointing", "--autocast-cache", "--bf16-residuals",
               "--eval-every", "2", "--save-every", "2"]
    initial = torch.load(tmp_path / "initial.pt", map_location="cpu", weights_only=False)
    # A synchronous continuous reference catches snapshot aliasing while updates
    # run during periodic writes. The split run also tests the final flush on
    # max-steps and the masked final rank when nine tokens remain across two GPUs.
    expected = run_training(continuous, common, 2, timeout=300, expect_validation_display=True)
    async_common = common + ["--async-checkpoint"]
    partial = run_training(resumed, async_common, 2, ["--max-steps", "1"],
                           timeout=300, expect_async=True, expect_validation_display=True)
    assert partial["tokens_seen"] == 32 and partial["step"] == 1
    assert partial["val_step"] == 1 and partial["val_tokens"] == 32
    actual = run_training(resumed, async_common, 2, timeout=300, expect_async=True,
                          expect_validation_display=True)

    assert actual["format"] == expected["format"] == "warpstate-server-v1"
    assert actual["tokens_seen"] == expected["tokens_seen"] == 73
    assert actual["step"] == expected["step"] == 3
    assert actual["origin"] == partial["origin"] == expected["origin"]
    assert actual["signature"] == partial["signature"] == expected["signature"]
    settings = actual["signature"]["settings"]
    assert settings["world_size"] == 2
    assert settings["loss_mode"] == "compiled-full"
    assert settings["compile_core"] is True
    assert settings["compile_mode"] == "default"
    assert settings["activation_checkpointing"] is False
    assert settings["bf16_residuals"] is True
    assert settings["seq_len"] * settings["micro_batch"] * settings["grad_accum"] * settings["world_size"] == 32
    assert len(actual["rng_by_rank"]) == 2
    for key in ("model", "optimizer", "rng_by_rank"):
        assert_tensors_equal(actual[key], expected[key])
    assert actual["model"].keys() == initial["model"].keys()
    assert all(not key.startswith(("module.", "_orig_mod.")) for key in actual["model"])
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
               for value in actual["model"].values())
    assert actual["optimizer"]["state"]
    for state in actual["optimizer"]["state"].values():
        assert state["step"].item() == 3
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
    assert math.isfinite(actual["val_loss"])
    assert actual["val_loss"] == expected["val_loss"]
    assert actual["val_step"] == expected["val_step"] == 3
    assert actual["val_tokens"] == expected["val_tokens"] == 73

    continuous_records = [json.loads(line) for line in
                          (continuous / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["val_step"] for record in continuous_records] == [0, 2, 3]
    assert [record["val_tokens"] for record in continuous_records] == [0, 64, 73]
    continuous_validations = [json.loads(line) for line in
                              (continuous / "run/validation.jsonl").read_text().splitlines()]
    assert [record["step"] for record in continuous_validations] == [2, 3]
    assert all(record["val_loss"] == validation["loss"] for record, validation in
               zip(continuous_records[1:], continuous_validations, strict=True))
    assert math.isfinite(continuous_records[0]["val_loss"])
    for directory in (continuous, resumed):
        runtime = json.loads((directory / "run/settings.json").read_text())["runtime"]
        assert runtime["eval_every"] == runtime["save_every"] == 2

    records = [json.loads(line) for line in (resumed / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3]
    assert [record["tokens"] for record in records] == [32, 64, 73]
    assert [record["lr"] for record in records] == [0.001, 0.002, 0.002]
    assert all(math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"])
               for record in records)
    validations = [json.loads(line) for line in (resumed / "run/validation.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in validations] == [32, 64, 73]
    assert all(math.isfinite(record["loss"]) for record in validations)
    assert [record["val_step"] for record in records] == [1, 2, 3]
    assert [record["val_tokens"] for record in records] == [32, 64, 73]
    assert all(record["val_loss"] == validation["loss"] for record, validation in
               zip(records, validations, strict=True))


@TWO_GPUS
def test_one_to_two_gpu_reconfiguration_preserves_training_state(tmp_path):
    (serial, distributed), common = fixture_runs(tmp_path, ["serial", "distributed"])
    expected = run_training(serial, common, 1)
    partial = run_training(distributed, common, 1, ["--max-steps", "1"])
    assert partial["tokens_seen"] == 32 and partial["step"] == 1
    actual = run_training(distributed, common, 2, ["--reconfigure"])
    assert actual["tokens_seen"] == expected["tokens_seen"] == 73
    assert actual["step"] == expected["step"] == 3
    assert actual["origin"] == partial["origin"] == expected["origin"]
    settings = actual["signature"]["settings"]
    assert settings["seq_len"] * settings["micro_batch"] * settings["grad_accum"] * settings["world_size"] == 32
    assert settings["world_size"] == 2
    assert len(actual["rng_by_rank"]) == 2
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for key, value in actual["model"].items():
        torch.testing.assert_close(value, expected["model"][key], rtol=0.01, atol=2e-5)
    for parameter, state in actual["optimizer"]["state"].items():
        reference = expected["optimizer"]["state"][parameter]
        assert state["step"].item() == reference["step"].item() == 3
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key], reference[key], rtol=0.03, atol=2e-7)
    records = [json.loads(line) for line in (distributed / "run/metrics.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in records] == [32, 64, 73]
    assert [record["lr"] for record in records] == [0.001, 0.002, 0.002]
