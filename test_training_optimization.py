"""Two-GPU full-state continuation checks for optional training optimizations."""

import json
import math

import pytest
import torch

from test_training_execution import (
    TWO_GPUS,
    assert_tensors_equal,
    fixture_runs,
    run_training,
)


CASES = [
    pytest.param(
        ["--gradient-mode", "compiled-autograd-overlap", "--overlap-producer-schedule"],
        {"gradient_mode": "compiled-autograd-overlap", "activation_mode": "baseline",
         "overlap_producer_schedule": True, "overlap_swap_collectives": False,
         "overlap_runtime_estimates": False, "overlap_memory_budget": 0.0,
         "ddp_bucket_cap_mb": 128},
        id="producer-scheduled-overlap",
    ),
    pytest.param(
        ["--gradient-mode", "compiled-autograd-overlap", "--overlap-swap-collectives",
         "--overlap-runtime-estimates"],
        {"gradient_mode": "compiled-autograd-overlap", "activation_mode": "baseline",
         "overlap_swap_collectives": True, "overlap_runtime_estimates": True,
         "overlap_memory_budget": 0.0, "ddp_bucket_cap_mb": 128},
        id="runtime-estimated-overlap",
    ),
    pytest.param(
        ["--gradient-mode", "compiled-autograd-overlap", "--overlap-swap-collectives",
         "--overlap-memory-budget", "0.02", "--ddp-bucket-cap-mb", "64"],
        {"gradient_mode": "compiled-autograd-overlap", "activation_mode": "baseline",
         "overlap_swap_collectives": True, "overlap_runtime_estimates": False,
         "overlap_memory_budget": 0.02, "ddp_bucket_cap_mb": 64},
        id="waits-past-collectives",
    ),
    pytest.param(
        ["--gradient-mode", "compiled-autograd-overlap", "--matmul-plan", "__fixture_plan__"],
        {"gradient_mode": "compiled-autograd-overlap", "activation_mode": "baseline",
         "cache_core_weight_casts": False, "ddp_batched_grad_copy": False, "ddp_bucket_cap_mb": 128},
        id="matmul-epilogue-with-communication-overlap",
    ),
    pytest.param(
        ["--gradient-mode", "eager", "--activation-mode", "all",
         "--cache-core-weight-casts", "--ddp-batched-grad-copy",
         "--ddp-bucket-cap-mb", "64"],
        {"gradient_mode": "eager", "activation_mode": "all",
         "cache_core_weight_casts": True, "ddp_batched_grad_copy": True,
         "ddp_bucket_cap_mb": 64},
        id="activations-cached-casts-batched-copy",
    ),
    pytest.param(
        ["--gradient-mode", "compiled-autograd-overlap", "--activation-mode", "baseline",
         "--no-cache-core-weight-casts", "--no-ddp-batched-grad-copy"],
        {"gradient_mode": "compiled-autograd-overlap", "activation_mode": "baseline",
         "cache_core_weight_casts": False, "ddp_batched_grad_copy": False,
         "ddp_bucket_cap_mb": 128},
        id="compiled-autograd-overlap",
    ),
]


@TWO_GPUS
@pytest.mark.parametrize(("flags", "selected"), CASES)
def test_optimized_two_gpu_async_resume_preserves_full_state(tmp_path, flags, selected):
    (continuous, resumed), common = fixture_runs(tmp_path, ["continuous", "resumed"])
    common += [
        "--loss-mode", "compiled-full", "--compile-core", "--compile-mode", "default",
        "--bf16-residuals", "--no-activation-checkpointing", "--autocast-cache",
        "--eval-every", "2", "--save-every", "2", *flags,
    ]
    selected = dict(selected)
    if "__fixture_plan__" in common:
        from ops.matmul_execution import load_matmul_plan
        tile = {"block_m": 16, "block_n": 32, "block_k": 16, "num_warps": 4, "num_stages": 2}
        document = {"format": "warpstate-matmul-plan-v1", "entries": [
            {"kind": "swiglu", "m": 16, "n": 64, "k": 16,
             "a_stride": [16, 1], "b_stride": [1, 16], "dtype": "bfloat16",
             "algorithm": "persistent", "tile": tile},
            {"kind": "mm", "m": 64, "n": 16, "k": 16,
             "a_stride": [1, 64], "b_stride": [16, 1], "dtype": "bfloat16",
             "algorithm": "persistent_tma", "tile": tile},
        ]}
        recipe = tmp_path / "matmul-plan.json"
        recipe.write_text(json.dumps(document))
        common[common.index("__fixture_plan__")] = str(recipe)
        selected["matmul_execution"] = load_matmul_plan(recipe).metadata
    initial = torch.load(tmp_path / "initial.pt", map_location="cpu", weights_only=False)
    # The 73-token fixture consumes 32, 32, then 9 tokens. On its final update,
    # rank zero consumes the remaining tokens and rank one has entirely masked
    # targets. Resume must preserve that weighting and every rank's RNG state.
    expected = run_training(continuous, common, 2, timeout=300,
                            expect_validation_display=True)
    async_common = common + ["--async-checkpoint"]
    partial = run_training(resumed, async_common, 2, ["--max-steps", "1"],
                          timeout=300, expect_async=True, expect_validation_display=True)
    assert partial["tokens_seen"] == 32 and partial["step"] == 1
    assert partial["val_step"] == 1 and partial["val_tokens"] == 32
    actual = run_training(resumed, async_common, 2, timeout=300, expect_async=True,
                         expect_validation_display=True)

    assert actual["format"] == expected["format"] == "warpstate-server-v1"
    assert actual["step"] == expected["step"] == 3
    assert actual["tokens_seen"] == expected["tokens_seen"] == 73
    assert actual["origin"] == partial["origin"] == expected["origin"]
    assert actual["execution_history"] == partial["execution_history"] == expected["execution_history"]
    assert actual["signature"] == partial["signature"] == expected["signature"]
    settings = actual["signature"]["settings"]
    for name, value in selected.items():
        assert settings[name] == value
    assert settings["world_size"] == 2
    assert settings["compile_core"] is True
    assert settings["compile_mode"] == "default"
    assert settings["loss_mode"] == "compiled-full"
    assert settings["bf16_residuals"] is True
    assert settings["activation_checkpointing"] is False
    assert settings["seq_len"] * settings["micro_batch"] * settings["grad_accum"] * settings["world_size"] == 32
    assert len(actual["rng_by_rank"]) == len(partial["rng_by_rank"]) == 2
    for name in ("model", "optimizer", "rng_by_rank"):
        assert_tensors_equal(actual[name], expected[name])

    assert actual["model"].keys() == initial["model"].keys()
    assert all(not name.startswith(("module.", "_orig_mod.")) for name in actual["model"])
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
               for value in actual["model"].values())
    assert actual["optimizer"]["state"]
    for state in actual["optimizer"]["state"].values():
        assert state["step"].item() == 3
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].dtype == torch.float32 and torch.isfinite(state[name]).all()
    assert actual["val_loss"] == expected["val_loss"]
    assert math.isfinite(actual["val_loss"])
    assert actual["val_step"] == expected["val_step"] == 3
    assert actual["val_tokens"] == expected["val_tokens"] == 73

    for directory, async_enabled in [(continuous, False), (resumed, True)]:
        runtime = json.loads((directory / "run/settings.json").read_text())["runtime"]
        for name, value in selected.items():
            assert runtime[name] == value
        assert runtime["async_checkpoint"] is async_enabled
        assert runtime["bf16_residuals"] is True
        assert runtime["eval_every"] == runtime["save_every"] == 2
        assert not (directory / "run/latest.pt.tmp").exists()
        records = [json.loads(line) for line in
                   (directory / "run/metrics.jsonl").read_text().splitlines()]
        assert [record["step"] for record in records] == [1, 2, 3]
        assert [record["tokens"] for record in records] == [32, 64, 73]
        assert [record["lr"] for record in records] == [0.001, 0.002, 0.002]
        assert all(math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"])
                   for record in records)
    validations = [json.loads(line) for line in
                   (resumed / "run/validation.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in validations] == [32, 64, 73]
    assert all(math.isfinite(record["loss"]) for record in validations)
