"""CPU checks of the full-training backward-layout validation oracle."""
import json

import pytest
import torch

from ops.validate_backward_layout_training import (
    ACCUMULATION, MICRO_TOKENS, TOLERANCES, arguments, compare_named_tensors, exact_tree_equal,
    fixture_batch_cpu, fixture_config, tensor_comparison, update_plan, within_tolerance,
)


def test_fixture_retains_production_attention_and_ffn_geometry():
    cfg = fixture_config()
    assert (cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.chunk_size) == (2048, 32, 64, 128)
    assert cfg.ffn_hidden == 7168 and cfg.parallel_memory and cfg.dropout == 0
    assert cfg.n_cores == cfg.logical_depth == 1
    assert not cfg.gradient_checkpointing


def test_multicore_fixture_has_distinct_parameters_at_each_logical_layer():
    cfg = fixture_config(True, cores=2)
    assert cfg.n_cores == cfg.logical_depth == 2
    assert cfg.d_model == 2048 and cfg.ffn_hidden == 7168


@pytest.mark.parametrize("cores", [0, -1, True, 1.5])
def test_fixture_rejects_invalid_core_counts(cores):
    with pytest.raises(ValueError, match="positive integer"):
        fixture_config(cores=cores)


def test_unique_fixture_removes_repeated_embedding_indices_reproducibly():
    assert fixture_config(True).vocab_size == MICRO_TOKENS
    first = fixture_batch_cpu(191, 0, 0, 0, True)
    again = fixture_batch_cpu(191, 0, 0, 0, True)
    next_micro = fixture_batch_cpu(191, 0, 0, 1, True)
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert not torch.equal(first[0], next_micro[0])
    assert torch.equal(first[0].reshape(-1).sort().values, torch.arange(MICRO_TOKENS))
    assert first[1].min() >= 0 and first[1].max() < MICRO_TOKENS
    repeated, _ = fixture_batch_cpu(191, 0, 0, 0, False)
    assert repeated.unique().numel() <= 512 < repeated.numel()


@pytest.mark.parametrize("world_size", [1, 2])
def test_update_plan_has_accumulation_then_masked_final_rank(world_size):
    assert [update_plan(world_size, update) for update in range(3)] == [
        MICRO_TOKENS * ACCUMULATION * world_size, MICRO_TOKENS * ACCUMULATION * world_size, 9]
    assert ACCUMULATION >= 3
    final_valid = [max(0, min(MICRO_TOKENS, 9 - rank * MICRO_TOKENS)) for rank in range(world_size)]
    assert sum(final_valid) == 9
    if world_size == 2:
        assert final_valid == [9, 0]


@pytest.mark.parametrize(("actual", "expected"), [
    (torch.ones(3), torch.ones(4)),
    (torch.ones(3), torch.ones(3, dtype=torch.bfloat16)),
])
def test_oracle_rejects_mismatched_tensor_contract(actual, expected):
    with pytest.raises(ValueError, match="shape/dtype"):
        tensor_comparison(actual, expected)


def test_oracle_distinguishes_signed_zero_and_nonfinite_values():
    expected = torch.zeros(3, dtype=torch.bfloat16)
    observed = expected.clone()
    observed[0] = -0.0
    metrics = tensor_comparison(observed, expected)
    assert metrics["unequal"] == 0 and metrics["bitwise_unequal"] == 1
    assert not within_tolerance(metrics, "exact")
    observed[1] = float("nan")
    metrics = tensor_comparison(observed, expected)
    assert not metrics["finite"] and not within_tolerance(metrics, "gradient")
    json.dumps(metrics, allow_nan=False)


def test_oracle_checks_relative_norm_and_worst_element():
    expected = torch.ones(1000)
    observed = expected + TOLERANCES["gradient_relative_l2"] / 2
    assert within_tolerance(tensor_comparison(observed, expected), "gradient")
    observed = expected.clone()
    observed[0] += TOLERANCES["gradient_relative_max"] * 2
    metrics = tensor_comparison(observed, expected)
    assert metrics["relative_l2"] < TOLERANCES["gradient_relative_l2"]
    assert not within_tolerance(metrics, "gradient")
    assert not within_tolerance(tensor_comparison(expected * 1.02, expected), "gradient")


def test_oracle_uses_absolute_parameter_limit():
    expected = torch.zeros(3)
    assert within_tolerance(tensor_comparison(expected + 1e-5, expected), "parameter")
    assert not within_tolerance(tensor_comparison(expected + 3e-5, expected), "parameter")


def test_named_comparison_checks_every_parameter():
    expected = {"first": torch.ones(4), "second": torch.ones(2)}
    result = compare_named_tensors(expected, expected, "gradient")
    assert result["accepted"] and result["bitwise_equal"]
    with pytest.raises(ValueError, match="names"):
        compare_named_tensors({"first": torch.ones(4)}, expected, "gradient")
    result = compare_named_tensors({**expected, "second": torch.zeros(2)}, expected, "gradient")
    assert not result["accepted"] and not result["bitwise_equal"]


def test_optimizer_step_counter_must_match_exactly():
    wanted = {"weight/step": torch.tensor(40081.0)}
    actual = {"weight/step": torch.tensor(40082.0)}
    assert not compare_named_tensors(actual, wanted, "moment")["accepted"]


def test_resume_oracle_includes_dtype_signed_zero_and_metadata():
    expected = {"model": {"weight": torch.tensor([0.0])}, "rng": [torch.tensor([1, 2])],
                "execution": {"identity": "version1"}}
    assert exact_tree_equal(expected, expected)
    assert not exact_tree_equal({**expected, "model": {"weight": torch.tensor([-0.0])}}, expected)
    assert not exact_tree_equal({**expected, "execution": {"identity": "version2"}}, expected)
    assert not exact_tree_equal({**expected, "rng": [torch.tensor([1.0, 2.0])]}, expected)


def test_cli_rejects_outputs_outside_generated_validation_directory():
    with pytest.raises(SystemExit):
        arguments(["--output-dir", "/workspace/WarpState/rtx6000_40b/ops/bad-output"])
