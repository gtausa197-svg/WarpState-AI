"""Matrix tile validation, dispatch signatures, and explicit CUDA correctness."""

import json
import os

import pytest
import torch
from torch.library import triton_op

from ops.matmul_kernels import (
    DEFAULT_TILE, TUNING_TILES, launch_mm, matrix_signature, normalize_tile,
    policy_key, tile_dict,
)


@pytest.mark.parametrize("tile", [
    [], (128, 128, 32, 4), (128, 128, 32, True, 3), (120, 128, 32, 4, 3),
    (128, 128, 32, 2, 3), (128, 128, 32, 4, 8), (256, 256, 128, 8, 4),
    {"block_m": 128},
])
def test_invalid_or_oversized_tiles_fail(tile):
    with pytest.raises(ValueError):
        normalize_tile(tile)


def test_tuning_tiles_and_json_round_trip():
    for tile in TUNING_TILES:
        assert normalize_tile(json.loads(json.dumps(tile_dict(tile)))) == tile
    assert normalize_tile() == DEFAULT_TILE


def test_signature_distinguishes_strides_even_when_shapes_match():
    a = torch.empty((16, 32), dtype=torch.bfloat16)
    b = torch.empty((64, 32), dtype=torch.bfloat16).T
    signature = matrix_signature(a, b)
    assert signature == {"m": 16, "n": 64, "k": 32, "a_stride": [32, 1],
                         "b_stride": [1, 32], "dtype": "bfloat16"}
    assert json.loads(policy_key(a, b)) == signature
    assert policy_key(a, b) != policy_key(a, b.contiguous())


def test_no_cpu_or_unknown_algorithm_fallback():
    a = torch.empty((16, 32), dtype=torch.bfloat16)
    b = torch.empty((32, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        launch_mm(a, b)
    with pytest.raises(ValueError, match="Unknown"):
        launch_mm(a, b, "fp8")


def test_tuner_kind_defaults_and_selected_tiles():
    from ops.tune_matmul import arguments, cases

    generic = arguments(["--list-cases"])
    assert generic.operations == ["fwd", "dw"]
    assert generic.algorithms == ["tiled", "persistent", "persistent_tma"]
    fused = arguments(["--list-cases", "--kind", "swiglu", "--tile", "128,128,32,8,3"])
    assert fused.operations == ["fwd"]
    assert fused.algorithms == ["tiled", "persistent"]
    assert fused.tile == [(128, 128, 32, 8, 3)]
    forward, dw = cases(8192, generic.operations)
    assert forward["a_stride"] == [2048, 1] and forward["b_stride"] == [1, 2048]
    assert dw["a_stride"] == [1, 14336] and dw["b_stride"] == [2048, 1]


@pytest.mark.parametrize("flags", [
    ["--kind", "swiglu", "--operations", "dw"],
    ["--kind", "swiglu", "--algorithms", "persistent_tma"],
    ["--kind", "swiglu", "--tile", "256,256,32,8,3"],
    ["--tile", "128,128,32,4"],
    ["--input-scale", "nan"], ["--input-scale", "0"],
    ["--operations", "fwd", "fwd"],
    ["--algorithms", "tiled", "tiled"],
    ["--tile", "128,128,32,4,3", "--tile", "128,128,32,4,3"],
    ["--output-dir", "/tmp/outside-workspace"],
])
def test_tuner_rejects_unsupported_kind_or_unsafe_output(flags):
    from ops.tune_matmul import arguments

    with pytest.raises(SystemExit):
        arguments(["--list-cases", *flags])


def test_tuner_checks_all_outputs_and_preserves_bf16():
    from ops.tune_matmul import output_metrics

    expected = torch.ones((4, 8), dtype=torch.bfloat16)
    metrics = output_metrics((expected.clone(), expected * 2), (expected, expected))
    assert metrics["output_metrics"][0]["relative_rms_error"] == 0
    assert metrics["relative_rms_error"] == 1
    with pytest.raises(ValueError, match="BF16"):
        output_metrics(expected.float(), expected)
    with pytest.raises(ValueError, match="Nonfinite"):
        output_metrics((expected, expected * float("nan")), (expected, expected))


def test_tuner_cannot_select_a_median_winner_with_a_losing_trial():
    from ops.tune_matmul import paired_timing_metrics

    trials = [{"aten_before_ms": 2.0, "candidate_ms": candidate, "aten_after_ms": 2.0}
              for candidate in (1.5, 1.5, 2.1)]
    metrics = paired_timing_metrics(trials, minimum_gain=0.01)
    assert metrics["paired_gain"] > 0.3
    assert metrics["minimum_trial_gain"] < 0
    assert not metrics["timing_eligible"]
    drifted = [{"aten_before_ms": 1.0, "candidate_ms": 1.4, "aten_after_ms": 2.0}]
    metrics = paired_timing_metrics(drifted, minimum_gain=0.01)
    assert metrics["timing_eligible"]
    assert metrics["maximum_control_drift_fraction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("timing", [0, -1, float("nan"), float("inf")])
def test_tuner_rejects_invalid_timing_metrics(timing):
    from ops.tune_matmul import paired_timing_metrics

    with pytest.raises(ValueError, match="finite and positive"):
        paired_timing_metrics([{"aten_before_ms": 1, "candidate_ms": timing,
                                "aten_after_ms": 1}], minimum_gain=0.01)


@pytest.mark.parametrize("offset_operand", ["a", "b"])
def test_tma_rejects_misaligned_offset_without_launching(offset_operand):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    mode = FakeTensorMode()
    # Construct fake CUDA operands from meta storage; this test never starts
    # CUDA or reaches the custom-op launch, including on CPU-only machines.
    tensors = []
    for name, shape in (("a", (32, 64)), ("b", (64, 32))):
        offset = 1 if name == offset_operand else 0
        storage = torch.empty(shape[0] * shape[1] + offset, device="meta", dtype=torch.bfloat16)
        view = storage.as_strided(shape, (shape[1], 1), offset)
        tensors.append(FakeTensor(mode, view, torch.device("cuda:0")))
    with pytest.raises(ValueError, match="16-byte-aligned storage offsets"):
        launch_mm(*tensors, "persistent_tma")


cuda_window = pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                                 reason="requires an explicitly reserved CUDA device")


@triton_op("warpstate_test::candidate_mm", mutates_args={})
def _test_mm(a: torch.Tensor, b: torch.Tensor, algorithm: str) -> torch.Tensor:
    return launch_mm(a, b, algorithm, DEFAULT_TILE)


@cuda_window
@pytest.mark.parametrize("algorithm", ["tiled", "persistent"])
@pytest.mark.parametrize("layouts", [(False, False), (False, True), (True, False), (True, True)])
def test_cuda_odd_shapes_and_transposed_operands(algorithm, layouts):
    torch.manual_seed(77)
    m, n, k = 97, 113, 129
    a = torch.randn((k, m) if layouts[0] else (m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k) if layouts[1] else (k, n), device="cuda", dtype=torch.bfloat16)
    a = a.T if layouts[0] else a
    b = b.T if layouts[1] else b
    acopy, bcopy = a.clone(), b.clone()
    with torch.inference_mode():
        expected = torch.mm(a, b)
        output = launch_mm(a, b, algorithm)
    error = (output.float() - expected.float()).square().mean().sqrt()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert error / expected.float().square().mean().sqrt() < 0.003
    torch.testing.assert_close(a, acopy, rtol=0, atol=0)
    torch.testing.assert_close(b, bcopy, rtol=0, atol=0)


@cuda_window
@pytest.mark.parametrize("algorithm", ["tiled", "persistent", "persistent_tma"])
def test_cuda_launch_is_usable_under_compiled_triton_op(algorithm):
    a = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((96, 64), device="cuda", dtype=torch.bfloat16).T
    compiled = torch.compile(lambda x, y: _test_mm(x, y, algorithm), fullgraph=True)
    torch.testing.assert_close(compiled(a, b), _test_mm(a, b, algorithm), rtol=0, atol=0)


@cuda_window
@pytest.mark.parametrize("layouts", [(False, False), (False, True), (True, False), (True, True)])
def test_cuda_tma_layouts_with_masked_output_tiles(layouts):
    m, n, k = 96, 112, 128
    a = torch.randn((k, m) if layouts[0] else (m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k) if layouts[1] else (k, n), device="cuda", dtype=torch.bfloat16)
    a, b = a.T if layouts[0] else a, b.T if layouts[1] else b
    expected = torch.mm(a, b)
    actual = launch_mm(a, b, "persistent_tma")
    difference = actual.float() - expected.float()
    assert torch.isfinite(actual).all()
    assert difference.square().mean().sqrt() / expected.float().square().mean().sqrt() < 0.003
