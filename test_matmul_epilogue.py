"""Projection rounding, masking and output ownership for the fused FFN GEMM."""

import os

import pytest
import torch
import torch.nn.functional as F

from ops.matmul_epilogue import SUPPORTED_TILES, launch_swiglu_mm


@pytest.mark.parametrize("eager_rounding", [False, True])
def test_reference_preserves_projection_and_activation_rounding(eager_rounding):
    torch.manual_seed(554)
    a = torch.randn(7, 31, dtype=torch.bfloat16)
    b = torch.randn(46, 31, dtype=torch.bfloat16).T
    packed, activation = launch_swiglu_mm(a, b, "aten", SUPPORTED_TILES[0], eager_rounding)
    expected = a @ b
    up, gate = expected.chunk(2, dim=-1)
    wanted = F.silu(gate) * up if eager_rounding else (F.silu(gate.float()) * up.float()).bfloat16()
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(activation, wanted, rtol=0, atol=0)
    assert packed.is_contiguous() and activation.is_contiguous()
    assert packed.untyped_storage().data_ptr() != activation.untyped_storage().data_ptr()


@pytest.mark.parametrize("case", ["dtype", "shape", "odd_width", "empty", "cpu", "tile", "algorithm"])
def test_unsupported_execution_fails_before_a_kernel_launch(case):
    a = torch.ones(4, 8, dtype=torch.bfloat16)
    b = torch.ones(8, 16, dtype=torch.bfloat16)
    algorithm, tile = "aten", SUPPORTED_TILES[0]
    if case == "dtype":
        a = a.float()
    elif case == "shape":
        b = b[:7]
    elif case == "odd_width":
        b = b[:, :15]
    elif case == "empty":
        a = a[:0]
    elif case == "cpu":
        algorithm = "tiled"
    elif case == "tile":
        tile = (256, 256, 64, 8, 4)
    elif case == "algorithm":
        algorithm = "unsupported"
    with pytest.raises(ValueError):
        launch_swiglu_mm(a, b, algorithm, tile)


cuda_window = pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                                reason="requires an explicitly reserved CUDA device")


@cuda_window
@pytest.mark.parametrize("algorithm", ["tiled", "persistent"])
@pytest.mark.parametrize("eager_rounding", [False, True])
@pytest.mark.parametrize("shape", [(13, 46, 31), (128, 256, 128), (257, 192, 193)])
@pytest.mark.parametrize("tile", [SUPPORTED_TILES[0], SUPPORTED_TILES[-1]])
def test_cuda_projection_and_saved_activation_match_reference(algorithm, eager_rounding, shape, tile):
    m, packed_width, k = shape
    torch.manual_seed(561)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    b = (torch.randn(packed_width, k, device="cuda", dtype=torch.bfloat16) * 0.1).T
    packed, activation = launch_swiglu_mm(a, b, algorithm, tile, eager_rounding)
    expected, wanted = launch_swiglu_mm(a, b, "aten", SUPPORTED_TILES[0], eager_rounding)
    for result, reference in ((packed, expected), (activation, wanted)):
        assert result.dtype == torch.bfloat16 and result.is_contiguous()
        assert torch.isfinite(result).all()
        delta = result.float() - reference.float()
        relative_l2 = delta.norm() / reference.float().norm().clamp_min(1e-12)
        # cuBLAS may use reduced-precision reduction for ragged BF16 GEMMs.
        # The packed error can grow through the activation/product; bound both
        # and independently check our FP32 accumulation and saved projection.
        assert relative_l2.item() < 0.006
        torch.testing.assert_close(result, reference, rtol=0.02, atol=0.001)
    accumulated = (a.float() @ b.float()).bfloat16()
    projection_relative_l2 = ((packed.float() - accumulated.float()).norm()
                              / accumulated.float().norm().clamp_min(1e-12))
    assert projection_relative_l2.item() < 0.001
    # The activation must consume the RETURNED rounded projection. This detects
    # an incorrect shortcut that activates unrounded FP32 tensor-core accumulators.
    up, gate = packed.chunk(2, dim=-1)
    packed_reference = F.silu(gate) * up if eager_rounding else (F.silu(gate.float()) * up.float()).bfloat16()
    torch.testing.assert_close(activation, packed_reference, rtol=0.01, atol=0.0002)
