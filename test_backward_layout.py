"""Independent derivative/index checks for the isolated backward layout prototype."""

import os

import pytest
import torch

from ops.backward_layout import (PROFILE_SHAPE, launch_layout, make_inputs,
                                 prepare_layout, reference_layout, validate_inputs)


def token_major(value):
    b, n, h, c, d = value.shape
    return value.permute(0, 1, 3, 2, 4).reshape(b, n * c, h * d)


def test_pair_direction_and_projection_packing():
    inputs = make_inputs((1, 2, 2, 3, 4))
    value = torch.arange(1, 49, dtype=torch.bfloat16).reshape(inputs.q_memory.shape)
    inputs = inputs._replace(q_memory=value, q_local=torch.zeros_like(value),
        k_write=torch.zeros_like(inputs.k_write), k_local=value, v_local=value + 1,
        sin=torch.ones_like(inputs.sin), cos=torch.zeros_like(inputs.cos))
    q, k, v, _, _ = reference_layout(inputs).chunk(5, -1)
    wanted = token_major(torch.stack((value[..., 1::2], -value[..., ::2]), -1).flatten(-2))
    assert torch.equal(q, wanted) and torch.equal(k, wanted)
    assert torch.equal(v, token_major(value + 1))


def test_q_matches_model_rope_vjp_with_unequal_pair_coefficients():
    from warpstate.model import apply_rope
    inputs = make_inputs((2, 3, 2, 7, 6), seed=921)
    generator = torch.Generator().manual_seed(799)
    inputs = inputs._replace(sin=torch.randn(inputs.sin.shape, generator=generator),
                             cos=torch.randn(inputs.cos.shape, generator=generator))
    original = torch.zeros_like(inputs.q_memory, dtype=torch.float32, requires_grad=True)
    output = apply_rope(original, inputs.cos, inputs.sin)
    gradient, = torch.autograd.grad(output, original,
        grad_outputs=inputs.q_memory.float() + inputs.q_local.float())
    assert torch.equal(reference_layout(inputs).chunk(5, -1)[0], token_major(gradient.bfloat16()))


def test_k_intermediate_bf16_rounding_is_observable():
    inputs = make_inputs((1, 2, 2, 7, 16), seed=19)
    k = (inputs.k_write.transpose(-2, -1).float()
         * (1 - inputs.tanh_k.float().square()) + inputs.k_local.float())
    sine = k * inputs.sin
    unrounded_rotary = torch.stack((sine[..., 1::2], -sine[..., ::2]), -1).flatten(-2)
    wrong = token_major((unrounded_rotary + k * inputs.cos).bfloat16())
    assert torch.any(reference_layout(inputs).chunk(5, -1)[1] != wrong)


def test_gate_matches_independent_autograd_and_u_packing():
    inputs = make_inputs((1, 2, 2, 3, 4), seed=987)
    inputs = inputs._replace(tanh_u=torch.zeros_like(inputs.tanh_u))
    gate = inputs.gate.float().requires_grad_()
    blended = gate * inputs.local.float() + (1 - gate) * inputs.memory.float() / 2
    dy = inputs.grad_y.reshape(1, 2, 3, 2, 4).permute(0, 1, 3, 2, 4).float()
    derivative, = torch.autograd.grad(blended, gate, grad_outputs=dy)
    expected = token_major((derivative * gate * (1 - gate)).bfloat16())
    _, _, _, actual, u = reference_layout(inputs).chunk(5, -1)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-6)
    assert torch.equal(u, token_major(inputs.u_write))


@pytest.mark.parametrize("case", ["dtype", "k_shape", "rope_dtype", "strides", "device", "odd_head"])
def test_contract_rejects_unsupported_inputs(case):
    inputs = make_inputs((1, 2, 2, 3, 4))
    if case == "dtype":
        inputs = inputs._replace(q_memory=inputs.q_memory.float())
    elif case == "k_shape":
        inputs = inputs._replace(k_write=inputs.k_write[..., :-1])
    elif case == "rope_dtype":
        inputs = inputs._replace(sin=inputs.sin.bfloat16())
    elif case == "strides":
        inputs = inputs._replace(v_local=inputs.v_local.transpose(1, 2))
    elif case == "device":
        inputs = inputs._replace(gate=torch.empty(inputs.gate.shape, device="meta", dtype=torch.bfloat16))
    else:
        inputs = inputs._replace(q_memory=inputs.q_memory[..., :-1])
    with pytest.raises(ValueError):
        validate_inputs(inputs)


def test_cuda_launch_contract_rejects_cpu():
    with pytest.raises(ValueError, match="require CUDA"):
        prepare_layout(make_inputs((1, 1, 1, 1, 2)))


def test_benchmark_distinguishes_signed_zero_and_serializes_nonfinite_outputs():
    import json
    from ops.backward_layout_benchmark import output_errors
    expected = torch.zeros((1, 1, 10), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0, 0] = -0.0
    report = output_errors(actual, expected)
    assert report["q"]["unequal"] == 0
    assert report["q"]["bitwise_unequal"] == 1
    actual[0, 0, 3] = float("nan")
    report = output_errors(actual, expected)
    assert not report["k"]["finite"] and report["k"]["relative_rms"] is None
    json.dumps(report, allow_nan=False)


cuda_window = pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                                reason="requires an explicitly reserved CUDA device")


@cuda_window
@pytest.mark.parametrize("algorithm", ["paired", "single"])
@pytest.mark.parametrize("shape", [(1, 2, 3, 7, 10), (2, 3, 2, 16, 64)])
def test_cuda_ragged_and_production_head_arithmetic(algorithm, shape):
    inputs = make_inputs(shape, device="cuda", seed=455)
    # Distinct pair coefficients detect accidentally selecting the wrong RoPE lane.
    inputs = inputs._replace(sin=torch.randn_like(inputs.sin), cos=torch.randn_like(inputs.cos))
    saved = [value.clone() for value in inputs]
    actual = launch_layout(inputs, algorithm, block=256)
    expected = reference_layout(inputs)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    for value, wanted in zip(actual.chunk(5, -1), expected.chunk(5, -1), strict=True):
        assert torch.isfinite(value).all()
        error = (value.float() - wanted.float()).norm() / wanted.float().norm().clamp_min(1e-20)
        assert float(error) < 0.002
        torch.testing.assert_close(value, wanted, rtol=0.015, atol=1e-4)
    assert all(torch.equal(value, original) for value, original in zip(inputs, saved, strict=True))


@cuda_window
@pytest.mark.parametrize("algorithm", ["paired", "single"])
def test_cuda_production_shape_matches_exact_profile_kernels(algorithm):
    inputs = make_inputs(PROFILE_SHAPE, device="cuda", seed=817)
    expected = launch_layout(inputs, "profile")
    actual = launch_layout(inputs, algorithm, block=512)
    for value, wanted in zip(actual.chunk(5, -1), expected.chunk(5, -1), strict=True):
        assert torch.isfinite(value).all()
        assert torch.equal(value.view(torch.int16), wanted.view(torch.int16))


@cuda_window
def test_cuda_profile_scratch_cannot_alias():
    inputs = make_inputs(PROFILE_SHAPE, device="cuda")
    scratch = torch.empty_like(inputs.q_memory)
    for values in ((scratch, scratch), (inputs.q_local, scratch)):
        with pytest.raises(ValueError, match="Scratch tensors may not alias"):
            prepare_layout(inputs, "profile", scratch=values)
