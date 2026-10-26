"""Optional activation kernels preserve state, rounding and usable gradients."""

from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn.functional as F

from ops.activation_execution import apply_activation_execution, bf16_rmsnorm, bf16_swiglu
from warpstate import WarpStateConfig, WarpStateLM
from warpstate.model import RMSNorm


def _model():
    torch.manual_seed(943)
    return WarpStateLM(WarpStateConfig(vocab_size=48, d_model=32, n_heads=2,
        n_cores=1, logical_depth=2, ffn_hidden=56, chunk_size=4,
        gradient_checkpointing=False, parallel_memory=True)).train()


@pytest.mark.parametrize("mode", ["baseline", "swiglu", "rmsnorm", "all"])
@pytest.mark.parametrize("autocast", [False, True])
def test_cpu_fallback_retains_parameters_state_logits_and_gradients(mode, autocast):
    reference = _model()
    actual = copy.deepcopy(reference)
    parameters = tuple(actual.parameters())
    keys = tuple(actual.state_dict())
    apply_activation_execution(actual, mode)
    tokens = torch.randint(48, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = reference(tokens)
        result = actual(tokens)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    expected.float().square().mean().backward()
    result.float().square().mean().backward()
    assert tuple(actual.state_dict()) == keys
    assert all(a is b for a, b in zip(actual.parameters(), parameters, strict=True))
    for p, q in zip(actual.parameters(), reference.parameters(), strict=True):
        assert p.dtype == p.grad.dtype == torch.float32
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    reference.eval()
    actual.eval()
    with torch.no_grad():
        torch.testing.assert_close(actual(tokens), reference(tokens), rtol=0, atol=0)


def test_mode_validation_and_noop():
    model = _model()
    forward = model.cores[0].ffn.forward
    apply_activation_execution(model, "baseline")
    assert model.cores[0].ffn.forward == forward
    with pytest.raises(ValueError, match="Unknown activation"):
        apply_activation_execution(model, "invalid")
    apply_activation_execution(model, "all")
    with pytest.raises(ValueError, match="already been applied"):
        apply_activation_execution(model, "all")
    with pytest.raises(ValueError, match="FP32 master"):
        apply_activation_execution(_model().bfloat16(), "all")


# CUDA tests require an explicit maintenance window. Normal CPU CI never starts
# CUDA or competes with the running training process merely on test collection.
cuda_window = pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                                 reason="set WARPSTATE_TEST_CUDA=1 in an exclusive GPU window")


def _check_error(actual, expected, *, relative_rms=0.003, atol=0.0625):
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    difference = actual.float() - expected.float()
    relative = difference.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-20)
    assert relative.item() <= relative_rms, relative.item()
    assert difference.abs().max().item() <= atol, difference.abs().max().item()


@cuda_window
@pytest.mark.parametrize("shape", [(3, 224), (2, 32, 256), (257, 14336)])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_swiglu_forward_and_backward(shape, compiled):
    torch.manual_seed(947)
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    reference = source.clone().requires_grad_()
    actual = source.clone().requires_grad_()
    def reference_fn(x):
        up, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * up
    fn = lambda x: bf16_swiglu(x, eager_rounding=not compiled)
    if compiled:
        reference_fn = torch.compile(reference_fn, fullgraph=True)
        fn = torch.compile(fn, fullgraph=True)
    expected = reference_fn(reference)
    result = fn(actual)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    result.backward(grad)
    _check_error(result, expected, relative_rms=0.0002)
    _check_error(actual.grad, reference.grad, relative_rms=0.001)


@cuda_window
@pytest.mark.parametrize("shape", [(3, 24), (2, 16, 128), (257, 2048)])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_rmsnorm_rounding_and_master_gradient(shape, compiled):
    torch.manual_seed(949)
    reference = RMSNorm(shape[-1]).cuda()
    with torch.no_grad():
        reference.weight.uniform_(0.1, 2.0)
    weight = reference.weight.detach().clone().requires_grad_()
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x_ref = source.clone().requires_grad_()
    x_actual = source.clone().requires_grad_()
    fn = lambda x, w: bf16_rmsnorm(x, w, reference.eps, eager_rounding=not compiled)[0]
    ref_fn = reference
    if compiled:
        ref_fn = torch.compile(reference, fullgraph=True)
        fn = torch.compile(fn, fullgraph=True)
    expected = ref_fn(x_ref)
    result = fn(x_actual, weight)
    # A strided upstream gradient verifies backward's contiguous handling.
    grad = torch.randn((*shape, 2), device="cuda", dtype=torch.bfloat16)[..., 0]
    expected.backward(grad)
    result.backward(grad)
    _check_error(result, expected, relative_rms=0.0002)
    _check_error(x_actual.grad, x_ref.grad, relative_rms=0.001)
    _check_error(weight.grad, reference.weight.grad,
                 relative_rms=0.003, atol=0.5)
    assert weight.grad.dtype == torch.float32


@cuda_window
def test_cuda_compiled_model_with_bf16_residuals_and_optimizer():
    from ops.residual_experiment import apply_bf16_residuals
    from run_40b import compile_training_cores
    reference = _model().cuda()
    actual = copy.deepcopy(reference)
    apply_bf16_residuals(reference)
    apply_bf16_residuals(actual)
    parameters = tuple(actual.parameters())
    keys = tuple(actual.state_dict())
    apply_activation_execution(actual, "all")
    compile_training_cores(reference)
    compile_training_cores(actual)
    optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-4)
    tokens = torch.randint(48, (2, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_loss = reference(tokens, tokens, return_logits=False)[1]
        loss = actual(tokens, tokens, return_logits=False)[1]
    ref_loss.backward()
    loss.backward()
    torch.testing.assert_close(loss, ref_loss, rtol=0.001, atol=0.001)
    grads = torch.cat([p.grad.flatten() for p in parameters])
    ref_grads = torch.cat([p.grad.flatten() for p in reference.parameters()])
    assert F.cosine_similarity(grads, ref_grads, dim=0).item() > 0.999
    assert (grads - ref_grads).norm().item() / ref_grads.norm().item() < 0.02
    optimizer.step()
    assert tuple(actual.state_dict()) == keys
    assert all(a is b for a, b in zip(actual.parameters(), parameters, strict=True))
    assert all(p.dtype == p.grad.dtype == optimizer.state[p]["exp_avg"].dtype
               == optimizer.state[p]["exp_avg_sq"].dtype == torch.float32 for p in parameters)
    restored = _model().cuda()
    restored.load_state_dict(actual.state_dict(), strict=True)
    actual.eval()
    with torch.no_grad():
        # Outside autocast the BF16 residual wrapper and activation patches both
        # retain exactly the archived model's original FP32 inference behavior.
        torch.testing.assert_close(actual(tokens), restored.eval()(tokens), rtol=0, atol=0)
