"""Exact plan selection and autograd ownership for optional BF16 matrix kernels."""

import copy
import json
import os

import pytest
import torch

from ops.matmul_execution import (
    PLAN_FORMAT, apply_matmul_execution, load_matmul_plan,
    matmul_plan_from_dict, planned_matmul,
)
from warpstate import WarpStateConfig, WarpStateLM


def _entry(*, kind="mm", m=8, n=32, k=16, algorithm="tiled"):
    return {"kind": kind, "m": m, "n": n, "k": k,
            "a_stride": [k, 1], "b_stride": [1, k], "dtype": "bfloat16",
            "algorithm": algorithm,
            "tile": {"block_m": 32, "block_n": 32, "block_k": 32,
                     "num_warps": 4, "num_stages": 3}}


def _document(entries=None):
    return {"format": PLAN_FORMAT, "entries": entries if entries is not None else [_entry()]}


def _model():
    torch.manual_seed(492)
    return WarpStateLM(WarpStateConfig(vocab_size=48, d_model=16, n_heads=2,
        n_cores=1, logical_depth=1, ffn_hidden=32, chunk_size=4,
        parallel_memory=True, gradient_checkpointing=False)).train()


def test_plan_exact_shape_stride_key_and_source_provenance(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_document()))
    plan = load_matmul_plan(path)
    assert plan.select("mm", 8, 32, 16, (16, 1), (1, 16)).algorithm == "tiled"
    assert plan.select("mm", 8, 32, 16, (16, 1), (32, 1)) is None
    assert plan.select("mm", 7, 32, 16, (16, 1), (1, 16)) is None
    assert plan.select("swiglu", 8, 32, 16, (16, 1), (1, 16)) is None
    assert len(plan.metadata["sha256"]) == 64
    assert "ops/matmul_execution.py" in plan.metadata["source_sha256"]
    assert "ops/matmul_kernels.py" in plan.metadata["source_sha256"]
    path.write_text(json.dumps(_document(), indent=4))
    assert load_matmul_plan(path).metadata == plan.metadata
    altered = _document()
    altered["entries"][0]["tile"]["block_m"] = 64
    assert matmul_plan_from_dict(altered).sha256 != plan.sha256
    fused = matmul_plan_from_dict(_document([_entry(kind="swiglu")]))
    assert "ops/matmul_epilogue.py" in fused.metadata["source_sha256"]
    assert "ops/activation_execution.py" in fused.metadata["source_sha256"]


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(format="other"),
    lambda d: d.update(unknown=True),
    lambda d: d["entries"][0].update(dtype="float16"),
    lambda d: d["entries"][0].update(algorithm="unknown"),
    lambda d: d["entries"][0].update(m=True),
    lambda d: d["entries"][0].update(n=0),
    lambda d: d["entries"][0].update(a_stride=[0, 1]),
    lambda d: d["entries"][0].update(b_stride=[1]),
    lambda d: d["entries"][0].update(kind="swiglu", n=31),
    lambda d: d["entries"][0].update(kind="swiglu", algorithm="persistent_tma"),
    lambda d: d["entries"][0]["tile"].update(block_k=0),
    lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])),
])
def test_invalid_plan_fails_before_model_or_gpu_work(mutation):
    document = _document()
    mutation(document)
    with pytest.raises(ValueError):
        matmul_plan_from_dict(document)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"format":"warpstate-matmul-plan-v1","entries":[],"entries":[]}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_matmul_plan(path)


@pytest.mark.parametrize("compiled", [False, True])
def test_unsupported_cpu_matmul_retains_cast_and_fp32_gradient_accumulation(compiled):
    torch.manual_seed(495)
    plan = matmul_plan_from_dict(_document())
    expected = torch.nn.Parameter(torch.randn(32, 16))
    actual = torch.nn.Parameter(expected.detach().clone())
    original = lambda x, weight: torch.mm(x, weight.to(torch.bfloat16).T)
    selected = lambda x, weight: planned_matmul(x, weight.to(torch.bfloat16).T, plan)
    if compiled:
        original = torch.compile(original, backend="aot_eager", fullgraph=True)
        selected = torch.compile(selected, backend="aot_eager", fullgraph=True)
    for _ in range(8):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        grad = torch.randn(8, 32, dtype=torch.bfloat16)
        y = original(x, expected)
        z = selected(x, actual)
        torch.testing.assert_close(z, y, rtol=0, atol=0)
        y.backward(grad)
        z.backward(grad)
        assert actual.grad.dtype == torch.float32
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_model_patch_preserves_parameters_state_and_cpu_eval():
    reference = _model()
    actual = copy.deepcopy(reference)
    masters = tuple(actual.parameters())
    keys = tuple(actual.state_dict())
    apply_matmul_execution(actual, matmul_plan_from_dict(_document([_entry(kind="swiglu")])))
    inputs = torch.randint(48, (2, 8))
    for training in [True, False]:
        reference.train(training)
        actual.train(training)
        with torch.set_grad_enabled(training):
            torch.testing.assert_close(actual(inputs), reference(inputs), rtol=0, atol=0)
    assert tuple(actual.state_dict()) == keys
    assert all(a is b for a, b in zip(masters, actual.parameters(), strict=True))
    with pytest.raises(ValueError, match="already been applied"):
        apply_matmul_execution(actual, matmul_plan_from_dict(_document()))
    with pytest.raises(ValueError, match="FP32 master"):
        apply_matmul_execution(_model().bfloat16(), matmul_plan_from_dict(_document()))


def test_shape_policy_only_patches_relevant_projection_modules():
    model = _model()
    core = model.cores[0]
    originals = {name: module.forward for name, module in core.named_modules()
                 if isinstance(module, torch.nn.Linear)}
    apply_matmul_execution(model, matmul_plan_from_dict(
        _document([_entry(m=16, n=64, k=16)])))
    assert core.ffn.up_gate.forward != originals["ffn.up_gate"]
    assert core.ffn.down.forward == originals["ffn.down"]
    assert core.in_proj.forward == originals["in_proj"]
    assert core.out_proj.forward == originals["out_proj"]


cuda_window = pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                                 reason="requires an explicit exclusive GPU test window")


@cuda_window
@pytest.mark.parametrize("algorithm", ["tiled", "persistent", "persistent_tma"])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_selected_forward_dx_dw_keep_fp32_leaf_gradients(algorithm, compiled):
    torch.manual_seed(497)
    forward = _entry(m=64, n=128, k=64, algorithm=algorithm)
    backward = _entry(m=128, n=64, k=64, algorithm=algorithm)
    # Match the original Linear dW orientation: dY.T @ X.
    backward["a_stride"] = [1, 128]
    backward["b_stride"] = [64, 1]
    activation_backward = _entry(m=64, n=64, k=128, algorithm=algorithm)
    activation_backward["b_stride"] = [64, 1]
    plan = matmul_plan_from_dict(_document([forward, backward, activation_backward]))
    reference = torch.nn.Parameter(torch.randn(128, 64, device="cuda") * 0.02)
    actual = torch.nn.Parameter(reference.detach().clone())
    expected_fn = lambda x, w: x.to(torch.bfloat16) @ w.to(torch.bfloat16).T
    selected_fn = lambda x, w: planned_matmul(x.to(torch.bfloat16), w.to(torch.bfloat16).T, plan)
    if compiled:
        expected_fn = torch.compile(expected_fn, fullgraph=True, dynamic=False)
        selected_fn = torch.compile(selected_fn, fullgraph=True, dynamic=False)
    for _ in range(8):
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        selected_x = x.detach().clone().requires_grad_(True)
        dy = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        expected = expected_fn(x, reference)
        result = selected_fn(selected_x, actual)
        torch.testing.assert_close(result, expected, rtol=0.015, atol=0.015)
        expected.backward(dy)
        result.backward(dy)
        assert selected_x.grad.dtype == torch.float32
        assert ((selected_x.grad - x.grad).norm() / x.grad.norm()).item() < 0.005
    assert actual.dtype == actual.grad.dtype == torch.float32
    relative = (actual.grad - reference.grad).norm() / reference.grad.norm()
    assert relative.item() < 0.005


@cuda_window
def test_cuda_unsupported_tma_layout_falls_back_to_aten():
    a = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)[:, ::2]
    entry = _entry(m=32, n=32, k=32, algorithm="persistent_tma")
    entry["b_stride"] = list(b.stride())
    plan = matmul_plan_from_dict(_document([entry]))
    torch.testing.assert_close(planned_matmul(a, b, plan), a @ b, rtol=0, atol=0)


@cuda_window
@pytest.mark.parametrize("misaligned_operand", ["a", "b"])
def test_cuda_compiled_tma_rechecks_storage_offset_before_launch(misaligned_operand):
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16).T
    plan = matmul_plan_from_dict(_document([
        _entry(m=64, n=128, k=64, algorithm="persistent_tma")]))
    compiled = torch.compile(lambda x, y: planned_matmul(x, y, plan),
                             fullgraph=True, dynamic=False)
    control = torch.compile(lambda x, y: x @ y, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(a, b), a @ b, rtol=0.015, atol=0.015)
    original = a if misaligned_operand == "a" else b
    storage = torch.randn(original.numel() + 1, device="cuda", dtype=torch.bfloat16)
    shifted = storage.as_strided(original.shape, original.stride(), storage_offset=1)
    assert shifted.shape == original.shape and shifted.stride() == original.stride()
    assert shifted.data_ptr() % 16 == 2
    a, b = (shifted, b) if misaligned_operand == "a" else (a, shifted)
    torch.testing.assert_close(planned_matmul(a, b, plan), a @ b, rtol=0, atol=0)
    # Use the compiled ATen control: Inductor may realign an input before its
    # fallback call, and aligned cuBLAS can differ from eager unaligned cuBLAS.
    # Realignment may also make TMA valid. Check the compiled result without
    # assuming the caller's original offset survives to the opaque boundary.
    torch.testing.assert_close(compiled(a, b), control(a, b), rtol=0, atol=0)


@cuda_window
@pytest.mark.parametrize("fused", [False, True])
def test_cuda_compiled_model_selected_matmul_preserves_fp32_adam_and_reload(fused):
    from ops.residual_experiment import apply_bf16_residuals
    from run_40b import compile_training_cores

    reference = _model().cuda()
    actual = copy.deepcopy(reference)
    entries = [_entry(kind="swiglu" if fused else "mm", m=16, n=64, k=16)]
    plan = matmul_plan_from_dict(_document(entries))
    keys = tuple(actual.state_dict())
    masters = tuple(actual.parameters())
    apply_matmul_execution(actual, plan)
    for model in [reference, actual]:
        compile_training_cores(model)
        apply_bf16_residuals(model)
    optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-4)
    inputs = torch.randint(48, (2, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(inputs, inputs, return_logits=False)[1]
        result = actual(inputs, inputs, return_logits=False)[1]
    expected.backward()
    result.backward()
    torch.testing.assert_close(result, expected, rtol=0.001, atol=0.001)
    grads = torch.cat([p.grad.flatten() for p in actual.parameters()])
    ref_grads = torch.cat([p.grad.flatten() for p in reference.parameters()])
    assert torch.nn.functional.cosine_similarity(grads, ref_grads, dim=0).item() > 0.999
    assert (grads - ref_grads).norm().item() / ref_grads.norm().item() < 0.02
    optimizer.step()
    assert tuple(actual.state_dict()) == keys
    assert all(a is b for a, b in zip(masters, actual.parameters(), strict=True))
    assert all(p.dtype == p.grad.dtype == optimizer.state[p]["exp_avg"].dtype
               == optimizer.state[p]["exp_avg_sq"].dtype == torch.float32 for p in masters)
    restored = _model().cuda().eval()
    restored.load_state_dict(actual.state_dict(), strict=True)
    actual.eval()
    with torch.no_grad():
        torch.testing.assert_close(actual(inputs), restored(inputs), rtol=0, atol=0)


@cuda_window
@pytest.mark.parametrize("algorithm", ["tiled", "persistent", "persistent_tma"])
@pytest.mark.parametrize("recompute", [False, True])
def test_cuda_compiled_autograd_fused_projection_dx_dw_and_recomputation(algorithm, recompute):
    from ops.gradient_execution import make_backward
    from torch.utils.checkpoint import checkpoint

    # Ordinary AOT forward/backward testing misses Dynamo re-tracing AOT's
    # functionalized Triton HOP during Compiled Autograd. Exercise that second
    # trace and activation-checkpoint recomputation with every matrix algorithm.
    with torch._dynamo.config.patch(compiled_autograd=False,
                                    compiled_autograd_kwargs_override={}, optimize_ddp=True):
        torch._dynamo.reset()
        backward = make_backward("compiled-autograd")
        reference_model = _model().cuda()
        selected_model = copy.deepcopy(reference_model)
        fused = _entry(kind="swiglu", m=16, n=64, k=16, algorithm="persistent")
        dx = _entry(m=16, n=16, k=64, algorithm=algorithm)
        dx["b_stride"] = [16, 1]
        dw = _entry(m=64, n=16, k=16, algorithm=algorithm)
        dw["a_stride"], dw["b_stride"] = [1, 64], [16, 1]
        apply_matmul_execution(selected_model, matmul_plan_from_dict(_document([fused, dx, dw])))
        reference = reference_model.cores[0].ffn
        selected = selected_model.cores[0].ffn
        # Capture recomputation inside AOT. An eager checkpoint outside the
        # compiled forward has a separate installed-PyTorch CA limitation:
        # Dynamo cannot trace its saved-tensor unpack hook.
        wanted_eager = (lambda x: checkpoint(reference, x, use_reentrant=False)) if recompute else reference
        actual_eager = (lambda x: checkpoint(selected, x, use_reentrant=False)) if recompute else selected
        wanted_fn = torch.compile(wanted_eager, fullgraph=True, dynamic=False)
        actual_fn = torch.compile(actual_eager, fullgraph=True, dynamic=False)
        for _ in range(2):
            x = torch.randn(16, 16, device="cuda", requires_grad=True)
            actual_x = x.detach().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                wanted = wanted_fn(x)
                actual = actual_fn(actual_x)
            torch.testing.assert_close(actual, wanted, rtol=0.015, atol=0.0001)
            wanted.float().square().mean().backward()
            backward(actual.float().square().mean())
            for p, q in [(x, actual_x), *zip(reference.parameters(), selected.parameters(), strict=True)]:
                assert q.grad.dtype == torch.float32
                relative = (q.grad - p.grad).norm() / p.grad.norm().clamp_min(1e-12)
                assert relative.item() < 0.02
    torch._dynamo.reset()
