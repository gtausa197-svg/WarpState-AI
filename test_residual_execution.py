"""BF16 residual execution keeps FP32 masters and changes only compute rounding."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ops.residual_experiment import apply_bf16_residuals
from run_40b import compile_training_cores
from warpstate import WarpStateConfig, WarpStateLM


def _model():
    torch.manual_seed(685)
    config = WarpStateConfig(vocab_size=128, d_model=64, n_heads=4,
                             n_cores=1, logical_depth=2, ffn_hidden=112,
                             chunk_size=4, parallel_memory=True,
                             gradient_checkpointing=False, loss_chunk_size=8)
    return WarpStateLM(config).train()


@pytest.mark.parametrize("compile_cores", [False, True])
def test_bf16_residual_numerics_and_fp32_state(compile_cores, tmp_path, monkeypatch):
    # CPU eager backend checks Dynamo capture and functional parameter swapping;
    # production CUDA Inductor speed and numerics require separate GPU screening.
    if compile_cores:
        actual_compile = torch.compile
        monkeypatch.setattr(torch, "compile", lambda fn, **kw: actual_compile(fn, backend="eager", **kw))
    reference = _model()
    experiment = copy.deepcopy(reference)
    parameters = tuple(experiment.parameters())
    keys = tuple(experiment.state_dict())
    optimizer = torch.optim.AdamW(experiment.parameters(), lr=1e-4)
    apply_bf16_residuals(experiment)
    if compile_cores:
        compile_training_cores(reference)
        compile_training_cores(experiment)
    inputs = torch.randint(128, (2, 16))
    targets = torch.randint(128, inputs.shape)
    targets[:, -3:] = -100
    residual_dtypes = []
    hook = experiment.cores[0].register_forward_pre_hook(lambda _module, args: residual_dtypes.append(args[0].dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, ref_loss = reference(inputs, targets, return_logits=False)
        _, loss = experiment(inputs, targets, return_logits=False)
    hook.remove()
    assert residual_dtypes == [torch.bfloat16] * experiment.cfg.logical_depth
    ref_loss.backward()
    loss.backward()
    assert tuple(experiment.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, experiment.parameters(), strict=True))
    assert all(parameter.dtype == parameter.grad.dtype == torch.float32 for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    actual_gradients = torch.cat([parameter.grad.flatten() for parameter in parameters])
    reference_gradients = torch.cat([parameter.grad.flatten() for parameter in reference.parameters()])
    cosine = float(torch.nn.functional.cosine_similarity(actual_gradients, reference_gradients, dim=0))
    norm_ratio = float(actual_gradients.norm() / reference_gradients.norm())
    relative_loss_difference = float((loss.detach() - ref_loss.detach()).abs() / ref_loss.detach().abs())
    report = {
        "compiled_cores": compile_cores, "reference_loss": float(ref_loss.detach()),
        "loss": float(loss.detach()), "relative_loss_difference": relative_loss_difference,
        "gradient_cosine": cosine, "gradient_norm_ratio": norm_ratio,
    }
    print("BF16_RESIDUAL_NUMERICS " + json.dumps(report, sort_keys=True), flush=True)
    (tmp_path / "bf16-residual-numerics.json").write_text(json.dumps(report, indent=2) + "\n")
    assert relative_loss_difference <= 0.01, report
    assert cosine >= 0.99, report
    assert 0.95 <= norm_ratio <= 1.05, report
    optimizer.step()
    for parameter in parameters:
        assert parameter.dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    restored = _model()
    restored.load_state_dict(experiment.state_dict(), strict=True)
    assert tuple(restored.state_dict()) == keys
    for name, expected in experiment.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_bf16_residual_restores_parameters_on_error_and_preserves_non_autocast():
    model = _model()
    reference = copy.deepcopy(model)
    original_parameters = tuple(model.parameters())
    apply_bf16_residuals(model)
    inputs = torch.randint(128, (1, 16))
    torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError, match="must be divisible"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(inputs[:, :-1])
    assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    with pytest.raises(ValueError, match="already been applied"):
        apply_bf16_residuals(model)


def test_bf16_residual_ddp_reducer_owns_fp32_master_gradients(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires the CPU Gloo distributed backend")
    if dist.is_initialized():
        pytest.skip("requires ownership of an isolated default process group")
    assert tmp_path.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    dist.init_process_group("gloo", init_method=(tmp_path / "ddp-init").as_uri(), rank=0, world_size=1)
    try:
        model = _model()
        masters = tuple(model.parameters())
        apply_bf16_residuals(model)
        distributed = DistributedDataParallel(model)
        inputs = torch.randint(128, (2, 16))
        for _ in range(2):  # Second forward detects unfinished reducer work.
            distributed.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss = distributed(inputs, inputs, return_logits=False)
            loss.backward()
            assert all(parameter.grad is not None and parameter.grad.dtype == torch.float32 for parameter in masters)
        assert all(a is b for a, b in zip(masters, model.parameters(), strict=True))
    finally:
        dist.destroy_process_group()
