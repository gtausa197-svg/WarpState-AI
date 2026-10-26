"""Casting reuse must preserve rounding, FP32 leaves, and update boundaries."""

from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ops.residual_experiment import apply_bf16_residuals
from ops.weight_cast_execution import apply_weight_cast_cache
from warpstate import WarpStateConfig, WarpStateLM


def _model(*, dropout=0.0):
    torch.manual_seed(2718)
    cfg = WarpStateConfig(vocab_size=64, d_model=32, n_heads=4, n_cores=1,
                          logical_depth=1, ffn_hidden=64, chunk_size=4,
                          parallel_memory=True, gradient_checkpointing=False,
                          loss_chunk_size=8, dropout=dropout)
    model = WarpStateLM(cfg).train()
    apply_bf16_residuals(model)
    return model


def _batch(seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(64, (1, 8), generator=generator),
            torch.randint(64, (1, 8), generator=generator))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("set_to_none", [False, True])
def test_eight_backward_calls_and_optimizer_updates_match_exactly(compiled, set_to_none):
    reference = _model(dropout=0.1)
    experiment = _model(dropout=0.1)
    original_parameters = tuple(experiment.parameters())
    original_keys = tuple(experiment.state_dict())
    if compiled:
        for model in [reference, experiment]:
            for core in model.cores:
                core.forward = torch.compile(core.forward, backend="aot_eager",
                                             fullgraph=True, dynamic=False)
    cache = apply_weight_cast_cache(experiment)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.0001, foreach=False)
                  for model in [reference, experiment]]
    for update in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
        with cache.update():
            assert cache.cached_tensor_count == 4 and cache.cached_bytes > 0
            for microstep in range(8):
                inputs, targets = _batch(500 + update * 8 + microstep)
                torch.manual_seed(1000 + update * 8 + microstep)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    _, expected_loss = reference(inputs, targets, return_logits=False)
                (expected_loss / 8).backward()
                torch.manual_seed(1000 + update * 8 + microstep)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    _, actual_loss = experiment(inputs, targets, return_logits=False)
                (actual_loss / 8).backward()
                torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                for actual, expected in zip(experiment.parameters(), reference.parameters()):
                    assert actual.dtype == actual.grad.dtype == torch.float32
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
            assert tuple(experiment.state_dict()) == original_keys
            assert all(a is b for a, b in zip(original_parameters, experiment.parameters()))
        assert not cache.active and cache.cached_tensor_count == cache.cached_bytes == 0
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in zip(experiment.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state = optimizers[1].state[actual]
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_checkpoint_load_between_scopes_uses_fresh_casts_and_preserves_fp32_state():
    model = _model()
    cache = apply_weight_cast_cache(model)
    with cache.update():
        old = cache._views["wrapped.cores.0.in_proj.weight"].detach().clone()
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint["cores.0.in_proj.weight"].add_(0.5)
    model.load_state_dict(checkpoint, strict=True)
    with cache.update():
        current = cache._views["wrapped.cores.0.in_proj.weight"]
        assert not torch.equal(old, current)
        torch.testing.assert_close(current, model.cores[0].in_proj.weight.bfloat16(), rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_scope_failure_restores_parameters_and_releases_casts():
    model = _model()
    masters = tuple(model.parameters())
    cache = apply_weight_cast_cache(model)
    inputs, targets = _batch(40)
    with pytest.raises(ValueError, match="must be divisible"):
        with cache.update(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(inputs[:, :-1], targets[:, :-1])
    assert not cache.active and cache.cached_bytes == 0
    assert all(a is b for a, b in zip(masters, model.parameters()))
    with cache.update(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss = model(inputs, targets, return_logits=False)
        loss.backward()


@pytest.mark.parametrize("mutation", ["in_place", "load_state", "replace"])
def test_parameter_mutation_inside_scope_is_detected_and_cleared(mutation):
    model = _model()
    cache = apply_weight_cast_cache(model)
    with pytest.raises(RuntimeError, match="changed"):
        with cache.update():
            if mutation == "in_place":
                with torch.no_grad():
                    model.cores[0].in_proj.weight.add_(0.01)
            elif mutation == "load_state":
                model.load_state_dict(model.state_dict())
            else:
                model.cores[0].in_proj.weight = torch.nn.Parameter(
                    model.cores[0].in_proj.weight.detach().clone())
    assert not cache.active and cache.cached_bytes == 0


def test_outside_scope_and_inference_are_unchanged_and_do_not_create_cache():
    reference = _model()
    model = _model()
    cache = apply_weight_cast_cache(model)
    inputs, _ = _batch(71)
    for candidate in [reference, model]:
        candidate.eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
    assert not cache.active and cache.cached_bytes == 0
    with pytest.raises(RuntimeError, match="training with gradients"):
        with cache.update():
            pass


def test_incompatible_modes_and_nested_scopes_fail_explicitly():
    model = _model()
    model.cfg.logical_depth = 2
    with pytest.raises(ValueError, match="one use"):
        apply_weight_cast_cache(model)
    model.cfg.logical_depth = 1
    model.cfg.gradient_checkpointing = True
    with pytest.raises(ValueError, match="checkpointing off"):
        apply_weight_cast_cache(model)
    model.cfg.gradient_checkpointing = False
    cache = apply_weight_cast_cache(model)
    with pytest.raises(ValueError, match="already been applied"):
        apply_weight_cast_cache(model)
    with cache.update():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with cache.update():
                pass
        with pytest.raises(RuntimeError, match="requires BF16 autocast"):
            model(_batch(2)[0])


def _ddp_worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        reference = DistributedDataParallel(_model(), gradient_as_bucket_view=True)
        raw = _model()
        cache = apply_weight_cast_cache(raw)
        experiment = DistributedDataParallel(raw, gradient_as_bucket_view=True)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=0.0001, foreach=False)
                      for model in [reference, experiment]]
        for update in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with cache.update():
                for microstep in range(8):
                    inputs, targets = _batch(100 + rank * 100 + update * 8 + microstep)
                    for model in [reference, experiment]:
                        with model.no_sync() if microstep < 7 else nullcontext():
                            with torch.autocast("cpu", dtype=torch.bfloat16):
                                _, loss = model(inputs, targets, return_logits=False)
                            (loss / 8).backward()
                for actual, expected in zip(experiment.parameters(), reference.parameters()):
                    assert actual.grad.dtype == expected.grad.dtype == torch.float32
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
            assert not cache.active
            for optimizer in optimizers:
                optimizer.step()
            for actual, expected in zip(experiment.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (directory / f"rank-{rank}.json").write_text(json.dumps({"updates": 3, "equal": True}))
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_preserves_ddp_leaf_hooks_exactly(tmp_path):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    processes = []
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(rank), str(tmp_path)],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=90)
            assert process.returncode == 0, f"Rank {rank} failed:\n{output}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait(timeout=10)
    assert all(json.loads((tmp_path / f"rank-{rank}.json").read_text())["equal"]
               for rank in range(2))


@pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1",
                    reason="requires an explicitly reserved CUDA device")
def test_cuda_inductor_core_weight_cache_matches_bf16_updates(tmp_path):
    from ops.loss_experiment import apply_loss_execution

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = WarpStateConfig(vocab_size=256, d_model=256, n_heads=4, n_cores=2,
                          logical_depth=2, ffn_hidden=896, chunk_size=128,
                          parallel_memory=True, gradient_checkpointing=False,
                          loss_chunk_size=256)
    models = []
    for _ in range(2):
        torch.manual_seed(314)
        model = WarpStateLM(cfg).cuda().train()
        for core in model.cores:
            core.forward = torch.compile(core.forward, fullgraph=True, dynamic=False)
        apply_loss_execution(model, "compiled-full")
        apply_bf16_residuals(model)
        models.append(model)
    reference, experiment = models
    cache = apply_weight_cast_cache(experiment)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.0001, fused=True)
                  for model in models]
    report = {"maximum_loss_difference": 0.0, "maximum_gradient_difference": 0.0,
              "maximum_parameter_difference": 0.0, "updates": 3, "microsteps": 8}
    for update in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with cache.update():
            for microstep in range(8):
                generator = torch.Generator().manual_seed(42 + update * 8 + microstep)
                inputs = torch.randint(256, (1, 256), generator=generator).cuda()
                targets = torch.randint(256, (1, 256), generator=generator).cuda()
                losses = []
                for model in models:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(inputs, targets, return_logits=False)
                    (loss / 8).backward()
                    losses.append(loss.detach())
                report["maximum_loss_difference"] = max(report["maximum_loss_difference"],
                    (losses[0] - losses[1]).abs().item())
                for actual, expected in zip(experiment.parameters(), reference.parameters()):
                    report["maximum_gradient_difference"] = max(report["maximum_gradient_difference"],
                        (actual.grad - expected.grad).abs().max().item())
                    assert actual.grad.dtype == expected.grad.dtype == torch.float32
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in zip(experiment.parameters(), reference.parameters()):
            report["maximum_parameter_difference"] = max(report["maximum_parameter_difference"],
                (actual - expected).abs().max().item())
    (tmp_path / "cuda-weight-cast-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CUDA_WEIGHT_CAST_COMPARISON " + json.dumps(report), flush=True)
    assert report["maximum_loss_difference"] == 0.0, report
    assert report["maximum_gradient_difference"] == 0.0, report
    assert report["maximum_parameter_difference"] == 0.0, report


if __name__ == "__main__":
    _ddp_worker(int(sys.argv[1]), sys.argv[2])
