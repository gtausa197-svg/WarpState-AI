"""CPU/Gloo checks for DDP bucket execution and BF16 accumulation semantics."""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ops.distributed_execution import ddp_kwargs, ddp_runtime_settings


@pytest.mark.parametrize("changes", [
    {"bucket_cap_mb": 0}, {"bucket_cap_mb": -1}, {"bucket_cap_mb": True},
    {"bucket_cap_mb": float("nan")}, {"bucket_cap_mb": float("inf")},
    {"bucket_cap_mb": "128"}, {"bucket_cap_mb": 1e-12},
    {"batched_grad_copy": 1}, {"batched_grad_copy": "false"},
    {"bucket_cap_mb_list": []}, {"bucket_cap_mb_list": "64,128"},
    {"bucket_cap_mb_list": [64, 0]}, {"bucket_cap_mb_list": [64, False]},
])
def test_invalid_runtime_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        ddp_runtime_settings(**changes)


def test_legacy_runtime_does_not_receive_unsupported_optional_arguments(monkeypatch):
    def legacy_ddp(module, broadcast_buffers=False, gradient_as_bucket_view=False,
                   bucket_cap_mb=None):
        pass

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", legacy_ddp)
    assert ddp_kwargs({}) == {
        "broadcast_buffers": False, "gradient_as_bucket_view": True,
        "bucket_cap_mb": 128,
    }
    for requested in [{"batched_grad_copy": True}, {"bucket_cap_mb_list": [64, 128]}]:
        with pytest.raises(RuntimeError, match="Installed PyTorch DDP does not support"):
            ddp_kwargs(ddp_runtime_settings(**requested))


def test_capacity_list_is_copied_and_records_effective_capacity():
    capacities = [16, 64, 32]
    settings = ddp_runtime_settings(bucket_cap_mb=128, batched_grad_copy=True,
                                    bucket_cap_mb_list=capacities)
    capacities[0] = 999
    assert settings["ddp_bucket_cap_mb"] == 64
    assert settings["ddp_bucket_cap_mb_list"] == [16, 64, 32]
    assert json.loads(json.dumps(settings)) == settings
    assert ddp_kwargs(settings)["bucket_cap_mb_list"] == [16, 64, 32]


@pytest.mark.parametrize("requested", [
    {"batched_grad_copy": True}, {"bucket_cap_mb_list": [64, 128]},
])
def test_python_reducer_rejects_inapplicable_cpp_bucket_options(requested):
    with torch._dynamo.config.patch(optimize_ddp="python_reducer"):
        with pytest.raises(ValueError, match="require the C\\+\\+ DDP reducer"):
            ddp_kwargs(ddp_runtime_settings(**requested))
        assert ddp_kwargs(ddp_runtime_settings(bucket_cap_mb=64))["bucket_cap_mb"] == 64


def _model():
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(32, 32), nn.SiLU()])
    layers.append(nn.Linear(32, 8))
    return nn.Sequential(*layers)


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    reports = []
    try:
        cases = [(capacity, batched, None)
                 for capacity in [64, 128, 256, 0.002]
                 for batched in [False, True]]
        cases.append((128, True, [0.002, 0.004]))
        for capacity, batched, capacities in cases:
            torch.manual_seed(321 + rank)
            settings = ddp_runtime_settings(bucket_cap_mb=capacity,
                                            batched_grad_copy=batched,
                                            bucket_cap_mb_list=capacities)
            model = DistributedDataParallel(_model(), **ddp_kwargs(settings))
            reference = deepcopy(model.module)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001,
                                                    foreach=False)
            maximum_gradient_error = 0.0
            for update in range(3):
                # Reallocation on every update exercises the production
                # set_to_none=True path both before and after bucket rebuilding.
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                assert all(parameter.grad is None for parameter in model.parameters())
                for microstep in range(8):
                    generator = torch.Generator().manual_seed(
                        10000 + rank * 1000 + update * 8 + microstep)
                    inputs = torch.randn(3, 32, generator=generator)
                    target = torch.randn(3, 8, generator=generator)
                    context = model.no_sync() if microstep < 7 else nullcontext()
                    with context:
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            output = model(inputs)
                            reference_output = reference(inputs)
                        assert output.dtype == reference_output.dtype == torch.bfloat16
                        loss = (output.float() - target).square().mean() / 8
                        reference_loss = (reference_output.float() - target).square().mean() / 8
                        loss.backward()
                        reference_loss.backward()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    assert actual.dtype == expected.dtype == torch.float32
                    assert actual.grad.dtype == expected.grad.dtype == torch.float32
                    # Independent reference: accumulate each rank's distinct
                    # microbatches locally, then explicitly average their FP32
                    # gradients. This detects missing/repeated microsteps and
                    # incorrect reducer scaling without assuming bucket layout.
                    dist.all_reduce(expected.grad)
                    expected.grad.div_(2)
                    maximum_gradient_error = max(maximum_gradient_error,
                        (actual.grad - expected.grad).abs().max().item())
                    torch.testing.assert_close(actual.grad, expected.grad,
                                               rtol=1e-6, atol=1e-7)
                    assert torch.isfinite(actual.grad).all()
                optimizer.step()
                reference_optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                for state in optimizer.state.values():
                    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
            reports.append({"settings": settings,
                            "maximum_gradient_error": maximum_gradient_error,
                            "updates": 3, "microsteps_per_update": 8})
            del model, reference, optimizer, reference_optimizer
        (directory / f"rank-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


def test_two_rank_bf16_accumulation_matches_explicit_fp32_gradient_average(tmp_path):
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
    for rank in range(2):
        reports = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert len(reports) == 9
        assert all(report["maximum_gradient_error"] <= 1e-7 for report in reports)


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2])
