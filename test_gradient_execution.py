"""Numerical, accumulation, tied-weight, and DDP checks for compiled autograd."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ops.gradient_execution import OverlapSettings, make_backward, scale_loss


@pytest.fixture(autouse=True)
def compiler_settings():
    # make_backward intentionally configures the lifetime of a training process.
    # Tests share a process, so restore configuration and graph caches between them.
    from torch._inductor import config as inductor_config, config_comms

    with torch._dynamo.config.patch(
        compiled_autograd=False, compiled_autograd_kwargs_override={}, optimize_ddp=True
    ), inductor_config.patch(
        reorder_for_compute_comm_overlap=False, reorder_for_compute_comm_overlap_passes=[],
        deterministic=False, runtime_estimations_mms_benchmark=False,
    ), config_comms.patch(
        sink_iterative_peak_memory_budget=0.2,
        sink_waits_iterative_unsafe_collectives_reorder=True,
        sink_waits_iterative_swap_with_collectives=False,
        sink_iterative_use_runtime_estimations=False,
        runtime_estimations_align_across_all_distributed_ranks=False,
        reorder_iterative_unsafe_collectives_reorder=True,
        reorder_iterative_peak_memory_budget=0.2,
        reorder_iterative_group_with_collectives=False,
        reorder_iterative_use_runtime_estimations=False,
    ):
        yield
    torch._dynamo.reset()


def test_eager_selection_does_not_change_compiler_configuration():
    backward = make_backward()
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    backward(parameter.square().sum())
    assert parameter.grad.item() == 4.0
    assert torch._dynamo.config.compiled_autograd is False
    assert torch._dynamo.config.optimize_ddp is True
    with pytest.raises(ValueError, match="Unknown gradient"):
        make_backward("not-a-mode")
    with pytest.raises(ValueError, match="Unknown gradient"):
        scale_loss(parameter, 0.125, "not-a-mode")


@pytest.mark.parametrize("mode", ["eager", "compiled-autograd", "compiled-autograd-overlap"])
def test_cpu_loss_scaling_preserves_masked_and_partial_batch_gradients(mode):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    for factor in (0.0, 0.125, 7.0 / 19.0):
        parameter.grad = None
        scale_loss(parameter.sum(), factor, mode).backward()
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, factor), rtol=0, atol=0)


def test_overlap_mode_preserves_collective_order_and_peak_memory_budget():
    from torch._inductor import config, config_comms

    operation = make_backward("compiled-autograd-overlap", distributed=True, backend="eager")
    assert callable(operation)
    assert torch._dynamo.config.optimize_ddp == "python_reducer"
    assert torch._dynamo.config.compiled_autograd is True
    assert config.reorder_for_compute_comm_overlap is True
    assert config.reorder_for_compute_comm_overlap_passes == ["sink_waits_iterative"]
    assert config_comms.sink_iterative_peak_memory_budget == 0.0
    assert config_comms.sink_waits_iterative_unsafe_collectives_reorder is False
    assert config_comms.sink_waits_iterative_swap_with_collectives is False
    assert config_comms.sink_iterative_use_runtime_estimations is False
    assert config.runtime_estimations_mms_benchmark is False
    assert config_comms.runtime_estimations_align_across_all_distributed_ranks is False
    assert OverlapSettings().producer_schedule is False


def test_overlap_options_are_immutable_and_have_stable_json_metadata():
    options = OverlapSettings(True, True, 0.02)
    assert json.loads(json.dumps(options.metadata)) == {
        "swap_with_collectives": True, "use_runtime_estimations": True,
        "peak_memory_budget": 0.02, "producer_schedule": False}
    with pytest.raises(FrozenInstanceError):
        options.peak_memory_budget = 0.1
    modified_copy = options.metadata
    modified_copy["peak_memory_budget"] = 1
    assert options.peak_memory_budget == 0.02
    assert OverlapSettings(peak_memory_budget=0).metadata["peak_memory_budget"] == 0.0
    assert OverlapSettings(peak_memory_budget=0.25).peak_memory_budget == 0.25


@pytest.mark.parametrize("settings", [
    {"swap_with_collectives": 1}, {"swap_with_collectives": "false"},
    {"use_runtime_estimations": 0}, {"use_runtime_estimations": None},
    {"producer_schedule": 1}, {"producer_schedule": "true"}, {"producer_schedule": None},
    {"peak_memory_budget": True}, {"peak_memory_budget": "0.02"},
    {"peak_memory_budget": -0.01}, {"peak_memory_budget": 0.251},
    {"peak_memory_budget": float("nan")}, {"peak_memory_budget": float("inf")},
    {"peak_memory_budget": 10**999},
])
def test_invalid_overlap_options_fail_before_global_configuration(settings):
    with pytest.raises(ValueError):
        OverlapSettings(**settings)
    assert torch._dynamo.config.compiled_autograd is False


@pytest.mark.parametrize("other", [
    {"swap_with_collectives": True},
    {"use_runtime_estimations": True},
    {"peak_memory_budget": 0.02},
])
def test_producer_scheduling_is_a_separate_bounded_experiment(other):
    with pytest.raises(ValueError, match="producer_schedule initially requires"):
        OverlapSettings(producer_schedule=True, **other)
    assert torch._dynamo.config.compiled_autograd is False


@pytest.mark.parametrize("distributed", [False, True])
def test_producer_scheduling_enforces_order_and_memory_guards(distributed):
    from torch._inductor import config, config_comms

    # These are unsafe installed defaults; selecting the pass alone is not safe.
    assert config_comms.reorder_iterative_unsafe_collectives_reorder is True
    assert config_comms.reorder_iterative_peak_memory_budget == 0.2
    options = OverlapSettings(producer_schedule=True)
    assert options.metadata["producer_schedule"] is True
    make_backward("compiled-autograd-overlap", distributed=distributed,
                  backend="eager", overlap_settings=options)
    assert config.reorder_for_compute_comm_overlap_passes == [
        "reorder_communication_preserving_peak_memory", "sink_waits_iterative"]
    assert config_comms.reorder_iterative_unsafe_collectives_reorder is False
    assert config_comms.reorder_iterative_peak_memory_budget == 0.0
    assert config_comms.reorder_iterative_group_with_collectives is False
    assert config_comms.reorder_iterative_use_runtime_estimations is False
    assert config_comms.sink_iterative_peak_memory_budget == 0.0
    assert config_comms.sink_waits_iterative_unsafe_collectives_reorder is False
    assert config_comms.sink_waits_iterative_swap_with_collectives is False
    assert config_comms.sink_iterative_use_runtime_estimations is False
    assert config.runtime_estimations_mms_benchmark is False
    assert config_comms.runtime_estimations_align_across_all_distributed_ranks is False


@pytest.mark.parametrize("missing", [
    "reorder_communication_preserving_peak_memory", "sink_waits_iterative"])
def test_producer_scheduling_requires_both_passes_before_global_changes(monkeypatch, missing):
    from torch._inductor import comms, config, config_comms

    monkeypatch.setattr(comms, missing, None)
    with pytest.raises(RuntimeError, match="lacks the " + missing):
        make_backward("compiled-autograd-overlap", overlap_settings=OverlapSettings(producer_schedule=True))
    assert config.reorder_for_compute_comm_overlap is False
    assert config_comms.reorder_iterative_unsafe_collectives_reorder is True
    assert config_comms.reorder_iterative_peak_memory_budget == 0.2
    assert torch._dynamo.config.compiled_autograd is False


@pytest.mark.parametrize("mode", ["eager", "compiled-autograd"])
def test_overlap_options_rejected_outside_overlap_mode(mode):
    with pytest.raises(ValueError, match="requires compiled-autograd-overlap"):
        make_backward(mode, overlap_settings=OverlapSettings())
    assert torch._dynamo.config.compiled_autograd is False


def test_overlap_options_require_validated_type():
    with pytest.raises(TypeError, match="OverlapSettings instance"):
        make_backward("compiled-autograd-overlap", overlap_settings={"peak_memory_budget": 0.02})


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("runtime_estimates", [False, True])
def test_explicit_overlap_options_keep_launch_order_and_map_runtime_controls(distributed, runtime_estimates):
    from torch._inductor import config, config_comms

    options = OverlapSettings(swap_with_collectives=True,
                              use_runtime_estimations=runtime_estimates, peak_memory_budget=0.02)
    make_backward("compiled-autograd-overlap", distributed=distributed,
                  backend="eager", overlap_settings=options)
    assert config_comms.sink_waits_iterative_swap_with_collectives is True
    assert config_comms.sink_waits_iterative_unsafe_collectives_reorder is False
    assert config_comms.sink_iterative_peak_memory_budget == 0.02
    assert config_comms.sink_iterative_use_runtime_estimations is runtime_estimates
    assert config.runtime_estimations_mms_benchmark is runtime_estimates
    assert config_comms.runtime_estimations_align_across_all_distributed_ranks is (
        distributed and runtime_estimates)


def test_runtime_estimate_alignment_must_exist_before_settings_change(monkeypatch):
    from torch._inductor import comms, config

    monkeypatch.setattr(comms, "align_runtime_estimations_across_all_distributed_ranks", None)
    with pytest.raises(RuntimeError, match="lacks distributed runtime-estimate alignment"):
        make_backward("compiled-autograd-overlap", distributed=True,
                      overlap_settings=OverlapSettings(use_runtime_estimations=True))
    assert torch._dynamo.config.compiled_autograd is False
    assert config.reorder_for_compute_comm_overlap is False
    assert config.runtime_estimations_mms_benchmark is False


def test_overlap_mode_rejects_a_disabled_or_missing_scheduler(monkeypatch):
    from torch._inductor import comms, config

    with config.patch(deterministic=True):
        with pytest.raises(ValueError, match="deterministic mode disables"):
            make_backward("compiled-autograd-overlap")
    monkeypatch.setattr(comms, "sink_waits_iterative", None)
    with pytest.raises(RuntimeError, match="lacks the sink_waits_iterative"):
        make_backward("compiled-autograd-overlap")
    assert torch._dynamo.config.compiled_autograd is False


@pytest.mark.parametrize("reset_to_none", [True, False])
@pytest.mark.parametrize("mode", ["compiled-autograd", "compiled-autograd-overlap"])
def test_bf16_gemm_rounding_and_fp32_accumulation_are_preserved(reset_to_none, mode):
    from torch._dynamo.utils import counters

    captures_before = counters["compiled_autograd"]["captures"]
    torch.manual_seed(901)
    expected = torch.nn.Parameter(torch.randn(12, 16))
    actual = torch.nn.Parameter(expected.detach().clone())
    backward = make_backward(mode)
    expected_optimizer = torch.optim.AdamW([expected], lr=0.01)
    actual_optimizer = torch.optim.AdamW([actual], lr=0.01)
    for update in range(2):
        expected_optimizer.zero_grad(set_to_none=reset_to_none)
        actual_optimizer.zero_grad(set_to_none=reset_to_none)
        reference = torch.zeros_like(expected)
        for microbatch in range(4):
            inputs = torch.randn(8, 16, dtype=torch.bfloat16)
            derivative = torch.randn(8, 12, dtype=torch.bfloat16)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected_loss = (F.linear(inputs, expected) * derivative).sum()
                actual_loss = (F.linear(inputs, actual) * derivative).sum()
            expected_loss.backward()
            backward(actual_loss)
            # This reference deliberately rounds dW to BF16 before FP32 addition.
            reference.add_((derivative.T @ inputs).float())
            torch.testing.assert_close(expected.grad, reference, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, reference, rtol=0, atol=0)
            assert actual.grad.dtype == torch.float32
        for parameter in (expected, actual):
            torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
        expected_optimizer.step()
        actual_optimizer.step()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_optimizer.state[actual][key],
                                       expected_optimizer.state[expected][key], rtol=0, atol=0)
    assert counters["compiled_autograd"]["captures"] >= captures_before + 2


class TiedProjection(torch.nn.Module):
    """Tied embedding/head and repeated linear calls, with FP32 master leaves."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.projection = torch.nn.Linear(8, 8, bias=False)

    def forward(self, tokens):
        weight = self.embed.weight.to(torch.bfloat16)
        value = F.embedding(tokens, weight)
        value = self.projection(value) + self.projection(value * 0.5)
        return F.linear(value, weight).float()


def test_compiled_forward_tied_weights_and_checkpoint_continuation(tmp_path):
    torch.manual_seed(112)
    expected = TiedProjection()
    actual = deepcopy(expected)
    backward = make_backward("compiled-autograd")
    expected.forward = torch.compile(expected.forward, fullgraph=True, dynamic=False)
    actual.forward = torch.compile(actual.forward, fullgraph=True, dynamic=False)
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=0.002)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=0.002)
    tokens = torch.tensor([[0, 2, 0, 1], [5, 3, 9, 3]])
    for _ in range(3):
        for model, operation in ((expected, lambda loss: loss.backward()), (actual, backward)):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(tokens).square().mean()
            operation(loss)
        for wanted, observed in zip(expected.parameters(), actual.parameters(), strict=True):
            torch.testing.assert_close(observed.grad, wanted.grad, rtol=0, atol=0)
    for model, optimizer in ((expected, expected_optimizer), (actual, actual_optimizer)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    path = tmp_path / "resume.pt"
    torch.save({"model": actual.state_dict(), "optimizer": actual_optimizer.state_dict()}, path)
    state = torch.load(path, weights_only=True)
    resumed = TiedProjection()
    resumed.load_state_dict(state["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.002)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed.forward = torch.compile(resumed.forward, fullgraph=True, dynamic=False)
    assert tuple(state["model"]) == ("embed.weight", "projection.weight")
    for wanted, observed in zip(expected.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    # Resume with ordinary autograd: state contains no compiler-specific objects.
    for model, optimizer in ((expected, expected_optimizer), (resumed, resumed_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(tokens).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    for wanted, observed in zip(expected.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


def _distributed_worker(rank, rendezvous, output_directory, explicit_overlap=False):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(31)
        expected = TiedProjection()
        actual = deepcopy(expected)
        eager = torch.nn.parallel.DistributedDataParallel(expected, bucket_cap_mb=1)
        # Eager backend keeps this a lightweight CPU distributed-semantics test;
        # the separate tests above exercise Inductor's generated accumulation.
        if explicit_overlap:
            from torch._inductor.comms import align_runtime_estimations_across_all_distributed_ranks

            backward = make_backward("compiled-autograd-overlap", distributed=True, backend="eager",
                overlap_settings=OverlapSettings(True, True, 0.02))

            class Estimate:
                def __init__(self, value):
                    self.value = value

                def get_estimated_runtime(self):
                    return self.value

            estimates = [Estimate(1.0 + rank), Estimate(2.0 + 2 * rank)]
            # Exercise the actual installed scheduler's cross-rank alignment
            # collective on CPU, separately from the lightweight eager backend.
            align_runtime_estimations_across_all_distributed_ranks(estimates)
            assert [node.override_estimated_runtime for node in estimates] == [1.0, 2.0]
        else:
            backward = make_backward("compiled-autograd", distributed=True, backend="eager")
        compiled = torch.nn.parallel.DistributedDataParallel(actual, bucket_cap_mb=1)
        for microbatch in range(3):
            tokens = (torch.arange(8).reshape(2, 4) + rank * 2 + microbatch) % 16
            for model, operation in ((eager, lambda loss: loss.backward()), (compiled, backward)):
                synchronization = model.no_sync() if microbatch < 2 else nullcontext()
                with synchronization:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss = model(tokens).square().mean() / 3
                    operation(loss)
            for wanted, observed in zip(expected.parameters(), actual.parameters(), strict=True):
                torch.testing.assert_close(observed.grad, wanted.grad, rtol=0, atol=0)
        torch.save({name: parameter.grad for name, parameter in actual.named_parameters()},
                   Path(output_directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("explicit_overlap", [False, True])
def test_distributed_no_sync_and_fp32_reduction_match_eager(tmp_path, explicit_overlap):
    torch.multiprocessing.spawn(_distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), explicit_overlap), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    for name in ranks[0]:
        assert ranks[0][name].dtype == torch.float32
        torch.testing.assert_close(ranks[0][name], ranks[1][name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness fixture")
def test_cuda_warpstate_compiled_cores_loss_residuals_and_accumulation():
    from ops.loss_experiment import apply_loss_execution
    from ops.residual_experiment import apply_bf16_residuals
    from warpstate import WarpStateConfig, WarpStateLM
    from run_40b import compile_training_cores

    torch.manual_seed(119)
    cfg = WarpStateConfig(vocab_size=128, d_model=128, n_heads=2, n_cores=1,
                          logical_depth=2, ffn_hidden=256, chunk_size=16,
                          gradient_checkpointing=False, parallel_memory=True)
    expected = WarpStateLM(cfg).cuda()
    actual = deepcopy(expected)
    backward = make_backward("compiled-autograd")
    for model in (expected, actual):
        compile_training_cores(model)
        apply_loss_execution(model, "compiled-full")
        apply_bf16_residuals(model)
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=0.0001, fused=True)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=0.0001, fused=True)
    # Exercise continuation with existing FP32 Adam moments. Fresh Adam can
    # turn a tiny BF16 cancellation around zero into an entire signed learning
    # rate on its first update; it is not representative of this resumed run.
    for model, optimizer in ((expected, expected_optimizer), (actual, actual_optimizer)):
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(40081.0, device="cuda"),
                "exp_avg": torch.full_like(parameter, 0.001),
                "exp_avg_sq": torch.full_like(parameter, 0.0001),
            }
    for update in range(2):
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
        for microbatch in range(3):
            tokens = torch.randint(128, (2, 32), device="cuda")
            labels = torch.randint(128, (2, 32), device="cuda")
            for model, operation in ((expected, lambda loss: loss.backward()), (actual, backward)):
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    _, loss = model(tokens, labels, return_logits=False)
                operation(scale_loss(loss, 1.0 / 3.0, "compiled-autograd"))
            for (name, wanted), (_, observed) in zip(expected.named_parameters(),
                                                    actual.named_parameters(), strict=True):
                assert observed.dtype == observed.grad.dtype == torch.float32, name
                assert torch.isfinite(observed.grad).all(), name
                # CA also fuses BF16 work across the old core/loss boundaries.
                # Individual BF16 rounding can differ: bound both total error
                # and worst-element error instead of claiming bitwise identity.
                difference = observed.grad - wanted.grad
                relative = difference.norm() / wanted.grad.norm().clamp_min(1e-12)
                assert difference.norm().item() <= wanted.grad.norm().item() * 0.015 + 1e-8, (name, relative.item())
                assert difference.abs().max().item() <= wanted.grad.abs().max().item() * 0.02 + 1e-8, name
        for model, optimizer in ((expected, expected_optimizer), (actual, actual_optimizer)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        for wanted, observed in zip(expected.parameters(), actual.parameters(), strict=True):
            assert torch.isfinite(observed).all()
            torch.testing.assert_close(observed, wanted, rtol=0, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scalar capture workaround")
def test_cuda_loss_scalar_materialization_preserves_fp32_arithmetic():
    torch.manual_seed(711)
    values = torch.randn(97, device="cuda", dtype=torch.float32)
    for factor in (0.0, 0.125, 1.0 / 3.0, 7.0 / 19.0, 1.0):
        expected = scale_loss(values, factor, "eager")
        actual = scale_loss(values, factor, "compiled-autograd")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
