"""Opt-in capture of weight-gradient conversion and FP32 accumulation.

Ordinary AOTAutograd compilation ends before leaf ``AccumulateGrad`` nodes.
Compiled Autograd exposes those nodes, casts, and DDP hooks to Inductor so a
BF16 gradient can be converted while adding directly into its FP32 destination.
Master parameters, optimizer, and state-dict keys are unchanged. No custom hook
writes a gradient or bypasses autograd. The isolated linear cast/accumulation
path preserves BF16 GEMM output rounding exactly. Whole-backward compilation
can also fuse a tied-head BF16 matrix product and gradient add into BF16 addmm,
or fuse other BF16 operations across existing region boundaries. These alter
intermediate rounding; complete-model gradients use explicit BF16 tolerances.

This uses the experimental Compiled Autograd integration in the installed
PyTorch. Select it before compiling the forward or constructing DDP. Benchmark
it on the actual model; larger backward graphs can increase compilation costs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch


GRADIENT_MODES = ("eager", "compiled-autograd", "compiled-autograd-overlap")
MAX_OVERLAP_MEMORY_BUDGET = 0.25


@dataclass(frozen=True)
class OverlapSettings:
    """Explicit scheduler choices; precision and collective launch order stay fixed.

    ``peak_memory_budget`` is a fractional allowance in the scheduler's memory
    estimate, not a reservation or a substitute for the measured CUDA cap.
    Runtime estimates opt into GEMM benchmarking during compilation, and align
    estimates across ranks for distributed execution. These may lengthen cold
    compilation; they do not benchmark or communicate during configuration.

    ``producer_schedule`` separately experiments with moving a collective's
    packing/casting producers earlier. It retains collective order, permits no
    estimated peak-memory increase, and initially excludes other overlap knobs.
    """

    swap_with_collectives: bool = False
    use_runtime_estimations: bool = False
    peak_memory_budget: float = 0.0
    producer_schedule: bool = False

    def __post_init__(self):
        for name in ("swap_with_collectives", "use_runtime_estimations", "producer_schedule"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        value = self.peak_memory_budget
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("peak_memory_budget must be a finite fraction between 0 and 0.25")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError("peak_memory_budget must be a finite fraction between 0 and 0.25") from error
        if not math.isfinite(value) or not 0 <= value <= MAX_OVERLAP_MEMORY_BUDGET:
            raise ValueError("peak_memory_budget must be a finite fraction between 0 and 0.25")
        object.__setattr__(self, "peak_memory_budget", value)
        if self.producer_schedule and (self.swap_with_collectives or self.use_runtime_estimations or value):
            raise ValueError("producer_schedule initially requires default wait swapping, "
                             "runtime estimates, and zero peak-memory budget")

    @property
    def metadata(self):
        """JSON-compatible immutable choices for a run's execution signature."""
        return asdict(self)


def _configure_communication_overlap(options: OverlapSettings, *, distributed: bool) -> None:
    """Delay wait/copy groups while preserving collective order and estimated peak.

    The default DDP FX pass only sinks waits until their first consumer, which
    is immediately the gradient copy-back. Inductor's iterative scheduler can
    move that dependent group together past unrelated gradient computation.
    Defaults retain collective ordering, stop before the next collective, and
    allow no increase in estimated peak memory. Optional wait-group movement
    past later launches still prohibits changing their relative launch order.
    """
    from torch._inductor import comms, config, config_comms

    settings = {
        "reorder_for_compute_comm_overlap": True,
        "reorder_for_compute_comm_overlap_passes": ["sink_waits_iterative"],
    }
    communication_settings = {
        "sink_iterative_peak_memory_budget": options.peak_memory_budget,
        "sink_waits_iterative_unsafe_collectives_reorder": False,
        "sink_waits_iterative_swap_with_collectives": options.swap_with_collectives,
        "sink_iterative_use_runtime_estimations": options.use_runtime_estimations,
    }
    if options.producer_schedule:
        # Installed config.py recommends producer reordering followed by wait
        # sinking (then optionally another reorder). Its comms.py docstring
        # mentions a stale sink_comms_and_waits prerequisite that is not a
        # callable in this version. The ordinary peak-memory scheduler runs
        # before these passes. Start with just one guarded producer pass and
        # retain the existing final sink. Without runtime estimates this moves
        # pack/cast producer groups but stops when a GEMM producer is reached.
        settings["reorder_for_compute_comm_overlap_passes"] = [
            "reorder_communication_preserving_peak_memory", "sink_waits_iterative"]
        communication_settings.update(
            reorder_iterative_unsafe_collectives_reorder=False,
            reorder_iterative_peak_memory_budget=0.0,
            reorder_iterative_group_with_collectives=False,
            reorder_iterative_use_runtime_estimations=False,
        )
    if options.use_runtime_estimations:
        settings["runtime_estimations_mms_benchmark"] = True
        communication_settings["runtime_estimations_align_across_all_distributed_ranks"] = distributed
        if distributed and not callable(getattr(
                comms, "align_runtime_estimations_across_all_distributed_ranks", None)):
            raise RuntimeError("Installed PyTorch lacks distributed runtime-estimate alignment")
    for module, values in ((config, settings), (config_comms, communication_settings)):
        missing = [key for key in values if not hasattr(module, key)]
        if missing:
            raise RuntimeError("Installed PyTorch lacks overlap settings: " + ", ".join(missing))
    for name in settings["reorder_for_compute_comm_overlap_passes"]:
        if not callable(getattr(comms, name, None)):
            raise RuntimeError(f"Installed PyTorch lacks the {name} scheduler pass")
    if config.deterministic:
        raise ValueError("Inductor deterministic mode disables communication reordering; "
                         "compiled-autograd-overlap would have no effect")
    for module, values in ((config, settings), (config_comms, communication_settings)):
        for name, value in values.items():
            setattr(module, name, value)


def scale_loss(loss: torch.Tensor, scale: float, mode: str = "eager") -> torch.Tensor:
    """Apply the existing microbatch weight without CPU-scalar capture on CUDA.

    PyTorch 2.14 Compiled Autograd can emit invalid CUDA wrapper code for an
    eager multiplication's saved CPU scalar. Materializing the same coefficient
    in the loss's dtype/device avoids that path without changing the arithmetic.
    For repeated fixed weights a caller can equivalently prepare the scalar
    tensor once per update and multiply directly. This helper is convenient for
    the variable valid-token fraction in the final training update.
    """
    if mode not in GRADIENT_MODES:
        raise ValueError(f"Unknown gradient execution mode: {mode}")
    if mode != "eager" and loss.device.type == "cuda":
        return loss * loss.new_tensor(scale)
    return loss * scale


def _backward(loss: torch.Tensor) -> None:
    loss.backward()


def make_backward(
    mode: str = "eager", *, distributed: bool = False, backend: str | Callable = "inductor",
    overlap_settings: OverlapSettings | None = None,
) -> Callable[[torch.Tensor], None]:
    """Return the backward operation for a training process.

    Call once, before any ``torch.compile`` or DDP construction. The compiled
    mode enables process-wide Dynamo settings; each experiment must use its own
    process. Distributed execution uses PyTorch's Python reducer, whose ordinary
    post-accumulation hooks respect ``DDP.no_sync()`` and retain FP32 all-reduce.
    Its communication fusion uses the DDP bucket capacity. C++ reducer options
    such as ``batched_grad_copy`` are separate experiments, not combined modes.

    Leave forward, ``no_sync`` contexts, ``zero_grad``, clipping, and optimizer
    stepping with the caller. Use ``scale_loss`` for the existing loss weight,
    or multiply by a scalar tensor already on the CUDA device. The callable changes only
    how the existing autograd graph is executed. Existing checkpoint loading and
    saving require no model or optimizer conversion. A change in this setting
    should nevertheless be recorded as an execution reconfiguration.

    ``fullgraph`` applies to the captured autograd graph. PyTorch 2.14 still
    requires the outer wrapper to permit the ``Tensor.backward`` graph break.
    The first microbatch (no gradients) and later accumulating microbatches may
    create different graphs; the final DDP synchronization creates another.
    Compilation failures propagate rather than silently changing algorithms.

    ``compiled-autograd-overlap`` additionally enables a conservative scheduler
    pass that sinks collective waits and gradient copy-back together. FP32
    collective order is preserved; the default estimated peak-memory budget is
    zero. ``overlap_settings`` is accepted only in this mode. Runtime estimates
    also benchmark GEMMs during compilation and, when distributed, align timing
    estimates through the existing process group. Both ranks must compile the
    same collective-bearing graphs; normal DDP construction and no-sync ordering
    remain the caller's responsibility.
    The separate ``producer_schedule`` experiment first raises pack/cast
    producer groups and their collective, then keeps the conservative final
    wait sink. It cannot reorder collective launches or increase the scheduler's
    estimated peak; actual CUDA memory and throughput still require measurement.
    Select and record this separately because scheduling benefits require an
    actual distributed benchmark and compatible installed PyTorch internals.
    """
    if mode not in GRADIENT_MODES:
        raise ValueError(f"Unknown gradient execution mode: {mode}")
    if overlap_settings is not None:
        if not isinstance(overlap_settings, OverlapSettings):
            raise TypeError("overlap_settings must be an OverlapSettings instance")
        if mode != "compiled-autograd-overlap":
            raise ValueError("overlap_settings requires compiled-autograd-overlap")
    if mode == "eager":
        return _backward

    from torch._dynamo import config

    if mode == "compiled-autograd-overlap":
        _configure_communication_overlap(overlap_settings or OverlapSettings(), distributed=distributed)
    config.compiled_autograd = True
    config.compiled_autograd_kwargs_override = {
        **config.compiled_autograd_kwargs_override,
        "fullgraph": True,
    }
    if distributed:
        config.optimize_ddp = "python_reducer"
    return torch.compile(_backward, backend=backend, fullgraph=False, dynamic=False)
