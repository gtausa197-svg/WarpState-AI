"""Validated, opt-in DDP execution settings without changing gradient precision."""

import inspect
import math
from numbers import Real


def _capacity(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number of MiB")
    if not math.isfinite(value) or value <= 0 or value * 1024 * 1024 < 1:
        raise ValueError(f"{name} must be finite and at least one byte in MiB")
    return float(value)


def ddp_runtime_settings(*, bucket_cap_mb=128, batched_grad_copy=False,
                         bucket_cap_mb_list=None):
    """Return serializable settings; a capacity list overrides the uniform cap.

    These options tune the existing FP32 gradient reducer. They do not install a
    communication hook, change reduction precision, or alter accumulation. The
    default keeps the historical 128 MiB buckets and individual gradient copies.
    """
    capacity = _capacity(bucket_cap_mb, "bucket_cap_mb")
    if not isinstance(batched_grad_copy, bool):
        raise ValueError("batched_grad_copy must be a boolean")
    capacities = None
    if bucket_cap_mb_list is not None:
        if not isinstance(bucket_cap_mb_list, (list, tuple)) or not bucket_cap_mb_list:
            raise ValueError("bucket_cap_mb_list must be a nonempty list of capacities")
        capacities = [_capacity(value, "bucket_cap_mb_list entry")
                      for value in bucket_cap_mb_list]
        capacity = max(capacities)
    return {
        "ddp_bucket_cap_mb": capacity,
        "ddp_batched_grad_copy": batched_grad_copy,
        "ddp_bucket_cap_mb_list": capacities,
    }


def ddp_kwargs(settings):
    """Validate stored settings and build kwargs for the installed DDP version.

    Unsupported experimental options fail explicitly instead of silently running
    a different benchmark. Unselected new arguments are omitted so historical
    defaults still work with PyTorch versions that predate those arguments.
    """
    from torch.nn.parallel import DistributedDataParallel

    resolved = ddp_runtime_settings(
        bucket_cap_mb=settings.get("ddp_bucket_cap_mb", 128),
        batched_grad_copy=settings.get("ddp_batched_grad_copy", False),
        bucket_cap_mb_list=settings.get("ddp_bucket_cap_mb_list"),
    )
    kwargs = {
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": resolved["ddp_bucket_cap_mb"],
    }
    requested = {}
    if resolved["ddp_batched_grad_copy"]:
        requested["batched_grad_copy"] = True
    if resolved["ddp_bucket_cap_mb_list"] is not None:
        requested["bucket_cap_mb_list"] = resolved["ddp_bucket_cap_mb_list"]
    parameters = inspect.signature(DistributedDataParallel).parameters
    for name, value in requested.items():
        if name not in parameters:
            raise RuntimeError(f"Installed PyTorch DDP does not support requested {name}")
        kwargs[name] = value
    if requested:
        import torch._dynamo

        if torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer":
            # Compiled autograd uses per-parameter Python hooks and compiler
            # communication fusion instead of the C++ reducer's copy/bucket
            # implementation. Do not report inactive flags as measured options.
            raise ValueError("batched_grad_copy and bucket_cap_mb_list require "
                             "the C++ DDP reducer; disable them for python_reducer")
    return kwargs
