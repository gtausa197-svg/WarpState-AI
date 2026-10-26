"""BF16 residual-stream experiment with unchanged FP32 trainable parameters.

Only embedding and depth modulation use temporary, differentiable BF16 views.
The existing forward owns every model operation. Core matrix weights, parameter
gradients, normalization reductions, optimizer state, and checkpoint names retain
their existing representation. Residual rounding changes; this is experimental.
"""

from __future__ import annotations

import torch
from torch.func import functional_call


class _OriginalForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model
        self.original_forward = model.forward

    def forward(self, *args, **kwargs):
        return self.original_forward(*args, **kwargs)


def apply_bf16_residuals(model: torch.nn.Module) -> None:
    """Apply after loading weights/placing the model and before wrapping in DDP.

    ``functional_call`` restores all master parameter references after each
    forward, including on errors. Casts remain differentiable, so gradients flow
    to the original FP32 leaves and the existing optimizer continues to own them.
    The temporary wrapper is held in the forward closure, never registered as a
    child of the model, which preserves the canonical state dictionary.

    BF16 views are enabled only inside BF16 autocast. Inference calls outside
    autocast retain the original forward's behavior.
    """
    if getattr(model, "_bf16_residuals", False):
        raise ValueError("BF16 residual execution has already been applied")
    parameters = {
        "wrapped.embed.weight": model.embed.weight,
        "wrapped.depth_scale": model.depth_scale,
        "wrapped.depth_bias": model.depth_bias,
    }
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise ValueError("BF16 residual execution requires FP32 master parameters")
    wrapper = _OriginalForward(model)

    def forward(input_ids, targets=None, *, return_logits=True):
        device_type = input_ids.device.type
        kwargs = {"targets": targets, "return_logits": return_logits}
        if (not torch.is_autocast_enabled(device_type)
                or torch.get_autocast_dtype(device_type) != torch.bfloat16):
            return wrapper.original_forward(input_ids, **kwargs)
        views = {name: parameter.to(torch.bfloat16) for name, parameter in parameters.items()}
        return functional_call(wrapper, views, (input_ids,), kwargs, strict=False)

    model.forward = forward
    model._bf16_residuals = True
