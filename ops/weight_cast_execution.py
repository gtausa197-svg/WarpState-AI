"""Experimental per-update BF16 core weight views with FP32 master gradients."""

from collections import Counter
from contextlib import contextmanager

import torch
from torch import nn
from torch.func import functional_call


class _OriginalForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model
        self.original_forward = model.forward

    def forward(self, *args, **kwargs):
        return self.original_forward(*args, **kwargs)


class CoreWeightCastCache:
    """Own differentiable casts for one complete accumulation update only.

    Install after selecting the model's forward/loss/residual execution and
    before DDP construction. Enclose all accumulation forwards and backwards in
    ``with cache.update():`` and exit before clipping, optimizer, evaluation, or
    checkpoint operations. Ordinary execution outside that scope is unchanged.

    ``ToCopyBackward`` returns each BF16 gradient to its original FP32 leaf on
    every backward. Only that stateless conversion node is reused across
    microsteps; completed forward graphs are never retained. FP32 parameters,
    optimizer ownership, DDP hooks, and state dictionary names remain unchanged.
    """

    def __init__(self, model):
        if getattr(model, "_core_weight_cast_cache", None) is not None:
            raise ValueError("Core weight cast caching has already been applied")
        cfg = getattr(model, "cfg", None)
        if cfg is None or cfg.logical_depth != cfg.n_cores:
            raise ValueError("Core weight cast caching requires one use of each core per forward")
        if cfg.gradient_checkpointing:
            raise ValueError("Core weight cast caching requires core activation checkpointing off")
        self._model = model
        self._masters = {}
        for core_name, core in model.cores.named_children():
            for name, module in core.named_modules():
                if isinstance(module, nn.Linear):
                    parameter_name = f"cores.{core_name}.{name}.weight"
                    parameter = module.weight
                    if parameter.dtype != torch.float32 or not parameter.requires_grad:
                        raise ValueError("Core weight cast caching requires trainable FP32 linear weights")
                    self._masters[parameter_name] = parameter
        if not self._masters:
            raise ValueError("No core linear weights were found")
        references = Counter(id(parameter) for _, parameter in
                             model.named_parameters(remove_duplicate=False))
        if any(references[id(parameter)] != 1 for parameter in self._masters.values()):
            raise ValueError("Core weight cast caching does not support aliased linear weights")
        self._views = None
        self._versions = None
        self._wrapper = _OriginalForward(model)
        model.forward = self._forward
        model._core_weight_cast_cache = self

    @property
    def active(self):
        return self._views is not None

    @property
    def cached_tensor_count(self):
        return len(self._views) if self._views is not None else 0

    @property
    def cached_bytes(self):
        return sum(tensor.numel() * tensor.element_size()
                   for tensor in (self._views or {}).values())

    def _validate_masters(self, *, check_versions):
        current = dict(self._model.named_parameters())
        for name, parameter in self._masters.items():
            if current.get(name) is not parameter:
                raise RuntimeError("Core master parameter identity changed; recreate the model and cast cache")
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise RuntimeError("Core master weights must remain trainable FP32 parameters")
            if check_versions and parameter._version != self._versions[name]:
                raise RuntimeError("Core weights changed inside an accumulation scope; "
                                   "finish the cast-cache scope before optimizer or checkpoint loading")
        cfg = self._model.cfg
        if cfg.logical_depth != cfg.n_cores or cfg.gradient_checkpointing:
            raise RuntimeError("Core reuse or activation checkpointing changed after cast-cache setup")

    @contextmanager
    def update(self):
        """Create fresh casts, then release them even when an update fails."""
        if self.active:
            raise RuntimeError("Core weight cast accumulation scopes cannot be nested")
        if not self._model.training or not torch.is_grad_enabled():
            raise RuntimeError("Core weight cast accumulation requires training with gradients enabled")
        self._validate_masters(check_versions=False)
        self._versions = {name: parameter._version for name, parameter in self._masters.items()}
        try:
            self._views = {f"wrapped.{name}": parameter.to(torch.bfloat16)
                           for name, parameter in self._masters.items()}
            yield self
            self._validate_masters(check_versions=True)
        finally:
            self._views = None
            self._versions = None

    def _forward(self, input_ids, *args, **kwargs):
        if not self.active:
            return self._wrapper.original_forward(input_ids, *args, **kwargs)
        if not self._model.training or not torch.is_grad_enabled():
            raise RuntimeError("Exit the cast-cache accumulation scope before inference or evaluation")
        device_type = input_ids.device.type
        if (not torch.is_autocast_enabled(device_type)
                or torch.get_autocast_dtype(device_type) != torch.bfloat16):
            raise RuntimeError("An active core weight cast cache requires BF16 autocast")
        self._validate_masters(check_versions=True)
        return functional_call(self._wrapper, self._views, (input_ids, *args), kwargs,
                               strict=False)


def apply_weight_cast_cache(model):
    """Install the opt-in forward adapter and return its accumulation controller."""
    return CoreWeightCastCache(model)
