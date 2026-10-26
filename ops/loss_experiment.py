"""Disposable full-vocabulary loss execution options for throughput experiments.

These options retain the existing model, tied embedding/head parameter, objective,
and canonical state dictionary. They trade additional logits/softmax memory for
avoiding the default chunked head's activation recomputation. Apply before DDP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _full_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.reshape(-1)
    values = logits.reshape(-1, logits.size(-1)).float()
    # Sum/valid-count also gives finite zero on a rank with no valid final tokens.
    total = F.cross_entropy(values, labels, ignore_index=-100, reduction="sum")
    return total / labels.ne(-100).sum().clamp_min(1)


def apply_loss_execution(model: torch.nn.Module, mode: str) -> None:
    """Select checkpointed, full, or compiled-full loss without replacing weights.

    Only calls with targets and ``return_logits=False`` use the experiment.
    Inference and explicit-logit calls retain the model's public behavior. The
    original forward owns every core operation and the tied output projection,
    so the experiment does not duplicate or diverge from the model architecture.
    ``compiled-full`` compiles the loss reduction, using eager loss in evaluation
    to avoid compiling another graph solely for validation's smaller microbatch.
    """
    if mode not in {"checkpointed", "full", "compiled-full"}:
        raise ValueError(f"Unknown loss execution mode: {mode}")
    if getattr(model, "_loss_execution_mode", "checkpointed") != "checkpointed":
        raise ValueError("Loss execution has already been patched on this model")
    if mode == "checkpointed":
        return

    original_forward = model.forward
    training_loss = _full_loss
    if mode == "compiled-full":
        training_loss = torch.compile(_full_loss, fullgraph=True, dynamic=False)

    def forward(input_ids, targets=None, *, return_logits=True):
        if targets is None or return_logits:
            return original_forward(input_ids, targets, return_logits=return_logits)
        logits = original_forward(input_ids, targets=None, return_logits=True)
        loss_fn = training_loss if model.training and torch.is_grad_enabled() else _full_loss
        return None, loss_fn(logits, targets)

    model.forward = forward
    model._loss_execution_mode = mode
