"""The experimental loss retains the full vocabulary objective and model state."""

import copy

import pytest
import torch

from ops.loss_experiment import apply_loss_execution
from warpstate.config import WarpStateConfig
from warpstate.model import WarpStateLM


@pytest.mark.parametrize("mode", ["full", "compiled-full"])
@pytest.mark.parametrize("mask", ["none", "partial", "all"])
def test_full_loss_preserves_objective_gradients_and_parameters(mode, mask, monkeypatch):
    # Exercise full graph capture on CPU without hardware-specific Inductor code.
    if mode == "compiled-full":
        actual_compile = torch.compile
        monkeypatch.setattr(torch, "compile", lambda fn, **kw: actual_compile(fn, backend="eager", **kw))
    torch.manual_seed(572)
    cfg = WarpStateConfig(vocab_size=32, d_model=16, n_heads=2, n_cores=1,
                          logical_depth=2, ffn_hidden=24, chunk_size=4,
                          gradient_checkpointing=False, parallel_memory=True,
                          loss_chunk_size=3)
    reference = WarpStateLM(cfg).train()
    experiment = copy.deepcopy(reference)
    keys = list(experiment.state_dict())
    parameters = list(experiment.parameters())
    embedding = experiment.embed.weight
    apply_loss_execution(experiment, mode)
    assert list(experiment.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, experiment.parameters(), strict=True))
    assert experiment.embed.weight is embedding
    inputs = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    if mask == "partial":
        labels.flatten()[3:11] = -100
    elif mask == "all":
        labels.fill_(-100)
    ref_logits, ref_loss = reference(inputs, labels, return_logits=False)
    logits, loss = experiment(inputs, labels, return_logits=False)
    assert ref_logits is None and logits is None
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
    ref_loss.backward()
    loss.backward()
    for (name, expected), (other_name, actual) in zip(reference.named_parameters(), experiment.named_parameters(), strict=True):
        assert name == other_name
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-6)
    if mask == "all":
        assert loss.item() == 0.0
        assert all(torch.count_nonzero(p.grad) == 0 for p in experiment.parameters())
    with torch.no_grad():
        reference.eval()
        experiment.eval()
        torch.testing.assert_close(experiment(inputs), reference(inputs), rtol=0, atol=0)
        torch.testing.assert_close(experiment(inputs, labels, return_logits=False)[1], ref_loss.detach(), rtol=1e-6, atol=1e-6)


def test_loss_mode_validation_and_checkpointed_noop():
    model = WarpStateLM(WarpStateConfig(vocab_size=16, d_model=8, n_heads=1,
                                      n_cores=1, logical_depth=1, ffn_hidden=8,
                                      chunk_size=4))
    forward = model.forward
    apply_loss_execution(model, "checkpointed")
    assert model.forward == forward
    with pytest.raises(ValueError, match="Unknown loss"):
        apply_loss_execution(model, "invalid")
    apply_loss_execution(model, "full")
    with pytest.raises(ValueError, match="already been patched"):
        apply_loss_execution(model, "full")
