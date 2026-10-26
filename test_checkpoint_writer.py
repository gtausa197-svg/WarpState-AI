"""CPU regressions for immutable, bounded asynchronous checkpoint publication."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sys
import threading

import numpy as np
import pytest
import torch

from ops import checkpoint_writer
from run_40b import arguments


def training_state():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": 1, "tokens_seen": 8,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.arange(4)},
        "metadata": {"history": [{"batch": 2}], "tuple": ([1, 2], "bf16")},
    }
    return model, optimizer, state


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == "cpu"
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, other in zip(actual, expected, strict=True):
            assert_state_equal(value, other)
    else:
        assert actual == expected


def test_pending_save_keeps_model_optimizer_and_metadata_snapshot_immutable(tmp_path, monkeypatch):
    model, optimizer, state = training_state()
    expected = deepcopy(state)
    entered, release = threading.Event(), threading.Event()
    original = checkpoint_writer.atomic_torch_save

    def blocked_write(path, snapshot):
        entered.set()
        assert release.wait(10), "Test did not release checkpoint writer"
        original(path, snapshot)

    monkeypatch.setattr(checkpoint_writer, "atomic_torch_save", blocked_write)
    path = tmp_path / "latest.pt"
    writer = checkpoint_writer.AsyncCheckpointWriter(path)
    try:
        writer.submit(state)
        assert entered.wait(10)
        # A real second AdamW update mutates parameters, moments, and step counters.
        optimizer.zero_grad(set_to_none=True)
        model(torch.full((2, 2), 2.0)).square().mean().backward()
        optimizer.step()
        state["optimizer"]["param_groups"][0]["lr"] = 0.5
        state["rng"]["numpy"][:] = -1
        state["rng"]["torch"].zero_()
        state["metadata"]["history"][0]["batch"] = 99
        state["metadata"]["tuple"][0].append(3)
        assert not path.exists()
        release.set()
        writer.flush()
        actual = torch.load(path, map_location="cpu", weights_only=False)
        assert_state_equal(actual, expected)
        assert not torch.equal(state["model"]["weight"], expected["model"]["weight"])
        assert state["optimizer"]["state"][0]["step"].item() == 2
    finally:
        release.set()
        writer.close()


def test_atomic_save_keeps_old_checkpoint_until_complete_publication(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    torch.save({"step": 1}, path)
    written, release = threading.Event(), threading.Event()
    original = torch.save

    def pause_after_serialization(state, stream, *args, **kwargs):
        original(state, stream, *args, **kwargs)
        written.set()
        assert release.wait(10), "Test did not release serialized checkpoint"

    monkeypatch.setattr(torch, "save", pause_after_serialization)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(checkpoint_writer.atomic_torch_save, path, {"step": 2})
        try:
            assert written.wait(10)
            assert torch.load(path, weights_only=False) == {"step": 1}
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=10)
    assert torch.load(path, weights_only=False) == {"step": 2}


def test_failed_serialization_keeps_old_checkpoint_and_removes_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    torch.save({"step": 1}, path)

    def partial_write_then_fail(state, stream):
        stream.write(b"incomplete checkpoint")
        raise OSError("serialization disk failure")

    monkeypatch.setattr(torch, "save", partial_write_then_fail)
    with pytest.raises(OSError, match="serialization disk failure"):
        checkpoint_writer.atomic_torch_save(path, {"step": 2})
    assert torch.load(path, weights_only=False) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_second_submit_waits_before_allocating_another_snapshot(tmp_path, monkeypatch):
    entered, release, attempting, snapshotted_second = [threading.Event() for _ in range(4)]
    original_write = checkpoint_writer.atomic_torch_save
    original_snapshot = checkpoint_writer.snapshot_to_cpu
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def snapshot(state):
        if state["step"] == 2:
            snapshotted_second.set()
        return original_snapshot(state)

    def blocked_write(path, state):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if state["step"] == 1:
                entered.set()
                assert release.wait(10), "Test did not release first writer"
            original_write(path, state)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(checkpoint_writer, "snapshot_to_cpu", snapshot)
    monkeypatch.setattr(checkpoint_writer, "atomic_torch_save", blocked_write)
    path = tmp_path / "latest.pt"
    writer = checkpoint_writer.AsyncCheckpointWriter(path)
    try:
        writer.submit({"step": 1, "tokens_seen": 8})
        assert entered.wait(10)

        def submit_second():
            attempting.set()
            writer.submit({"step": 2, "tokens_seen": 16})

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(submit_second)
            try:
                assert attempting.wait(10)
                assert not snapshotted_second.wait(0.2)
                assert not future.done()
            finally:
                release.set()
            future.result(timeout=10)
        writer.flush()
        assert snapshotted_second.is_set()
        assert maximum_active == 1
        assert torch.load(path, weights_only=False) == {"step": 2, "tokens_seen": 16}
    finally:
        release.set()
        writer.close()


def test_background_failure_is_persistent_and_preserves_old_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    torch.save({"step": 1}, path)

    def failed_write(path, state):
        raise OSError("checkpoint disk failure")

    monkeypatch.setattr(checkpoint_writer, "atomic_torch_save", failed_write)
    writer = checkpoint_writer.AsyncCheckpointWriter(path)
    writer.submit({"step": 2, "tokens_seen": 16})
    for action in (writer.flush, writer.check,
                   lambda: writer.submit({"step": 3, "tokens_seen": 24}), writer.close):
        with pytest.raises(OSError, match="checkpoint disk failure"):
            action()
    assert torch.load(path, weights_only=False) == {"step": 1}


@pytest.mark.parametrize("exceptional_exit", [False, True])
def test_context_exit_flushes_pending_checkpoint(tmp_path, monkeypatch, exceptional_exit):
    entered, release, leaving_context, finished = [threading.Event() for _ in range(4)]
    original = checkpoint_writer.atomic_torch_save
    path = tmp_path / "latest.pt"

    def blocked_write(path, state):
        entered.set()
        assert release.wait(10), "Test did not release writer during exit"
        original(path, state)

    monkeypatch.setattr(checkpoint_writer, "atomic_torch_save", blocked_write)

    def run_context():
        try:
            with checkpoint_writer.AsyncCheckpointWriter(path) as writer:
                writer.submit({"step": 7, "tokens_seen": 73})
                assert entered.wait(10)
                leaving_context.set()
                if exceptional_exit:
                    raise ValueError("training failed")
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_context)
        try:
            assert leaving_context.wait(10)
            assert not finished.wait(0.2)
            assert not path.exists()
        finally:
            release.set()
        if exceptional_exit:
            with pytest.raises(ValueError, match="training failed"):
                future.result(timeout=10)
        else:
            future.result(timeout=10)
    assert torch.load(path, weights_only=False) == {"step": 7, "tokens_seen": 73}


@pytest.mark.parametrize(("flags", "enabled"), [
    ([], False),
    (["--async-checkpoint"], True),
    (["--async-checkpoint", "--no-async-checkpoint"], False),
])
def test_async_checkpoint_cli_is_explicit_and_reversible(monkeypatch, flags, enabled):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--stage", "train", *flags])
    assert arguments().async_checkpoint is enabled


def test_async_checkpoint_cli_accepts_multiple_training_ranks(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(sys, "argv", ["run_40b.py", "--stage", "train", "--async-checkpoint"])
    assert arguments().async_checkpoint is True


def test_training_exception_preserved_when_background_write_also_fails(tmp_path, monkeypatch):
    def failed_write(path, state):
        raise OSError("checkpoint disk failure")

    monkeypatch.setattr(checkpoint_writer, "atomic_torch_save", failed_write)
    with pytest.raises(ValueError, match="training failed") as error:
        with checkpoint_writer.AsyncCheckpointWriter(tmp_path / "latest.pt") as writer:
            writer.submit({"step": 1})
            raise ValueError("training failed")
    assert any("checkpoint disk failure" in note for note in error.value.__notes__)
