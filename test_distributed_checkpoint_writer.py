"""Two-process CPU/Gloo regressions for coordinated checkpoint publication."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch
import torch.distributed as dist

from ops import checkpoint_writer


def _worker(rank, directory, scenario):
    directory = Path(directory)
    report = {"rank": rank, "writes": [], "syncs": 0, "factory_calls": 0}
    writer = None
    release = threading.Event()
    original_write = checkpoint_writer.atomic_torch_save
    original_snapshot = checkpoint_writer.snapshot_to_cpu
    original_fsync = os.fsync
    original_broadcast = dist.broadcast

    def save(path, state):
        report["writes"].append({
            "rank": rank, "thread": threading.current_thread().name,
            "device": str(state["weight"].device),
        })
        if scenario in {"check", "submit", "flush", "close"}:
            raise OSError("injected checkpoint disk failure")
        assert release.wait(10), "Test did not release checkpoint serialization"
        original_write(path, state)

    def snapshot(state):
        if scenario == "snapshot":
            raise MemoryError("injected CPU snapshot failure")
        return original_snapshot(state)

    def fsync(descriptor):
        original_fsync(descriptor)
        report["syncs"] += 1

    def forbidden_collective(*args, **kwargs):
        raise AssertionError("Exceptional context exit attempted a collective")

    checkpoint_writer.atomic_torch_save = save
    checkpoint_writer.snapshot_to_cpu = snapshot
    os.fsync = fsync
    try:
        dist.init_process_group(
            "gloo", init_method=(directory / "rendezvous").as_uri(),
            rank=rank, world_size=2, timeout=timedelta(seconds=15),
        )
        try:
            with checkpoint_writer.DistributedCheckpointWriter(
                directory / "latest.pt", rank=rank,
                device=torch.device("cpu"), distributed=True,
            ) as writer:
                report["owns_writer"] = writer._writer is not None
                state = {"weight": torch.tensor([1.0, 2.0]), "metadata": [3]}

                def build_state():
                    report["factory_calls"] += 1
                    assert rank == 0, "Only rank zero may construct checkpoint state"
                    if scenario == "factory":
                        raise ValueError("injected checkpoint construction failure")
                    return state

                report["staging_seconds"] = (
                    writer.submit(state_factory=build_state)
                    if scenario in {"factory", "normal_factory"}
                    else writer.submit(state if rank == 0 else None)
                )
                if scenario in {"check", "submit", "flush", "close"}:
                    # Ensure the disk error has happened before asking every rank
                    # to observe it through each distinct public operation.
                    if rank == 0:
                        try:
                            writer._writer._future.result(timeout=10)
                        except OSError:
                            pass
                    dist.barrier()
                    if scenario == "submit":
                        writer.submit(state if rank == 0 else None)
                    else:
                        getattr(writer, scenario)()
                    raise AssertionError("Checkpoint failure was not raised")
                state["weight"].add_(10)
                state["metadata"].append(4)
                writer.check()  # A pending write is allowed to overlap training.
                report["exists_before_release"] = (directory / "latest.pt").exists()
                dist.barrier()
                release.set()
                if scenario == "exceptional_exit":
                    dist.broadcast = forbidden_collective
                    raise ValueError("injected training failure")
            report["normal_exit"] = True
        except BaseException as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        if writer is not None:
            report["closed"] = writer._writer is None or writer._writer._closed
    finally:
        release.set()
        checkpoint_writer.atomic_torch_save = original_write
        checkpoint_writer.snapshot_to_cpu = original_snapshot
        os.fsync = original_fsync
        dist.broadcast = original_broadcast
        if dist.is_initialized():
            dist.destroy_process_group()
        (directory / f"rank-{rank}.json").write_text(json.dumps(report))


def _run_two_ranks(directory, scenario):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    processes = []
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 str(rank), str(directory), scenario],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=40)
            assert process.returncode == 0, f"Rank {rank} failed:\n{output}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait(timeout=10)
    return [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(2)]


@pytest.mark.parametrize("operation", ["check", "submit", "flush", "close", "snapshot", "factory"])
def test_rank_zero_failure_reaches_both_ranks_without_hanging(tmp_path, operation):
    path = tmp_path / "latest.pt"
    torch.save({"old_checkpoint": True}, path)
    reports = _run_two_ranks(tmp_path, operation)
    expected_type, expected_message = {
        "snapshot": ("MemoryError", "CPU snapshot failure"),
        "factory": ("ValueError", "checkpoint construction failure"),
    }.get(operation, ("OSError", "checkpoint disk failure"))
    assert reports[0]["error"]["type"] == expected_type
    assert reports[1]["error"]["type"] == "RuntimeError"
    for rank, report in enumerate(reports):
        assert expected_message in report["error"]["message"]
        assert report["owns_writer"] is (rank == 0)
        assert report["closed"]
    assert reports[1]["writes"] == []
    assert reports[1]["factory_calls"] == 0
    assert reports[0]["factory_calls"] == int(operation == "factory")
    assert torch.load(path, weights_only=False) == {"old_checkpoint": True}
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("scenario", ["normal_exit", "normal_factory"])
def test_normal_exit_publishes_one_immutable_cpu_snapshot_durably(tmp_path, scenario):
    reports = _run_two_ranks(tmp_path, scenario)
    for rank, report in enumerate(reports):
        assert "error" not in report, report
        assert report["owns_writer"] is (rank == 0)
        assert report["normal_exit"] and report["closed"]
        assert report["exists_before_release"] is False
    assert reports[0]["staging_seconds"] >= 0
    assert reports[1]["staging_seconds"] is None
    assert reports[1]["factory_calls"] == 0
    assert reports[0]["factory_calls"] == int(scenario == "normal_factory")
    assert reports[1]["writes"] == []
    assert reports[1]["syncs"] == 0
    assert reports[0]["syncs"] == 2  # File contents, then directory publication.
    assert len(reports[0]["writes"]) == 1
    write = reports[0]["writes"][0]
    assert write["device"] == "cpu" and write["thread"].startswith("checkpoint-writer")
    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    torch.testing.assert_close(state["weight"], torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    assert state["metadata"] == [3]
    assert not (tmp_path / "latest.pt.tmp").exists()


def test_exceptional_exit_joins_local_writer_without_any_collectives(tmp_path):
    reports = _run_two_ranks(tmp_path, "exceptional_exit")
    for report in reports:
        assert report["error"] == {"type": "ValueError", "message": "injected training failure"}
        assert report["closed"]
    assert (tmp_path / "latest.pt").is_file()
    assert reports[1]["writes"] == []


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2], sys.argv[3])
