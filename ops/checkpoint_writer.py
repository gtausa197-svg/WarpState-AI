"""Stage immutable CPU checkpoints, then serialize one at a time in a thread.

GPU-to-CPU copies finish synchronously before training can update live tensors.
Only serialization and durable publication overlap subsequent training. The
snapshot needs host memory for model parameters and full optimizer state; no
additional full checkpoint is allocated on the GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist


def snapshot_to_cpu(state):
    """Copy every tensor and mutable metadata without copying tensors on CUDA.

    Prepopulate deepcopy's memo with detached CPU tensors. This also retains
    repeated tensor references, OrderedDict metadata, and NumPy/Python RNG
    objects while preventing any alias to subsequently mutated live state.
    """
    memo = {}
    visited = set()

    def stage(value):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            memo[identity] = value.detach().to(device='cpu', copy=True, non_blocking=False)
        elif isinstance(value, dict):
            for key, item in value.items():
                stage(key)
                stage(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                stage(item)

    stage(state)
    return copy.deepcopy(state, memo)


def atomic_torch_save(path, state):
    """Publish the existing single-file torch.save format only after fsync."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AsyncCheckpointWriter:
    """One outstanding immutable snapshot; errors surface on check/flush/close.

    submit() blocks on the previous save before allocating the next snapshot,
    bounding both queued writes and additional host checkpoint memory to one.
    Use as a context manager, or always close in a finally block: an exception
    in training must still join an already staged, valid checkpoint write.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-writer')
        self._future = None
        self._error = None
        self._closed = False
        self._lock = threading.RLock()

    def _raise_error(self):
        if self._error is not None:
            raise self._error

    def _finish(self):
        self._raise_error()
        if self._future is not None:
            try:
                self._future.result()
            except BaseException as error:
                self._error = error
                raise
            finally:
                self._future = None

    def _write(self, snapshot):
        begin = time.perf_counter()
        atomic_torch_save(self.path, snapshot)
        progress = ''
        if isinstance(snapshot, dict) and 'step' in snapshot and 'tokens_seen' in snapshot:
            progress = f'step {snapshot["step"]}, tokens={snapshot["tokens_seen"]:,}; '
        print(f'Saved {self.path} ({progress}background write {time.perf_counter()-begin:.2f}s)', flush=True)

    def submit(self, state):
        """Wait for the prior write, stage a frozen CPU copy, return staging seconds."""
        with self._lock:
            self._raise_error()
            if self._closed:
                raise RuntimeError('Checkpoint writer is closed')
            self._finish()
            begin = time.perf_counter()
            snapshot = snapshot_to_cpu(state)
            staging_seconds = time.perf_counter() - begin
            self._future = self._executor.submit(self._write, snapshot)
            return staging_seconds

    def check(self):
        """Propagate a completed background failure without waiting on a live save."""
        with self._lock:
            self._raise_error()
            if self._future is not None and self._future.done():
                self._finish()

    def flush(self):
        """Join the outstanding save and surface any previous failure."""
        with self._lock:
            self._finish()

    def close(self):
        with self._lock:
            if self._closed:
                self._raise_error()
                return
            try:
                self._finish()
            finally:
                self._closed = True
                self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            _close_after_error(self, exc_value)
        return False


def _close_after_error(writer, original_error):
    """Join locally without replacing the exception that stopped training."""
    try:
        writer.close()
    except BaseException as error:
        if error is not original_error:
            original_error.add_note(
                f'Checkpoint cleanup also failed: {type(error).__name__}: {error}'
            )


class DistributedCheckpointWriter:
    """Coordinate one rank-zero writer from the training thread on every rank.

    All ranks must call check/submit/flush/close in the same order. A status
    broadcast after each rank-zero operation keeps peers from entering the next
    training collective if staging, waiting, or serialization failed. submit()
    returns only after the snapshot is immutable; the subsequent CPU-only file
    write overlaps training. No background thread uses CUDA or distributed APIs.

    Normal context exit joins the writer collectively. Exceptional exit only
    joins the local writer: a training failure may be rank-local, so another
    collective could hang instead of allowing the launcher to stop its peers.
    """

    def __init__(self, path, *, rank=0, device=None, distributed=False):
        self.rank = rank
        self.distributed = distributed
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self._writer = None
        self._status = (
            torch.empty(1, dtype=torch.int32, device=self.device)
            if distributed else None
        )

        def initialize():
            self._writer = AsyncCheckpointWriter(path)

        self._coordinate('initialization', initialize)

    def _coordinate(self, operation, action):
        result = None
        error = None
        if self.rank == 0:
            try:
                result = action()
            except BaseException as caught:
                error = caught
        if self.distributed:
            if self.rank == 0:
                self._status.fill_(int(error is not None))
            dist.broadcast(self._status, src=0)
            if self._status.item():
                details = [
                    f'{type(error).__name__}: {error}' if self.rank == 0 else None
                ]
                dist.broadcast_object_list(details, src=0, device=self.device)
                if error is None:
                    raise RuntimeError(
                        f'Rank 0 checkpoint {operation} failed: {details[0]}'
                    )
        if error is not None:
            raise error
        return result

    def check(self):
        """Surface a completed write failure to all ranks before training."""
        return self._coordinate('check', lambda: self._writer.check())

    def submit(self, state=None, *, state_factory=None):
        """Freeze rank zero's state; peers wait for staging or a shared error.

        Pass a state_factory to build the checkpoint dictionary inside the
        coordinated operation. Only rank zero calls it, after the previous
        write finishes, so construction errors also reach every rank.
        """
        def stage():
            if state_factory is not None:
                if state is not None:
                    raise ValueError('Pass checkpoint state or state_factory, not both')
                self._writer.flush()
                return self._writer.submit(state_factory())
            return self._writer.submit(state)

        return self._coordinate('submit', stage)

    def flush(self):
        """Wait for durable publication and report any failure to every rank."""
        return self._coordinate('flush', lambda: self._writer.flush())

    def close(self):
        """Join the writer and report its final status on every rank."""
        return self._coordinate('close', lambda: self._writer.close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        elif self._writer is not None:
            _close_after_error(self._writer, exc_value)
        return False
