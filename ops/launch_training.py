#!/usr/bin/env python3
"""Supervise torchrun while allowing workers time to save on a stop request."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


WORKSPACE = Path('/workspace/WarpState').resolve()
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_40b import process_lock


def workspace_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f'Created paths must stay inside {WORKSPACE}: {path}')
    return path


def process_info(pid):
    """Return parent and creation time; creation time guards against PID reuse."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def remember_descendants(parent, known, parent_created):
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = process_info(int(path.name))
            if info is not None:
                processes[int(path.name)] = info
    ancestors = {parent} if processes.get(parent, (None, None))[1] == parent_created else set()
    ancestors.update(pid for pid, created in known.items()
                     if processes.get(pid, (None, None))[1] == created)
    while True:
        added = {pid for pid, (ppid, _) in processes.items() if ppid in ancestors} - ancestors
        if not added:
            break
        ancestors.update(added)
    for pid in ancestors - {parent}:
        if pid in processes:
            known[pid] = processes[pid][1]


def kill_descendants(known):
    """Only signal processes whose recorded identities still match."""
    for pid, created in known.items():
        current = process_info(pid)
        if current is not None and current[1] == created:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def supervise(command, stop_path, environment, *, grace_seconds=820, terminate_seconds=45,
              pass_fds=()):
    """A stop file reaches workers even though torchrun gives each a new session."""
    stop_path = workspace_path(stop_path)
    stopped_at = None
    handlers = {}
    child = None
    child_created = None
    descendants = {}

    def request_stop(signum, frame):
        nonlocal stopped_at
        if stopped_at is None:
            stopped_at = time.monotonic()
        descriptor = os.open(stop_path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        handlers[sig] = signal.signal(sig, request_stop)
    try:
        if stopped_at is not None or stop_path.exists():
            return 0
        # Handoff jobs pass their exclusion locks to torchrun, which retains them
        # even if the supervising Python process is unexpectedly killed.
        child = subprocess.Popen(command, env=environment, start_new_session=True,
                                 pass_fds=pass_fds)
        identity = process_info(child.pid)
        child_created = identity[1] if identity is not None else None
        print(f'Training launcher started PID {child.pid}; stop requests use {stop_path}', flush=True)
        last_scan = 0.0
        reported_stop = False
        termination_sent = None
        while child.poll() is None:
            now = time.monotonic()
            if now - last_scan >= 2:
                remember_descendants(child.pid, descendants, child_created)
                last_scan = now
            if stopped_at is not None:
                if not reported_stop:
                    print('Stop requested; waiting for the current update and checkpoint.', flush=True)
                    reported_stop = True
                if termination_sent is None and now - stopped_at >= grace_seconds:
                    remember_descendants(child.pid, descendants, child_created)
                    print('Graceful-stop deadline exceeded; asking torchrun to terminate its workers.', flush=True)
                    try:
                        child.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    termination_sent = now
                elif termination_sent is not None and now - termination_sent >= terminate_seconds:
                    remember_descendants(child.pid, descendants, child_created)
                    print('Termination deadline exceeded; killing remaining training processes.', flush=True)
                    kill_descendants(descendants)
                    child.kill()
                    child.wait(timeout=5)
                    break
            try:
                child.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        return child.returncode
    finally:
        if child is not None:
            remember_descendants(child.pid, descendants, child_created)
            if child.poll() is None:
                kill_descendants(descendants)
                child.kill()
                child.wait(timeout=5)
            kill_descendants(descendants)
        for sig, previous in handlers.items():
            signal.signal(sig, previous)


def training_environment(work):
    cache = workspace_path(WORKSPACE / '.cache')
    hf = workspace_path(work / 'hf-cache')
    paths = {
        'TMPDIR': WORKSPACE / 'tmp',
        'XDG_CACHE_HOME': cache, 'UV_CACHE_DIR': cache / 'uv',
        'PIP_CACHE_DIR': cache / 'pip', 'TORCH_HOME': cache / 'torch',
        'CUDA_CACHE_PATH': cache / 'cuda', 'TRITON_CACHE_DIR': cache / 'triton',
        'TORCHINDUCTOR_CACHE_DIR': cache / 'torchinductor',
        'TORCH_EXTENSIONS_DIR': cache / 'torch-extensions',
        'HF_HOME': hf, 'HF_DATASETS_CACHE': hf / 'datasets', 'HF_XET_CACHE': hf / 'xet',
    }
    environment = os.environ.copy()
    for key, value in paths.items():
        path = workspace_path(value)
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['WARPSTATE_STOP_FILE'] = str(workspace_path(work / 'control/training-stop-request'))
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--work-dir', type=Path, default=PACKAGE / 'work')
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error('--processes must be positive')
    extra = args.runner_args[1:] if args.runner_args[:1] == ['--'] else args.runner_args
    if any(value == '--work-dir' or value.startswith('--work-dir=') for value in extra):
        parser.error('Pass --work-dir to the launcher, before the -- separator')
    try:
        work = workspace_path(args.work_dir)
        control = workspace_path(work / 'control')
        logs = workspace_path(work / 'logs/torchrun-training')
        stop_path = workspace_path(control / 'training-stop-request')
        lock_path = workspace_path(control / 'training-launcher.lock')
    except ValueError as error:
        parser.error(str(error))
    control.mkdir(parents=True, exist_ok=True)
    with process_lock(lock_path):
        stop_path.unlink(missing_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        environment = training_environment(work)
        command = [sys.executable]
        if args.processes > 1:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                        '--nproc-per-node', str(args.processes), '--log-dir', str(logs)]
        command += [str(PACKAGE / 'run_40b.py'), '--work-dir', str(work), *extra]
        result = supervise(command, stop_path, environment)
    return 128 - result if result < 0 else result


if __name__ == '__main__':
    sys.exit(main())
