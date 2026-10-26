"""CPU process test: orphan SFT ranks continue excluding base training."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from run_40b import process_lock


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def _identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _wait(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for the CPU lock-test process")


def _kill(identity):
    pid, created = identity
    current = _identity(pid)
    if current is not None and current[0] != "Z" and current[1] == created:
        os.kill(pid, signal.SIGKILL)


def _dead(identity):
    current = _identity(identity[0])
    return current is None or current[0] == "Z" or current[1] != identity[1]


def _exclusive_available(path):
    try:
        with process_lock(path):
            return True
    except BlockingIOError:
        return False


def test_detached_shared_ranks_exclude_base_after_both_ancestors_die():
    """Exercise the real worker helper and base lock across a fatal orphan tree.

    Fake torchrun deliberately closes inherited descriptors and starts workers
    in separate sessions, matching the installed elastic subprocess launcher.
    Either surviving rank must independently prevent an exclusive base/handoff
    lock, even after the guardian and torchrun process have both been SIGKILLed.
    """
    temporary_root = WORKSPACE / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="posttrain-worker-lock-", dir=temporary_root) as folder:
        directory = Path(folder)
        pipeline = directory / "pipeline.lock"
        pipeline.write_text("Existing lock inode must be preserved.\n")
        initial_inode = pipeline.stat().st_ino
        initial_bytes = pipeline.read_bytes()
        helper = directory / "tree.py"
        helper.write_text(
            "import json, os, pathlib, subprocess, sys, time\n"
            "mode, root = sys.argv[1], pathlib.Path(sys.argv[2])\n"
            "def ready(name):\n"
            "    fields = pathlib.Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()\n"
            "    target = root / (name + '.json')\n"
            "    temporary = target.with_suffix('.tmp')\n"
            "    temporary.write_text(json.dumps([os.getpid(), int(fields[19])]))\n"
            "    temporary.replace(target)\n"
            "if mode == 'guardian':\n"
            "    child = subprocess.Popen([sys.executable, __file__, 'agent', str(root)],\n"
            "                             start_new_session=True, close_fds=True)\n"
            "    ready('guardian')\n"
            "    sys.exit(child.wait())\n"
            "elif mode == 'agent':\n"
            "    ready('agent')\n"
            "    children = [subprocess.Popen([sys.executable, __file__, 'rank' + str(i), str(root)],\n"
            "                                 start_new_session=True, close_fds=True) for i in range(2)]\n"
            "    sys.exit(max(child.wait() for child in children))\n"
            "else:\n"
            "    from posttrain import shared_pipeline_lock\n"
            "    with shared_pipeline_lock(root / 'pipeline.lock'):\n"
            "        ready(mode)\n"
            "        while True: time.sleep(0.05)\n"
        )
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                           CUDA_VISIBLE_DEVICES="", TMPDIR=str(temporary_root))
        identities = {}
        with (directory / "process.log").open("w+") as log:
            guardian = subprocess.Popen([sys.executable, str(helper), "guardian", str(directory)],
                                        env=environment, stdout=log, stderr=log)
            try:
                _wait(lambda: all((directory / f"{name}.json").exists()
                                  for name in ("guardian", "agent", "rank0", "rank1")))
                identities = {name: json.loads((directory / f"{name}.json").read_text())
                              for name in ("guardian", "agent", "rank0", "rank1")}
                assert len({os.getsid(value[0]) for value in identities.values()}) == 4
                with pytest.raises(BlockingIOError), process_lock(pipeline):
                    pass
                guardian.kill()
                guardian.wait(timeout=5)
                assert not _dead(identities["agent"])
                with pytest.raises(BlockingIOError), process_lock(pipeline):
                    pass
                _kill(identities["agent"])
                _wait(lambda: _dead(identities["agent"]))
                assert all(not _dead(identities[name]) for name in ("rank0", "rank1"))
                with pytest.raises(BlockingIOError), process_lock(pipeline):
                    pass
                _kill(identities["rank0"])
                _wait(lambda: _dead(identities["rank0"]))
                assert not _dead(identities["rank1"])
                with pytest.raises(BlockingIOError), process_lock(pipeline):
                    pass
                _kill(identities["rank1"])
                _wait(lambda: _dead(identities["rank1"]))
                # A killed process's main task may be a zombie while other
                # threads are still closing the shared descriptor table.
                _wait(lambda: _exclusive_available(pipeline), seconds=5)
                with process_lock(pipeline):
                    assert pipeline.stat().st_ino == initial_inode
                    assert pipeline.read_bytes() == initial_bytes
            except Exception as error:
                log.flush()
                log.seek(0)
                error.add_note(log.read())
                raise
            finally:
                # Recover ready identities even if setup failed midway. Creation
                # times ensure cleanup never signals an unrelated reused PID.
                for name in ("guardian", "agent", "rank0", "rank1"):
                    ready = directory / f"{name}.json"
                    if ready.exists():
                        identities[name] = json.loads(ready.read_text())
                for identity in identities.values():
                    _kill(identity)
                if guardian.poll() is None:
                    guardian.kill()
                guardian.wait(timeout=5)
                for identity in identities.values():
                    _wait(lambda identity=identity: _dead(identity), seconds=5)
