"""CPU-only regression tests for graceful and forced supervised shutdown."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
LAUNCHER = ROOT / "ops/launch_training.py"


@pytest.fixture
def workspace_tmp():
    directory = WORKSPACE / "tmp"
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="launcher-test-", dir=directory) as temporary:
        yield Path(temporary)


@pytest.mark.parametrize("cooperative", [True, False])
def test_stop_reaches_or_cleans_up_worker_in_another_session(workspace_tmp, cooperative):
    stop_file = workspace_tmp / "stop-request"
    ready_file = workspace_tmp / "ready"
    observed_file = workspace_tmp / "observed"
    worker = workspace_tmp / "worker.py"
    worker.write_text(
        "import os, pathlib, signal, sys, time\n"
        f"cooperative = {cooperative!r}\n"
        "if not cooperative: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "stop, ready, observed = map(pathlib.Path, sys.argv[1:])\n"
        "ready.write_text(str(os.getpid()))\n"
        "while not cooperative or not stop.exists(): time.sleep(0.02)\n"
        "observed.write_text('graceful stop')\n"
    )
    agent = workspace_tmp / "agent.py"
    agent.write_text(
        "import signal, subprocess, sys\n"
        f"if not {cooperative!r}: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, *sys.argv[1:]], start_new_session=True)\n"
        "sys.exit(child.wait())\n"
    )
    harness = (
        "import importlib.util, os, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('launcher', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "sys.exit(module.supervise([sys.executable, *sys.argv[3:]], pathlib.Path(sys.argv[2]), "
        "os.environ.copy(), grace_seconds=0.3, terminate_seconds=0.3))"
    )
    environment = os.environ.copy()
    environment.update(TMPDIR=str(WORKSPACE / "tmp"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-c", harness, str(LAUNCHER), str(stop_file),
               str(agent), str(worker), str(stop_file), str(ready_file), str(observed_file)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=environment)
    try:
        deadline = time.monotonic() + 10
        while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready_file.exists(), "Fake distributed worker did not start"
        worker_pid = int(ready_file.read_text())
        assert os.getpgid(worker_pid) != os.getpgid(process.pid)
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert stop_file.exists()
        if cooperative:
            assert process.returncode == 0, stdout + "\n" + stderr
            assert observed_file.read_text() == "graceful stop"
        else:
            assert process.returncode != 0, stdout + "\n" + stderr
            assert "Termination deadline exceeded" in stdout
            stat = Path(f"/proc/{worker_pid}/stat")
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_launcher_refuses_work_outside_workspace():
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--work-dir", "/tmp"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "Created paths must stay inside" in result.stderr
