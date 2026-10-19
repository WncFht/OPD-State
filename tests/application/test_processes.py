"""Real subprocess cleanup must preserve unrelated concurrent processes."""

import os
import signal
import subprocess
import sys
import time

import pytest

from opd_state.processes import OwnedProcess, finish_cleanup, process_record


def test_owned_descendant_cleanup_preserves_peer(tmp_path):
    peer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    child_pid = tmp_path / "child.pid"
    program = "import subprocess,sys,time; from pathlib import Path; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(c.pid)); time.sleep(60)"
    try:
        with (tmp_path / "log").open("wb") as log:
            job = OwnedProcess.launch(
                [sys.executable, "-c", program, str(child_pid)],
                cwd=tmp_path,
                environment=dict(os.environ),
                log=log,
            )
            deadline = time.monotonic() + 5
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert child_pid.exists()
            job.observe()
            pid = int(child_pid.read_text())
            assert pid in job.identities
            job.stop(timeout=1)
            assert process_record(pid) is None
            assert peer.poll() is None
    finally:
        peer.terminate()
        peer.wait(timeout=5)


def test_cleanup_remembers_cancellation_until_after_cleanup():
    actions = []
    with pytest.raises(InterruptedError):
        with finish_cleanup():
            os.kill(os.getpid(), signal.SIGINT)
            actions.append("cleanup finished")
    assert actions == ["cleanup finished"]


def test_reused_identity_is_not_a_signal_target(tmp_path):
    with (tmp_path / "log").open("wb") as log:
        job = OwnedProcess.launch(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            environment=dict(os.environ),
            log=log,
        )
        real = job.identities[job.process.pid]
        try:
            job.identities[job.process.pid] = "different-birth"
            job._signal(job.process.pid, signal.SIGKILL)
            assert job.process.poll() is None
        finally:
            job.identities[job.process.pid] = real
            job.stop(timeout=1)
