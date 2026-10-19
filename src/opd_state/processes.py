"""Linux foreground ownership, including detached job descendants.

The application owns one subprocess tree. Identity checks and a per-launch
environment marker prevent PID reuse or unrelated processes becoming targets.
No process is selected by its executable name or by a global Ray command.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def process_record(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "birth": fields[19],
            "parent": int(fields[1]),
            "session": int(fields[3]),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def enable_subreaper():
    """Called only by the dedicated CLI before starting its one job."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Could not enable child subreaping")


@contextlib.contextmanager
def finish_cleanup():
    previous, pending = {}, []
    already_raising = sys.exc_info()[0] is not None
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(
                sig, lambda number, _frame: pending.append(number)
            )
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if pending and not already_raising:
        raise InterruptedError(f"Cancellation received during cleanup: {pending[0]}")


class ProcessCleanupError(RuntimeError):
    pass


class OwnedProcess:
    def __init__(self, process, token, *, subreaper=False):
        self.process, self.token, self.subreaper = process, token, subreaper
        self.owner_pid = os.getpid()
        self.identities = {}
        self.pidfds = {}
        self._remember(process.pid, process_record(process.pid))

    @classmethod
    def launch(cls, command, *, cwd, environment, log, subreaper=False):
        token = uuid.uuid4().hex
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**environment, "OPD_PROCESS_OWNER": token},
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return cls(process, token, subreaper=subreaper)

    def _remember(self, pid, record):
        if record is None or pid in self.identities:
            return
        self.identities[pid] = record["birth"]
        if hasattr(os, "pidfd_open"):
            try:
                fd = os.pidfd_open(pid)
            except (ProcessLookupError, PermissionError):
                return
            if process_record(pid) == record:
                self.pidfds[pid] = fd
            else:
                os.close(fd)

    def _tagged(self, pid):
        try:
            return ("OPD_PROCESS_OWNER=" + self.token).encode() in Path(
                f"/proc/{pid}/environ"
            ).read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return False

    def observe(self):
        records = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                record = process_record(int(path.name))
                if record is not None:
                    records[int(path.name)] = record
        live = {
            pid
            for pid, birth in self.identities.items()
            if records.get(pid, {}).get("birth") == birth
        }
        # A detached child can outlive its parent between polls. A session or
        # adopted-child candidate must carry this launch's unique marker.
        for pid, record in records.items():
            candidate = record["session"] == self.process.pid or (
                self.subreaper and record["parent"] == self.owner_pid
            )
            if candidate and self._tagged(pid):
                self._remember(pid, record)
                live.add(pid)
        while True:
            children = {
                pid for pid, record in records.items() if record["parent"] in live
            } - live
            if not children:
                break
            for pid in children:
                self._remember(pid, records[pid])
            live.update(children)

    def live(self):
        return [
            pid
            for pid, birth in self.identities.items()
            if (process_record(pid) or {}).get("birth") == birth
        ]

    def _signal(self, pid, sig):
        if (process_record(pid) or {}).get("birth") != self.identities[pid]:
            return
        try:
            if pid in self.pidfds and hasattr(signal, "pidfd_send_signal"):
                signal.pidfd_send_signal(self.pidfds[pid], sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, timeout=20):
        """Return only after observed/tagged descendants stop; otherwise retain state."""
        try:
            for sig, seconds in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5)):
                deadline = time.monotonic() + seconds
                while True:
                    self.observe()
                    live = self.live()
                    if not live:
                        break
                    for pid in reversed(live):
                        self._signal(pid, sig)
                    self.process.poll()
                    if self.subreaper:
                        for pid in self.identities:
                            if pid != self.process.pid:
                                with contextlib.suppress(ChildProcessError):
                                    os.waitpid(pid, os.WNOHANG)
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                if not self.live():
                    break
            self.observe()
            if self.live():
                raise ProcessCleanupError(
                    f"Owned descendants remain alive: {self.live()}"
                )
            self.process.wait(timeout=5)
        finally:
            for fd in self.pidfds.values():
                os.close(fd)
            self.pidfds.clear()
