"""Inspect an existing run hourly and archive complete validation prefixes.

The first inspection waits a full interval. This observer does not modify,
restart or stop training, and it sends no messages. GPU telemetry is separate.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("collector", Path(__file__).with_name("collect_math_experiment.py"))
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".json.new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def complete_steps(run, rows, interval):
    """A raw response file alone does not prove validation metrics were logged."""
    return [
        step
        for step, row in sorted(rows.items())
        if step % interval == 0 and "eval/macro/mean@16" in row and (run / f"validation/{step}.jsonl").exists()
    ]


def inspect(run, interval):
    logs = list((run / "logs").glob("*.log"))
    if len(logs) != 1:
        raise ValueError("Expected one training log")
    _, rows = COLLECTOR.parse_steps(logs[0].read_text(), through_step=10**9)
    return {
        "last_complete_training_log_step": max(rows, default=0),
        "complete_validation_steps": complete_steps(run, rows, interval),
        "plateau": json.loads((run / "plateau.json").read_text()) if (run / "plateau.json").exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--validation-interval", type=int, default=50)
    args = parser.parse_args()
    if args.interval_seconds < 3600 or args.validation_interval < 1:
        raise ValueError("Check at most hourly and use a positive validation interval")
    run = args.run.resolve()
    output = args.output or run / "hourly-watch"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "state.json").exists():
        raise FileExistsError("Use a fresh observer output directory")
    identity = process_start(args.pid)
    if identity is None:
        raise ValueError("The watched process must be live at attachment")
    collector_path = Path(COLLECTOR.__file__)
    collector_hash = hashlib.sha256(collector_path.read_bytes()).hexdigest()
    state = {
        "observer_pid": os.getpid(),
        "watched_pid": args.pid,
        "watched_start_id": identity,
        "run_directory": str(run),
        "interval_seconds": args.interval_seconds,
        "validation_interval": args.validation_interval,
        "created_at_utc": iso(time.time()),
        "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "collector_source_sha256": collector_hash,
        "last_checked_at_utc": None,
        "next_check_at_utc": iso(time.time() + args.interval_seconds),
        "status": "waiting_for_first_hourly_check",
        "archived_through_step": 0,
    }
    save(output / "state.json", state)
    while True:
        # This sleeps only in the detached observer, not in an agent tool call.
        time.sleep(args.interval_seconds)
        checked_at = time.time()
        current_identity = process_start(args.pid)
        same_process = current_identity == identity
        state.update(last_checked_at_utc=iso(checked_at), next_check_at_utc=iso(checked_at + args.interval_seconds))
        try:
            progress = inspect(run, args.validation_interval)
            state["progress"] = progress
            ready = progress["complete_validation_steps"]
            latest = max(ready, default=0)
            if latest > state["archived_through_step"]:
                if hashlib.sha256(collector_path.read_bytes()).hexdigest() != collector_hash:
                    raise ValueError("Collector changed; retain the prior audit and review before collecting")
                destination = output / f"step-{latest}-check-{int(checked_at)}"
                with (output / f"collect-step-{latest}.log").open("w") as stream:
                    subprocess.run(
                        [
                            sys.executable,
                            str(collector_path),
                            "--run",
                            str(run),
                            "--output",
                            str(destination),
                            "--steps",
                            str(latest),
                            "--interval",
                            str(args.validation_interval),
                            "--snapshot",
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                state["archived_through_step"] = latest
                state["latest_snapshot_directory"] = str(destination)
            state.pop("error", None)
            state["status"] = "waiting_for_next_hourly_check" if same_process else "watched_process_ended_or_replaced"
        except Exception as exc:
            state.update(status="inspection_error", error=f"{type(exc).__name__}: {exc}")
        if not same_process:
            state["next_check_at_utc"] = None
            state["watched_process_live"] = False
        else:
            state["watched_process_live"] = True
        save(output / "state.json", state)
        save(output / f"check-{int(checked_at)}.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if not same_process:
            break


if __name__ == "__main__":
    main()
