"""One append-only event stream per explicit producer."""

import json
import os
import re
import time
from pathlib import Path


class EventWriter:
    def __init__(self, directory, run_id, producer, *, source_id=None, config_id=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", producer):
            raise ValueError("Event producer must be a path-safe identity")
        self.path = Path(directory) / (producer + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x"):
            pass
        self.context = {
            "schema_version": 1,
            "run_id": run_id,
            "source_id": source_id,
            "config_id": config_id,
            "producer": producer,
        }
        self.sequence = 0

    def emit(self, event, **details):
        row = {
            **self.context,
            "sequence": self.sequence,
            "event": event,
            "epoch_s": time.time(),
            "details": details,
        }
        line = (
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        with self.path.open("a") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
