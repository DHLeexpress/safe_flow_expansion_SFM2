"""Atomic heartbeat status for long SFM2 expansion runs.

A supervisor polls a single ``STATUS.json`` written with the same atomic
temporary-file idiom as the diagnostic markers; it never tails logs or trace
tensors.  Every beat is also appended to ``progress.jsonl`` so a crashed run
retains its full timeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time


VERSION = "sfm_hp100_expansion_status_v1"


class Heartbeat:
    """Atomically publish supervisor-pollable progress for one process."""

    def __init__(self, path: str | Path | None, *, interval_seconds: float = 30.0):
        self.path = None if path is None else Path(path)
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0.0:
            raise ValueError("heartbeat interval must be positive")
        self.started = time.time()
        self._last_beat = 0.0
        self._static = {
            "version": VERSION,
            "pid": int(os.getpid()),
            "host": socket.gethostname(),
        }

    def beat(self, *, force: bool = True, **fields) -> None:
        if self.path is None:
            return
        now = time.time()
        if not force and now - self._last_beat < self.interval_seconds:
            return
        self._last_beat = now
        payload = {
            **self._static,
            **fields,
            "elapsed_s": float(now - self.started),
            "last_beat_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)
            ),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(self.path)
        with self.path.with_name("progress.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
