from __future__ import annotations

import threading
from pathlib import Path

import orjson

from catalog_organizer.core.schemas import AuditEvent


class AuditWriter:
    _FLUSH_EVERY = 50
    _FLUSH_INTERVAL_S = 30.0

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._fh = log_path.open("ab")
        self._lock = threading.Lock()
        self._buf: list[bytes] = []
        self._timer: threading.Timer | None = None
        self._schedule_flush()

    def write(self, event: AuditEvent) -> None:
        line = orjson.dumps(event.model_dump(mode="json")) + b"\n"
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self._FLUSH_EVERY:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        with self._lock:
            self._flush_locked()
            self._fh.close()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        for line in self._buf:
            self._fh.write(line)
        self._fh.flush()
        import os
        os.fsync(self._fh.fileno())
        self._buf.clear()

    def _schedule_flush(self) -> None:
        self._timer = threading.Timer(self._FLUSH_INTERVAL_S, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        with self._lock:
            self._flush_locked()
        self._schedule_flush()
