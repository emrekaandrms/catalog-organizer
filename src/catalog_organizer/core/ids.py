from __future__ import annotations

import threading
from pathlib import Path

from catalog_organizer.core.paths import data_dir


class IdAllocator:
    def __init__(self, id_file: Path | None = None) -> None:
        self._path = id_file or (data_dir() / "next_id.txt")
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("1", encoding="utf-8")

    def next_id(self) -> str:
        with self._lock:
            n = int(self._path.read_text(encoding="utf-8").strip())
            self._path.write_text(str(n + 1), encoding="utf-8")
        return f"JCAD-{n:09d}"


_default_allocator: IdAllocator | None = None
_alloc_lock = threading.Lock()


def get_allocator() -> IdAllocator:
    global _default_allocator
    with _alloc_lock:
        if _default_allocator is None:
            _default_allocator = IdAllocator()
    return _default_allocator


def next_id() -> str:
    return get_allocator().next_id()
