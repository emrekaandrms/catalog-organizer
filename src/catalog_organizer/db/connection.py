"""SQLite bağlantı fabrikası.

WAL zorunlu: analiz hattı JSONL'e yazarken kullanıcı Ara ekranında sorgu
çalıştırabilmeli. `foreign_keys` zorunlu: SQLite'ta varsayılan olarak KAPALI
ve o haliyle selection_items'ın ON DELETE CASCADE'i sessizce çalışmaz.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

from catalog_organizer.core.paths import data_dir
from catalog_organizer.db.schema import apply_schema


class Database:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = Path(db_path) if db_path else (data_dir() / "catalog.db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        apply_schema(self._conn)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
