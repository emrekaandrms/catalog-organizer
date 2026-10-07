from __future__ import annotations

import threading
from pathlib import Path

from catalog_organizer.catalog.writer import read_all_records
from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import CatalogRecord


class CatalogIndex:
    """In-RAM map file_id → CatalogRecord. Hot-reloaded from JSONL on startup."""

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._jsonl_path = jsonl_path or (data_dir() / "catalog_master.jsonl")
        self._records: dict[str, CatalogRecord] = {}
        self._lock = threading.RLock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._records.clear()
            for rec in read_all_records(self._jsonl_path):
                # Later occurrences supersede earlier ones with the same file_id.
                self._records[rec.file_id] = rec

    def get(self, file_id: str) -> CatalogRecord | None:
        with self._lock:
            return self._records.get(file_id)

    def upsert(self, record: CatalogRecord) -> None:
        with self._lock:
            self._records[record.file_id] = record

    def all(self) -> list[CatalogRecord]:
        with self._lock:
            return list(self._records.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __contains__(self, file_id: str) -> bool:
        with self._lock:
            return file_id in self._records
