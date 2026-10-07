"""Resume protocol per §C.8.

State transitions per file:
    new → snapshot_done → vlm_done → measured → weighted → final
                                            └→ needs_review
                                            └→ failed

For v1 the pipeline runs a whole file end-to-end in one function call
(`pipeline.process_one`), so resume eligibility is binary: a file is
"done" if its record already exists in the catalog index with a terminal
state, otherwise it's eligible to (re)run.
"""
from __future__ import annotations

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.core.schemas import ManifestEntry

_TERMINAL_STATES: frozenset[str] = frozenset({"final", "needs_review", "failed"})


def filter_eligible(
    entries: list[ManifestEntry],
    index: CatalogIndex,
    reprocess_failed: bool = False,
) -> list[ManifestEntry]:
    """
    Return the subset of `entries` that should be processed.

    Skips:
      - duplicates (`duplicate_of` set)
      - missing files
      - files whose catalog record is already in a terminal state
        (`final` or `needs_review`). Failed records are re-enqueued
        when `reprocess_failed=True`.
    """
    out: list[ManifestEntry] = []
    for e in entries:
        if e.duplicate_of:
            continue
        if e.state == "missing":
            continue

        existing = index.get(e.file_id)
        if existing is None:
            out.append(e)
            continue
        if existing.state == "failed" and reprocess_failed:
            out.append(e)
            continue
        if existing.state not in _TERMINAL_STATES:
            out.append(e)
    return out
