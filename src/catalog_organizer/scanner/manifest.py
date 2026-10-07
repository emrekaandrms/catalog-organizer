from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import orjson

from catalog_organizer.core.ids import IdAllocator
from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import ManifestEntry
from catalog_organizer.scanner.hasher import hash_files
from catalog_organizer.scanner.walker import walk_for_cad_files


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _read_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Load existing manifest into {source_path: ManifestEntry}."""
    entries: dict[str, ManifestEntry] = {}
    if not path.exists():
        return entries
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ManifestEntry.model_validate(orjson.loads(line))
                entries[entry.source_path] = entry
            except Exception:
                continue
    return entries


def _write_manifest(path: Path, entries: dict[str, ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for entry in entries.values():
            fh.write(orjson.dumps(entry.model_dump(mode="json")) + b"\n")


def scan(
    roots: list[Path],
    batch_name: str = "default",
    manifest_path: Path | None = None,
    allocator: IdAllocator | None = None,
) -> dict[str, ManifestEntry]:
    """
    Walk roots, hash files, update manifest.jsonl idempotently.

    - New files are added with state='new'.
    - Files already in the manifest are left unchanged (existing state preserved).
    - Files previously in the manifest but no longer on disk are marked state='missing'.
    - Duplicate files (same SHA256) have duplicate_of set to the first file_id seen.

    Returns the full updated manifest as {source_path: ManifestEntry}.
    """
    if manifest_path is None:
        manifest_path = data_dir() / "manifest.jsonl"
    if allocator is None:
        from catalog_organizer.core.ids import get_allocator
        allocator = get_allocator()

    existing = _read_manifest(manifest_path)
    found_paths = walk_for_cad_files(roots)
    found_strs = {str(p) for p in found_paths}

    # Mark previously known files that are now missing
    for src, entry in existing.items():
        if entry.state != "missing" and src not in found_strs:
            existing[src] = entry.model_copy(update={"state": "missing"})

    # Hash only new files (not already in manifest)
    new_paths = [p for p in found_paths if str(p) not in existing]
    new_hashes = hash_files(new_paths) if new_paths else {}

    # Build sha256 → first file_id map for duplicate detection (existing + new)
    sha_to_id: dict[str, str] = {}
    for entry in existing.values():
        if entry.sha256 and entry.duplicate_of is None:
            sha_to_id.setdefault(entry.sha256, entry.file_id)

    scan_ts = _now()

    for path in found_paths:
        src = str(path)
        if src in existing:
            continue  # already registered; don't touch

        sha = new_hashes.get(path, "")
        stat = path.stat()

        duplicate_of: str | None = None
        if sha and sha in sha_to_id:
            duplicate_of = sha_to_id[sha]

        file_id = allocator.next_id()

        if sha and duplicate_of is None:
            sha_to_id[sha] = file_id

        entry = ManifestEntry(
            file_id=file_id,
            source_path=src,
            file_extension=path.suffix.lower(),  # type: ignore[arg-type]
            file_size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            sha256=sha,
            duplicate_of=duplicate_of,
            state="new",
            processing_batch=batch_name,
            scan_timestamp=scan_ts,
        )
        existing[src] = entry

    _write_manifest(manifest_path, existing)
    return existing


def iter_manifest(manifest_path: Path | None = None) -> Iterator[ManifestEntry]:
    if manifest_path is None:
        manifest_path = data_dir() / "manifest.jsonl"
    if not manifest_path.exists():
        return
    with manifest_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield ManifestEntry.model_validate(orjson.loads(line))
                except Exception:
                    continue
