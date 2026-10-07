from __future__ import annotations

from pathlib import Path

import pytest

from catalog_organizer.core.ids import IdAllocator
from catalog_organizer.scanner.hasher import sha256_file
from catalog_organizer.scanner.manifest import iter_manifest, scan
from catalog_organizer.scanner.walker import walk_for_cad_files

FIXTURES = Path(__file__).parent / "fixtures"


def test_walker_finds_all_fixtures():
    found = walk_for_cad_files([FIXTURES])
    exts = {p.suffix.lower() for p in found}
    assert ".stl" in exts
    assert ".3dm" in exts
    assert len(found) == 5


def test_walker_skips_nonexistent_root(tmp_path: Path):
    found = walk_for_cad_files([tmp_path / "does_not_exist"])
    assert found == []


def test_hasher_produces_stable_digest():
    stl = FIXTURES / "sample_ring.stl"
    d1 = sha256_file(stl)
    d2 = sha256_file(stl)
    assert d1 == d2
    assert len(d1) == 64


def test_scan_produces_5_entries(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    allocator = IdAllocator(tmp_path / "next_id.txt")
    result = scan([FIXTURES], batch_name="test", manifest_path=manifest_path, allocator=allocator)
    assert len(result) == 5
    for entry in result.values():
        assert entry.state == "new"
        assert entry.file_id.startswith("JCAD-")
        assert entry.sha256 != ""


def test_scan_is_idempotent(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    allocator = IdAllocator(tmp_path / "next_id.txt")
    first = scan([FIXTURES], batch_name="test", manifest_path=manifest_path, allocator=allocator)
    second = scan([FIXTURES], batch_name="test", manifest_path=manifest_path, allocator=allocator)
    assert len(first) == len(second)
    for src, entry in first.items():
        assert second[src].file_id == entry.file_id
        assert second[src].sha256 == entry.sha256
