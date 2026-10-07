from __future__ import annotations

import hashlib
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_MMAP_THRESHOLD = 50 * 1024 * 1024   # 50 MB
_CHUNK = 1 * 1024 * 1024              # 1 MB streaming chunk


def sha256_file(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    if size == 0:
        return h.hexdigest()
    if size >= _MMAP_THRESHOLD:
        with path.open("rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                h.update(mm)
    else:
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
    return h.hexdigest()


def hash_files(paths: list[Path], max_workers: int = 4) -> dict[Path, str]:
    """Hash multiple files in parallel. Returns {path: hex_digest}."""
    results: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(sha256_file, p): p for p in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception:
                results[path] = ""
    return results
