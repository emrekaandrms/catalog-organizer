from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_EXTENSIONS = frozenset([".3dm", ".stl"])


def walk_for_cad_files(roots: list[Path]) -> list[Path]:
    """
    Recursively yield all .3dm and .stl files under each root.
    Skips unreadable directories and files silently.
    Returns a deduplicated list sorted by absolute path.
    """
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            _walk_dir(root, found)
        except PermissionError:
            pass
    return sorted(found)


def _walk_dir(directory: Path, found: set[Path]) -> None:
    try:
        entries = list(os.scandir(directory))
    except PermissionError:
        return
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                _walk_dir(Path(entry.path), found)
            elif entry.is_file(follow_symlinks=False):
                p = Path(entry.path)
                if p.suffix.lower() in _ALLOWED_EXTENSIONS:
                    found.add(p.resolve())
        except (PermissionError, OSError):
            continue
