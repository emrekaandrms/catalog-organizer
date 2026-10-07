from __future__ import annotations

from pathlib import Path

# Two parents up: core/ -> catalog_organizer/ -> src/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return PROJECT_ROOT / "config"


def data_dir() -> Path:
    d = PROJECT_ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = PROJECT_ROOT / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshots_dir(file_id: str) -> Path:
    d = cache_dir() / "snapshots" / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumbnails_dir() -> Path:
    d = cache_dir() / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d


def geometry_cache_dir() -> Path:
    d = cache_dir() / "geometry"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vlm_raw_dir() -> Path:
    d = cache_dir() / "vlm_raw"
    d.mkdir(parents=True, exist_ok=True)
    return d
