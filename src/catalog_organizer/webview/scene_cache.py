"""The scenes the in-app viewer shows, kept in a cache the user never sees.

The viewer (three.js inside the app's own Chromium) needs a product as a glTF scene plus the
shared environment and materials. Nothing here is an "export": the pieces are made on first
use, stored under `cache/web/site/`, and reused until the source file, the category, the look
or the exporter changes -- the same contract the PNG render cache has.

    ensure_site()                      viewer pages + three.js + shared environment (once)
    ensure_scene(source, file_id, ..)  data/<file_id>/piece.glb, current or rebuilt
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from catalog_organizer.core.paths import cache_dir
from catalog_organizer.webview import export
from catalog_organizer.webview.glb import read_json_chunk
from catalog_organizer.webview.look import STUDIO_LOOK, Look, shared_meta

# Bump when anything the scene file contains changes (exporter, AO, pose, stone frames).
SCENE_VERSION = 7          # 6: seats found along all six axes, facing outward; 5: seat centres in the file frame; 4: stones placed in detected seats

_site_lock = threading.Lock()
_scene_locks: dict[str, threading.Lock] = {}
_scene_locks_guard = threading.Lock()


@dataclass(frozen=True)
class Scene:
    file_id: str
    glb: Path
    key: str
    rotation: np.ndarray        # file frame -> canonical camera frame
    stones: int
    from_cache: bool
    views: dict                 # {"front": {"dir", "up"}, "iso": {...}} in the canonical frame
    placed: bool = False        # the stones were placed in detected seats, not read from the file

    def canonical(self, vector) -> list[float]:
        """A direction in the file's frame as the viewer's camera frame sees it."""
        return [float(x) for x in self.rotation @ np.asarray(vector, dtype=float)]


def site_dir() -> Path:
    d = cache_dir() / "web" / "site"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _look_key(look: Look) -> str:
    return hashlib.sha1(json.dumps(asdict(look), sort_keys=True).encode()).hexdigest()[:10]


def _assets_changed(src: Path, dst: Path) -> bool:
    return (not dst.exists()) or dst.stat().st_size != src.stat().st_size \
        or dst.read_bytes() != src.read_bytes()


def ensure_site(look: Look = STUDIO_LOOK) -> Path:
    """Viewer files and shared textures, written when missing or out of date. Returns the root."""
    root = site_dir()
    with _site_lock:
        for name in ("viewer.html", "viewer.js", "index.html"):
            src = export.ASSETS / name
            if _assets_changed(src, root / name):
                shutil.copyfile(src, root / name)
        vendor_src = export.ASSETS / "vendor"
        marker = root / "vendor" / "three" / "three.module.js"
        if _assets_changed(vendor_src / "three" / "three.module.js", marker):
            shutil.copytree(vendor_src, root / "vendor", dirs_exist_ok=True)

        shared = root / "data" / "_shared"
        state_file = root / "site.json"
        want = {"version": SCENE_VERSION, "look": _look_key(look)}
        have = {}
        if state_file.exists():
            try:
                have = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                have = {}
        if not ((shared / "meta.json").exists() and (shared / "env_metal.f16").exists()
                and (shared / "env_gem.f16").exists()
                and {k: have.get(k) for k in want} == want):
            export.write_shared(root, look)
            state_file.write_text(json.dumps(want), encoding="utf-8")
        else:
            # Materials can change without a version bump; meta.json is cheap to refresh.
            saved_meta = json.loads((shared / "meta.json").read_text(encoding="utf-8"))
            sizes = {k: saved_meta[k] for k in ("envMetal", "envGem")}
            fresh = json.dumps(shared_meta(look, sizes), ensure_ascii=False, indent=1)
            if (shared / "meta.json").read_text(encoding="utf-8") != fresh:
                (shared / "meta.json").write_text(fresh, encoding="utf-8")
    return root


def scene_key(source: Path | None, category: str | None, look: Look,
              max_triangles: int, place_stones: bool = False) -> str:
    h = hashlib.sha1()
    if source is not None:
        st = Path(source).stat()
        h.update(f"{Path(source).resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(f"|{category}|{SCENE_VERSION}|{_look_key(look)}|{max_triangles}|{int(place_stones)}".encode())
    return h.hexdigest()[:16]


def _lock_for(file_id: str) -> threading.Lock:
    with _scene_locks_guard:
        return _scene_locks.setdefault(file_id, threading.Lock())


def ensure_scene(source: Path | None, file_id: str, *, category: str | None = None,
                 look: Look = STUDIO_LOOK, max_triangles: int = export.MAX_TRIANGLES,
                 arrays=None, force: bool = False, place_stones: bool = False) -> Scene:
    """The product's scene, built if the cached one is stale. Safe to call from any thread."""
    root = ensure_site(look)
    key = scene_key(source if arrays is None else None, category, look, max_triangles, place_stones)
    if arrays is not None:
        key += hashlib.sha1(np.ascontiguousarray(arrays[0][0]).tobytes()).hexdigest()[:8]
    folder = root / "data" / file_id
    marker = folder / "scene.json"
    with _lock_for(file_id):
        glb = folder / "piece.glb"
        if marker.exists() and glb.exists() and not force:
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = {}
            if saved.get("key") == key:
                return _read(file_id, glb, key, from_cache=True)
        marker.unlink(missing_ok=True)
        result = export.export_product(source, file_id, root, category=category, look=look,
                                       max_triangles=max_triangles, arrays=arrays,
                                       place_stones=place_stones)
        marker.write_text(json.dumps({"key": key, "stones": result.stones}), encoding="utf-8")
        return _read(file_id, glb, key, from_cache=False)


def _read(file_id: str, glb: Path, key: str, *, from_cache: bool) -> Scene:
    doc = read_json_chunk(glb)
    stones = sum(1 for n in doc["nodes"] if n.get("extras", {}).get("role") == "gem")
    pose = doc["scenes"][0]["extras"]["pose"]
    return Scene(file_id=file_id, glb=glb, key=key,
                 rotation=np.asarray(pose["rotation"], dtype=float),
                 stones=stones, from_cache=from_cache, views=pose.get("views", {}),
                 placed=bool(doc["scenes"][0]["extras"].get("placedStones", False)))
