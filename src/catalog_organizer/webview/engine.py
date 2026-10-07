"""Draw a product's two catalogue views with the web engine and return files, like the VTK one did.

    scene  = scene_cache.ensure_scene(...)        heavy, any thread, cached
    pngs   = service.capture(scene, ...)          GUI thread's hidden Chromium, ~0.05 s a view
    files  = cache/renders/<id>/w<version>_...png the same RenderResult the PDF and the Render tab use

Same pose choice as the VTK render (`pose.product_pose` calls the same `camera_axes` with the
category), the same hand-set camera overrides, the same cache-by-file contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from catalog_organizer.render import materials as mat
from catalog_organizer.render import product_render as pr
from catalog_organizer.webview import scene_cache, service as service_module
from catalog_organizer.webview.look import STUDIO_LOOK, Look

WEB_RENDER_VERSION = 1
FRAMES = 16


def _tag(parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(np.round(np.asarray(p, dtype=float), 4).tobytes())
    return h.hexdigest()[:8]


def render_views(source_path: Path, *, file_id: str, metal_key: str, stone_key: str,
                 resolution: int = 1400, camera: dict | None = None,
                 category: str | None = None, out_dir: Path | None = None,
                 overwrite: bool = False, look: Look = STUDIO_LOOK,
                 service=None, place_stones: bool = False) -> pr.RenderResult:
    mat.metal(metal_key), mat.stone(stone_key)               # unknown keys fail early and by name
    scene = scene_cache.ensure_scene(Path(source_path), file_id, category=category, look=look,
                                     place_stones=place_stones)
    out_dir = out_dir or pr.renders_dir(file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    views: dict[str, tuple] = {}
    targets: dict[str, Path] = {}
    for name in pr.VIEW_NAMES:
        base = scene.views[name]
        direction, up, tag = base["dir"], base["up"], ""
        if camera and name in camera:
            d, u = camera[name]
            direction, up = scene.canonical(d), scene.canonical(u)
            tag = "_c" + _tag([d, u])
        views[name] = (direction, up)
        targets[name] = out_dir / (f"w{WEB_RENDER_VERSION}_{scene.key}_{metal_key}_{stone_key}"
                                   f"_{resolution}_{name}{tag}.png")
    result = pr.RenderResult(file_id, metal_key, stone_key, targets, scene.stones > 0, "web")
    if not overwrite and all(p.exists() for p in targets.values()):
        return result

    svc = service or service_module.current()
    if svc is None:
        raise service_module.WebRenderError("web render servisi çalışmıyor")
    pngs = svc.capture(scene, metal_key=metal_key, stone_key=stone_key, size=resolution,
                       views=views, frames=FRAMES)
    for name, data in pngs.items():
        targets[name].write_bytes(data)
    return result
