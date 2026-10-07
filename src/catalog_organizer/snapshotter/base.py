from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

# View names → camera direction (looking *toward* origin from this offset).
_VIEWS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "side":  (1.0, 0.0, 0.0),
    "top":   (0.0, 0.0, 1.0),
    "iso":   (1.0, -1.0, 1.0),
}

_CLAY_COLOR = "#cfcfcf"
_BG_COLOR = "white"


def render_views(
    mesh: pv.PolyData,
    out_dir: Path,
    resolution: int = 1024,
    thumbnail_path: Path | None = None,
    thumbnail_size: int = 256,
) -> dict[str, Path]:
    """
    Render 4 orthographic snapshots (front/side/top/iso) of the mesh
    into out_dir/{view}.png. Optionally writes a webp thumbnail of the
    isometric view at thumbnail_path. Returns {view: png_path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    center = np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ])
    extent = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1e-6,
    )

    for view_name, direction in _VIEWS.items():
        png_path = out_dir / f"{view_name}.png"
        plotter = pv.Plotter(off_screen=True, window_size=(resolution, resolution))
        plotter.background_color = _BG_COLOR
        plotter.add_mesh(
            mesh,
            color=_CLAY_COLOR,
            smooth_shading=True,
            specular=0.3,
            specular_power=15,
        )
        plotter.enable_parallel_projection()
        cam_offset = np.array(direction) * extent * 2.5
        plotter.camera.position = tuple(center + cam_offset)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.up = (0.0, 0.0, 1.0) if view_name != "top" else (0.0, 1.0, 0.0)
        plotter.camera.parallel_scale = extent * 0.65
        plotter.show(screenshot=str(png_path), auto_close=True)
        plotter.close()
        results[view_name] = png_path

    if thumbnail_path is not None and "iso" in results:
        _make_thumbnail(results["iso"], thumbnail_path, thumbnail_size)

    return results


def _make_thumbnail(src_png: Path, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_png).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    img.save(dest, format="WEBP", quality=85)
