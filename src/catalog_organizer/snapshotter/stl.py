from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from catalog_organizer.core.paths import snapshots_dir, thumbnails_dir
from catalog_organizer.snapshotter.base import render_views


def _trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # PyVista face format: [N, v0, v1, ..., vN-1] flat array per face
    n = faces.shape[0]
    pv_faces = np.empty((n, 4), dtype=np.int64)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces
    return pv.PolyData(verts, pv_faces.flatten())


def snapshot_stl(
    stl_path: Path,
    file_id: str,
    resolution: int = 1024,
    thumbnail_size: int = 256,
) -> dict[str, Path]:
    """
    Load STL via trimesh, convert to PolyData, render 4 views + thumbnail.
    Returns {view_name: png_path, "thumbnail": webp_path}.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"STL produced no mesh: {stl_path}")

    polydata = _trimesh_to_polydata(mesh)
    out_dir = snapshots_dir(file_id)
    thumb_path = thumbnails_dir() / f"{file_id}.webp"

    pngs = render_views(
        polydata,
        out_dir=out_dir,
        resolution=resolution,
        thumbnail_path=thumb_path,
        thumbnail_size=thumbnail_size,
    )
    pngs["thumbnail"] = thumb_path
    return pngs
