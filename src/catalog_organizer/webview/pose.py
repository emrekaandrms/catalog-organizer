"""Which way a product is photographed, and the rotation that bakes it into the file.

The camera is chosen by the SAME call the server render makes (`camera_axes`, driven by the
category), so a ring is framed on the web the way it is in the catalogue PDF. What changes is
where the choice is stored: instead of shipping a camera per product, the product itself is
turned into the camera's frame -- camera on +Z looking at the origin, up = +Y. Then

  * the environment is identical for every product (one file, not one per product),
  * the pose travels inside the GLB and cannot drift from the geometry,
  * orbit controls work on a +Y-up scene, the way they expect.

This is the usual way to ship a viewer model: normalised at authoring time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from catalog_organizer.render import gem_trace
from catalog_organizer.render import product_render as pr


@dataclass(frozen=True)
class ProductPose:
    direction: np.ndarray       # centre -> camera, in the file's own frame
    up: np.ndarray
    rotation: np.ndarray        # file frame -> canonical frame (rows: right, up, toward camera)
    centre: np.ndarray
    extent: float               # largest bounding-box dimension
    zoom: float                 # half-height of the view as a fraction of `extent`
    bore: int | None            # which axis looks through a ring's opening, if any
    views: dict | None = None   # {"front": (direction, up), "iso": (direction, up)} in the file's frame

    def canonical(self, vector) -> np.ndarray:
        """A direction in the file's frame, expressed in the canonical camera frame."""
        return self.rotation @ np.asarray(vector, dtype=float)


def _bounds(arrays) -> tuple:
    lo = np.min([a[0].min(axis=0) for a in arrays], axis=0)
    hi = np.max([a[0].max(axis=0) for a in arrays], axis=0)
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def product_pose(metal, gem, *, category: str | None, source: Path,
                 view: str = "iso") -> ProductPose:
    parts = [a for a in (metal, gem) if a is not None]
    bounds = _bounds(parts)
    # An STL's "stones" are inferred and must not aim the camera (see render_product).
    guides = Path(source).suffix.lower() != ".stl"
    facing = (gem if (gem is not None and guides) else metal) or gem
    if metal is not None and gem is not None:
        silhouette = (np.vstack([metal[0], gem[0]]),
                      np.vstack([metal[1], gem[1] + metal[0].shape[0]]))
    else:
        silhouette = metal or gem
    bore = pr.open_bore_axis(silhouette)
    front, iso, up = pr.camera_axes(
        bounds, facing, avoid_bore=True, silhouette_arrays=silhouette,
        known_bore=bore, category=category)
    direction = np.asarray(iso if view == "iso" else front, dtype=float)
    direction = direction / np.linalg.norm(direction)
    up = np.asarray(up, dtype=float)
    centre = np.array([0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]),
                       0.5 * (bounds[4] + bounds[5])])
    extent = float(max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                       bounds[5] - bounds[4], 1e-6))
    zoom = 0.62 * (pr._RING_ZOOM_FACTOR if bore is not None else 1.0)
    unit = lambda v: np.asarray(v, dtype=float) / np.linalg.norm(v)          # noqa: E731
    return ProductPose(direction=direction, up=up,
                       rotation=gem_trace.view_basis(direction, up), centre=centre,
                       extent=extent, zoom=zoom, bore=bore,
                       views={"front": (unit(front), up), "iso": (unit(iso), up)})


def camera_distance(pose: ProductPose, fov_degrees: float) -> float:
    """How far back the canonical camera sits so the piece fills the same fraction of the
    frame as it does in the server render, whatever the lens."""
    return float(pose.extent * pose.zoom / np.tan(np.radians(fov_degrees) / 2.0))
