"""The two environment textures the web viewer shares across every product.

Every product is exported already turned into its own camera frame (camera on +Z looking at
the origin, up = +Y; see `pose.py`), so the environment is the SAME for all of them and is
written once instead of once per product.

Two textures, two jobs, exactly as in the server render:
  metal   the studio rig, image-based light for the metal. Baked into a three.js equirect,
          softened with a Gaussian: luxury-retail reflections are soft and bright, and the
          unblurred rig measurably made the gold's brightness distribution worse.
  gem     the supplied showroom HDR (desaturated, scaled) that a stone's refracted rays look
          up. It needs real structure -- a stone is a window onto the room.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Gaussian sigma in pixels of the 512-wide map; swept 0-11 against the target gold.
METAL_BLUR = 6.0
METAL_SIZE = (256, 512)           # rows, columns


def texture_array(tex) -> np.ndarray:
    """(H, W, 3) float32 rows in the order the VTK texture stores them (bottom -> top)."""
    from vtkmodules.util import numpy_support as ns
    img = tex.GetInput()
    w, h, _ = img.GetDimensions()
    arr = ns.vtk_to_numpy(img.GetPointData().GetScalars()).astype(np.float32)
    return arr.reshape(h, w, -1)[:, :, :3]


def sample_equirect(data: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Bilinear lookup in a VTK-ordered equirect with the stone shader's own mapping:
    u = atan2(z, x) / 2pi + 0.5, v = 1 - acos(y) / pi, rows running bottom -> top."""
    h, w, _ = data.shape
    u = np.arctan2(s[..., 2], s[..., 0]) / (2 * np.pi) + 0.5
    v = 1.0 - np.arccos(np.clip(s[..., 1], -1, 1)) / np.pi
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x1 = (x0 + 1) % w
    x0 = x0 % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    y0 = np.clip(y0, 0, h - 1)
    top = data[y0, x0] * (1 - fx) + data[y0, x1] * fx
    bot = data[y1, x0] * (1 - fx) + data[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def bake_metal(src: np.ndarray, *, blur: float = METAL_BLUR,
               size: tuple[int, int] = METAL_SIZE) -> np.ndarray:
    """A three.js equirect (rows bottom -> top, v = asin(y)/pi + .5) of the metal's studio rig.

    In the canonical camera frame a view-space direction IS a world direction, so the lookup
    needs no rotation. The (x, +z) orientation is the one measured to reproduce the server's
    gold (luminance correlation 0.962; the other three signs: -0.07, +0.55, -0.09).

    Softening is a blur, NOT contrast reduction about the mean: the rig is heavy-tailed HDR,
    so mean + k*(x - mean) adds a huge floor everywhere and washes the frame white.
    """
    height, width = size
    j = (np.arange(width) + 0.5) / width
    i = (np.arange(height) + 0.5) / height
    u, v = np.meshgrid(j, i)
    lon = (u - 0.5) * 2 * np.pi
    lat = (v - 0.5) * np.pi
    d = np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], axis=-1)
    env = sample_equirect(src, d).astype(np.float32)
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        env = gaussian_filter(env, sigma=(blur, blur, 0), mode=("nearest", "wrap", "nearest"))
    return env


def write_f16(path: Path, rgb: np.ndarray) -> None:
    """RGBA half-float, row-major: what the viewer uploads as a DataTexture unchanged."""
    h, w, _ = rgb.shape
    rgba = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], axis=-1)
    Path(path).write_bytes(np.clip(rgba, 0, 65000).astype("<f2").tobytes())


def write_shared(directory: Path) -> dict:
    """Write env_metal.f16 and env_gem.f16; returns their sizes for meta.json."""
    from catalog_organizer.render import studio
    from catalog_organizer.render.product_render import environment_for

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metal_tex, _from_file = environment_for("metal")
    metal = bake_metal(texture_array(metal_tex))
    gem = texture_array(studio.gem_lookup())
    write_f16(directory / "env_metal.f16", metal)
    write_f16(directory / "env_gem.f16", gem)
    return {"envMetal": {"w": metal.shape[1], "h": metal.shape[0]},
            "envGem": {"w": gem.shape[1], "h": gem.shape[0]}}
