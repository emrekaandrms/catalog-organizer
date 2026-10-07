"""Which engine draws the pictures the app shows and prints.

The app's engine is the web viewer (three.js in the app's own Chromium: `webview.service`),
because that is the look the catalogue is judged against. The VTK renderer (`product_render`)
stays as the fallback for where no Chromium is running -- scripts, tests, a machine without
QtWebEngine -- so those still produce pictures.

    CATALOG_RENDER_ENGINE=vtk   forces the old engine (comparisons, bisecting a regression)
"""
from __future__ import annotations

import os
from pathlib import Path

from catalog_organizer.render import product_render as pr


def web_engine_available() -> bool:
    if os.environ.get("CATALOG_RENDER_ENGINE", "").lower() == "vtk":
        return False
    try:
        from catalog_organizer.webview import service
    except ImportError:                 # no PyQt6-WebEngine
        return False
    return service.current() is not None


def load_and_render(source_path: Path, *, file_id: str, metal_key: str, stone_key: str,
                    **kwargs) -> pr.RenderResult:
    """Same signature and result as `product_render.load_and_render`; the engine is chosen here."""
    if web_engine_available():
        from catalog_organizer.webview import engine
        web_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("resolution", "camera", "category", "out_dir", "overwrite",
                               "place_stones")}
        return engine.render_views(Path(source_path), file_id=file_id, metal_key=metal_key,
                                   stone_key=stone_key, **web_kwargs)
    kwargs.pop("place_stones", None)            # the VTK engine has no seat placement
    return pr.load_and_render(source_path, file_id=file_id, metal_key=metal_key,
                              stone_key=stone_key, **kwargs)
