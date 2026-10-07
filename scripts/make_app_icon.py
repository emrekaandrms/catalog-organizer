"""Generate assets/app_icon.ico — the window / taskbar / shortcut icon.

Kept as a script rather than a checked-in binary blob nobody can edit: the
mark is derived from the app's own accent colour, so re-running this after
a theme change keeps the icon in step.

Run:  py -3.12 scripts/make_app_icon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catalog_organizer.gui import theme  # noqa: E402

# Drawn large and downsampled by Pillow when writing the .ico. Supersampling
# is what keeps the ring's stroke smooth at 32 px and below.
CANVAS = 1024
SS = 4  # supersample factor for the shapes we draw by hand
ICO_SIZES = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def build() -> Image.Image:
    size = CANVAS * SS
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bg = _rgb(theme.BG_SURFACE)
    accent = _rgb(theme.ACCENT_DEFAULT)
    border = _rgb(theme.BORDER_STRONG)

    # Rounded-square plate. Windows renders shortcut icons on wildly
    # different backgrounds, so the mark carries its own.
    pad = size // 16
    d.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=size // 6,
        fill=bg + (255,),
        outline=border + (255,),
        width=size // 128,
    )

    # The mark: a band (ring) seen face-on with a gem seated at the top.
    # A plain circle read as a loading spinner at small sizes; the gem is
    # what makes it legible as jewellery at 32 px.
    cx, cy = size // 2, int(size * 0.56)
    r_outer = int(size * 0.27)
    stroke = int(size * 0.085)
    d.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=accent + (255,),
        width=stroke,
    )

    # Gem: a simple brilliant silhouette (table + pavilion) straddling the
    # top of the band.
    gem_w = int(size * 0.30)
    gem_h = int(size * 0.22)
    gem_cy = cy - r_outer - int(gem_h * 0.18)
    top = gem_cy - gem_h // 2
    girdle = gem_cy - gem_h // 6
    tip = gem_cy + gem_h // 2
    left, right = cx - gem_w // 2, cx + gem_w // 2
    table_l, table_r = cx - gem_w // 5, cx + gem_w // 5

    light = tuple(min(255, c + 60) for c in accent)
    d.polygon(
        [(table_l, top), (table_r, top), (right, girdle),
         (cx, tip), (left, girdle)],
        fill=light + (255,),
    )
    # Facet hint: the crown break, drawn in the base accent so the gem
    # doesn't flatten into a solid blob.
    d.line([(left, girdle), (right, girdle)], fill=accent + (255,),
           width=max(1, size // 200))
    d.line([(table_l, top), (cx, tip)], fill=accent + (200,),
           width=max(1, size // 256))
    d.line([(table_r, top), (cx, tip)], fill=accent + (200,),
           width=max(1, size // 256))

    return img.resize((CANVAS, CANVAS), Image.LANCZOS)


def main() -> int:
    out_dir = ROOT / "assets"
    out_dir.mkdir(exist_ok=True)
    icon = build()
    ico_path = out_dir / "app_icon.ico"
    icon.save(ico_path, format="ICO", sizes=ICO_SIZES)
    icon.save(out_dir / "app_icon.png", format="PNG")
    print(f"wrote {ico_path} ({ico_path.stat().st_size:,} bytes)")
    print(f"wrote {out_dir / 'app_icon.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
