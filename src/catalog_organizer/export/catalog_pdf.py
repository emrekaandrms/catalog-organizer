"""Ürün katalog PDF'i.

Sayfa başına bir ürün: karşıdan ve çapraz görünüş yan yana, altında ID,
kategori, ağırlık ve ölçü. Kapak sayfasında liste adı, tarih ve seçilen
maden/taş rengi.

PDF, Qt'nin `QPdfWriter`'ı ile yazılıyor — projede zaten bulunan PyQt6 dışında
bağımlılık gerektirmiyor ve metni vektör olarak gömüyor, yani 300 DPI'da bile
yazılar keskin ve PDF içinde aranabilir kalıyor. (Pillow ile sayfa
resimleştirmek de mümkündü ama metin piksele dönüşürdü.)

Gösterilen ağırlık, render'da görünen madenin ağırlığıdır: gümüş render'ında
gümüş gramı, altın render'ında 14 ayar gramı. Farklı bir madeni çizip başka bir
madenin ağırlığını yazmak yanıltıcı olurdu.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PyQt6.QtCore import QMarginsF, QRectF, Qt
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter,
)

from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.render import materials as mat

RESOLUTION_DPI = 300
MARGIN_MM = 14.0

_INK = QColor("#1A1A1E")
_MUTED = QColor("#6B6B75")
_RULE = QColor("#D8D8DE")


class CatalogPdfError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogItem:
    file_id: str
    heading: str
    category: str
    weight_line: str
    dimension_line: str
    front_png: Path | None
    iso_png: Path | None
    note: str = ""


# Which stored weight belongs to which rendered metal. Yellow/rose/white gold
# all read the 14 ayar column because that is the only gold gram the pipeline
# keeps as a reference — the same decision the pricing module works from.
_WEIGHT_FIELD = {
    "yellow_gold": ("gold_14k_yellow_g", "14 Ayar Altın"),
    "rose_gold": ("gold_14k_yellow_g", "14 Ayar Altın"),
    "white_gold": ("gold_14k_yellow_g", "14 Ayar Altın"),
    "platinum": ("platinum_g", "Platin"),
    "silver": ("silver_925_g", "Gümüş 925"),
    "antique": ("silver_925_g", "Gümüş 925"),
    # Wax is a pattern, not a finished piece, and the pipeline stores no wax
    # gram. Rather than quietly print a silver weight under a purple render,
    # the label says which metal the figure belongs to. Converting it would
    # need a density ratio, and density conversions are explicitly out of
    # scope for this project.
    "wax_purple": ("silver_925_g", "Döküm karşılığı · Gümüş 925"),
}


def weight_line(rec: CatalogRecord, metal_key: str) -> str:
    field, label = _WEIGHT_FIELD.get(metal_key, ("silver_925_g", "Gümüş 925"))
    weights = rec.metal_weights
    value = getattr(weights, field, 0.0) if weights else 0.0
    if not value:
        return f"{label} · ağırlık ölçülemedi"
    return f"{label} · {value:.2f} g"


def dimension_line(rec: CatalogRecord) -> str:
    m = rec.measurements
    return (f"{m.bbox_width_mm:.1f} × {m.bbox_height_mm:.1f} × "
            f"{m.bbox_depth_mm:.1f} mm")


def build_item(
    rec: CatalogRecord,
    metal_key: str,
    front_png: Path | None,
    iso_png: Path | None,
    *,
    heading: str | None = None,
    note: str = "",
) -> CatalogItem:
    return CatalogItem(
        file_id=rec.file_id,
        heading=heading or rec.file_id,
        category=f"{rec.main_category} / {rec.subcategory}",
        weight_line=weight_line(rec, metal_key),
        dimension_line=dimension_line(rec),
        front_png=front_png,
        iso_png=iso_png,
        note=note,
    )


def _font(painter: QPainter, size_pt: int, *, bold: bool = False,
          letter_spacing: float = 100.0) -> QFont:
    """Build a font from the painter's current one.

    Letter spacing is always set, never left alone: QFont copies it from the
    source font, so one tracked-out label leaked its 180% spacing into every
    font built afterwards and the cover title came out as "Ö r n e k".
    """
    font = QFont(painter.font())
    font.setPointSizeF(size_pt)
    font.setBold(bold)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, letter_spacing)
    return font


def _draw_image_fitted(painter: QPainter, path: Path | None,
                       box: QRectF) -> None:
    """Draw the render centred inside `box`, never upscaled past its own size.

    A missing or unreadable file leaves a light placeholder rather than
    aborting the whole catalogue — one bad render should not cost 39 good
    pages.
    """
    image = QImage(str(path)) if path is not None else QImage()
    if image.isNull():
        painter.save()
        painter.setPen(_RULE)
        painter.drawRect(box)
        painter.setPen(_MUTED)
        painter.setFont(_font(painter, 9))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter),
                         "görsel yok")
        painter.restore()
        return

    scale = min(box.width() / image.width(), box.height() / image.height())
    w = image.width() * scale
    h = image.height() * scale
    target = QRectF(box.x() + (box.width() - w) / 2,
                    box.y() + (box.height() - h) / 2, w, h)
    painter.drawImage(target, image)


def _draw_cover(painter: QPainter, page: QRectF, *, title: str,
                metal_key: str, stone_key: str, count: int,
                subtitle: str) -> None:
    painter.setPen(_INK)
    y = page.y() + page.height() * 0.28

    painter.setFont(_font(painter, 11, letter_spacing=180))
    painter.drawText(QRectF(page.x(), y, page.width(), page.height() * 0.05),
                     int(Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignTop),
                     subtitle.upper())

    y += page.height() * 0.06
    painter.setFont(_font(painter, 30, bold=True))
    painter.drawText(QRectF(page.x(), y, page.width(), page.height() * 0.12),
                     int(Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignTop),
                     title)

    y += page.height() * 0.13
    painter.setPen(_RULE)
    rule_w = page.width() * 0.18
    painter.drawLine(int(page.center().x() - rule_w / 2), int(y),
                     int(page.center().x() + rule_w / 2), int(y))

    y += page.height() * 0.045
    painter.setPen(_MUTED)
    painter.setFont(_font(painter, 12))
    lines = [
        f"Maden: {mat.metal(metal_key).label_tr}",
        f"Taş: {mat.stone(stone_key).label_tr}",
        f"{count} ürün",
        date.today().strftime("%d.%m.%Y"),
    ]
    for line in lines:
        painter.drawText(
            QRectF(page.x(), y, page.width(), page.height() * 0.04),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            line)
        y += page.height() * 0.035


def _draw_item(painter: QPainter, page: QRectF, item: CatalogItem,
               *, page_no: int, total: int, footer: str) -> None:
    gap = page.width() * 0.035
    image_h = page.height() * 0.58
    box_w = (page.width() - gap) / 2

    left = QRectF(page.x(), page.y(), box_w, image_h)
    right = QRectF(page.x() + box_w + gap, page.y(), box_w, image_h)
    _draw_image_fitted(painter, item.front_png, left)
    _draw_image_fitted(painter, item.iso_png, right)

    painter.setPen(_MUTED)
    painter.setFont(_font(painter, 8, letter_spacing=140))
    label_h = page.height() * 0.025
    for box, label in ((left, "KARŞIDAN"), (right, "ÇAPRAZ")):
        painter.drawText(
            QRectF(box.x(), box.bottom(), box.width(), label_h),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            label)

    y = page.y() + image_h + label_h + page.height() * 0.02
    painter.setPen(_RULE)
    painter.drawLine(int(page.x()), int(y), int(page.right()), int(y))

    y += page.height() * 0.028
    painter.setPen(_INK)
    painter.setFont(_font(painter, 19, bold=True))
    painter.drawText(QRectF(page.x(), y, page.width(), page.height() * 0.06),
                     int(Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignTop),
                     item.heading)

    y += page.height() * 0.055
    painter.setPen(_MUTED)
    painter.setFont(_font(painter, 11))
    detail_lines = [item.category, item.weight_line, item.dimension_line]
    if item.note:
        detail_lines.append(item.note)
    for line in detail_lines:
        painter.drawText(
            QRectF(page.x(), y, page.width(), page.height() * 0.035),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            line)
        y += page.height() * 0.032

    painter.setFont(_font(painter, 8))
    painter.setPen(_MUTED)
    painter.drawText(
        QRectF(page.x(), page.bottom() - page.height() * 0.028,
               page.width(), page.height() * 0.028),
        int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom),
        footer)
    painter.drawText(
        QRectF(page.x(), page.bottom() - page.height() * 0.028,
               page.width(), page.height() * 0.028),
        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
        f"{page_no} / {total}")


def export_catalog_pdf(
    items: list[CatalogItem],
    out_path: Path,
    *,
    title: str,
    metal_key: str,
    stone_key: str,
    subtitle: str = "Ürün Kataloğu",
    cover: bool = True,
) -> Path:
    """Write the catalogue. Returns the path written."""
    if not items:
        raise CatalogPdfError("katalog için ürün yok")

    # Validate the colour keys up front: a typo should fail before a minute of
    # rendering, not on the cover page after it.
    mat.metal(metal_key)
    mat.stone(stone_key)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = QPdfWriter(str(out_path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setResolution(RESOLUTION_DPI)
    writer.setPageMargins(
        QMarginsF(MARGIN_MM, MARGIN_MM, MARGIN_MM, MARGIN_MM),
        QPageLayout.Unit.Millimeter,
    )
    writer.setTitle(title)
    writer.setCreator("Catalog Organizer")

    painter = QPainter()
    if not painter.begin(writer):
        raise CatalogPdfError(f"PDF açılamadı: {out_path}")
    try:
        page = QRectF(0, 0, writer.width(), writer.height())
        footer = f"{title} · {mat.metal(metal_key).label_tr}"

        if cover:
            _draw_cover(painter, page, title=title, metal_key=metal_key,
                        stone_key=stone_key, count=len(items),
                        subtitle=subtitle)
            writer.newPage()

        for index, item in enumerate(items, start=1):
            if index > 1:
                writer.newPage()
            _draw_item(painter, page, item, page_no=index,
                       total=len(items), footer=footer)
    finally:
        painter.end()

    return out_path


def render_and_export(
    records: list[CatalogRecord],
    out_path: Path,
    *,
    metal_key: str,
    stone_key: str,
    title: str,
    resolution: int = 1400,
    cover: bool = True,
    progress=None,
    db_conn=None,
) -> tuple[Path, dict[str, str]]:
    """Render every record in the chosen colours, then write the catalogue.

    Returns (pdf path, {file_id: error}). A product whose source file has moved
    or whose geometry fails still gets a page — with a "görsel yok" placeholder
    and a note — because a missing CAD file is worth seeing in the catalogue
    rather than silently dropping the item.

    `progress(done, total, file_id)` is called before each render so the GUI
    can show which piece is being drawn; a 40-item catalogue takes minutes.

    `db_conn`, when given, is checked for a hand-set camera angle per product
    (the Render tab's manual-orbit dialog writes these). Without it every page
    falls back to the automatic guess — an angle picked by hand in the Render
    tab is pointless if the actual catalogue never uses it.
    """
    from catalog_organizer.render.engine import load_and_render

    if not records:
        raise CatalogPdfError("katalog için ürün yok")

    errors: dict[str, str] = {}
    items: list[CatalogItem] = []
    total = len(records)

    for index, rec in enumerate(records, start=1):
        if progress is not None:
            progress(index, total, rec.file_id)
        front = iso = None
        note = ""
        source = Path(rec.source_path)
        if not source.exists():
            errors[rec.file_id] = f"kaynak dosya bulunamadı: {source}"
            note = "kaynak dosya bulunamadı"
        else:
            try:
                camera = None
                if db_conn is not None:
                    from catalog_organizer.db.camera_overrides import (
                        get_camera_overrides,
                    )
                    overrides = get_camera_overrides(db_conn, rec.file_id)
                    if overrides:
                        camera = {v: (o.direction, o.up)
                                  for v, o in overrides.items()}
                # Stones in the seats of a stoneless STL: the same answer the Render tab shows
                # (the record's stone status, or the user's explicit choice for this piece).
                from catalog_organizer.db.render_options import wants_placed_stones
                result = load_and_render(
                    source, file_id=rec.file_id, metal_key=metal_key,
                    stone_key=stone_key, resolution=resolution, camera=camera,
                    category=rec.main_category,
                    place_stones=wants_placed_stones(db_conn, rec),
                )
                front = result.views.get("front")
                iso = result.views.get("iso")
                if not result.had_stones and rec.stone_summary.status == "stone":
                    # The analysis says stones, the file gave us none to colour.
                    # Say so on the page instead of quietly shipping a catalogue
                    # where the chosen stone colour is nowhere to be seen.
                    note = "taş geometrisi bulunamadı — taş rengi uygulanmadı"
            except Exception as exc:
                errors[rec.file_id] = f"{type(exc).__name__}: {exc}"
                note = "render başarısız"
        items.append(build_item(rec, metal_key, front, iso, note=note))

    path = export_catalog_pdf(
        items, out_path, title=title, metal_key=metal_key,
        stone_key=stone_key, cover=cover,
    )
    return path, errors
