from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from catalog_organizer.core.schemas import MetalWeights  # noqa: E402
from catalog_organizer.export.catalog_pdf import (  # noqa: E402
    CatalogItem, CatalogPdfError, build_item, dimension_line,
    export_catalog_pdf, weight_line,
)
from catalog_organizer.render import materials as mat  # noqa: E402
from catalog_organizer.render.product_render import camera_axes  # noqa: E402


# ---------------------------------------------------------------- malzemeler

def test_material_palettes_are_non_empty_and_labelled():
    assert mat.DEFAULT_METAL in mat.METALS
    assert mat.DEFAULT_STONE in mat.STONES
    assert all(m.label_tr for m in mat.METALS.values())
    assert all(s.label_tr for s in mat.STONES.values())


def test_metals_are_fully_metallic_and_the_wax_is_not():
    """Every real metal sits at 1.0. Castable wax shares the drop-down because
    it shares the workflow slot — it is what the piece is sold as when the
    customer is buying the printed pattern — but it is a dyed dielectric and
    rendering it as a metal would give it reflections no wax has."""
    metals = [m for m in mat.METALS.values() if not m.key.startswith("wax")]
    assert metals and all(m.metallic == 1.0 for m in metals)
    assert mat.metal("wax_purple").metallic == 0.0


def test_stones_are_described_by_their_refractive_index():
    """Every gem here is a real mineral with a measured index, and the index
    is the only shading input the refraction shader needs: it fixes both how
    the ray bends and how much the surface reflects."""
    assert mat.stone("white").ior == pytest.approx(2.417)      # diamond
    assert mat.stone("blue").ior == pytest.approx(1.77)        # corundum
    assert mat.stone("green").ior == pytest.approx(1.58)       # beryl
    assert all(s.ior > 1.0 for s in mat.STONES.values())


def test_diamond_reflects_hardest_because_its_index_is_highest():
    """f0 = ((n-1)/(n+1))^2, not a preference. Diamond's 2.417 gives 0.172 —
    four times a common glass, and a large part of why a diamond's surface
    alone looks brighter than any imitation's."""
    from catalog_organizer.render.gem_trace import fresnel_f0

    assert fresnel_f0(2.417) == pytest.approx(0.172, abs=0.002)
    assert fresnel_f0(1.77) == pytest.approx(0.077, abs=0.002)
    assert fresnel_f0(mat.stone("white").ior) > fresnel_f0(mat.stone("blue").ior)


def test_only_onyx_is_opaque():
    """Onyx has no interior to see into, so running it through the refraction
    shader would invent one. Every other stone here transmits."""
    assert mat.stone("black").opaque
    assert not any(s.opaque for s in mat.STONES.values() if s.key != "black")


def test_a_coloured_stone_tints_transmission_not_its_surface():
    """The colour of a gem is absorption along the path through it, so it
    belongs on the refracted ray. An earlier model had to tint the surface
    instead — the layered draw's coloured layer reflected the rig's dark
    lower hemisphere and came back grey, measuring a saturation of 0.005 on
    the catalogue's own pieces. Keeping the tint in transmission is why a real
    sapphire still shows colourless glints around its rim."""
    from catalog_organizer.render.gem_trace import absorption_from

    red, _green, blue = absorption_from(mat.stone("blue").color)
    assert red > blue + 0.4, "a sapphire has to absorb red, not blue"
    assert absorption_from(mat.stone("white").color) == (0.0, 0.0, 0.0)


def _bounds(dx: float, dy: float, dz: float):
    return (0.0, dx, 0.0, dy, 0.0, dz)


def test_camera_looks_down_the_thinnest_axis():
    """A pendant modelled flat in XY must be viewed from above. A fixed
    world-axis camera rendered it edge-on as a gold sliver."""
    front, _iso, _up = camera_axes(_bounds(30.0, 28.0, 4.0))
    assert np.allclose(front, [0, 0, 1])


def test_camera_handles_a_ring_standing_in_yz():
    front, _iso, up = camera_axes(_bounds(4.0, 22.0, 20.0))
    assert np.allclose(front, [1, 0, 0])
    assert np.allclose(up, [0, 1, 0])


def test_up_axis_is_the_longest_remaining_axis():
    _front, _iso, up = camera_axes(_bounds(40.0, 10.0, 3.0))
    assert np.allclose(up, [1, 0, 0])


def test_iso_is_unit_length_and_off_axis():
    front, iso, _up = camera_axes(_bounds(30.0, 28.0, 4.0))
    assert np.isclose(np.linalg.norm(iso), 1.0)
    assert not np.allclose(iso, front)
    # Still mostly face-on: a three-quarter view, not a side view.
    assert float(np.dot(iso, front)) > 0.7


def test_camera_survives_a_degenerate_flat_mesh():
    front, iso, up = camera_axes(_bounds(10.0, 10.0, 0.0))
    for vec in (front, iso, up):
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------- alan metni

def test_weight_line_follows_the_rendered_metal(record_factory):
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=4.0, gold_14k_yellow_g=5.1, platinum_g=7.9))
    assert weight_line(rec, "silver") == "Gümüş 925 · 4.00 g"
    assert weight_line(rec, "yellow_gold") == "14 Ayar Altın · 5.10 g"
    assert weight_line(rec, "platinum") == "Platin · 7.90 g"


def test_weight_line_says_so_when_the_weight_is_missing(record_factory):
    rec = record_factory(file_id="A", metal_weights=None,
                         weight_source="unavailable",
                         weight_confidence="unavailable")
    assert "ölçülemedi" in weight_line(rec, "yellow_gold")


def test_wax_says_which_metal_its_weight_belongs_to(record_factory):
    """A purple wax render carries no wax gram — the pipeline never measures
    one. Printing the silver figure unlabelled under a wax pattern would read
    as the wax's own weight, so the label names the metal instead."""
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=4.0, gold_14k_yellow_g=5.1, platinum_g=7.9))
    line = weight_line(rec, "wax_purple")
    assert "Gümüş 925" in line and "4.00 g" in line
    assert line != weight_line(rec, "silver"), "wax must not pose as silver"


def test_dimension_line(record_factory):
    rec = record_factory(file_id="A")
    assert dimension_line(rec) == "20.0 × 22.0 × 8.0 mm"


def test_build_item_carries_id_category_and_weight(record_factory):
    rec = record_factory(file_id="JCAD-000000009", main_category="pendant",
                         subcategory="heart",
                         metal_weights=MetalWeights(gold_14k_yellow_g=3.0))
    item = build_item(rec, "yellow_gold", None, None)
    assert item.heading == "JCAD-000000009"
    assert item.category == "pendant / heart"
    assert item.weight_line == "14 Ayar Altın · 3.00 g"


# ---------------------------------------------------------------- PDF

def _png(path: Path, size: int = 64) -> Path:
    from PyQt6.QtGui import QColor, QImage
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(QColor("#E8B84B"))
    image.save(str(path))
    return path


def _item(tmp_path: Path, name: str = "JCAD-000000001") -> CatalogItem:
    return CatalogItem(
        file_id=name, heading=name, category="pendant / heart",
        weight_line="14 Ayar Altın · 5.72 g",
        dimension_line="29.1 × 27.3 × 5.8 mm",
        front_png=_png(tmp_path / f"{name}-front.png"),
        iso_png=_png(tmp_path / f"{name}-iso.png"),
    )


def _page_count(path: Path) -> int:
    from PyQt6.QtPdf import QPdfDocument
    doc = QPdfDocument(None)
    doc.load(str(path))
    return doc.pageCount()


def test_pdf_has_a_page_per_product_plus_cover(qapp, tmp_path):
    items = [_item(tmp_path, f"JCAD-{i:09d}") for i in range(3)]
    out = tmp_path / "katalog.pdf"
    export_catalog_pdf(items, out, title="Kış 2026",
                       metal_key="yellow_gold", stone_key="white")
    assert out.exists()
    assert _page_count(out) == 4


def test_pdf_without_cover(qapp, tmp_path):
    items = [_item(tmp_path, f"JCAD-{i:09d}") for i in range(2)]
    out = tmp_path / "katalog.pdf"
    export_catalog_pdf(items, out, title="T", metal_key="silver",
                       stone_key="red", cover=False)
    assert _page_count(out) == 2


def test_pdf_refuses_an_empty_catalogue(qapp, tmp_path):
    with pytest.raises(CatalogPdfError, match="ürün yok"):
        export_catalog_pdf([], tmp_path / "x.pdf", title="T",
                           metal_key="silver", stone_key="red")


def test_pdf_rejects_bad_colour_before_rendering_anything(qapp, tmp_path):
    """A typo must fail immediately, not after a minute of rendering."""
    out = tmp_path / "katalog.pdf"
    with pytest.raises(KeyError):
        export_catalog_pdf([_item(tmp_path)], out, title="T",
                           metal_key="bronze", stone_key="white")
    assert not out.exists()


def test_pdf_survives_a_missing_render(qapp, tmp_path):
    """One failed render must not cost the other pages."""
    good = _item(tmp_path, "GOOD")
    bad = CatalogItem(
        file_id="BAD", heading="BAD", category="ring / other",
        weight_line="Gümüş 925 · 1.00 g", dimension_line="1 × 1 × 1 mm",
        front_png=tmp_path / "yok.png", iso_png=None,
        note="render başarısız")
    out = tmp_path / "katalog.pdf"
    export_catalog_pdf([good, bad], out, title="T", metal_key="silver",
                       stone_key="white", cover=False)
    assert _page_count(out) == 2


def test_pdf_creates_missing_output_directory(qapp, tmp_path):
    out = tmp_path / "yeni" / "klasor" / "katalog.pdf"
    export_catalog_pdf([_item(tmp_path)], out, title="T",
                       metal_key="silver", stone_key="white", cover=False)
    assert out.exists()


def test_pdf_embeds_text_not_pixels(qapp, tmp_path):
    """QPdfWriter writes vector text — the ID under each product stays crisp
    at print size and is searchable inside the PDF."""
    out = tmp_path / "katalog.pdf"
    export_catalog_pdf([_item(tmp_path, "JCAD-000000042")], out, title="T",
                       metal_key="silver", stone_key="white", cover=False)
    raw = out.read_bytes()
    assert b"/Font" in raw


# ---------------------------------------------------------------- diyalog

def test_dialog_defaults_and_options(qtbot, tmp_path):
    from catalog_organizer.gui.panels.catalog_pdf_dialog import CatalogPdfDialog
    dialog = CatalogPdfDialog(default_title="Kış 2026", item_count=40,
                              out_path=tmp_path / "k.pdf")
    qtbot.addWidget(dialog)
    options = dialog.options()
    assert options.metal_key == mat.DEFAULT_METAL
    assert options.stone_key == mat.DEFAULT_STONE
    assert options.title == "Kış 2026"
    assert options.cover is True
    assert dialog.slice_bounds() == (0, 40)


def test_dialog_lists_every_material(qtbot, tmp_path):
    from catalog_organizer.gui.panels.catalog_pdf_dialog import CatalogPdfDialog
    dialog = CatalogPdfDialog(default_title="T", item_count=1,
                              out_path=tmp_path / "k.pdf")
    qtbot.addWidget(dialog)
    assert dialog._metal.count() == len(mat.METALS)
    assert dialog._stone.count() == len(mat.STONES)


def test_dialog_blank_title_falls_back(qtbot, tmp_path):
    from catalog_organizer.gui.panels.catalog_pdf_dialog import CatalogPdfDialog
    dialog = CatalogPdfDialog(default_title="T", item_count=1,
                              out_path=tmp_path / "k.pdf")
    qtbot.addWidget(dialog)
    dialog._title.setText("   ")
    assert dialog.options().title == "Ürün Kataloğu"


# ---------------------------------------------------------------- blok cozumu

def test_xform_matrix_and_application():
    """Gems are placed as block instances carrying a 4x4 transform; applying
    it wrong puts every stone at the model origin instead of in its setting."""
    from catalog_organizer.snapshotter.threedm import (
        _apply_xform, _xform_to_matrix,
    )

    class FakeXform:
        M00, M01, M02, M03 = 2.0, 0.0, 0.0, 5.0
        M10, M11, M12, M13 = 0.0, 2.0, 0.0, -1.0
        M20, M21, M22, M23 = 0.0, 0.0, 2.0, 0.5
        M30, M31, M32, M33 = 0.0, 0.0, 0.0, 1.0

    matrix = _xform_to_matrix(FakeXform())
    assert matrix.shape == (4, 4)
    moved = _apply_xform(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), matrix)
    assert np.allclose(moved[0], [7.0, 1.0, 2.5])
    assert np.allclose(moved[1], [5.0, -1.0, 0.5])


def test_rotation_transform_is_not_treated_as_translation_only():
    from catalog_organizer.snapshotter.threedm import (
        _apply_xform, _xform_to_matrix,
    )

    class Rot90Z:
        M00, M01, M02, M03 = 0.0, -1.0, 0.0, 0.0
        M10, M11, M12, M13 = 1.0, 0.0, 0.0, 0.0
        M20, M21, M22, M23 = 0.0, 0.0, 1.0, 0.0
        M30, M31, M32, M33 = 0.0, 0.0, 0.0, 1.0

    moved = _apply_xform(np.array([[1.0, 0.0, 0.0]]),
                         _xform_to_matrix(Rot90Z()))
    assert np.allclose(moved[0], [0.0, 1.0, 0.0])


_SAMPLES = Path("PilotBatch")
_needs_samples = pytest.mark.skipif(
    not _SAMPLES.exists(), reason="PilotBatch örnek dosyaları yok")


@_needs_samples
def test_split_loader_resolves_block_instance_gems():
    """Regression: MatrixGold stores stones as InstanceReference blocks. Before
    the block resolver this returned zero gem vertices on stone-set files, so
    the catalogue's stone colour had nothing to colour."""
    from catalog_organizer.snapshotter.threedm import load_3dm_split_arrays

    stone_files = [p for p in sorted(_SAMPLES.rglob("*.3dm"))
                   if "tasli" in p.name.lower()][:4]
    if not stone_files:
        pytest.skip("taşlı örnek dosya yok")

    with_gems = 0
    for path in stone_files:
        metal, gems = load_3dm_split_arrays(path)
        assert metal is not None and metal[0].shape[0] > 0
        if gems is not None and gems[0].shape[0] > 0:
            with_gems += 1
    assert with_gems > 0, "hiçbir taşlı dosyada taş geometrisi çözülemedi"


@_needs_samples
def test_split_loader_drops_engraved_note_text():
    """The first real catalogue page rendered "17 boy cilalı" larger than the
    product, and the inflated bounding box shrank the piece into a corner."""
    from catalog_organizer.snapshotter.threedm import load_3dm_split_arrays

    path = next((p for p in sorted(_SAMPLES.rglob("*.3dm"))
                 if "17_" in p.name or "boy" in p.name.lower()), None)
    files = [path] if path else sorted(_SAMPLES.rglob("*.3dm"))[:1]
    metal, _gems = load_3dm_split_arrays(files[0])
    assert metal is not None
    verts = metal[0]
    extent = verts.max(0) - verts.min(0)
    # A note sits beside the piece, so its presence shows up as an outsized
    # footprint relative to thickness. Not a tight assertion — just proof the
    # loader returns a plausible single object rather than piece-plus-billboard.
    assert float(extent.max() / max(extent.min(), 1e-6)) < 40.0


# ---------------------------------------------------------------- kadraj

def _cube(size=1.0, offset=(0.0, 0.0, 0.0)):
    """Axis-aligned box as (verts, faces) — two triangles per face."""
    sx, sy, sz = (size, size, size) if np.isscalar(size) else size
    ox, oy, oz = offset
    v = np.array([
        [0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
        [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz],
    ], dtype=float) + np.array([ox, oy, oz])
    f = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ])
    return v, f


def test_facing_weights_point_at_the_broad_face():
    """A slab's surface mostly faces its thin axis."""
    from catalog_organizer.render.product_render import facing_weights
    verts, faces = _cube(size=(20.0, 18.0, 1.0))
    weights = facing_weights(verts, faces)
    assert int(np.argmax(weights)) == 2


def test_camera_uses_geometry_when_given_it():
    """A wide ring band's flat sides beat its setting on bbox alone; the
    facing signal is what puts the camera on the design."""
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _cube(size=(20.0, 18.0, 1.0))
    front, _iso, _up = camera_axes((0, 20, 0, 18, 0, 1), (verts, faces))
    assert np.allclose(front, [0, 0, 1])


def test_camera_falls_back_to_bounds_without_geometry():
    from catalog_organizer.render.product_render import camera_axes
    front, _iso, _up = camera_axes((0, 30, 0, 28, 0, 4), None)
    assert np.allclose(front, [0, 0, 1])


def test_camera_ignores_empty_face_array():
    from catalog_organizer.render.product_render import camera_axes
    front, _iso, _up = camera_axes(
        (0, 30, 0, 28, 0, 4),
        (np.zeros((0, 3)), np.zeros((0, 3), dtype=int)))
    assert np.allclose(front, [0, 0, 1])


# ---------------------------------------------------------------- ortam (HDR)

def _supplied_hdrs_present() -> bool:
    """The two optional showroom/studio HDRs are NOT shipped with the project (they were a third
    party's); a machine that has them dropped into assets/hdr/ gets the extra tests."""
    from catalog_organizer.render.product_render import GEM_HDR, HDR_DIR, METAL_HDR
    return (HDR_DIR / GEM_HDR).exists() and (HDR_DIR / METAL_HDR).exists()


_NEEDS_HDRS = pytest.mark.skipif(not _supplied_hdrs_present(),
                                 reason="optional assets/hdr/*.hdr are not present (none are shipped)")


@_NEEDS_HDRS
def test_supplied_hdrs_are_reported_as_unusable():
    """Both files in assets/hdr/ are 512x512. `SetEnvironmentTexture` reads an
    equirectangular panorama and an equirect is always 2:1, so neither was ever
    mapped onto the sphere correctly — and a misprojected environment still
    renders, just wrongly, which is why this went unnoticed. The status has to
    separate "present" from "usable" or the check is vacuous."""
    from catalog_organizer.render.product_render import hdr_status
    status = hdr_status()
    assert status["metal"]["present"] and status["gem"]["present"]
    assert not status["metal"]["usable"]
    assert not status["gem"]["usable"]


def test_environment_falls_back_to_the_rig_not_to_the_bad_file():
    from catalog_organizer.render.product_render import environment_for
    texture, from_file = environment_for("metal")
    assert texture is not None
    assert from_file is False


def test_metal_and_gem_get_different_environments():
    """One VTK renderer carries one environment texture; that is why the
    renderer draws two passes. If both kinds resolved to the same texture the
    split would be pointless."""
    from catalog_organizer.render.product_render import environment_for
    metal_tex, _ = environment_for("metal")
    gem_tex, _ = environment_for("gem")
    assert metal_tex is not gem_tex


def test_render_version_is_in_the_cache_filename(tmp_path):
    """Cached PNGs from a previous lighting pipeline must not be served next to
    new ones — the version prefix is what prevents that.

    The prefix now carries a hash of the tuned constants as well, because the
    manual bump was forgotten once and the app served a whole catalogue of
    stale frames; see `product_render._look_key`."""
    from catalog_organizer.render.product_render import (RENDER_VERSION,
                                                         _look_key)

    verts, faces = _cube(size=(10.0, 9.0, 2.0))
    from catalog_organizer.render.product_render import render_product
    result = render_product((verts, faces), None, file_id="CACHE",
                            metal_key="silver", stone_key="white",
                            resolution=120, out_dir=tmp_path, overwrite=True)
    for path in result.views.values():
        assert path.name.startswith(f"v{RENDER_VERSION}{_look_key()}_")
        assert path.exists()


def _rgba(shape, value, alpha=255):
    out = np.zeros(shape + (4,), dtype=np.uint8)
    out[..., :3] = value
    out[..., 3] = alpha
    return out


def test_composite_prefers_gem_pixels_where_the_mask_says_gem():
    from catalog_organizer.render.product_render import composite
    metal = _rgba((4, 4), 10)
    gem = _rgba((4, 4), 200)
    mask = np.zeros((4, 4, 3), dtype=np.int16)          # black = background
    mask[0:2] = [255, 0, 0]                             # metal key
    mask[2:4] = [0, 255, 0]                             # gem key
    out = composite(metal, gem, mask)
    assert (out[0:2, :, :3] == 10).all()
    assert (out[2:4, :, :3] == 200).all()


def test_composite_keeps_partly_covered_edge_pixels():
    """The defect behind "round edges look stepped".

    Coverage belongs to the passes' own alpha; the mask only says which
    material a pixel is. An earlier version classified with a hard threshold —
    a pixel counted as metal only within 90 of the key colour — and a
    silhouette pixel at 65% coverage renders as (165, 0, 0), exactly 90 away.
    Every edge pixel below two thirds coverage was deleted, collapsing a
    silhouette that left the renderer with 89 distinct coverage levels into a
    1-bit edge."""
    from catalog_organizer.render.product_render import composite
    metal = _rgba((1, 4), 200, alpha=0)
    metal[0, :, 3] = [255, 165, 60, 0]                  # an anti-aliased edge
    gem = _rgba((1, 4), 0, alpha=0)
    mask = np.zeros((1, 4, 3), dtype=np.int16)
    mask[0, :, 0] = [255, 165, 60, 0]                   # the same ramp, in red
    out = composite(metal, gem, mask)
    assert list(out[0, :, 3]) == [255, 165, 60, 0]
    assert (out[0, :3, :3] == 200).all(), "edge pixels kept the metal's colour"


def test_composite_leaves_uncovered_pixels_transparent():
    from catalog_organizer.render.product_render import composite
    metal = _rgba((2, 2), 10, alpha=0)
    gem = _rgba((2, 2), 200, alpha=0)
    mask = np.zeros((2, 2, 3), dtype=np.int16)
    assert (composite(metal, gem, mask)[..., 3] == 0).all()


# ---------------------------------------------------------------- bore duzeltmesi

def _annulus(r_min=8.0, r_max=10.0, height=3.0):
    """A hollow washer — flat top/bottom, genuinely open in the middle. Stands
    in for a real ring: same failure mode (facing_weights dominated by the
    flat rim, straight down the bore) without needing a CAD file on disk."""
    import trimesh
    m = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=height)
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_central_hole_is_zero_for_a_solid_slab():
    """Regression: the first version of this measure sampled triangle corners
    and centroids, which put 20 points on a 64x64 grid for a 12-triangle box
    and read a solid slab as almost entirely hollow."""
    from catalog_organizer.render.product_render import central_hole_fraction
    verts, faces = _cube(size=(20.0, 18.0, 1.0))
    assert central_hole_fraction(verts, faces, 2) == 0.0


def test_central_hole_is_large_for_a_ring_shape():
    from catalog_organizer.render.product_render import central_hole_fraction
    verts, faces = _annulus()
    assert central_hole_fraction(verts, faces, 2) > 0.9


def test_central_hole_is_small_viewed_across_a_ring():
    """Looking at a ring edge-on, the band fills the middle — only the bore
    axis shows the hole, which is what makes this usable as a test."""
    from catalog_organizer.render.product_render import central_hole_fraction
    verts, faces = _annulus()
    assert central_hole_fraction(verts, faces, 0) < 0.6


def test_bore_avoidance_is_off_by_default(qapp=None):
    """Regression: camera_axes must not change behaviour unless a caller
    explicitly opts in — this is what keeps every gem-driven ring (validated
    on four real files) rendering exactly as before."""
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _annulus()
    bounds = (verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max(),
              verts[:, 2].min(), verts[:, 2].max())
    front, _iso, _up = camera_axes(bounds, (verts, faces))
    assert np.allclose(front, [0, 0, 1]), "varsayilan davranis degismemeli"


def test_bore_avoidance_moves_off_the_hole_axis():
    """The actual bug: a plain ring band rendered from an STL (no gem layer to
    anchor the camera on) came out as a nearly featureless circle — the
    camera was looking straight through the finger hole. Reproduced here with
    a synthetic washer so the test needs no CAD file on disk."""
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _annulus()
    bounds = (verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max(),
              verts[:, 2].min(), verts[:, 2].max())
    front, iso, up = camera_axes(bounds, (verts, faces), avoid_bore=True)
    assert not np.allclose(front, [0, 0, 1]), "hala duz delikten bakiyor"
    # "up" is the design direction, so the setting sits at the top of the
    # frame; it is perpendicular to the bore rather than along it.
    assert abs(float(up[2])) < 1e-6, "up, delik ekseni olmamali"
    assert np.isclose(np.linalg.norm(up), 1.0)
    assert not np.allclose(iso, front), "iso, front'tan gorunur sekilde farkli olmali"


def test_bore_is_found_on_an_axis_the_facing_weights_did_not_pick():
    """The reported "details are unreadable" framing, in one assertion.

    The hole test used to run only on the axis `facing_weights` chose. Of four
    real rings, three had a bore fraction of 1.00 on a DIFFERENT axis and 0.00
    on the chosen one, so they were never recognised as rings and were framed
    as though they were flat plates.

    The annulus here is built around Z while its facing weights land on X or
    Y — the same disagreement, reproduced without needing a CAD file."""
    from catalog_organizer.render.product_render import (
        facing_weights, open_bore_axis,
    )
    verts, faces = _annulus(r_min=8.0, r_max=10.0, height=3.0)
    chosen = int(np.argmax(facing_weights(verts, faces)))
    found = open_bore_axis((verts, faces))
    assert found == 2, "the bore is the Z axis"
    if chosen != 2:
        assert found != chosen, "found despite the facing weights disagreeing"


def test_open_bore_axis_says_none_for_a_solid_piece():
    """Scanning all three axes is only safe because pendants stay far from the
    threshold. Measured per axis on six real pendants, the highest reading of
    any axis of any piece was 0.18, against 1.00 for every ring."""
    from catalog_organizer.render.product_render import open_bore_axis
    assert open_bore_axis(_cube(size=(20.0, 18.0, 1.0))) is None
    assert open_bore_axis(_cube(size=(20.0, 20.0, 2.0))) is None
    assert open_bore_axis(None) is None


def test_a_precomputed_bore_is_honoured_and_does_not_shadow_the_local():
    """`camera_axes` takes the bore from `render_product` so the million-face
    hole test is not run twice — it costs about a second. The parameter is
    deliberately not called `bore_axis`: a local of that name is initialised
    inside the function, and naming the parameter the same thing overwrote it
    before it was ever read, which classified every piece as a ring."""
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _cube(size=(20.0, 18.0, 1.0))
    bounds = (0, 20, 0, 18, 0, 1)
    told_none = camera_axes(bounds, (verts, faces), avoid_bore=True,
                            known_bore=None)[0]
    assert np.allclose(told_none, [0, 0, 1])
    told_ring = camera_axes(bounds, (verts, faces), avoid_bore=True,
                            known_bore=1)[0]
    assert not np.allclose(told_ring, [0, 0, 1])


def test_rings_are_framed_wider_than_flat_pieces():
    """A ring is tall in frame with shoulders running out to the sides, and at
    the common zoom it crowded its own edges."""
    from catalog_organizer.render.product_render import _RING_ZOOM_FACTOR
    assert 1.0 < _RING_ZOOM_FACTOR < 1.35


def test_bore_avoidance_does_not_fire_on_a_solid_piece():
    """A flat pendant has no hole at its centre — the correction must leave it
    alone even when avoid_bore=True is passed (e.g. it has no stones)."""
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _cube(size=(20.0, 18.0, 1.0))
    bounds = (0, 20, 0, 18, 0, 1)
    front, _iso, _up = camera_axes(bounds, (verts, faces), avoid_bore=True)
    assert np.allclose(front, [0, 0, 1])


def test_render_product_arms_bore_avoidance_only_without_gems(qapp, tmp_path):
    """The wiring in render_product: avoid_bore is tied to gem_arrays being
    None, not to a separate flag a caller could forget to set."""
    from catalog_organizer.render.product_render import render_product

    metal = _annulus()
    # taş kaynağı yok — STL'in ya da taşsız bir parçanın tam durumu
    result = render_product(metal, None, file_id="RING-NOGEM",
                            metal_key="yellow_gold", stone_key="white",
                            resolution=150, out_dir=tmp_path, overwrite=True)
    assert result.had_stones is False
    for path in result.views.values():
        assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------- kamera override

def test_render_product_camera_override_changes_direction(qapp, tmp_path):
    """A hand-set angle from the orbit dialog must actually change the picture,
    not just be silently accepted."""
    from catalog_organizer.render.product_render import render_product

    metal = _annulus()
    auto = render_product(metal, None, file_id="OV", metal_key="silver",
                          stone_key="white", resolution=120,
                          out_dir=tmp_path, overwrite=True)
    custom = render_product(
        metal, None, file_id="OV", metal_key="silver", stone_key="white",
        resolution=120, out_dir=tmp_path, overwrite=True,
        camera={"front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))})
    assert auto.views["front"] != custom.views["front"]
    assert custom.views["front"].exists()


def test_render_product_camera_override_leaves_other_view_automatic(qapp, tmp_path):
    from catalog_organizer.render.product_render import render_product

    metal = _annulus()
    result = render_product(
        metal, None, file_id="OV2", metal_key="silver", stone_key="white",
        resolution=120, out_dir=tmp_path, overwrite=True,
        camera={"front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))})
    auto_only = render_product(metal, None, file_id="OV2", metal_key="silver",
                               stone_key="white", resolution=120,
                               out_dir=tmp_path, overwrite=True)
    assert result.views["iso"] == auto_only.views["iso"], (
        "front icin ozel aci, iso'ya sizmamali")


def test_render_product_ignores_override_for_unknown_view(qapp, tmp_path):
    from catalog_organizer.render.product_render import render_product

    metal = _annulus()
    result = render_product(
        metal, None, file_id="OV3", metal_key="silver", stone_key="white",
        resolution=120, out_dir=tmp_path, overwrite=True,
        camera={"top": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))})
    assert set(result.views) == {"front", "iso"}


def test_repeated_override_reuses_its_own_cache_slot(qapp, tmp_path):
    """Same override, second call: must hit the cache, not re-render."""
    from catalog_organizer.render.product_render import render_product

    metal = _annulus()
    cam = {"front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}
    r1 = render_product(metal, None, file_id="OV4", metal_key="silver",
                        stone_key="white", resolution=120, out_dir=tmp_path,
                        overwrite=True, camera=cam)
    mtime1 = r1.views["front"].stat().st_mtime_ns
    r2 = render_product(metal, None, file_id="OV4", metal_key="silver",
                        stone_key="white", resolution=120, out_dir=tmp_path,
                        overwrite=False, camera=cam)
    assert r2.views["front"] == r1.views["front"]
    assert r2.views["front"].stat().st_mtime_ns == mtime1


# ---------------------------------------------------------------- halka kadraji

def _twisted_band(design_angle_deg=-90.0):
    """A washer with extra height over one half — stands in for a ring whose
    design (twist, setting, shoulders) sits on one side of the loop."""
    import trimesh
    # Plenty of segments: feature_azimuth bins vertices into 36 sectors and
    # ignores any sector with fewer than five, so a coarse washer reports no
    # design at all.
    # Thin enough that the flat rim dominates the facing weights and the bore
    # is the axis the camera would pick — which is what real rings measure
    # like, and the case the ring composition exists for.
    ring = trimesh.creation.annulus(r_min=8.0, r_max=10.0, height=1.0,
                                    sections=240)
    verts = np.asarray(ring.vertices).copy()
    angle = np.arctan2(verts[:, 1], verts[:, 0])
    target = np.radians(design_angle_deg)
    # Angular distance to the design position, wrapped to [-pi, pi].
    delta = np.abs(np.arctan2(np.sin(angle - target), np.cos(angle - target)))
    boost = np.clip(1.0 - delta / (np.pi / 2), 0.0, 1.0)
    verts[:, 2] *= 1.0 + 1.5 * boost
    return verts, np.asarray(ring.faces)


def test_feature_azimuth_finds_the_thickened_side():
    """Knowing to shoot a ring from the side is half the answer; the other
    half is which side. A world-axis guess landed 90° off on a real ring and
    rendered the plain back of the band."""
    from catalog_organizer.render.product_render import feature_azimuth
    verts, faces = _twisted_band(design_angle_deg=-90.0)
    bounds = (verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max(),
              verts[:, 2].min(), verts[:, 2].max())
    direction = feature_azimuth(verts, 2, bounds)
    assert direction is not None
    found = np.degrees(np.arctan2(direction[1], direction[0]))
    assert abs(found - (-90.0)) < 20.0, f"tasarim yonu {found:.0f}° bulundu"


def test_feature_azimuth_follows_the_design_around():
    from catalog_organizer.render.product_render import feature_azimuth
    for expected in (0.0, 90.0, 180.0):
        verts, faces = _twisted_band(design_angle_deg=expected)
        bounds = (verts[:, 0].min(), verts[:, 0].max(),
                  verts[:, 1].min(), verts[:, 1].max(),
                  verts[:, 2].min(), verts[:, 2].max())
        d = feature_azimuth(verts, 2, bounds)
        found = np.degrees(np.arctan2(d[1], d[0]))
        gap = abs((found - expected + 180) % 360 - 180)
        assert gap < 25.0, f"{expected}° beklenirken {found:.0f}° bulundu"


def test_feature_azimuth_returns_none_for_an_even_band():
    """A plain washer has no design side — the caller must keep its own
    axis-aligned choice rather than be handed a meaningless direction."""
    from catalog_organizer.render.product_render import feature_azimuth
    verts, faces = _annulus()
    bounds = (verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max(),
              verts[:, 2].min(), verts[:, 2].max())
    assert feature_azimuth(verts, 2, bounds) is None


def test_ring_composition_puts_the_design_up_and_tilts_off_the_bore():
    from catalog_organizer.render.product_render import camera_axes
    verts, faces = _twisted_band(design_angle_deg=-90.0)
    bounds = (verts[:, 0].min(), verts[:, 0].max(),
              verts[:, 1].min(), verts[:, 1].max(),
              verts[:, 2].min(), verts[:, 2].max())
    front, iso, up = camera_axes(bounds, (verts, faces), avoid_bore=True)
    # up points at the design, so the setting lands at the top of the frame
    assert np.dot(up, [0, -1, 0]) > 0.9
    # and the camera is well off the bore, but not perpendicular to it either
    assert 0.3 < abs(float(front[2])) < 0.95
    assert not np.allclose(front, iso)


def test_silhouette_arrays_override_the_hole_test():
    """The hole test must run on the whole piece: a halo's ring of stones has
    a gap in the middle by itself, and testing the stones alone reported a
    hole where the metal underneath plainly fills the frame."""
    from catalog_organizer.render.product_render import camera_axes
    stones, stone_faces = _annulus(r_min=6.0, r_max=8.0, height=1.0)
    solid, solid_faces = _cube(size=(20.0, 20.0, 2.0))
    bounds = (0, 20, 0, 20, 0, 2)
    hollow = camera_axes(bounds, (stones, stone_faces), avoid_bore=True)[0]
    backed = camera_axes(bounds, (stones, stone_faces), avoid_bore=True,
                         silhouette_arrays=(solid, solid_faces))[0]
    assert not np.allclose(hollow, backed)
    assert np.allclose(backed, [0, 0, 1])


# ---------------------------------------------------------------- STL rolleri

def test_stl_roles_leave_a_cad_assembly_alone():
    """The guard that makes role-splitting safe. Measured on eight real
    catalogue pieces the largest component held 2-23 % of the faces; guessing
    roles there mislabelled hundreds of metal parts as stones."""
    import trimesh
    from catalog_organizer.render.product_render import _split_stl_roles

    a = trimesh.creation.box(extents=(4, 4, 4))
    b = trimesh.creation.box(extents=(4, 4, 4))
    b.apply_translation((20, 0, 0))
    metal, gems = _split_stl_roles(trimesh.util.concatenate([a, b]))
    assert gems is None, "montaj dosyasinda rol tahmini yapilmamali"
    assert metal[1].shape[0] == len(a.faces) + len(b.faces)


def test_stl_roles_split_a_single_body_export():
    """One dominant body plus small extras — the shape of an exported render
    model, where the split was validated against the file's own material
    assignment."""
    import trimesh
    from catalog_organizer.render.product_render import _split_stl_roles

    body = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    stone = trimesh.creation.icosphere(subdivisions=2, radius=1.2)
    stone.apply_translation((0, 0, 11.0))
    engraving = trimesh.creation.box(extents=(4.0, 1.0, 0.05))
    engraving.apply_translation((0, 0, -11.0))

    mesh = trimesh.util.concatenate([body, stone, engraving])
    metal, gems = _split_stl_roles(mesh)
    assert gems is not None, "tas bulunmali"
    assert gems[1].shape[0] == len(stone.faces)
    # the flat engraving is dropped, not coloured and not left as metal
    assert metal[1].shape[0] == len(body.faces)


def test_stl_roles_keep_a_clean_single_part_untouched():
    import trimesh
    from catalog_organizer.render.product_render import _split_stl_roles
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    metal, gems = _split_stl_roles(sphere)
    assert gems is None
    assert metal[1].shape[0] == len(sphere.faces)


# -------------------------------------------- aynı kategori, aynı açı

def _rotate(verts, axis, degrees):
    """Same piece, saved in a different CAD orientation."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
    return verts @ R.T


def _principal(verts):
    centred = verts - verts.mean(axis=0)
    _w, vectors = np.linalg.eigh(centred.T @ centred)
    return vectors


def _bounds_of(verts):
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def test_one_category_is_photographed_from_one_angle():
    """The "bazı ürünler dik bazı ürünler çapraz" report.

    A catalogue is a comparison: two pendants side by side have to be shot the
    same way, or the reader takes the difference in framing for a difference
    in the product. The pose was being chosen per PIECE from whatever its own
    geometry suggested, so the same pendant saved at an angle -- which is
    entirely the designer's habit, not a property of the jewellery -- was
    photographed at that angle.

    The pose is compared in each piece's OWN axes, because that is what a
    viewer sees; the world frame is just how the file happened to be written.
    """
    from catalog_organizer.render.product_render import camera_axes

    verts, faces = _cube(size=(18.0, 30.0, 2.0))
    poses = []
    for angle in (0.0, 23.0, 61.0, -47.0):
        turned = _rotate(verts, (0.3, 0.5, 0.8), angle)
        front, _iso, up = camera_axes(
            _bounds_of(turned), (turned, faces),
            silhouette_arrays=(turned, faces), category="pendant")
        basis = _principal(turned)
        poses.append(np.concatenate([np.abs(front @ basis),
                                     np.abs(up @ basis)]))
    spread = np.asarray(poses).std(axis=0).max()
    assert spread < 0.01, f"same pendant, different angles: spread {spread:.3f}"


def test_a_pendant_hangs_by_its_narrow_end():
    """"Up" is only defined to a sign, and the wrong sign hangs the whole
    catalogue upside down -- consistently, but upside down. A pendant tapers
    toward its bail, so the narrow end is the end it hangs by."""
    from catalog_organizer.render.product_render import flat_pose

    # A wedge: wide at y = 0, narrow at y = 30.
    rows = []
    for i in range(31):
        half = 9.0 * (1.0 - i / 30.0) + 0.5
        rows += [[-half, float(i), 0.0], [half, float(i), 0.0],
                 [-half, float(i), 1.0], [half, float(i), 1.0]]
    _normal, up, _side = flat_pose(np.asarray(rows))
    assert up[1] > 0.9, f"pendant hangs upside down: up = {up.round(2)}"


def test_a_ring_is_shot_down_its_bore_even_with_stones_on_it():
    """A set band and a plain band are the same ring shape.

    Before, the "design" direction came from the gems when a piece had any,
    so two rings of one category arrived turned differently depending on
    whether stones happened to be modelled. The bore is a property of the
    ring; it decides the pose now.
    """
    from catalog_organizer.render.product_render import camera_axes

    ring, faces = _ring_mesh()
    front, _iso, up = camera_axes(
        _bounds_of(ring), (ring, faces), silhouette_arrays=(ring, faces),
        category="ring")
    # The camera sits off the bore axis, not down it and not level with it.
    tilt = np.degrees(np.arccos(abs(float(front @ np.array([0.0, 0.0, 1.0])))))
    assert 20.0 < tilt < 75.0, f"ring shot at {tilt:.0f}° off its bore"
    assert abs(float(up @ np.array([0.0, 0.0, 1.0]))) < 0.4


def test_a_ring_labelled_plate_is_not_forced_down_a_bore():
    """JCAD-000000001 is filed as "ring/other" with no inner diameter at all
    and a 29 x 27 x 5.8 mm box -- a flat plate. The category is a label and
    the geometry is the fact, so a labelled ring with no hole to look through
    is shot like the flat piece it is rather than down whichever axis happened
    to have the largest gap."""
    from catalog_organizer.render.product_render import camera_axes

    verts, faces = _cube(size=(29.0, 27.0, 5.8))
    front, _iso, _up = camera_axes(
        _bounds_of(verts), (verts, faces), silhouette_arrays=(verts, faces),
        category="ring")
    assert abs(float(front @ np.array([0.0, 0.0, 1.0]))) > 0.9


def _ring_mesh(sectors=64, radius=9.0, thickness=2.0, width=5.0):
    """A plain band: bore along +z."""
    angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    rings = []
    for r in (radius, radius + thickness):
        for z in (0.0, width):
            rings.append(np.column_stack([r * np.cos(angles),
                                          r * np.sin(angles),
                                          np.full(sectors, z)]))
    verts = np.vstack(rings)
    faces = []
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
        for i in range(sectors):
            j = (i + 1) % sectors
            faces += [[a * sectors + i, b * sectors + i, b * sectors + j],
                      [a * sectors + i, b * sectors + j, a * sectors + j]]
    return verts, np.asarray(faces)
