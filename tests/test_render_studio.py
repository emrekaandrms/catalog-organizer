"""Tests for the lighting rig.

These exist because the renderer's two worst faults were both invisible: a
misprojected environment still renders, and a missing tone mapping operator
still renders. Nothing failed, nothing warned, and the gold just quietly looked
like brass. Every test here is aimed at one of those silent failures.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")

import pyvista as pv  # noqa: E402

from catalog_organizer.render import materials as mat  # noqa: E402
from catalog_organizer.render import studio  # noqa: E402
from catalog_organizer.render.product_render import _to_polydata  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------- the rig

def test_the_rig_is_a_two_to_one_equirect():
    """The single fact that invalidated the supplied HDRs. VTK reads the
    environment as an equirectangular panorama; an equirect is 2:1. Both
    supplied files were 512x512 and were being wrapped onto the sphere
    wrongly."""
    for arr in (studio.studio_equirect(), studio.gem_equirect()):
        height, width, channels = arr.shape
        assert channels == 3
        assert width == 2 * height


def test_the_rig_has_genuinely_dark_regions():
    """The root cause of the dull metal, and the one that would have bitten at
    any aspect ratio. Polished metal is a mirror; a mirror in a uniformly lit
    room is a uniformly lit surface. The supplied metal HDR had 81% of its
    pixels above 1.0 and a mean of 1.84 — a lightbox with no black cards in
    it, and nothing for the bright bands to read against."""
    lum = studio.studio_equirect() @ LUMA
    assert (lum < 0.2).mean() > 0.2, "rig has no dark surround left"
    assert lum.mean() < 12.0


def test_the_rig_carries_real_high_dynamic_range():
    """If the panels ever fall to LDR the highlights stop rolling off and the
    tone mapping operator has nothing left to do."""
    lum = studio.studio_equirect() @ LUMA
    assert lum.max() > 15.0


def test_the_gem_rig_adds_point_sources_the_metal_rig_lacks():
    """Scintillation comes from small intense sources. A softbox reflected in
    a 1 mm facet is one dull grey square."""
    metal = studio.studio_equirect() @ LUMA
    gem = studio.gem_equirect() @ LUMA
    assert gem.max() > 5 * metal.max()
    assert (gem > 50).mean() < 0.05, "sparkles should be small, not panels"


def test_the_gem_rig_is_reproducible():
    """Sparkle positions are random but must not move between two renders of
    the same product, or its two catalogue views disagree."""
    assert np.array_equal(studio.gem_equirect(), studio.gem_equirect())


def test_metal_exposure_is_low_enough_to_keep_colour():
    """ACES desaturates as it approaches white. Measured on a real ring:
    exposure 1.00 -> saturation 0.258, exposure 0.30 -> 0.357."""
    assert studio.EXPOSURE_METAL < 0.5


def test_gem_exposure_suits_a_refracted_stone():
    """This value has now moved in both directions, and each move was the
    material model changing underneath it. Under the layered model most of a
    stone was the dark rig seen between facets and it needed MORE than metal
    (0.80). With refraction most of a stone is the environment seen THROUGH
    it, so it needs much less. Swept on a real 4.5 mm catalogue stone: 0.45
    gave a 5th-percentile of 157 with local contrast 2.00 — washed out; 0.12
    gave 57 / 3.28 — flashing but going dark again."""
    assert 0.12 < studio.EXPOSURE_GEM < 0.35


def test_the_rig_lights_the_camera_axis():
    """A flat piece mirrors whatever is behind the camera. With nothing there
    a stamped pendant rendered at a median luminance of 37 face-on while its
    three-quarter view was fine; the on-axis card took it to 187 and moved a
    ring only from 163 to 177."""
    lit = studio.studio_equirect() @ LUMA
    dark = studio.studio_equirect(frontal=0.0) @ LUMA
    height, width = lit.shape
    # -z is toward the camera, which is the centre column of the equirect.
    axis = slice(int(width * 0.70), int(width * 0.80))
    band = slice(int(height * 0.40), int(height * 0.60))
    assert lit[band, axis].mean() > dark[band, axis].mean() * 1.5


# ---------------------------------------------------------------- tone mapping

@pytest.mark.gpu
def test_tone_mapping_is_actually_installed_on_the_renderer():
    """The whole fault in one assertion: the pipeline had no tone mapping pass
    at all, so every reflected value above 1.0 clipped flat to white."""
    from vtkmodules.vtkRenderingOpenGL2 import vtkToneMappingPass

    plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
    studio.install_tone_mapping(plotter, 0.35)
    installed = plotter.renderer.GetPass()
    assert isinstance(installed, vtkToneMappingPass)
    assert installed.GetExposure() == pytest.approx(0.35)
    assert installed.GetUseACES()
    plotter.close()


@pytest.mark.gpu
def test_tone_mapping_wraps_an_existing_pass_rather_than_dropping_it():
    """SSAO and friends install their own pass. Tone mapping has to sit
    outside whatever is already there, not replace it."""
    from vtkmodules.vtkRenderingOpenGL2 import vtkRenderStepsPass

    plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
    first = vtkRenderStepsPass()
    plotter.renderer.SetPass(first)
    studio.install_tone_mapping(plotter, 0.35)
    assert plotter.renderer.GetPass().GetDelegatePass() is first
    plotter.close()


@pytest.mark.gpu
def test_environment_is_oriented_from_the_placed_camera():
    """Without this the lighting depends on how the piece happened to be
    oriented in CAD, so two rings off the same tray photograph differently."""
    plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
    plotter.camera.position = (0.0, 0.0, 10.0)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    studio.orient_environment_to_camera(plotter)
    assert np.allclose(plotter.renderer.GetEnvironmentUp(), (0, 1, 0), atol=1e-6)
    # Looking down -Z with up +Y, the camera's right is +X.
    assert np.allclose(plotter.renderer.GetEnvironmentRight(), (1, 0, 0),
                       atol=1e-6)
    plotter.close()


# ---------------------------------------------------------------- compositing

def test_background_is_a_gradient_not_a_flat_white_page():
    """A polished edge against flat white has nothing to be seen against."""
    bg = studio.gradient_background(100, 40)
    assert bg.shape == (100, 40, 3)
    assert bg[0].mean() > bg[-1].mean() + 15


def test_composite_over_respects_alpha():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = (200, 100, 50, 255)          # opaque
    rgba[1, 1] = (200, 100, 50, 0)            # fully transparent
    bg = np.full((2, 2, 3), 240.0)
    out = studio.composite_over(rgba, bg)
    assert tuple(out[0, 0]) == (200, 100, 50)
    assert tuple(out[1, 1]) == (240, 240, 240)


def test_contact_shadow_darkens_below_the_piece_and_not_above():
    """Grounding the product. Against a plain gradient an unshadowed piece
    reads as a cut-out pasted onto the page."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[60:120, 70:130] = 255               # a block in the upper middle
    shade = studio.contact_shadow(alpha)[..., 0]
    assert shade.shape == (200, 200)
    assert shade.max() <= 1.0
    # It is a *contact* shadow: the core sits at the piece's lowest edge
    # (measured darkest at row 117 for a block ending at 119), not adrift
    # below it. An earlier tuning put it a tenth of a frame lower and it read
    # as a separate blob the ring was hovering over.
    contact = shade[110:135].min()
    assert contact < 0.97, "no shadow was laid under the piece"
    assert shade[:55].min() > 0.999, "shadow leaked above the piece"
    assert shade[160:].min() > 0.999, "shadow drifted away from the piece"


def test_contact_shadow_is_subtle():
    """It is a grounding cue, not a second subject. A heavy blob under a ring
    reads as a separate object."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[60:120, 70:130] = 255
    assert studio.contact_shadow(alpha).min() > 0.6


def test_contact_shadow_survives_an_empty_frame():
    shade = studio.contact_shadow(np.zeros((32, 32), dtype=np.uint8))
    assert np.all(shade == 1.0)


def test_floor_reflection_appears_below_the_piece_only():
    """Grounding, part two. A shadow says there is a floor; a reflection says
    what the floor is made of, and without one the page reads as a flat colour
    swatch the product was pasted onto."""
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[60:120, 70:130] = (200, 160, 90, 255)
    refl = studio.floor_reflection(rgba)
    assert refl.shape == rgba.shape
    assert refl[:119, :, 3].max() == 0, "reflection leaked above the piece"
    assert refl[120:, :, 3].max() > 0, "no reflection below the piece"


def test_floor_reflection_fades_with_distance():
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[40:120, 70:130] = (200, 160, 90, 255)
    alpha = studio.floor_reflection(rgba)[..., 3].astype(float)
    near = alpha[122:132].max()
    far = alpha[170:180].max()
    assert near > far, "reflection should weaken further from the contact line"
    assert near < 255 * 0.5, "a floor reflection is never as bright as the piece"


def test_floor_reflection_survives_an_empty_frame():
    assert studio.floor_reflection(
        np.zeros((32, 32, 4), dtype=np.uint8))[..., 3].max() == 0


def test_downsample_halves_the_frame():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    assert studio.downsample(img, 2).shape == (40, 40, 3)
    assert studio.downsample(img, 1).shape == (80, 80, 3)


# ---------------------------------------------------------------- gems

def _brilliant(diameter=10.0, sectors=16):
    """A round brilliant at standard proportions, as a test subject.

    It exists because the catalogue cannot supply one: the largest stone in
    any real file is 2.2 mm, perhaps ten pixels across, so a gem material
    cannot be judged on the products themselves. Facets big enough to measure
    have to be built.
    """
    r, gh, ch, pd = diameter / 2.0, diameter * 0.03, diameter * 0.145, diameter * 0.43
    ang = np.arange(sectors) * 2 * np.pi / sectors
    half = ang + np.pi / sectors

    def ring(radius, z, angles):
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles),
                                np.full(len(angles), z)])

    verts = np.vstack([
        ring(r, gh / 2, ang), ring(r, -gh / 2, ang),
        ring(r * 0.80, gh / 2 + ch * 0.55, half), ring(r * 0.56, gh / 2 + ch, ang),
        ring(r * 0.52, -gh / 2 - pd * 0.55, half), [[0.0, 0.0, -gh / 2 - pd]]])
    n = sectors
    GT, GB, ST, TB, PM, CU = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[GT + i, GB + i, GB + j], [GT + i, GB + j, GT + j],
                  [GT + i, ST + i, TB + i], [GT + j, TB + j, ST + i],
                  [ST + i, TB + j, TB + i],
                  [GB + i, PM + i, GB + j], [GB + j, PM + i, CU],
                  [GB + i, CU, PM + i]]
        if i >= 2:
            faces += [[TB + 0, TB + i - 1, TB + i]]
    f = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(verts, np.column_stack([np.full(len(f), 3), f]).ravel())


def _gem_luminance(up=(0.0, 0.0, 1.0), direction=(0.25, 0.55, 0.80),
                   stone="white"):
    from catalog_organizer.render.product_render import _render_pass

    verts, faces = _real_stone()
    gem = _to_polydata((verts, faces))
    d = np.asarray(direction, float)
    rgba = _render_pass(gem, kind="gem", arrays=(verts, faces),
                        metal=mat.metal("yellow_gold"),
                        stone=mat.stone(stone), bounds=gem.bounds,
                        direction=d / np.linalg.norm(d),
                        up=np.asarray(up, float), zoom=0.60, resolution=260)
    obj = rgba[..., 3] > 200
    assert obj.sum() > 500, "nothing was drawn"
    return rgba, obj


def _local_contrast(rgba, obj) -> float:
    """Mean neighbouring-pixel difference over the stone.

    This, not a percentile spread, is what a flash physically IS: two adjacent
    facets sending light in completely different directions. A percentile
    spread saturates once the stone is bright -- it reads 110 on a stone whose
    facets are plainly blazing -- whereas this tracks the thing being looked
    at, and it is the number every gem model in this project was judged on.
    """
    lum = rgba[..., :3].astype(float) @ LUMA
    gy, gx = np.gradient(lum)
    return float(np.sqrt(gx ** 2 + gy ** 2)[obj].mean())


def test_the_gem_shows_facets_rather_than_one_flat_tone():
    """The "it looks like white ceramic" report. The first model gave a gem a
    light body colour and metallic 0.05 — almost entirely diffuse, and diffuse
    is uniform by definition: no facets at all."""
    rgba, obj = _gem_luminance()
    contrast = _local_contrast(rgba, obj)
    # Measured on this stone at this resolution: 12.02 with the facet trace,
    # 8.06 with it switched off. The threshold sits in that gap, so switching
    # the trace off fails this test rather than merely lowering a number.
    assert contrast > 10.0, (
        f"gem is one flat tone: local contrast {contrast:.2f}")


def test_the_gem_does_not_go_black_from_either_angle():
    """The other half of the same report — "reflections make the stone black",
    worst of all from straight overhead. Tracing the real facets is what fixed
    it: light that enters finds its way back out instead of a single mirror
    direction landing on the rig's dark surround and returning nothing."""
    for up, direction in (((0, 0, 1), (0.25, 0.55, 0.80)),
                          ((0, 1, 0), (0.02, 0.05, 1.00))):
        rgba, obj = _gem_luminance(up=up, direction=direction)
        lum = rgba[obj][:, :3].astype(float) @ LUMA
        assert np.percentile(lum, 5) > 30, (
            f"stone reads black from {direction}: p5 = "
            f"{np.percentile(lum, 5):.1f}")


def test_a_ruby_reads_red_and_not_pink():
    """The "kirmizi renk ruby tasi pembe gorunuyor" report, measured.

    Pink is not a hue -- it is red with green and blue left in. The mean pixel
    before the absorption density went in was (229, 147, 149): green and blue
    almost equal and both more than half of red. A ruby's colour comes from
    the green being GONE. Same for the sapphire, which was reported as reading
    like an amethyst for the mirror-image reason: its red had survived.
    """
    for stone, bright, dim in (("red", 0, (1, 2)), ("blue", 2, (0, 1))):
        rgba, obj = _gem_luminance(stone=stone)
        mean = rgba[obj][:, :3].astype(float).mean(axis=0)
        lead = mean[bright]
        rest = max(mean[i] for i in dim)
        assert lead / max(rest, 1e-6) > 2.5, (
            f"{stone} stone is washed out: mean pixel {mean.round(0)}")


def test_the_white_stone_is_not_clipped_flat_at_the_top():
    """The other half of the same report: "beyaz tasta fasetler hic belli
    olmuyor".

    A facet is a DIFFERENCE between two neighbouring directions, so a stone
    whose tones are all pressed against 255 has nowhere to put one. With the
    gain the tracer inherited from the shader before it, the median sat at 247
    and 18% of the stone was pure white; local contrast measured 11.4. The
    gain was re-swept against the tracer (see studio.GEM_LOOKUP_GAIN) and the
    same stone now measures a median of 235 with 6% pure white and a contrast
    of 21.6 -- brightness barely moved, the flash nearly doubled.
    """
    rgba, obj = _gem_luminance(stone="white")
    px = rgba[obj][:, :3].astype(float)
    pinned = float((px >= 254).all(axis=1).mean())
    assert pinned < 0.12, f"{pinned:.0%} of the stone is pure white"
    assert np.median(px @ LUMA) > 180, "stone is no longer bright"
    assert _local_contrast(rgba, obj) > 16.0, "the flash went with the gain"


def test_tracing_the_real_facets_beats_every_approximation_tried():
    """The measurement that settles it, on a real catalogue stone. Local
    contrast -- neighbouring-pixel difference, which is what a flash IS:

        layered specular, no interior        2.30
        analytic cone proxy                  1.15   (converges: WORSE)
        screen-space back faces              4.36
        real facet planes, 4 bounces         8.06

    A cone is smooth, so rays inside meet the same normal and leave together.
    Real facets make them diverge, and that divergence is the sparkle."""
    rgba, obj = _gem_luminance()
    contrast = _local_contrast(rgba, obj)
    assert contrast > 10.0, f"stone has no flash left: {contrast:.2f}"


def _supplied_hdrs_present() -> bool:
    """The two optional showroom/studio HDRs are NOT shipped with the project (they were a third
    party's); a machine that has them dropped into assets/hdr/ gets the extra tests."""
    from catalog_organizer.render.product_render import GEM_HDR, HDR_DIR, METAL_HDR
    return (HDR_DIR / GEM_HDR).exists() and (HDR_DIR / METAL_HDR).exists()


_NEEDS_HDRS = pytest.mark.skipif(not _supplied_hdrs_present(),
                                 reason="optional assets/hdr/*.hdr are not present (none are shipped)")


@_NEEDS_HDRS
def test_the_gem_lookup_uses_the_supplied_showroom_hdr():
    """The supplied env-gem-1.hdr is rejected as an IBL environment because it
    is 512x512 and an equirect is 2:1 — but the gem shader computes its own
    equirect lookup, so the projection is this module's decision there, and
    the file wins on the thing that matters for a refracting stone: real
    structure. A stone is a window; a smooth room seen through a window is a
    smooth blob. Measured on a 4.5 mm catalogue stone, the file raised local
    contrast — which is what sparkle is — from 2.84 to 4.31."""
    from catalog_organizer.render.product_render import GEM_HDR, HDR_DIR

    assert (HDR_DIR / GEM_HDR).exists(), "the showroom HDR is gone"
    lookup = studio.gem_lookup()
    assert lookup is not studio.environment("gem"), "fell back to the rig"


@_NEEDS_HDRS
def test_the_gem_lookup_is_colourless():
    """The showroom has a teal cast, and a colourless diamond showing teal is
    wrong twice: the cast is the room's rather than the stone's, and a
    coloured stone takes its colour from the transmission tint anyway.
    Desaturating keeps every bit of the structure and none of the cast."""
    from vtkmodules.util import numpy_support as ns

    image = studio.gem_lookup().GetInput()
    arr = ns.vtk_to_numpy(image.GetPointData().GetScalars()).astype(float)
    assert np.allclose(arr[:, 0], arr[:, 1]) and np.allclose(arr[:, 1], arr[:, 2])


def test_the_gem_lookup_gain_sits_between_dark_and_washed_out():
    """Brightness and sparkle trade against each other the whole way down this
    sweep, taken on the 4.5 mm stone (5th-pct / median / local contrast):
    gain 3 gave 4 / 71 / 5.21, gain 12 gave 63 / 241 / 3.39, gain 70 gave
    248 / 255 / 0.29 — a solid white blob."""
    assert 5.0 < studio.GEM_LOOKUP_GAIN < 15.0


def _real_stone():
    """The largest stone out of a real catalogue file, welded and isolated.

    Not the synthetic brilliant: the whole point of the facet tracer is that
    it runs on the customer's own geometry, and the geometry is where the
    surprises live -- a Brep writes a separate render mesh per face, so
    without welding every FACET reads as its own body, and the windings do not
    agree between them either.
    """
    from pathlib import Path as _Path

    import pytest as _pytest

    from catalog_organizer.render import facets
    from catalog_organizer.render.product_render import load_source_arrays

    source = _Path("samples/sampleA_12.3dm")
    if not source.exists():
        _pytest.skip("catalogue sample not present")
    _metal, gem = load_source_arrays(source)
    bodies = facets.components(*gem)
    biggest = max(bodies, key=lambda c: float(c.extents.max()))
    return np.asarray(biggest.vertices), np.asarray(biggest.faces)


def test_facet_planes_actually_bound_the_stone():
    """The bug this catches cost a whole render: a Brep writes one render mesh
    per face and their windings disagree, so a third of the facet normals came
    out pointing INTO the stone. Vertices then sat up to 2.0 units -- two full
    stone radii -- on the wrong side of their own facet's plane, and a ray
    traced against that set leaves on its first step.

    A cut stone is convex, so every vertex must lie on or inside every plane.
    """
    from catalog_organizer.render import facets

    verts, faces = _real_stone()
    shape = facets.extract(verts, faces)
    basis = facets.local_basis(shape.axis)
    local = (verts - shape.centre) @ basis.T / shape.radius
    offsets = np.einsum("ij,ij->i", shape.normals, shape.points)
    outside = (local @ shape.normals.T - offsets).max()
    assert outside < 1e-3, f"vertices sit {outside:.4f} outside their own planes"


def test_a_brilliant_reduces_to_a_workable_number_of_planes():
    """Every facet is baked into the GLSL as a constant and the intersection is
    unrolled over the lot, which only works because a cut stone is a small
    number of planes. A round brilliant is 57 facets plus a girdle."""
    from catalog_organizer.render import facets

    shape = facets.extract(*_real_stone())
    assert 40 < len(shape) < 400
    assert np.allclose(np.linalg.norm(shape.normals, axis=1), 1.0, atol=1e-6)


def test_every_stone_gets_its_own_frame():
    """One compiled shader serves the whole piece; what tells it where each
    stone sits and how it is turned is a per-actor uniform. If the frames were
    ever fewer than the stones, some would be traced in another's pose."""
    from pathlib import Path as _Path

    from catalog_organizer.render import facets
    from catalog_organizer.render.product_render import load_source_arrays

    source = _Path("samples/sampleA_42.3dm")
    if not source.exists():
        pytest.skip("catalogue sample not present")
    _metal, gem = load_source_arrays(source)
    frames = facets.stone_frames(*gem)
    assert len(frames) == len(facets.components(*gem))
    assert all(f["radius"] > 0 for f in frames)
    for frame in frames[:5]:
        basis = frame["basis"]
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)


def test_the_shader_bakes_the_facets_in_as_constants():
    from catalog_organizer.render import facets, gem_trace

    shape = facets.extract(*_real_stone())
    code = gem_trace.build(shape, dispersion=0.0)
    assert f"vec3[{len(shape)}]" in code, "plane arrays were not emitted"
    assert "uStoneRight" in code and "uStoneRadius" in code
    assert "refract(" in code and "reflect(" in code       # TIR path


def test_dispersion_traces_one_index_per_channel():
    """The construction:
        indices = (n - .5*d, n, n + .5*d)
    Three traces, three indices; that separation IS the coloured fire."""
    from catalog_organizer.render import facets, gem_trace

    shape = facets.extract(*_real_stone())
    with_fire = gem_trace.build(shape, ior=2.417, dispersion=0.044)
    without = gem_trace.build(shape, ior=2.417, dispersion=0.0)
    assert "vec3(2.395000, 2.417000, 2.439000)" in with_fire
    assert "vec3(2.417000, 2.417000, 2.417000)" in without
    assert "ch < 3" in with_fire and "ch < 1" in without   # 3x the cost


def test_absorption_is_the_complement_of_the_body_colour():
    """Their `uAbsorption = (1 - r, 1 - g, 1 - b)`, times a density.

    The SHAPE is theirs and is what this pins: a white stone absorbs nothing,
    and a sapphire absorbs nearly all the red while barely touching the blue.
    The density is a separate, measured scalar -- see `absorption_from` for
    why the raw coefficient is too dilute over the path lengths this geometry
    produces -- so the test reads the ratio, which the density cannot move.
    """
    from catalog_organizer.render.gem_trace import (ABSORPTION_DENSITY,
                                                    absorption_from)

    assert absorption_from("#FFFFFF") == (0.0, 0.0, 0.0)
    red, green, blue = absorption_from("#1838D8")
    assert red > 0.85 * ABSORPTION_DENSITY
    assert blue < 0.2 * ABSORPTION_DENSITY
    assert red / blue > 4.0            # the ratio is the stone's colour
    assert ABSORPTION_DENSITY > 1.0    # a dilute stone renders pale


def test_crowded_pieces_trace_fewer_bounces():
    """Their viewer drops this too (`decorationCount >= stonesThreshold`) and
    for the same reason: a pave stone is ten pixels across, nobody reads its
    interior, and every one of them otherwise pays a full trace per bounce
    per channel."""
    from catalog_organizer.render import gem_trace

    assert gem_trace.CROWDED_RAY_BOUNCES < gem_trace.RAY_BOUNCES
    assert gem_trace.CROWDED_STONE_COUNT > 1


# ---------------------------------------------------------------- end to end

def _gold_sphere_luminance() -> np.ndarray:
    """Render a gold ball under the rig and return its object pixels' luma."""
    from catalog_organizer.render.product_render import _render_pass

    ball = pv.Sphere(radius=5.0, theta_resolution=64, phi_resolution=64)
    rgba = _render_pass(ball, kind="metal", metal=mat.metal("yellow_gold"),
                        stone=mat.stone("white"), bounds=ball.bounds,
                        direction=np.array([0.0, 0.0, 1.0]),
                        up=np.array([0.0, 1.0, 0.0]), zoom=0.62,
                        resolution=220)
    obj = rgba[..., 3] > 200
    assert obj.sum() > 500, "nothing was drawn"
    return rgba[obj][:, :3].astype(float) @ LUMA


@pytest.mark.gpu
def test_polished_gold_shows_reflection_structure():
    """The end-to-end guard on the whole fault. A mirror ball under a rig with
    dark areas has a wide luminance spread; under a uniform lightbox it
    collapses to one value, which is what the old environment produced and
    what made the metal read as painted."""
    lum = _gold_sphere_luminance()
    spread = np.percentile(lum, 90) - np.percentile(lum, 10)
    assert spread > 60, f"metal is flat: p90-p10 = {spread:.1f}"


@pytest.mark.gpu
def test_polished_gold_is_not_blown_out():
    """The other half: with no tone mapping the reflected rig clipped, and a
    large share of the metal became pure white with no colour left in it."""
    lum = _gold_sphere_luminance()
    assert (lum >= 254).mean() < 0.10


def test_retuning_the_look_renames_every_cached_frame():
    """The stale-cache trap, which cost an hour when it was sprung.

    The gem gain and the absorption density were retuned, every measurement
    confirmed the new look -- and the app kept serving the frames it had
    already written under the same filename, so the retune looked like it had
    done nothing. The cache key now carries a hash of the constants the look
    depends on, which makes forgetting the version bump harmless.
    """
    from catalog_organizer.render import gem_trace, product_render, studio

    before = product_render._look_key()
    original = studio.GEM_LOOKUP_GAIN
    try:
        studio.GEM_LOOKUP_GAIN = original + 1.0
        assert product_render._look_key() != before
    finally:
        studio.GEM_LOOKUP_GAIN = original
    assert product_render._look_key() == before

    original = gem_trace.ABSORPTION_DENSITY
    try:
        gem_trace.ABSORPTION_DENSITY = original + 0.5
        assert product_render._look_key() != before
    finally:
        gem_trace.ABSORPTION_DENSITY = original
    assert product_render._look_key() == before


def _revolved_puck(sectors=64, rings=12):
    """A stone-shaped placeholder: a flat table over a smooth revolve.

    This is what JCAD-000000140 actually carries in its seats -- a puck the
    designer dropped in to mark where a stone goes. It is built here rather
    than read from that file so the test does not depend on a path outside the
    repository, and it is deliberately smooth: no facets anywhere.
    """
    angles = np.linspace(0.0, 2 * np.pi, sectors, endpoint=False)
    verts = [[0.0, 0.0, 1.0]]
    for k in range(rings + 1):
        t = k / rings
        radius = 0.55 + 0.45 * np.sin(t * np.pi / 2)      # smooth shoulder
        height = 1.0 - 1.8 * t
        verts += [[radius * np.cos(a), radius * np.sin(a), height]
                  for a in angles]
    verts.append([0.0, 0.0, -1.0])
    faces = []
    for i in range(sectors):
        j = (i + 1) % sectors
        faces.append([0, 1 + i, 1 + j])
        for k in range(rings):
            a, b = 1 + k * sectors, 1 + (k + 1) * sectors
            faces += [[a + i, b + i, b + j], [a + i, b + j, a + j]]
        last = 1 + rings * sectors
        faces.append([last + i, len(verts) - 1, last + j])
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def test_the_standard_brilliant_really_is_one():
    """The substitute has to be a cut, not a decoration."""
    from catalog_organizer.render import facets

    shape = facets.standard_brilliant()
    assert 55 <= len(shape) <= 90, f"{len(shape)} planes is not a brilliant"
    assert facets.cut_coverage(*facets._brilliant_mesh()) > 0.9
    # Table: one plane square on the axis, and the widest facet on the stone.
    table = shape.normals @ np.array([0.0, 0.0, 1.0])
    assert table.max() > 0.999, "no table facet"
    # Nothing sits outside the normalised body it is supposed to bound.
    verts, _faces = facets._brilliant_mesh()
    local = (verts - shape.centre) @ facets.local_basis(shape.axis).T
    local /= shape.radius
    outside = (local @ shape.normals.T
               - (shape.points * shape.normals).sum(axis=1)).max()
    assert outside < 1e-3, f"vertices sit {outside:.3f} outside their planes"


def test_a_real_cut_keeps_its_own_facets():
    """Substitution must never touch a stone that was properly cut in CAD."""
    from catalog_organizer.render import facets

    verts, faces = _real_stone()
    assert facets.cut_coverage(verts, faces) > 0.9
    shape, borrowed = facets.plane_set(verts, faces)
    assert not borrowed
    assert len(shape) == len(facets.extract(verts, faces))


def test_a_smooth_placeholder_borrows_a_brilliant():
    """The "fasetleri begenmedim" report on JCAD-000000140, run to ground.

    Its stones are not cut stones. The wireframe is a flat table over a
    smoothly revolved cone, and the measurement says the same thing: the 60
    largest planes carry 37% of its surface where a real brilliant's carry
    97%, and 364 of its 365 planes carry under 1% each. Traced honestly that
    is a polished bowl with a band of stripes where the revolve is tessellated
    -- which is exactly what the geometry is. No shader can reflect a facet
    that the CAD does not have, so the cut is borrowed: one standard
    brilliant is reused across the catalogue.

    The position, axis and size stay the placeholder's own: those are real.
    """
    from catalog_organizer.render import facets

    verts, faces = _revolved_puck()
    assert facets.cut_coverage(verts, faces) < 0.75
    shape, borrowed = facets.plane_set(verts, faces)
    assert borrowed
    ideal = facets.standard_brilliant()
    assert np.allclose(shape.normals, ideal.normals)
    own = facets.extract(verts, faces)
    assert np.allclose(shape.centre, own.centre)
    assert shape.radius == own.radius
