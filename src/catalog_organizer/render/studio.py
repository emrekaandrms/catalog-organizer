"""The lighting rig: a procedural jewellery studio, tone mapping, compositing.

Why this is built rather than loaded from an HDR file
----------------------------------------------------
Two Radiance .hdr files were supplied as the reference lighting. Measuring them
settled the question:

    env-metal-6.hdr   512x512   mean 1.84   max 7.3    81% of pixels above 1.0
    env-gem-1.hdr     512x512   mean 0.67   max 270     3% of pixels above 1.0

Both are square. An equirectangular panorama -- the projection
``vtkRenderer::SetEnvironmentTexture`` expects -- is always 2:1, so neither file
was ever mapped onto the sphere correctly.

The metal one is the more serious problem, and it would have been a problem at
any aspect ratio: with a mean of 1.84 and four fifths of its pixels above 1.0
it is a uniformly lit box. Polished metal is a mirror, and a mirror in a
uniformly bright room is a uniformly bright surface. That is why the gold read
as painted plastic. Jewellery photographers surround a piece with bright panels
AND black cards for exactly this reason: the dark gaps are what make the bright
bands read as polish.

So the rig here is built. It is a true 2:1 equirect of a dark studio holding
five soft panels, and being procedural buys two things a photograph cannot: it
is deterministic, so every product in a catalogue is lit identically, and it
can be rotated to follow the camera, so a piece is lit the same way regardless
of which way its modeller happened to orient it.

Tone mapping
------------
The rig peaks around 35x and polished metal reflects it. With no tone mapping
operator everything above 1.0 clips flat, destroying precisely the highlight
rolloff that makes metal look like metal. VTK ships ACES filmic; the three.js
configurators this was measured against use the same curve, which is where
their soft shoulder into white comes from.

Exposure is deliberately low. ACES desaturates as it approaches white, so an
over-exposed gold turns pale cream. Measured on a real ring: exposure 1.00 gave
mean object saturation 0.258, exposure 0.30 gave 0.357.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyvista as pv

# Bump when the rig changes, so cached PNGs of the previous look are dropped.
STUDIO_VERSION = 1

# Exposure fed to the ACES operator. Swept 0.20 / 0.30 / 0.45 / 0.65 / 1.00 on
# a real ring; saturation falls monotonically as it rises and the gold is pale
# cream by 0.65. 0.35 holds the metal in the saturated part of the curve while
# the brightest panels still reach the shoulder.
EXPOSURE_METAL = 0.35
# Gems refract now, so most of a stone is the environment seen THROUGH it
# rather than the dark rig seen between facets, and it needs far less
# exposure than the layered model did. Swept on a real 4.5 mm catalogue stone:
# 0.45 gave a 5th-percentile of 157 with local contrast 2.00 (washed out),
# 0.12 gave 57 / 3.28 (flashing but going dark again). 0.20 measured 89 / 2.84.
EXPOSURE_GEM = 0.20

# Midtone contrast of the filmic curve. VTK's default is 1.677; the highlight
# white point (HdrMax) was swept alongside it and moved object saturation by
# 0.005, while contrast moved it from 0.320 to 0.349. Gold loses its colour as
# it approaches white, so the curve that holds it below that point longer is
# the one that keeps it gold.
CURVE_CONTRAST = 2.05

# The catalogue background. Not white: against a flat white page a polished
# edge has nothing to be seen against and the piece loses its silhouette.
BG_TOP = "#F6F6F4"
BG_BOTTOM = "#D9D9D7"


# -- the rig -----------------------------------------------------------------

def _panel(directions, center, right_hint, half_a, half_b, feather=0.35):
    """A soft-edged rectangular panel on the sphere.

    Rectangular rather than round on purpose: a softbox is a rectangle, and the
    straight-edged highlight it lays along a band is a large part of what reads
    as a studio photograph rather than a CAD viewport.
    """
    c = np.asarray(center, float)
    c = c / np.linalg.norm(c)
    r = np.asarray(right_hint, float)
    r = r - c * np.dot(r, c)
    r = r / np.linalg.norm(r)
    u = np.cross(c, r)

    a = np.degrees(np.arcsin(np.clip(directions @ r, -1.0, 1.0)))
    b = np.degrees(np.arcsin(np.clip(directions @ u, -1.0, 1.0)))

    def edge(t, half):
        return np.clip((half - np.abs(t)) / max(half * feather, 1e-3) + 0.5,
                       0.0, 1.0)

    m = edge(a, half_a) * edge(b, half_b)
    m = m * m * (3.0 - 2.0 * m)                       # smoothstep
    # Behind the panel's own plane there is no panel.
    return np.where(directions @ c > 0.0, m, 0.0)


def sphere_directions(width: int, height: int) -> np.ndarray:
    """Equirect pixel grid -> unit directions. Row 0 is the zenith."""
    j, i = np.mgrid[0:height, 0:width]
    theta = (j + 0.5) / height * np.pi
    phi = (i + 0.5) / width * 2.0 * np.pi
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), np.cos(theta), s * np.sin(phi)], axis=-1)


def studio_equirect(width: int = 1024, height: int = 512, *, key: float = 22.0,
                    fill: float = 5.0, strip: float = 14.0, rim: float = 9.0,
                    bounce: float = 1.3, frontal: float = 1.2,
                    base: float = 0.035, top_lift: float = 0.09) -> np.ndarray:
    """The metal rig: five panels in a dark room, as a float32 2:1 equirect."""
    d = sphere_directions(width, height)
    # The dark surround -- the black cards. Only the ceiling lifts, and only a
    # little; the sides and back must stay dark or the panels stop reading.
    lum = base + top_lift * np.clip(d[..., 1], 0.0, 1.0) ** 2
    out = np.repeat(lum[..., None], 3, axis=-1).astype(np.float32)

    def add(mask, intensity, tint):
        out[:] += mask[..., None] * intensity * np.asarray(tint, np.float32)

    # Directions are CAMERA-relative once `orient_environment_to_camera` has
    # run, and the sign that matters is z: **-z points at the camera, +z away
    # from it**. That was established by measurement, not by reading it off
    # the axes -- an on-axis panel placed at +z lifted a flat pendant's median
    # luminance from 36.9 to 38.0, and the same panel at -z lifted it to 230.
    # The panels below are named for what they actually do under that sign.

    # The brightest source sits high and BEHIND, over the far shoulder. On
    # jewellery a back-top light is what draws the bright contour down a
    # polished edge; it does most of the work of making metal read as metal.
    add(_panel(d, (-0.45, 0.85, 0.55), (1, 0, 0), 40, 28), key,
        (1.00, 0.97, 0.92))
    # Cool kicker, low and behind to the right: opens the shadow side.
    add(_panel(d, (0.80, 0.18, 0.60), (0, 1, 0), 32, 40), fill,
        (0.93, 0.96, 1.00))
    # Overhead strip: long and thin. This is the one that lays the sweeping
    # highlight down the length of a band.
    add(_panel(d, (0.05, 1.00, 0.10), (1, 0, 0), 78, 9), strip,
        (1.00, 0.99, 0.97))
    # Front-upper-right, the modelling light. Despite the name this is the
    # panel a photographer would call the key: it is the one in front.
    add(_panel(d, (0.25, 0.55, -0.85), (1, 0, 0), 26, 18), rim,
        (0.97, 0.98, 1.00))
    # Bounce off the table, so undersides read as dark rather than dead.
    add(_panel(d, (0.00, -0.85, 0.45), (1, 0, 0), 60, 34), bounce,
        (1.00, 0.99, 0.96))
    # On-axis card, straight down the lens. Every other panel is off to a side,
    # overhead or behind, so a FLAT piece -- a stamped pendant, a signet face --
    # mirrors the one place nothing was lighting: straight back past the
    # camera. Such pieces rendered almost black face-on while their
    # three-quarter view was fine. Measured on a flat pendant and a ring
    # together: 1.2 takes the pendant from 37 to 187 while the ring only moves
    # 163 to 177, and neither gains a single blown pixel. Stronger starts
    # flattening the curved pieces, which is exactly what a bounce card does
    # when you push it.
    add(_panel(d, (0.00, 0.10, -1.00), (1, 0, 0), 46, 34), frontal,
        (1.00, 0.99, 0.98))
    return out


def gem_equirect(width: int = 1024, height: int = 512, *, count: int = 26,
                 sparkle: float = 260.0, seed: int = 7) -> np.ndarray:
    """The gem rig: the same room, dimmer, plus small very bright sources.

    Scintillation comes from point sources, not panels. A softbox reflected in
    a 1 mm facet is one dull grey square; a bright point in the same facet is a
    spark. The panels stay, so the stone still sits in the same room as the
    metal it is set into.

    `seed` is fixed: the sparkle positions are random but must not move between
    two renders of the same product.
    """
    # The floor bounce is far stronger here than in the metal rig. A gem's
    # pavilion is seen through the table from below, so what it returns is
    # whatever lies under the stone -- and under a dark rig that is nothing.
    # Raising it lifted the table's median from 6 to 11 while facet spread
    # only fell from 242 to 238. A jeweller's light tent has a white floor for
    # exactly this reason.
    out = studio_equirect(width, height, key=16.0, fill=3.0, strip=9.0,
                          rim=6.0, bounce=14.0, frontal=1.0, base=0.05,
                          top_lift=0.05)
    d = sphere_directions(width, height)
    rng = np.random.default_rng(seed)
    white = np.float32([1.0, 1.0, 1.0])
    for _ in range(count):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        if v[1] < -0.2:                   # keep them above the horizon
            v[1] = abs(v[1])
        out += (_panel(d, v, np.cross(v, (0.0, 1.0, 1e-3)), 2.4, 2.4,
                       feather=0.9)[..., None] * sparkle * white)
    return out


def as_texture(arr: np.ndarray) -> pv.Texture:
    """Float RGB array -> a VTK texture usable as a PBR environment.

    `ColorModeToDirectScalars` is mandatory: without it VTK maps the radiance
    values through a lookup table and everything above 1.0 -- the entire point
    of an HDR environment -- is clamped away before the shader sees it.
    """
    height, width, _ = arr.shape
    grid = pv.ImageData(dimensions=(width, height, 1))
    grid.point_data["rgb"] = np.ascontiguousarray(
        arr[::-1].reshape(-1, 3).astype(np.float32))
    grid.set_active_scalars("rgb")
    texture = pv.Texture()
    texture.SetInputDataObject(grid)
    texture.SetColorModeToDirectScalars()
    texture.MipmapOn()
    texture.InterpolateOn()
    return texture


# The gem shader samples a texture of its own (see `render/gem_shader.py`),
# and that texture is not the same job as the IBL environment. A refracting
# stone is a WINDOW: what a viewer sees inside it is the room itself, so the
# room needs real structure — window frames, ceiling strips, dark gaps. The
# procedural rig is deliberately smooth, and a smooth room seen through a
# window is a smooth blob.
#
# The supplied `env-gem-1.hdr` is a real showroom, and measured through the
# refraction shader on a 4.5 mm catalogue stone it beats the rig on the one
# thing that matters here — local contrast, which is what "sparkle" is:
#
#     procedural rig    5th-pct 89    local contrast 2.84
#     supplied HDR      5th-pct 44    local contrast 2.80 (desaturated)
#     supplied HDR raw  5th-pct 43    local contrast 2.67, visible teal cast
#
# The 2:1 objection that rules this file out as an IBL environment does not
# apply here: the shader computes its own equirect lookup, so the projection
# is this module's decision rather than VTK's.
#
# DESATURATED on purpose. The showroom has a teal cast, and a colourless
# diamond showing teal is wrong twice over — the cast is the room's, not the
# stone's, and a coloured stone gets its colour from the transmission tint
# anyway. Stripping it keeps every bit of the structure and none of the cast.
# How far the supplied HDR is scaled up before the shader looks through it.
#
# This was RE-SWEPT after the facet tracer went in, because the value it had
# (9.0) was chosen for the shader that came before it and no longer described
# anything. With the tracer, the stone's whole tonal range shifts upward -- so
# the same gain that used to be merely bright now puts the median at 247 out
# of 255 and pins 18% of the stone at pure white. Facets are DIFFERENCES
# between neighbouring directions, and there is no room left for a difference
# at the top of the scale. That is the "fasetler hiç belli olmuyor" report,
# and it is a clipping problem, not a lighting one.
#
# Re-swept on the 4.5 mm catalogue stone, reading the median, the fraction
# pinned at pure white, and local contrast (what "sparkle" actually is):
#
#     gain  median  pure white  contrast
#      2.0      99        3.3%     37.1   stone reads grey
#      3.0     175        4.3%     35.1
#      4.5     224        6.4%     26.3
#      5.5     235        7.9%     21.4   <- chosen
#      6.0     238        8.7%     19.3
#      9.0     247       18.2%     11.4   <- the old value
#
# Contrast falls monotonically with gain the whole way down that column: every
# bit of extra brightness is bought from the flash. 5.5 keeps the stone plainly
# bright (median 235) while nearly doubling the flash and halving the clipping.
# Checked at both scales, as the first sweep was: the big stone above, and a
# 1.1 mm pave stone about ten pixels across, which is the case that stops this
# from going lower still.
GEM_LOOKUP_GAIN = 5.5


@lru_cache(maxsize=1)
def gem_lookup() -> pv.Texture:
    """The texture the refraction shader looks through the stone at."""
    from catalog_organizer.render.product_render import GEM_HDR, HDR_DIR

    path = HDR_DIR / GEM_HDR
    if path.exists():
        try:
            from vtkmodules.util import numpy_support as ns
            from vtkmodules.vtkIOImage import vtkHDRReader

            reader = vtkHDRReader()
            reader.SetFileName(str(path))
            reader.Update()
            image = reader.GetOutput()
            width, height, _ = image.GetDimensions()
            arr = ns.vtk_to_numpy(
                image.GetPointData().GetScalars()).astype(np.float32)
            arr = arr.reshape(height, width, -1)[:, :, :3]
            luma = arr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            flat = np.repeat(luma[..., None], 3, axis=-1) * GEM_LOOKUP_GAIN
            return as_texture(flat[::-1])     # as_texture flips again
        except Exception:
            pass
    return environment("gem")


@lru_cache(maxsize=2)
def environment(kind: str) -> pv.Texture:
    """The rig for "metal" or "gem", cached -- building one costs ~0.2 s."""
    return as_texture(gem_equirect() if kind == "gem" else studio_equirect())


# -- render-time helpers ------------------------------------------------------

def install_tone_mapping(plotter: pv.Plotter, exposure: float) -> None:
    """Wrap whatever pass the renderer already has in an ACES filmic operator.

    Order matters and is why the existing pass is read first rather than
    replaced: SSAO installs its own pass, and tone mapping has to sit outside
    it so it operates on the finished frame.
    """
    from vtkmodules.vtkRenderingOpenGL2 import (
        vtkRenderStepsPass, vtkToneMappingPass,
    )
    tone = vtkToneMappingPass()
    tone.SetToneMappingType(vtkToneMappingPass.GenericFilmic)
    tone.SetUseACES(True)
    tone.SetGenericFilmicDefaultPresets()
    tone.SetContrast(CURVE_CONTRAST)
    tone.SetExposure(exposure)
    existing = plotter.renderer.GetPass()
    tone.SetDelegatePass(existing if existing is not None
                         else vtkRenderStepsPass())
    plotter.renderer.SetPass(tone)


def orient_environment_to_camera(plotter: pv.Plotter) -> None:
    """Rotate the rig so its key light is always up-and-left of the camera.

    Without this the lighting depends on how the piece happened to be oriented
    in CAD, so two rings from the same tray photograph differently. A product
    photographer moves the lights with the setup; this is the same move.
    """
    camera = plotter.camera
    forward = np.asarray(camera.focal_point, float) - np.asarray(
        camera.position, float)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(camera.up, float)
    up = up - forward * np.dot(up, forward)
    up = up / np.linalg.norm(up)
    plotter.renderer.SetEnvironmentUp(*up)
    plotter.renderer.SetEnvironmentRight(*np.cross(forward, up))


def gradient_background(height: int, width: int, top: str = BG_TOP,
                        bottom: str = BG_BOTTOM) -> np.ndarray:
    """The page the product is composited onto, as float RGB."""
    def rgb(text):
        return np.array([int(text[i:i + 2], 16) for i in (1, 3, 5)], float)

    ramp = np.linspace(0.0, 1.0, height)[:, None, None]
    return (rgb(top) * (1.0 - ramp) + rgb(bottom) * ramp).repeat(width, 1)


def contact_shadow(alpha: np.ndarray, *, squash: float = 0.12,
                   drop: float = 0.004, blur: float = 0.050,
                   strength: float = 0.20) -> np.ndarray:
    """A soft shadow on the table under the piece, as a darkening multiplier.

    Not a real shadow: nothing here casts onto a ground plane, and adding one
    would mean a visible table in every frame. This is the flattened silhouette
    blurred and laid under the piece, which is the same trick the WebGL
    configurators use and is indistinguishable at catalogue size.

    It earns its place by grounding the object. Against a plain gradient a
    product with no shadow reads as a cut-out pasted onto the page, and that
    single cue is a large part of what separates a render from a photograph.
    """
    from PIL import Image, ImageFilter

    height, width = alpha.shape[:2]
    rows = np.flatnonzero(alpha.max(axis=1) > 8)
    if rows.size == 0:
        return np.ones((height, width, 1))
    bottom = int(rows[-1])

    squashed_h = max(2, int(height * squash))
    plate = Image.fromarray(alpha).resize((width, squashed_h), Image.LANCZOS)
    canvas = Image.new("L", (width, height), 0)
    top = min(height - squashed_h,
              int(bottom + height * drop - squashed_h // 2))
    canvas.paste(plate, (0, max(0, top)))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur * height))

    shade = np.asarray(canvas, dtype=float) / 255.0
    return (1.0 - strength * shade)[..., None]


def floor_reflection(rgba: np.ndarray, *, strength: float = 0.20,
                     fade: float = 0.40, blur: float = 0.004) -> np.ndarray:
    """The piece mirrored in the surface it stands on, fading with distance.

    Without it the background reads as a flat colour swatch the product was
    pasted onto. A shadow says "there is a floor"; a reflection says what the
    floor is made of, and a catalogue shot is nearly always taken on something
    with a sheen.

    Returned as RGBA, to be composited under the piece and over the page.
    """
    from PIL import Image, ImageFilter

    height, width = rgba.shape[:2]
    rows = np.flatnonzero(rgba[..., 3].max(axis=1) > 8)
    if rows.size == 0:
        return np.zeros_like(rgba)
    bottom = int(rows[-1])

    mirrored = rgba[::-1].copy()
    # Flipping puts the base at (height - 1 - bottom); shift it back down so
    # the real base and its reflection meet on the contact line.
    shift = bottom - (height - 1 - bottom)
    out = np.zeros_like(rgba)
    if shift > 0:
        out[shift:] = mirrored[:height - shift]
    elif shift < 0:
        out[:height + shift] = mirrored[-shift:]
    else:
        out = mirrored

    depth = np.clip((np.arange(height) - bottom) / max(height * fade, 1.0),
                    0.0, 1.0)
    falloff = ((1.0 - depth) ** 2)[:, None]
    falloff[:bottom] = 0.0                    # nothing above the contact line
    alpha = out[..., 3].astype(float) * falloff * strength
    if blur > 0:
        alpha = np.asarray(Image.fromarray(
            np.clip(alpha, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=blur * height)), dtype=float)
        # The blur runs after the cut, so it smears a little back over the
        # contact line. Left alone it puts reflection ON the piece.
        alpha[:bottom] = 0.0
    out[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return out


def composite_over(rgba: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Alpha-composite a transparent render onto a background.

    The object is rendered on transparency and combined here rather than in
    front of a coloured background because tone mapping runs over the whole
    frame: at the exposure that keeps gold saturated, a light background is
    dragged down to mid grey with it.
    """
    alpha = rgba[..., 3:4].astype(float) / 255.0
    rgb = rgba[..., :3].astype(float)
    return np.clip(rgb * alpha + background * (1.0 - alpha), 0, 255).astype(
        np.uint8)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter a supersampled frame down to its delivered size.

    Rendering at 2x and filtering down beats VTK's own anti-aliasing here: a
    prong or a bezel edge is a sub-pixel sliver of near-mirror metal, and MSAA
    resolves the coverage but not the wildly different radiance either side of
    the edge.
    """
    if factor <= 1:
        return image
    from PIL import Image

    height, width = image.shape[:2]
    return np.asarray(Image.fromarray(image).resize(
        (width // factor, height // factor), Image.LANCZOS))
