"""Generate a gem shader that traces the stone's own facet planes.

The shader is CODE GENERATED: every facet of the stone is baked into the GLSL as a
compile-time constant and the intersection loop is unrolled over the lot --
no texture, no acceleration structure, no proxy geometry. A cut stone is
only about 120 planes, so the exact answer is affordable, and because the ray
meets a real facet each bounce, neighbouring rays DIVERGE. That divergence is
the flash, and it is what every approximation tried here failed to produce:

    analytic cone proxy       local contrast 4.48 -> 1.15   (converges: worse)
    screen-space back faces   local contrast 2.30 -> 4.36   (one surface only)

A convex body makes the intersection trivial. The exit point is simply the
nearest plane the ray is heading toward, so one pass over the planes taking a
running minimum is exact.

Dispersion is done by tracing each colour channel at its own index:

    indices = (n - 0.5 * dispersion, n, n + 0.5 * dispersion)

with diamond's real dispersion of 0.044. Absorption follows the stone's colour:
coefficients `(1 - r, 1 - g, 1 - b)`.
"""
from __future__ import annotations

import numpy as np

from catalog_organizer.render.facets import StoneShape

# Defaults: diamond's refractive index and dispersion (physical constants).
REFRACTION_INDEX = 2.41
DISPERSION = 0.044          # diamond's real dispersion coefficient
RAY_BOUNCES = 4

# How much stone the ray is absorbed by per radius travelled. See
# `absorption_from` for why the raw (1 - colour) coefficient is too dilute at
# the path lengths this geometry actually produces.
ABSORPTION_DENSITY = 1.6

# Their viewer drops this when a piece carries many stones
# (`if (optimize.enabled && decorationCount >= stonesThreshold)`), and for the
# same reason: a pave stone is ten pixels across and nobody reads its
# interior, but every one of them pays the full trace.
CROWDED_STONE_COUNT = 24
CROWDED_RAY_BOUNCES = 2

# Everything lives in ONE block, inside main, using arrays and loops rather
# than functions. GLSL cannot declare a function inside another, so the
# natural split would be to put the helpers in `//VTK::Light::Dec` -- but that
# anchor sits ABOVE the point where VTK declares the texture samplers, so a
# helper that reads `gemEnv` fails to compile there and VTK reports the
# failure as an empty error string and an empty frame. Arrays sidestep the
# ordering entirely.
_MAIN = """
  //---- the stone's facets, baked in ------------------------------------
  const vec3 PA[PLANE_COUNT] = vec3[PLANE_COUNT](
#INJECT(POINTS)
  );
  const vec3 PN[PLANE_COUNT] = vec3[PLANE_COUNT](
#INJECT(NORMALS)
  );
  const vec3 ETAS = vec3(ETA_R, ETA_G, ETA_B);
  const vec3 ABSORB = ABSORPTION;

  vec3 Nv = normalize(normalVCVSOutput);
  vec3 Iv = normalize(vertexVC.xyz);
  if (dot(Nv, Iv) > 0.0) Nv = -Nv;

  // Into the stone's own frame: centred, radius 1, table along +Z. One plane
  // set then serves every stone of the cut, whatever its size or pose.
  mat3 toWorld = mat3(uStoneRight, uStoneUp, uStoneAxis);
  mat3 toLocal = transpose(toWorld);
  vec3 localDir = normalize(toLocal * Iv);
  vec3 localNrm = normalize(toLocal * Nv);
  vec3 localOrg = (toLocal * (vertexVC.xyz - uStoneCentre)) / uStoneRadius;

  vec3 through = vec3(0.0);
  for (int ch = 0; ch < CHANNELS; ++ch) {
    float eta = ETAS[ch];
    vec3 d = refract(localDir, localNrm, 1.0 / eta);
    if (dot(d, d) < 1e-6) { d = reflect(localDir, localNrm); }
    vec3 o = localOrg;
    float pathLength = 0.0;

    for (int bounce = 0; bounce < RAY_BOUNCES; ++bounce) {
      // The stone is convex, so the nearest plane the ray is heading toward
      // IS where it leaves. No containment test is needed.
      float theta = 1.0e9;
      vec3 hitN = vec3(0.0, 0.0, 1.0);
      for (int i = 0; i < PLANE_COUNT; ++i) {
        float dn = dot(d, PN[i]);
        if (dn > 1.0e-6) {
          float t = (dot(PA[i], PN[i]) - dot(o, PN[i])) / dn;
          if (t > 1.0e-4 && t < theta) { theta = t; hitN = PN[i]; }
        }
      }
      if (theta > 1.0e8) break;
      o += d * theta;
      pathLength += theta;
      vec3 outDir = refract(d, hitN, eta);
      if (dot(outDir, outDir) > 1.0e-6) { d = outDir; break; }
      d = reflect(d, hitN);              // total internal reflection
    }

    vec3 w = normalize(toWorld * d);
    vec3 s = normalize(vec3(w.x, w.y, -w.z));
    vec2 uv = vec2(atan(s.z, s.x) / 6.2831853 + 0.5,
                   1.0 - acos(clamp(s.y, -1.0, 1.0)) / 3.14159265);
    float absorbed = exp(-ABSORB[ch] * pathLength);
    if (CHANNELS == 1) {
      through = texture(gemEnv, uv).rgb * exp(-ABSORB * pathLength);
    } else {
      through[ch] = texture(gemEnv, uv)[ch] * absorbed;
    }
  }

  vec3 m = reflect(Iv, Nv);
  vec3 ms = normalize(vec3(m.x, m.y, -m.z));
  vec2 muv = vec2(atan(ms.z, ms.x) / 6.2831853 + 0.5,
                  1.0 - acos(clamp(ms.y, -1.0, 1.0)) / 3.14159265);
  vec3 mirror = texture(gemEnv, muv).rgb;

  float ct = clamp(dot(-Iv, Nv), 0.0, 1.0);
  const float F0 = F0_VALUE;
  float F = F0 + (1.0 - F0) * pow(1.0 - ct, 5.0);
  gl_FragData[0] = vec4(mix(through, mirror, F) * COLOR_BOOST, 1.0);
"""


def fresnel_f0(ior: float) -> float:
    """Normal-incidence reflectance, straight from the refractive index.

    Diamond's 2.417 gives 0.172 -- four times a common glass, and a large part
    of why a diamond's surface alone looks brighter than any imitation's. It
    is derived here rather than dialled in, so it can never drift away from
    the index the ray is actually bent by.
    """
    return ((ior - 1.0) / (ior + 1.0)) ** 2


def _vec3(v) -> str:
    return "vec3(%.6f, %.6f, %.6f)" % tuple(float(x) for x in v)


def build(shape: StoneShape, *, ior: float = REFRACTION_INDEX,
          dispersion: float = DISPERSION, bounces: int = RAY_BOUNCES,
          absorption=(0.0, 0.0, 0.0),
          boost=(1.0, 1.0, 1.0)) -> str:
    """GLSL for one CUT, with its facets baked in as constants.

    The facets are compile-time constants because every stone of a cut is the
    same shape; where each individual stone sits and how it is turned arrives
    as uniforms instead. VTK caches compiled programs by source, so a pave
    piece with 191 stones compiles this once and draws 191 actors through it —
    so each stone behaves like its own scene node.
    """
    joiner = ",\n"
    points = joiner.join("    " + _vec3(p) for p in shape.points)
    normals = joiner.join("    " + _vec3(n) for n in shape.normals)

    # Their exact construction for dispersion: one index per channel, split
    # about the base index by half the dispersion either way. With none, a
    # single achromatic trace serves all three -- three times cheaper, and a
    # colourless stone loses nothing by it.
    if dispersion > 0.0:
        etas = (ior - 0.5 * dispersion, ior, ior + 0.5 * dispersion)
        channel_count = 3
    else:
        etas = (ior, ior, ior)
        channel_count = 1

    main = _MAIN.replace("#INJECT(POINTS)", points)
    main = main.replace("#INJECT(NORMALS)", normals)
    main = main.replace("PLANE_COUNT", str(len(shape)))
    main = main.replace("RAY_BOUNCES", str(int(bounces)))
    main = main.replace("CHANNELS", str(channel_count))
    for name, value in zip(("ETA_R", "ETA_G", "ETA_B"), etas):
        main = main.replace(name, f"{value:.6f}")

    main = main.replace("ABSORPTION", _vec3(absorption))
    main = main.replace("COLOR_BOOST", _vec3(boost))
    main = main.replace("F0_VALUE", f"{fresnel_f0(ior):.5f}")
    return main


def set_frame(actor, *, right, up, axis, centre, radius: float) -> None:
    """Tell one actor which stone it is and where, all in VIEW space."""
    uniforms = actor.GetShaderProperty().GetFragmentCustomUniforms()
    uniforms.SetUniform3f("uStoneRight", [float(x) for x in right])
    uniforms.SetUniform3f("uStoneUp", [float(x) for x in up])
    uniforms.SetUniform3f("uStoneAxis", [float(x) for x in axis])
    uniforms.SetUniform3f("uStoneCentre", [float(x) for x in centre])
    uniforms.SetUniformf("uStoneRadius", float(radius))


def view_basis(direction, up) -> np.ndarray:
    """World -> view rotation rows, for the camera `_frame` will place."""
    forward = -np.asarray(direction, dtype=float)
    forward = forward / np.linalg.norm(forward)
    u = np.asarray(up, dtype=float)
    u = u - forward * float(u @ forward)
    u = u / np.linalg.norm(u)
    return np.stack([np.cross(forward, u), u, -forward])


def absorption_from(hex_colour: str) -> tuple[float, float, float]:
    """Absorption coefficients `(1 - r, 1 - g, 1 - b)` of the stone's colour, times a density.

    A white stone absorbs nothing; a sapphire absorbs almost all of the red
    and green, which is what leaves the blue that comes back out.

    The density is not decoration. `1 - c` is an absorption COEFFICIENT, so
    what reaches the eye is `exp(-coefficient * path)` and the path is
    measured in stone radii. Most rays leave on their first bounce -- for
    corundum the critical angle is 34 degrees, so anything steeper than that
    refracts straight out -- and those rays cross well under one radius of
    material. A ruby's #D81838 gives a green coefficient of 0.906, and over a
    0.7-radius path that is exp(-0.63) = 0.53: more than half the green
    survives. Green surviving alongside red is pink, and pink is exactly what
    was reported ("kirmizi renk ruby tasi pembe gorunuyor"). The mean pixel
    measured (229, 147, 149) -- green and blue equal, both high.

    So the hex says WHAT the stone absorbs and the density says HOW MUCH of it
    there is along the way. Swept on the catalogue stone, mean pixel:

        1.0   ruby (206,  98, 108)   sapphire ( 96, 100, 200)   still pale
        1.6   ruby (179,  54,  62)   sapphire ( 52,  53, 176)   <- chosen
        1.8   ruby (172,  47,  54)   sapphire ( 45,  45, 170)
        2.5   ruby (160,  40,  48)   sapphire ( 37,  40, 159)   going muddy

    Above ~2 the stone keeps saturating but starts losing its flash with it
    (local contrast 26.5 at 1.6, 21.8 at 1.8, and falling), because a ray that
    has been absorbed to nothing carries no sparkle back out either.
    """
    rgb = [int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(ABSORPTION_DENSITY * (1.0 - c) for c in rgb)
