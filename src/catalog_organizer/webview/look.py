"""The look the web viewer renders in, and the shared metadata it reads.

A flat, bright product-photography look: one page colour, ACES filmic tone mapping, a very
long lens, baked occlusion on the metal. The numbers were fitted so that yellow gold's mean
colour, brightness distribution and edge contrast land where a luxury-retail viewer's do. They
are tuning values, not physical constants; `Look` can be replaced wholesale (see
docs/RENDER_ENGINE.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Look:
    # three.js ACESFilmic (enum 4 in r152), a flat page colour.
    tone: str = "three"
    background: str = "EAE8E2"                  # measured: identical at 8 patches of the canvas
    fov: float = 10.0                           # a very long lens
    roughness: float = 0.12
    env_intensity: float = 0.9                  # fitted so the gold's mean colour matched
    # Baked occlusion. 3.2 matched the bracelet's edge crispness best (edge 55.9 vs the
    # 55.1 target), but per-vertex occlusion mottles a flat face and leaves triangle-shaped patches
    # on enclosed surfaces, and a high exponent makes both visible. 1.6 had the best brightness
    # distribution of the sweep (0.169, edge 49.5) and is the safer general default.
    ao_exponent: float = 1.6
    stone_gain: float = 0.55                    # swept 0.30-1.0
    # Base colours. White metals are 0.5-0.7, not the physical 0.97: the pipeline is tuned around
    # them, and it reproduced the target yellow gold's mean colour to 2/255.
    metals: dict = field(default_factory=lambda: {
        "yellow_gold": "0.991,0.546,0.205",
        "rose_gold": "0.991,0.474,0.296",
        "white_gold": "0.723,0.738,0.745",
        "platinum": "0.521,0.521,0.521",
        "silver": "0.723,0.738,0.745",
    })
    # Orbit limits: min 7 / max 30 around a default distance of 21.18.
    controls: dict = field(default_factory=lambda: {
        "min": 0.33, "max": 1.42, "polar": 3.0, "zoom": 0.15, "damping": 0.1})


STUDIO_LOOK = Look()


def shared_meta(look: Look, sizes: dict) -> dict:
    """meta.json: materials, stone constants and the look, shared by every product."""
    from catalog_organizer.render import gem_trace, studio
    from catalog_organizer.render import materials as mat
    from catalog_organizer.webview.glb import rounded

    return {
        "metals": {k: {"label": m.label_tr, "color": m.color, "metallic": m.metallic,
                       "roughness": m.roughness} for k, m in mat.METALS.items()},
        "stones": {k: {"label": s.label_tr, "color": s.color, "ior": s.ior,
                       "dispersion": s.dispersion, "opaque": s.opaque,
                       "roughness": s.roughness,
                       "absorption": rounded(gem_trace.absorption_from(s.color))}
                   for k, s in mat.STONES.items()},
        "stoneBounces": gem_trace.RAY_BOUNCES,
        "crowdedBounces": gem_trace.CROWDED_RAY_BOUNCES,
        "crowdedCount": gem_trace.CROWDED_STONE_COUNT,
        "gemExposure": studio.EXPOSURE_GEM, "metalExposure": studio.EXPOSURE_METAL,
        "tone": {"contrast": studio.CURVE_CONTRAST, "shoulder": 0.9714,
                 "midIn": 0.18, "midOut": 0.18, "hdrMax": 11.0785},
        **sizes,
        "defaults": {
            "tone": look.tone, "bg": look.background, "rough": str(look.roughness),
            "envi": str(look.env_intensity), "gemgain": str(look.stone_gain),
            "ao": str(look.ao_exponent), "metals": look.metals, "controls": look.controls,
        },
    }
