"""Metal and stone material presets for catalogue renders.

For a metal, base colour is not taste and not a paint swatch -- it is F0, the
fraction of light the surface reflects at normal incidence, and it is a
measured physical constant. The values below are those constants:

    gold      (1.000, 0.766, 0.336)
    silver    (0.972, 0.960, 0.915)
    platinum  (0.679, 0.642, 0.588)

An earlier pass had gold at #C49732 = (0.769, 0.592, 0.196): a quarter too
dark and pulled toward brown. It was arrived at honestly -- fitted against a
reference image -- but it was fitting the wrong variable. The renderer had no
tone mapping operator at the time, so the environment's highlights clipped flat
to white, and darkening the metal was the only lever that appeared to help. It
traded a blown-out gold for a dull brass one. With ACES filmic now in the
pipeline (see `render/studio.py`) the physical values are usable directly, and
anything else is a distortion on top of a correct image.

`POLISHED_ROUGHNESS` moved for the same reason: 0.36 is a satin finish. A
polished ring is close to a mirror, and it only reads as one once the rig it
reflects has dark areas in it to be a mirror *of*.

Stones are modelled from their refractive index rather than as paint -- see
`StoneMaterial` for the reasoning and the numbers. The short version: a gem
has no diffuse term at all, and treating it as though it did is what made
every stone render as white ceramic.

The dialog builds both of its drop-downs straight from these dicts, so adding a
colour means adding one entry here and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

# A polished jewellery finish is a near-mirror. Swept against a real ring under
# the studio rig; above ~0.2 the reflected panels smear into a single soft
# gradient and the piece stops reading as polished.
POLISHED_ROUGHNESS = 0.12


@dataclass(frozen=True)
class MetalMaterial:
    key: str
    label_tr: str
    # Linear F0. Written as hex for the dialog's convenience, but these are
    # reflectance values, not sRGB paint colours -- see the module docstring.
    color: str
    metallic: float = 1.0
    roughness: float = POLISHED_ROUGHNESS


@dataclass(frozen=True)
class StoneMaterial:
    """A gem as a refractive index and a transmission colour.

    Two earlier models are recorded here because both failed in ways that were
    reported back, and the reasons are the useful part:

    1. A light body colour with `metallic = 0.05` — almost entirely diffuse.
       Diffuse is uniform by definition, so the stone had no facets at all: a
       10 mm brilliant measured a p90-p10 luminance spread of 11.7. White
       ceramic.
    2. Pure specular layers, crown over pavilion. That restored the facets but
       a facet whose mirror direction found the rig's dark surround came back
       black, and lifting the surround to remove the black removed the
       contrast with it. On a real 4.5 mm stone the whole trade was visible in
       one sweep: floor 0.0 gave 5th-percentile 28 with local contrast 2.73,
       floor 6.0 gave 238 with 0.30. No point on that curve was both.

    What was missing was refraction, and refraction for a gem is not a detail
    — a brilliant's table is a window onto its pavilion, not a mirror of the
    room. `render/gem_shader.py` computes it properly, and the same stone then
    measured 89 with a contrast of 2.84: bright, and still flashing.

    So the fields here are the physical inputs that shader needs, not shading
    parameters: an index of refraction (which also fixes the surface's Fresnel
    reflectance) and the colour the stone imparts to light passing through it.
    """
    key: str
    label_tr: str
    # Transmission colour: what the light picks up on the way through. NOT a
    # surface colour — the surface reflection stays white, which is why even a
    # deep sapphire has colourless glints around its rim.
    color: str
    # Refractive index. Diamond 2.417, corundum (ruby, sapphire) 1.77,
    # beryl (emerald) 1.58, quartz (amethyst) 1.54.
    ior: float = 1.77
    # Dispersion: how far the index spreads between red and blue, and so how
    # much coloured fire the stone throws. Diamond's 0.044 is the highest of
    # any stone here by a wide margin and is a large part of why a diamond
    # reads as a diamond. The shader traces one channel per index:
    #     (n - .5*dispersion, n, n + .5*dispersion)
    dispersion: float = 0.018
    # Opaque stones do not refract. Onyx is the only one here.
    opaque: bool = False
    roughness: float = 0.0


METALS: dict[str, MetalMaterial] = {
    m.key: m for m in (
        MetalMaterial("yellow_gold", "Sarı Altın", "#FFC356"),
        MetalMaterial("rose_gold", "Rose Altın", "#FAB591"),
        # Rhodium plated, which is what white gold actually presents as.
        MetalMaterial("white_gold", "Beyaz Altın", "#EBE8E5"),
        MetalMaterial("silver", "Gümüş", "#F8F5E9"),
        MetalMaterial("platinum", "Platin", "#ADA496"),
        # An oxidised finish is genuinely rough and genuinely dark; it should
        # not follow the polished value.
        MetalMaterial("antique", "Antik / Oksit", "#59544D", roughness=0.55),
        # Castable wax. Not a metal at all -- it is a dyed dielectric, so
        # metallic drops to 0 and the base colour becomes a real albedo rather
        # than a reflectance. Carried in this dict because it occupies the
        # same slot in the workflow: it is what the piece is quoted and sold
        # as when the customer is buying the printed pattern, not the casting.
        MetalMaterial("wax_purple", "Mor Mum (Döküm)", "#6B2D8F",
                      metallic=0.0, roughness=0.42),
    )
}

STONES: dict[str, StoneMaterial] = {
    s.key: s for s in (
        # Diamond: the highest index of any stone here, and colourless, so
        # what comes back through it is white.
        StoneMaterial("white", "Beyaz (Pırlanta)", "#FFFFFF", ior=2.417,
                      dispersion=0.044),
        # Onyx is opaque — there is nothing to see through and nothing to
        # refract, so it keeps a plain reflective surface.
        StoneMaterial("black", "Siyah (Oniks)", "#1A1A1E", opaque=True,
                      dispersion=0.0,
                      roughness=0.08),
        # Coloured stones now carry their colour where it physically lives:
        # in transmission. An earlier pass had to tint the SURFACE instead,
        # because the layered model's coloured layer reflected the rig's dark
        # lower hemisphere and came back grey (saturation 0.005 on the
        # catalogue's own pieces). With refraction the tint sits on the ray
        # that actually passes through the stone, so the rim glints stay
        # colourless the way a real gem's do.
        StoneMaterial("red", "Kırmızı (Yakut)", "#D81838", ior=1.77),
        StoneMaterial("blue", "Mavi (Safir)", "#1838D8", ior=1.77),
        StoneMaterial("green", "Yeşil (Zümrüt)", "#18A860", ior=1.58,
                      dispersion=0.014),
        StoneMaterial("purple", "Mor (Ametist)", "#8830D8", ior=1.54,
                      dispersion=0.013),
        StoneMaterial("champagne", "Şampanya", "#D8AC70", ior=1.77),
        StoneMaterial("pink", "Pembe", "#E87898", ior=1.77),
    )
}

DEFAULT_METAL = "yellow_gold"
DEFAULT_STONE = "white"


def metal(key: str) -> MetalMaterial:
    try:
        return METALS[key]
    except KeyError:
        raise KeyError(
            f"bilinmeyen maden rengi: {key!r}; seçenekler: {sorted(METALS)}"
        ) from None


def stone(key: str) -> StoneMaterial:
    try:
        return STONES[key]
    except KeyError:
        raise KeyError(
            f"bilinmeyen taş rengi: {key!r}; seçenekler: {sorted(STONES)}"
        ) from None
