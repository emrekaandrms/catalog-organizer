"""Generate `demo_ring.stl`: a four-prong solitaire ring built from primitives.

An original, rights-free model for trying the app (and for screenshots and tests) without anyone's
CAD files. Like a casting model it has NO stone, only the conical seat the stone would sit in, so it
also shows the "place stones in the seats of a stoneless STL" feature end to end.

    python examples/make_demo_ring.py            # writes examples/demo_ring.stl

Units are millimetres. The finger axis is Y and the head faces +Z, the way a ring is usually
modelled; the band is 18 mm across inside (a size 7.5-ish finger).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


def build_demo_ring() -> trimesh.Trimesh:
    major, minor = 9.8, 1.5                              # band centre-line radius, half-thickness
    band = trimesh.creation.torus(major_radius=major, minor_radius=minor,
                                  major_sections=128, minor_sections=40)
    band.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))   # axis -> Y
    band.apply_scale((1.0, 1.9, 1.0))                    # a band 5.7 mm wide, 3 mm thick

    top = major + minor                                   # z of the band's crest
    z_top = top + 2.8                                     # the cup's rim stands 2.8 mm above the crest
    cup_bottom = top - 1.0                                # ... and is sunk 1 mm into the band
    cup = trimesh.creation.cylinder(radius=4.6, height=z_top - cup_bottom, sections=96)
    cup.apply_translation((0, 0, 0.5 * (z_top + cup_bottom)))
    head = trimesh.boolean.union([band, cup], engine="manifold")

    # the seat: a cone, 7.2 mm across where it meets the top of the cup, narrowing to the culet
    stone_d = 7.2
    depth = 3.0
    r_top = stone_d / 2
    over = 0.5                                            # the cutter starts above the cup so the cut is clean
    r_base = r_top * (depth + over) / depth
    cutter = trimesh.creation.cone(radius=r_base, height=depth + over, sections=96)
    cutter.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))   # apex down
    cutter.apply_translation((0, 0, z_top + over))
    seated = trimesh.boolean.difference([head, cutter], engine="manifold")

    prongs = []
    for k in range(4):
        angle = np.pi / 4 + k * np.pi / 2
        prong = trimesh.creation.cylinder(radius=0.55, height=4.4, sections=24)
        prong.apply_translation((4.4 * np.cos(angle), 4.4 * np.sin(angle), z_top + 0.6))
        prongs.append(prong)
    ring = trimesh.boolean.union([seated, *prongs], engine="manifold")
    ring.merge_vertices()
    return ring


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else Path(__file__).with_name("demo_ring.stl")
    ring = build_demo_ring()
    ring.export(out)
    print(f"{out}: {len(ring.faces)} triangles, watertight={ring.is_watertight}, "
          f"size={np.round(ring.extents, 1).tolist()} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
