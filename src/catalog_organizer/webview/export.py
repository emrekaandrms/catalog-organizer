"""Export catalogue products as a static web viewer: one GLB per product plus shared files.

    export_many([...], out_dir)  ->  out_dir/index.html          the gallery
                                     out_dir/viewer.html         the viewer  (?p=<file_id>)
                                     out_dir/data/_shared/       environments + materials, once
                                     out_dir/data/<file_id>/     piece.glb + info.json

The output is plain static files: serve the folder from any web server (or `webview.server`).

Nothing here changes how the catalogue renders. It reuses the renderer's own loading, camera
choice, metal normals and stone frames, so the web view of a product is framed and cut the way
the PDF page is, and any difference is the web renderer's, not a re-interpretation.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from catalog_organizer.render import facets
from catalog_organizer.render import product_render as pr
from catalog_organizer.webview import ao as ao_module
from catalog_organizer.webview import seats as seats_module
from catalog_organizer.webview import environment
from catalog_organizer.webview.glb import ARRAY_BUFFER, ELEMENT_BUFFER, Glb, rounded
from catalog_organizer.webview.look import STUDIO_LOOK, Look, shared_meta
from catalog_organizer.webview.pose import camera_distance, product_pose

ASSETS = Path(__file__).parent / "assets"

# A web viewer has a download budget the server render never had. A retail viewer's whole GLB is
# a few MB at ~70k triangles; the catalogue holds meshes of 1.8 million. Measured on a first
# batch of 13 products, a GLB costs 30-55 bytes per triangle (a decimated surface splits
# nearly every vertex at the 30 degree crease angle): 250k triangles was 14 MB, over the
# ~8 MB a product page can afford. 120k stays under it and is far more than a screen shows.
MAX_TRIANGLES = 120_000


@dataclass
class ExportResult:
    file_id: str
    label: str
    category: str
    bytes: int
    triangles: int
    vertices: int
    stones: int
    planes: int
    borrowed_cut: bool
    decimated_from: int | None
    seconds: float
    placed_stones: bool = False     # the file has no stones: they were put in the detected seats


def _decimate(verts: np.ndarray, faces: np.ndarray, target: int):
    """Quadric decimation of a welded copy. Used only above the triangle budget."""
    import fast_simplification
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    reduction = 1.0 - target / max(len(mesh.faces), 1)
    v, f = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32), target_reduction=float(reduction))
    return v.astype(np.float64), f.astype(np.int64)


# Per-vertex occlusion interpolates between vertices, so a large flat face with vertices only
# on its rim inherits the rim's darkness across its whole interior: a brooch's disc, the one
# open flat area of the piece, rendered chocolate-brown against the server render's olive gold
# (measured: AO read ~0.8 at the vertices that were ON the disc, yet the disc looked black).
# Long edges are therefore split before the bake, so a big face has vertices of its own.
# The scale occlusion changes on is its ray length, so vertices need to be no further apart than
# about half of it.
# HOW they are split matters for the size: CAD tessellations are full of long thin slivers, and
# splitting every triangle 1 -> 4 (trimesh's subdivide_to_size) cost +10..80 % triangles (a ring
# went 88k -> 160k). Bisecting the longest edge (1 -> 2) cuts a sliver across its length only:
# +3..45 % on the same meshes, and it kept the smooth look of the 1 -> 4 split. Splitting by
# face AREA instead was cheaper still (+0.5 %) but left triangular patches on a necklace's
# inside, where slivers fan out from a few vertices.
MAX_EDGE_FRACTION = 0.5 * ao_module.RADIUS_FRACTION
_MAX_SPLIT_ROUNDS = 30


def _split_long_edges(verts: np.ndarray, faces: np.ndarray, extent: float):
    """Bisect the longest edge of every triangle whose longest edge exceeds the limit, until none do.

    The midpoint of an edge two selected triangles share is one shared vertex.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    limit = MAX_EDGE_FRACTION * extent
    for _ in range(_MAX_SPLIT_ROUNDS):
        p = v[f]
        edge = np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                         np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                         np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1)
        pick = np.nonzero(edge.max(axis=1) > limit)[0]
        if not len(pick):
            break
        k = edge.argmax(axis=1)[pick]
        rows = np.arange(len(pick))
        a, b, c = f[pick][rows, k], f[pick][rows, (k + 1) % 3], f[pick][rows, (k + 2) % 3]
        ends, inverse = np.unique(np.sort(np.stack([a, b], axis=1), axis=1), axis=0,
                                  return_inverse=True)
        mid = len(v) + inverse.ravel()
        v = np.vstack([v, 0.5 * (v[ends[:, 0]] + v[ends[:, 1]])])
        keep = np.ones(len(f), dtype=bool)
        keep[pick] = False
        f = np.vstack([f[keep], np.stack([a, mid, c], axis=1), np.stack([mid, b, c], axis=1)])
    return v, f


def _fit_to_budget(verts: np.ndarray, faces: np.ndarray, extent: float, budget: int):
    """Simplify, then split long edges, so the count that SHIPS is within the budget.

    Splitting after decimating adds some (3-45 % on real meshes), so the decimation target is
    corrected from what the split produced.
    Returns (verts, faces, decimated_from); decimated_from is None when never simplified.
    """
    original = len(faces)
    target = min(budget, original)
    for attempt in range(3):
        decimated = original > target
        v, f = _decimate(verts, faces, target) if decimated else (verts, faces)
        v, f = _split_long_edges(v, f, extent)
        if len(f) <= budget * 1.08 or attempt == 2:
            return v, f, (original if decimated else None)
        target = max(int(target * budget / len(f)), 500)


def export_product(source: Path | None, file_id: str, out_dir: Path, *,
                   category: str | None = None, label: str | None = None,
                   look: Look = STUDIO_LOOK, max_triangles: int = MAX_TRIANGLES,
                   arrays=None, place_stones: bool = False) -> ExportResult:
    """Write out_dir/data/<file_id>/piece.glb (+ info.json). `arrays` = (metal, gem) skips loading."""
    started = time.time()
    if arrays is None:
        metal, gem = pr.load_source_arrays(Path(source))
    else:
        metal, gem = arrays
    if metal is None:
        raise ValueError(f"{file_id}: metal geometrisi yok, web gorunumu uretilemez")
    source = Path(source) if source is not None else Path(f"{file_id}.3dm")

    # A casting-model STL is the piece before the machine: no stones, but the seats are cut in.
    # Only when ASKED (the product's record says it has stones, or the user said so): an STL cannot
    # tell a seat from any other hole, so the program does not decide this alone.
    # webview/seats.py says which seats get a stone and why.
    placed = False
    if place_stones and gem is None and source.suffix.lower() == ".stl":
        gem = seats_module.seat_stones(metal, source if arrays is None else None)
        placed = gem is not None

    pose = product_pose(metal, gem, category=category, source=source)
    rotation, centre = pose.rotation, pose.centre

    # ---- metal: decimate if over budget, turn into the camera frame, split normals ----------
    verts = np.asarray(metal[0], dtype=np.float64)
    faces = np.asarray(metal[1], dtype=np.int64)
    verts, faces, decimated_from = _fit_to_budget(verts, faces, pose.extent, max_triangles)
    canonical = (verts - centre) @ rotation.T
    # What `_add_metal` asks VTK for: smooth shading with split_sharp_edges (30 degrees).
    poly = pr._to_polydata((canonical, faces)).compute_normals(
        cell_normals=False, split_vertices=True, feature_angle=30.0)
    pos = np.asarray(poly.points, dtype=np.float32)
    nrm = np.asarray(poly.point_data["Normals"], dtype=np.float32)
    tri = np.asarray(poly.faces).reshape(-1, 4)[:, 1:].astype(np.uint32).ravel()
    occlusion = ao_module.bake_vertex_ao(
        canonical, faces, pos.astype(np.float64), nrm.astype(np.float64),
        extent=pose.extent).astype(np.float32)

    glb = Glb()
    a_pos = glb.add(pos, target=ARRAY_BUFFER, kind="VEC3", bounds=True)
    a_nrm = glb.add(nrm, target=ARRAY_BUFFER, kind="VEC3")
    a_ao = glb.add(occlusion, target=ARRAY_BUFFER, kind="SCALAR")
    a_idx = glb.add(tri, target=ELEMENT_BUFFER, kind="SCALAR")
    meshes = [{"name": "Metal", "primitives": [
        {"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "_AO": a_ao},
         "indices": a_idx, "material": 0}]}]
    nodes = [{"name": "Metal", "mesh": 0, "extras": {"role": "metal"}}]

    # ---- stones: each one's frame rides in its node; the cut's planes ride in the scene -----
    frames, shape, borrowed = [], None, False
    if gem is not None:
        frames = facets.stone_frames(np.asarray(gem[0]), np.asarray(gem[1]))
    if frames:
        biggest = max(frames, key=lambda f: f["radius"])
        shape, borrowed = facets.plane_set(biggest["verts"], biggest["faces"])
    for i, frame in enumerate(frames):
        sv = ((frame["verts"] - centre) @ rotation.T).astype(np.float32)
        sf = np.asarray(frame["faces"], dtype=np.uint32).ravel()
        meshes.append({"name": f"Stone_{i}", "primitives": [
            {"attributes": {"POSITION": glb.add(sv, target=ARRAY_BUFFER, kind="VEC3", bounds=True)},
             "indices": glb.add(sf, target=ELEMENT_BUFFER, kind="SCALAR"), "material": 1}]})
        nodes.append({"name": f"Stone_{i}", "mesh": len(meshes) - 1, "extras": {
            "role": "gem", "frame": {
                "centre": rounded(rotation @ (frame["centre"] - centre)),
                # basis rows are directions in the file's frame; rotate each into the new one
                "basis": rounded(frame["basis"] @ rotation.T),
                "radius": round(float(frame["radius"]), 6)}}})

    distance = camera_distance(pose, look.fov)
    scene_extras = {
        "source": source.name, "category": category, "label": label or file_id,
        "extentMm": round(pose.extent, 4), "decimatedFrom": decimated_from,
        "placedStones": placed,
        "pose": {"cameraPos": [0.0, 0.0, round(distance, 6)], "fov": look.fov,
                 "up": [0.0, 1.0, 0.0], "zoom": round(pose.zoom, 6),
                 "viewRows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 # file frame -> canonical frame, and both catalogue views in the canonical one
                 "rotation": rounded(rotation),
                 "views": {name: {"dir": rounded(pose.canonical(d)), "up": rounded(pose.canonical(u))}
                           for name, (d, u) in (pose.views or {}).items()}},
    }
    if shape is not None:
        scene_extras["cut"] = {"points": rounded(shape.points), "normals": rounded(shape.normals),
                               "borrowed": bool(borrowed), "planes": len(shape)}
    doc = {
        "asset": {"version": "2.0", "generator": "catalog-organizer webview"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes))), "extras": scene_extras}],
        "nodes": nodes, "meshes": meshes,
        "materials": [
            {"name": "metal", "pbrMetallicRoughness": {
                "baseColorFactor": [1, 0.766, 0.336, 1], "metallicFactor": 1,
                "roughnessFactor": look.roughness}},
            {"name": "gem", "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0}},
        ],
    }
    folder = Path(out_dir) / "data" / file_id
    folder.mkdir(parents=True, exist_ok=True)
    size = glb.write(folder / "piece.glb", doc)

    result = ExportResult(
        file_id=file_id, label=label or file_id, category=category or "",
        bytes=size, triangles=len(tri) // 3, vertices=len(pos), stones=len(frames),
        planes=len(shape) if shape is not None else 0, borrowed_cut=bool(borrowed),
        decimated_from=decimated_from, seconds=round(time.time() - started, 1),
        placed_stones=placed)
    (folder / "info.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return result


def write_shared(out_dir: Path, look: Look = STUDIO_LOOK) -> None:
    shared = Path(out_dir) / "data" / "_shared"
    sizes = environment.write_shared(shared)
    (shared / "meta.json").write_text(
        json.dumps(shared_meta(look, sizes), ensure_ascii=False, indent=1), encoding="utf-8")


def write_site(out_dir: Path, results: list[ExportResult]) -> None:
    """Copy the viewer pages next to the data and write the product list the gallery reads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "viewer.html", "viewer.js"):
        shutil.copyfile(ASSETS / name, out_dir / name)
    shutil.copytree(ASSETS / "vendor", out_dir / "vendor", dirs_exist_ok=True)   # three.js, no CDN
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "data" / "products.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=1), encoding="utf-8")


def export_many(items, out_dir: Path, *, look: Look = STUDIO_LOOK, progress=None,
                max_triangles: int = MAX_TRIANGLES):
    """items: iterable of (file_id, source_path, category, label).

    One bad product must not abort the others, so failures are collected, not raised.
    Returns (results, errors) with errors = {file_id: message}.
    """
    items = list(items)
    out_dir = Path(out_dir)
    write_shared(out_dir, look)
    results: list[ExportResult] = []
    errors: dict[str, str] = {}
    for done, (file_id, source, category, label) in enumerate(items, start=1):
        try:
            results.append(export_product(source, file_id, out_dir, category=category,
                                          label=label, look=look, max_triangles=max_triangles))
        except Exception as exc:                      # noqa: BLE001 - reported per product
            errors[file_id] = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(done, len(items), file_id)
    write_site(out_dir, results)
    return results, errors
