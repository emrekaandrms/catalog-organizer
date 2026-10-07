"""Sprue / casting stem detection.

Three geometric topologies are recognised, then combined with the VLM signal:

  * **Casting rails** (`detect_casting_rails`): the runner bars of a
    casting tree, present as separate slender components fatter than the
    pieces they feed. Tried first — a tree has no neck to find.

  * **End neck** (`detect_sprue_geometric`): walk the mesh in 1-2 mm slabs
    along each principal axis and look for a sustained low-area "neck" at
    either END of one axis. A strong neck → sprue stem; the aggregate slab
    area gives a usable volume estimate so the pipeline can subtract it
    from the metal mass.

  * **Bore bar** (`detect_bore_sprue`): a support stem threaded through a
    ring's finger hole, fused into the same shell as the ring. Invisible to
    both detectors above — it sits in the MIDDLE of the bounding box (no
    neck) and forms ONE component with the ring (no separate rail). Its
    volume is an exact boolean intersection with the bore. Runs last so
    multi-piece models already claimed by the neck detector keep that
    answer. Added 2026-07-30 after 13 of 14 real wedding bands came back
    with no sprue at all.

  * **VLM** (fallback, no volume): when every geometric signal fails we
    still believe the VLM's yes/no answer but can't subtract anything from
    the weight — the record says "sprue detected" while the reported mass
    silently still contains it.

Algorithm verified on `samples/` (2026-05-19):
  - FMR-32 11 boy.stl     → Z axis neck at the bottom, ~246 mm³ sprue
  - FMR-33 12 Boy Kasa.stl → Z axis neck at the bottom, ~72 mm³ sprue
  - honeycomb-bangle.stl → no clear neck on any axis (geometric
    false negative; VLM signal still used)

Tuning rationale:
  - `slab_count=64`: fine enough for 30+ mm pieces, coarse enough to
    average out triangulation noise. For pieces shorter than ~6 mm
    along the scanned axis each slab is < 0.1 mm thick and noise
    dominates — we skip those axes.
  - `neck_ratio_max=0.30`: a slab whose cross-section is below 30% of
    the median is in a neck. Empirical: real necks are 5-15% of
    median, while normal piece variation rarely dips below 40%.
  - `min_len_mm=3.0`: shorter "necks" are almost always end-cap noise
    on thin meshes.
  - `min_vol_mm3=20`: avoid flagging hair-thin protrusions as sprues.
"""
from __future__ import annotations

import numpy as np
import trimesh

from catalog_organizer.core.schemas import SpruInfo


# ── Geometric detection ──────────────────────────────────────────────────────

_SLAB_COUNT_DEFAULT = 64
_NECK_RATIO_MAX = 0.30        # slab area < 30 % of median ⇒ in a neck
_MIN_SPRUE_LEN_MM = 3.0       # below this the "neck" is likely noise
_MIN_SPRUE_VOL_MM3 = 20.0     # below this the stem isn't worth subtracting
_MIN_AXIS_EXTENT_MM = 6.0     # below this slab thickness < 0.1 mm → unreliable


def _slab_signature(
    mesh: trimesh.Trimesh,
    axis_idx: int,
    slab_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each slab perpendicular to `axis_idx`, approximate the slab's
    cross-section area by the AABB area of the vertices that fall in it.

    This is a cheap O(verts) approximation — it ignores holes in the
    cross section, so for ring-like shapes it slightly over-estimates,
    but the *ratio* between slabs (which is what neck detection cares
    about) stays correct.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pos = verts[:, axis_idx]
    pmin, pmax = float(pos.min()), float(pos.max())
    edges = np.linspace(pmin, pmax, slab_count + 1)
    other_axes = [a for a in range(3) if a != axis_idx]
    other = verts[:, other_axes]
    areas = np.zeros(slab_count, dtype=np.float64)
    for i in range(slab_count):
        mask = (pos >= edges[i]) & (pos < edges[i + 1])
        if int(mask.sum()) < 4:
            continue
        pts = other[mask]
        extents = pts.max(axis=0) - pts.min(axis=0)
        areas[i] = float(extents[0] * extents[1])
    return edges, areas


def _count_leading(mask: np.ndarray) -> int:
    """How many True values at the start of `mask`."""
    n = 0
    for v in mask:
        if v:
            n += 1
        else:
            break
    return n


def detect_sprue_geometric(
    mesh: trimesh.Trimesh,
    slab_count: int = _SLAB_COUNT_DEFAULT,
) -> tuple[bool, float | None, float, int | None, str | None]:
    """Find a sprue by scanning slab cross-sections along each axis.

    Returns:
        (detected, volume_mm3, confidence, axis_idx, end)
        - `detected` False when no axis shows a strong-enough neck.
        - `volume_mm3` is the rough sum of slab cross-section × slab
          height for the sprue end; accurate to ±25 % typically.
        - `confidence` 0..1, scaled by how deep the neck is.
        - `axis_idx` 0|1|2 (X/Y/Z) of the scan axis.
        - `end` "lead" (sprue at low-coordinate end) or "trail".
    """
    bmin, bmax = mesh.bounds
    extents = (bmax - bmin).tolist()

    best: tuple[float, float, int, str] | None = None  # (vol, conf, axis, end)
    for axis_idx in range(3):
        if extents[axis_idx] < _MIN_AXIS_EXTENT_MM:
            continue
        edges, areas = _slab_signature(mesh, axis_idx, slab_count)
        nonzero = areas[areas > 0]
        if len(nonzero) < 8:
            continue
        a_med = float(np.median(nonzero))
        if a_med <= 0:
            continue
        slab_h = float((edges[-1] - edges[0]) / slab_count)

        low_mask = areas < a_med * _NECK_RATIO_MAX
        lead = _count_leading(low_mask)
        trail = _count_leading(low_mask[::-1])

        if lead >= trail:
            sprue_slabs = lead
            low_areas = areas[:lead]
            end = "lead"
        else:
            sprue_slabs = trail
            low_areas = areas[-trail:]
            end = "trail"

        sprue_len_mm = sprue_slabs * slab_h
        # Volume estimate: median cross-section among the slabs that actually
        # got a measurement, times the sprue length. We discard zero-area
        # slabs because cylindrical sprues are often coarsely tessellated
        # (<4 vertices per slab) and would otherwise drag the sum to zero.
        # The median (not max) keeps us close to the true uniform-stem
        # volume rather than over-shooting at the widening neck.
        low_nonzero = low_areas[low_areas > 0]
        if len(low_nonzero) == 0:
            continue
        typical_area = float(np.median(low_nonzero))
        sprue_vol_mm3 = typical_area * sprue_len_mm
        if sprue_len_mm < _MIN_SPRUE_LEN_MM or sprue_vol_mm3 < _MIN_SPRUE_VOL_MM3:
            continue

        # Stronger neck (lower min-to-median ratio) → higher confidence.
        ratio = float(nonzero.min() / a_med)
        confidence = float(np.clip(1.0 - ratio / _NECK_RATIO_MAX, 0.40, 0.95))

        if best is None or sprue_vol_mm3 > best[0]:
            best = (sprue_vol_mm3, confidence, axis_idx, end)

    if best is None:
        return False, None, 0.0, None, None
    vol, conf, axis, end = best
    return True, vol, conf, axis, end


# ── Casting-rail detection (separate-component sprue trees) ──────────────────

def detect_casting_rails(
    mesh: trimesh.Trimesh,
    *,
    min_slenderness: float = 8.0,
    min_span_fraction: float = 0.5,
    overlap_samples: int = 1500,
    seed: int = 0,
) -> tuple[float, float, int] | None:
    """Detect sprue *rails* — the runner bars of a casting tree.

    `5 MM KOLYE.stl` (ground-truthed on the workshop scale, 2026-06-11)
    is 40 chain links hung on two 153 mm runner bars. The slab "neck"
    detector can't see this: the sprue is fatter than the pieces, so
    there is no low-area neck. But the rails are unmistakable as
    *components*: extremely slender (length/width ≈ 95) and spanning
    almost the whole bounding box.

    A component is a rail when:
      * slenderness = max_extent / mid_extent  >  `min_slenderness`, and
      * its max extent > `min_span_fraction` × the whole mesh's max extent.

    Because the links are modelled overlapping the rails (the feeder
    joints), the merged signed volume double-counts the overlap. We
    estimate the overlap by point-sampling each link∩rail AABB and
    testing containment in both bodies.

    Returns `(rails_volume_mm3, overlap_volume_mm3, n_rails)` or `None`
    when no rails are present (single-piece mesh — caller falls through
    to the slab detector).

    Validation vs scale: rails 650.1 mm³ → 6.73 g silver (real 6.86 g,
    −1.9 %); net piece 439.9 mm³ → 4.56 g (real 4.40 g, +3.6 %).
    """
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        return None
    if len(comps) < 2:
        return None

    overall = float(mesh.extents.max())
    rails: list[trimesh.Trimesh] = []
    pieces: list[trimesh.Trimesh] = []
    for c in comps:
        ext = np.sort(c.extents)[::-1]      # descending
        slender = ext[0] / max(1e-9, ext[1])
        if slender > min_slenderness and ext[0] > min_span_fraction * overall:
            rails.append(c)
        else:
            pieces.append(c)
    if not rails or not pieces:
        return None

    rails_vol = float(sum(abs(r.volume) for r in rails))
    if rails_vol <= 0:
        return None

    # Overlap: links are attached to the rails through small fused joints.
    rng = np.random.RandomState(seed)
    overlap = 0.0
    for piece in pieces:
        for rail in rails:
            lo = np.maximum(piece.bounds[0], rail.bounds[0])
            hi = np.minimum(piece.bounds[1], rail.bounds[1])
            if np.any(hi <= lo):
                continue
            box_vol = float(np.prod(hi - lo))
            if box_vol <= 0:
                continue
            pts = rng.uniform(lo, hi, (overlap_samples, 3))
            try:
                inside = piece.contains(pts) & rail.contains(pts)
            except Exception:
                continue
            overlap += box_vol * float(inside.mean())

    return rails_vol, overlap, len(rails)


# ── Bore-bar detection (support stem threaded through a ring's finger hole) ──

# A real finger bore is a large fraction of the ring's outer radius. Anything
# smaller is a decorative hole or a gap between components, not a bore, and the
# material "inside" it is the piece itself. Measured on the golden STLs
# (2026-07-30): ring 17 -> 0.02, pave-bracelet -> 0.09; the alyans rings that
# genuinely carry a bar -> 0.77-0.82.
_BORE_MIN_RADIUS_FRACTION = 0.35

# A casting support bar is a thin stick: across 14 real alyans files it was
# 4.2-13.6 % of the piece. "FMR-32 11 boy (3 ADET MUM).stl" -- three separate
# waxes arranged around a runner -- puts 52 % of its volume inside the empty
# space between them, which is not a sprue bar but the pieces themselves.
# This cap keeps that class of model out. It is defence in depth: such models
# are already claimed by the slab detector, which runs first.
_BORE_MAX_SPRUE_FRACTION = 0.25

_BORE_MIN_SPRUE_MM3 = 3.0
_BORE_COVERAGE_SAMPLES = 150_000
_BORE_COVERAGE_BINS = 120
_BORE_COVERAGE_SECTORS = 36
_BORE_COVERAGE_MIN = 0.70


def _ring_axis(mesh: trimesh.Trimesh) -> int:
    """Index of the axis the ring is threaded on (the finger axis).

    Seen down that axis a ring projects to a circle, so the *other two*
    extents are nearly equal — a more reliable cue than "smallest extent",
    which flips on wide bands.
    """
    e = mesh.extents
    return min(
        range(3),
        key=lambda a: abs(e[(a + 1) % 3] - e[(a + 2) % 3])
        / max(e[(a + 1) % 3], e[(a + 2) % 3], 1e-9),
    )


def _bore_radius_by_angular_coverage(
    mesh: trimesh.Trimesh,
    axis_idx: int,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Inner radius of the ring wall, immune to a bar crossing the hole.

    Returns ``(inner_radius_mm, outer_radius_mm, centre_2d)``.

    Area-weighted surface samples are binned by (radius, angular sector).
    The ring wall covers every sector at every radius from the inner wall
    outwards; a straight bar can only ever occupy two opposite sectors.

    The scan runs from the OUTSIDE IN, which matters: scanning outwards from
    the centre reports a false "fully covered" ring immediately, because a
    bar passing through the middle wraps around every angle within the first
    few tenths of a millimetre. Tried the wrong way round first — it returned
    bores of 0.12-0.38 mm on rings whose real bore is ~18 mm (2026-07-30).

    Unlike `measurements._largest_passable_bore`, this does not search for an
    *empty* circle: a bar splits the hole in two and that search then returns
    roughly half the true bore (8.9 mm instead of 17.8 mm on the
    ``alyans5 ... prc*.stl`` files, where the bar is finely tessellated).
    """
    pts, _ = trimesh.sample.sample_surface(mesh, _BORE_COVERAGE_SAMPLES, seed=seed)
    other = [a for a in range(3) if a != axis_idx]
    flat = pts[:, other]
    centre = (flat.max(axis=0) + flat.min(axis=0)) / 2.0
    delta = flat - centre
    radius = np.linalg.norm(delta, axis=1)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    r_outer = float(radius.max())
    if r_outer <= 0:
        return 0.0, 0.0, centre

    r_bin = np.clip(
        (radius / r_outer * _BORE_COVERAGE_BINS).astype(int),
        0, _BORE_COVERAGE_BINS - 1,
    )
    s_bin = np.clip(
        ((theta + np.pi) / (2 * np.pi) * _BORE_COVERAGE_SECTORS).astype(int),
        0, _BORE_COVERAGE_SECTORS - 1,
    )
    occupied = np.zeros((_BORE_COVERAGE_BINS, _BORE_COVERAGE_SECTORS), dtype=bool)
    occupied[r_bin, s_bin] = True
    coverage = occupied.mean(axis=1)

    covered = np.where(coverage >= _BORE_COVERAGE_MIN)[0]
    if len(covered) == 0:
        return 0.0, r_outer, centre
    i = int(covered[-1])
    while i > 0 and coverage[i - 1] >= _BORE_COVERAGE_MIN:
        i -= 1
    return float(i / _BORE_COVERAGE_BINS * r_outer), r_outer, centre


def _bore_cylinder(
    mesh: trimesh.Trimesh,
    axis_idx: int,
    centre_2d: np.ndarray,
    radius: float,
    pad_mm: float = 6.0,
) -> trimesh.Trimesh:
    """A cylinder filling the finger bore, longer than the band so it cuts
    clean through."""
    height = float(mesh.extents[axis_idx]) + pad_mm
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=192)
    transform = np.eye(4)
    if axis_idx == 0:
        transform[:3, :3] = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    elif axis_idx == 1:
        transform[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    other = [a for a in range(3) if a != axis_idx]
    centre_3d = np.zeros(3)
    centre_3d[other[0]] = centre_2d[0]
    centre_3d[other[1]] = centre_2d[1]
    centre_3d[axis_idx] = float(mesh.bounds[:, axis_idx].mean())
    transform[:3, 3] = centre_3d
    cyl.apply_transform(transform)
    return cyl


def _to_manifold(mesh: trimesh.Trimesh):
    """Wrap a trimesh in a manifold3d solid.

    Used instead of `trimesh.boolean.*` on purpose. Real workshop STLs
    routinely carry a handful of non-manifold edges (edges shared by 4 or 6
    faces, where the bar's shell interpenetrates the ring's). That makes
    `mesh.is_watertight` False and trimesh's boolean wrapper bail out — 8 of
    the 14 alyans files failed exactly that way. manifold3d accepts every one
    of them and reproduces trimesh's own volume to the last decimal, so the
    geometry is handed to it directly.
    """
    import manifold3d  # noqa: PLC0415  (optional dependency, imported lazily)

    solid = manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    )
    return manifold3d.Manifold(solid)


def detect_bore_sprue(
    mesh: trimesh.Trimesh,
) -> tuple[bool, float | None, float]:
    """Find a casting support bar running through a ring's finger bore.

    Returns ``(detected, volume_mm3, confidence)``.

    This is a third sprue topology that neither existing detector can see:

      * `detect_sprue_geometric` looks for a low-area neck at the END of an
        axis. This bar sits in the MIDDLE of the bounding box, so there is no
        neck on any axis — it returned False on all 14 alyans files.
      * `detect_casting_rails` looks for separate slender components. This bar
        is fused into the same shell as the ring; every alyans file splits
        into exactly one component.

    Physically the rule is simple: a finger bore is empty by definition, so
    any solid inside it is scaffolding, not the piece. The volume is then an
    exact boolean intersection rather than an estimate — two cheaper
    approximations were tried first and rejected, a surface-area cylinder
    formula (up to 43 % off) and a capped sub-mesh (up to 97 % off on finely
    tessellated bars).

    Ground truth (14 real ``alyans*.stl`` wedding bands, 2026-07-30): a
    ~1.2 mm bar spanning the bore, 20.3-28.8 mm3, i.e. 4.2-13.6 % of the
    piece — enough to matter on a scale, which is why it must not be silently
    rolled into the metal weight.
    """
    try:
        axis_idx = _ring_axis(mesh)
        r_inner, r_outer, centre = _bore_radius_by_angular_coverage(mesh, axis_idx)
    except Exception:
        return False, None, 0.0

    if r_outer <= 0 or r_inner <= 0:
        return False, None, 0.0
    if r_inner / r_outer < _BORE_MIN_RADIUS_FRACTION:
        return False, None, 0.0

    try:
        piece = _to_manifold(mesh)
        bore = _to_manifold(_bore_cylinder(mesh, axis_idx, centre, r_inner))
        volume = float((piece ^ bore).volume())
        total = float(piece.volume())
    except Exception:
        # manifold3d missing, or the solid was rejected — never break the
        # pipeline over an optional refinement.
        return False, None, 0.0

    if total <= 0 or volume < _BORE_MIN_SPRUE_MM3:
        return False, None, 0.0
    if volume / total > _BORE_MAX_SPRUE_FRACTION:
        return False, None, 0.0

    return True, volume, 0.85


# ── Public entry point: hybrid geometric + VLM ───────────────────────────────

def detect_sprue(
    vlm_says_sprue: bool,
    vlm_confidence: float,
    mesh: trimesh.Trimesh | None = None,
) -> SpruInfo:
    """Combine geometric + VLM signals into a single SpruInfo.

    Precedence:
      1. **Geometric hit** → trust it (we have a volume estimate). If
         VLM also says yes, mark source="vlm_geometry_combined" and
         boost confidence.
      2. **VLM hit, no geometric** → believe the yes/no but leave
         volume None (weight correction silently skipped).
      3. **Neither** → no sprue.
    """
    geom_detected = False
    geom_vol: float | None = None
    geom_conf = 0.0
    cut_overlap = 0.0
    if mesh is not None:
        # Casting-tree rails first (separate runner-bar components). These
        # are invisible to the slab detector because the sprue is *fatter*
        # than the pieces it feeds.
        try:
            rails = detect_casting_rails(mesh)
        except Exception:
            rails = None
        if rails is not None:
            rails_vol, overlap, _n = rails
            geom_detected = True
            geom_vol = rails_vol
            cut_overlap = overlap
            geom_conf = 0.90
        else:
            try:
                geom_detected, geom_vol, geom_conf, _, _ = detect_sprue_geometric(mesh)
            except Exception:
                # Sprue detection is best-effort — never break the pipeline.
                geom_detected = False
            if not geom_detected:
                # Last: a support bar threaded through a ring's finger bore.
                # Deliberately AFTER the slab detector, not before it —
                # multi-piece casting models such as `FMR-32 11 boy
                # (3 ADET MUM).stl` park 52 % of their volume in the empty
                # space between the waxes, which this detector would read as
                # one enormous bar. The slab detector claims those first
                # (golden value 265.6 mm3), so ordering keeps that result
                # intact instead of relying on the fraction cap alone.
                try:
                    geom_detected, geom_vol, geom_conf = detect_bore_sprue(mesh)
                except Exception:
                    geom_detected = False

    if geom_detected:
        # When both signals agree, blend the two confidences (capped at
        # 0.95 because we still don't have a ground-truth source).
        if vlm_says_sprue:
            blended = min(0.95, geom_conf + 0.20 * vlm_confidence)
            source = "vlm_geometry_combined"
        else:
            blended = geom_conf
            source = "geometry"
        return SpruInfo(
            detected=True,
            source=source,
            confidence=float(blended),
            estimated_volume_mm3=geom_vol,
            cut_overlap_mm3=float(cut_overlap),
        )

    if vlm_says_sprue:
        return SpruInfo(
            detected=True,
            source="vlm",
            confidence=float(min(1.0, max(0.0, vlm_confidence * 0.6))),
            estimated_volume_mm3=None,
        )

    return SpruInfo(
        detected=False,
        source="none",
        confidence=0.5,
        estimated_volume_mm3=None,
    )
