# The render engine

The application draws jewellery with a **web renderer running inside the app**: three.js in an
embedded Chromium (`PyQt6-WebEngine`). The same engine drives the live players in the Render tab, the
PNG export and the images of the PDF catalogue, so what you judge on screen is what is printed.
A VTK renderer remains as a fallback for scripts and tests (set `CATALOG_RENDER_ENGINE=vtk` to force
it). Nothing is exported by the user: scenes are prepared on first use and cached.

```
CAD file ──► webview.export ──► GLB scene + shared env ──► cache/web/site/
                                                              │ served on 127.0.0.1
         live player (visible Chromium) ◄────────────────────┤
         PNG / PDF images (hidden Chromium, synchronous capture) ◄┘
```

## Pieces

| Module | Role |
|---|---|
| `webview/export.py` | Loads a product, turns it into a glTF scene: decimated metal with baked occlusion, one node per stone with its frame, the cut's facet planes, the camera poses |
| `webview/scene_cache.py` | The cache. Key = source file (path, size, mtime) + category + look + options + `SCENE_VERSION`. A hit is instant |
| `webview/service.py` | `WebRenderService`: a hidden Chromium on the GUI thread plus a loopback HTTP server (`127.0.0.1`, random port). `capture()` can be called from any thread |
| `webview/engine.py`, `render/engine.py` | `render_views()` returns the same `RenderResult` the VTK path did; `render/engine.py` picks the engine |
| `webview/assets/viewer.js` | The viewer: scene, metal, stone shader, accumulation, tone mapping, the host API |
| `webview/ao.py` | Per-vertex ambient occlusion by marching rays through a voxel grid (numpy) |
| `webview/seats.py` | Finds stone seats in a stoneless STL and places stones in them |
| `webview/look.py` | The look parameters (`Look`), shared by every product |
| `gui/widgets/web_player.py` | The live player widget |

three.js (r152) is vendored under `webview/assets/vendor/three` (MIT), so the app runs offline.

## The look

A flat, bright product-photography look: one page colour, ACES filmic tone mapping applied **once**
(after accumulation, so stones and metal share a single operator), a very long lens (10° field of
view), no ground shadow. The numbers are in `webview/look.py`; they were fitted so that yellow gold's
mean colour, brightness distribution and edge contrast land where a luxury-retail viewer's do, and are
tuning values rather than physical constants. Replace the `Look` to change the whole appearance.

* **Progressive accumulation.** Up to 16 frames with a Halton-jittered sub-pixel camera are averaged
  in a half-float target, then tone-mapped. Free anti-aliasing, and a still image sharpens as you wait.
* **Camera.** Every product is turned into the camera's frame (camera on +Z looking at the origin,
  +Y up), so a single environment serves them all. The pose is chosen by the same function as the
  VTK renderer and depends on the category, so products of one category are framed alike.

## Metal

PBR with a baked per-vertex ambient occlusion attribute. CAD meshes have no UVs, so occlusion cannot be
a texture; it is baked per vertex by marching ~48 cosine-distributed rays through a voxel grid
(300 voxels across the piece, ray length 11.7 % of the piece). Notes from measuring it:

* A large flat face with vertices only on its rim inherits the rim's darkness. Long edges are split
  (longest-edge bisection, not 1→4, which inflates the long thin triangles CAD exports are full of)
  before the bake.
* Catalogue models are often *overlapping solids*. A vertex buried inside another body is correctly
  fully occluded, but its triangles poke through the visible surface as dark patches. Buried vertices
  (no open air within a voxel, by flood fill from outside the grid) are given neutral occlusion.
* The triangle budget (120 000) is applied *after* the splitting.

## Stones

Each stone is ray-traced **analytically against the planes of its own facets**: a cut stone is about
120 planes, so the exact answer is cheap. Facets are baked into the GLSL as constants and unrolled; each
stone's position and orientation arrive as uniforms, so a 191-stone pavé piece compiles one shader.
Dispersion traces each colour channel at its own refractive index; absorption follows the stone's
colour. Stones whose CAD has no real cut (smooth placeholders) borrow a standard 57-facet brilliant's
interior while keeping their own silhouette, position and size.

The refraction looks up an environment. A built-in procedural showroom is used; an optional HDR in
`assets/hdr/` adds structure (none is shipped).

## Placing stones in a stoneless STL

A casting-model STL has the seats but not the stones. `webview/seats.py`:

1. runs the seat detector (`cad/stl_stone_seats.py`, which slices along +Z) in **six directions** by
   turning the piece so each axis in turn faces +Z;
2. keeps a seat only if it is round (≥ 0.85), narrows like a cone, is neither a film nor a bore
   (depth between 0.1 and 1.5 diameters), is **open to the sky above its entrance**, and has metal
   near it; a seat seen from two axes is kept once;
3. puts a standard brilliant at each entrance, axis along the opening direction.

Every filter exists because of a measured false positive on real files (a ring's finger bore read as a
7.9 mm seat; relief pockets in a sculpted pendant; gallery holes drilled from inside a band; seats
reported far from any metal because centres were once measured in each slice's own frame). The raw scan
is cached per source file; filters run on the cached scan, so tuning them needs no rescan.

Placement is **a request, never a default**: the product's catalogue record decides, and the user's
per-product choice (stored in `render_options`) always wins.

## Performance (RTX 3090, measured)

| | |
|---|---|
| Both catalogue views at 2000 px | ≈ 1.1 s (3000 px ≈ 1.4 s) once the scene exists |
| First open of a product | 1–20 s (occlusion bake dominates) |
| Seat scan of a stoneless STL | 30–300 s, once per source file |
| A typical scene | 0.5–7 MB |

## Limits

* Needs WebGL 2 with half-float render targets.
* Per-vertex occlusion cannot resolve detail finer than the mesh; thin-sliver surfaces can show faint
  streaks. A UV-baked texture would fix it and is not done.
* STL stones are inferred from seats and are estimates. Seats at a steep angle between the six axes
  can be missed.
