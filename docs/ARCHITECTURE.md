# Architecture

Catalog Organizer is a desktop application (PyQt6) over a local pipeline. Everything runs on the
user's machine; the only network traffic is to the model provider the user configured (Ollama on
`127.0.0.1` by default).

## Data flow

```
 folder of .3dm / .stl
        │  scanner/        discover, hash, give each file a permanent id (JCAD-000000123)
        ▼
 snapshotter/ + cad/       load geometry, render the snapshot images, measure:
        │                  dimensions, ring bore, metal volume and weights, stones, sprue, scale check
        ▼
 vlm/                      send the snapshots to a vision model: category, sub-category,
        │                  controlled tags, brand hint, description (JSON, validated)
        ▼
 orchestrator/             merge geometry + model output into one CatalogRecord; flag low confidence
        ▼
 catalog/ + db/            catalog_master.jsonl and data/catalog.db (SQLite, FTS5)
        ▼
 gui/ search · publish     filter, select, correct → listing/ (copy, prices) → export/ (CSV, PDF)
        ▼
 render/ + webview/        live 3D players, PNG, PDF images
```

## Packages

| Package | Responsibility |
|---|---|
| `app.py` | Entry point: GUI plus the `analyze`, `run-pilot`, `validate`, `webview` commands |
| `core/` | Paths, config loading, schemas (pydantic), audit log, cost tracker, ids, OS credential store (`secrets`), start-up checks |
| `scanner/` | File discovery and the manifest |
| `snapshotter/` | `.3dm` (rhino3dm) and `.stl` (trimesh) loading, metal volume, snapshots |
| `cad/` | Measurements, stone extraction (`.3dm` layers), stone seats (`.stl`), sprue detection, scale check, optional Rhino engine |
| `vlm/` | Provider protocol, Ollama client, OpenAI / MiniMax / OpenAI-compatible providers, prompts, parsing, text-only provider for listing copy, cost table |
| `orchestrator/` | `pipeline` (one file), `batch` (many, threaded, pausable), `analyze` (geometry only) |
| `catalog/` | Index, writer, in-memory search |
| `listing/` | Listing copy generation, deterministic pricing, validation |
| `export/` | Etsy and WooCommerce CSV, the PDF catalogue (Qt `QPdfWriter`) |
| `render/` | VTK renderer (fallback), stone optics (`facets`, `gem_trace`), materials, studio lighting |
| `webview/` | The web render engine (see [RENDER_ENGINE.md](RENDER_ENGINE.md)) |
| `db/` | SQLite schema and access: products, overrides, selections, listings, exports, camera overrides, render options |
| `gui/` | Panels and widgets; long work runs in `QThread`s so the window never freezes |

## Storage

* `data/catalog_master.jsonl` — the master record, one JSON object per product.
* `data/catalog.db` — SQLite with FTS5 search, selections, listings and variants, exports, camera
  overrides, render options. Created and migrated idempotently on every start (`db/schema.py`).
* `cache/` — snapshots, rendered images and 3D scenes. Safe to delete.

Paths are relative to the project root (`core/paths.py`); an editable install is therefore required.

## Design decisions worth knowing

* **Deterministic where possible.** Geometry, weights and prices involve no model. The vision model
  only supplies what geometry cannot: category, tags, a description, a brand hint.
* **The model never decides a number.** Prices come from formulas in `listing/pricing.py` and your
  config; weights from volume and density tables.
* **API keys never touch disk.** They live in the OS credential store.
* **Estimates say so.** Records carry confidence labels; stone and weight figures are flagged as
  estimates where they are.
* **One user choice, one place.** Per-product render decisions (a saved camera angle; whether stones
  are placed in the seats of an STL) are stored in the database and read by the Render tab, PNG export
  and the PDF catalogue alike, so a picture and the printed page never disagree.

## Tests

`pytest` runs ~550 tests. They cover geometry, weights, pricing, exports, the GUI panels (with stubbed
players), the render engine (real Chromium; needs a GPU, skippable with `CATALOG_ORGANIZER_NO_GPU=1`)
and the stone-seat logic against synthetic geometry. Tests that need private CAD files skip themselves.
