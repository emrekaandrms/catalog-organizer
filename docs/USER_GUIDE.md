# Catalog Organizer — User Guide

The full step-by-step manual is in Turkish: [KULLANIM_KILAVUZU.md](KULLANIM_KILAVUZU.md). This is the
English version, a little more compact. Installation: [INSTALL.md](INSTALL.md).

## What it does, and does not do

It **finds** `.3dm` / `.stl` jewellery CAD files, **measures** them (dimensions, ring bore, metal
weights for eight alloys, stone sizes and carats, sprue detection, all from geometry), **classifies**
them with a vision model (category, sub-category, tags, brand hint, description), stores everything in
a **catalogue** you can search, and turns a selection into **listings** (Etsy / WooCommerce CSV) and a
**PDF catalogue**. It also shows any piece in a **live 3D viewer**.

It never modifies, moves or deletes your CAD files. It does not upload listings for you (it writes the
CSVs). Weights are estimates; check them against a scale.

## The tabs

`Ctrl+1` … `Ctrl+9` switch tabs.

| Tab | Purpose |
|---|---|
| Dashboard | Totals, catalogue composition, recent activity |
| Analyze | Geometry-only analysis of a file or folder (no AI needed); export CSV |
| Process | Full pipeline over a folder: snapshots → vision model → measurements → catalogue; CSV |
| Search | Filters (category, sub-category, stone status, brand, sprue) + free text |
| Publish | Selection lists, listing copy, prices, Etsy / WooCommerce export, PDF catalogue |
| Render | Live 3D players for the two catalogue views; metal / stone colours; saved angles; PNG |
| Diagnostics | Run one pipeline stage at a time and time it |
| Settings | Paths, pipeline, model provider, prices, theme |
| Logs | Live audit log with a severity filter |

## Workflows

### Quick analysis (no AI)

**Analyze** → *Analyze file…* or *Analyze folder…* → read the table → *Export CSV…*. The command line
does the same: `python -m catalog_organizer.app analyze --folder DIR --csv out.csv`.

### Catalogue an archive

**Process** → *Pick folder…* → *Start*. Each file gets a permanent id (`JCAD-000000123`), then is
loaded, photographed, sent to the vision model, measured and written to the catalogue. *Pause* and
*Cancel* are safe; a re-run skips files already done. *Open CSV* shows the result so far. Start with
5–20 files and look at the results before you run the whole archive.

### Search, select, publish

1. **Search**: filter, then *Tick all* / *Add checked* / *Add all matching* to add results to a list.
2. **Publish** → *New list*; use *Correct category / brand* where the model was wrong.
3. *Generate listing copy* (text model; ticked items only, or the whole list), *Recalculate prices*
   (from `config/pricing.yaml` and **Settings → Pricing**, **example values you must replace**),
   choose the channel and *Export*.
4. *PDF katalog* asks for a metal and a stone colour and writes an A4 PDF: a cover, then one page per
   product with the two views side by side, id, category, weight and size.

### Render

Pick a product (or *Dosyadan aç…* for any `.3dm` / `.stl`). The first open of a product takes seconds
(minutes for a big stoneless STL, which is scanned for stone seats); the result is cached and the next
open is instant. Two live players show **KARŞIDAN** (front) and **ÇAPRAZ** (three-quarter), the two
views a catalogue page prints. Orbit with the mouse, change metal and stone colours instantly, and use
*Bu açıyı kaydet* to keep an angle you like as that view for that product (the PDF and PNGs use it).
*PNG kaydet…* writes both views at the chosen size. Products of one category are always shot from the
same angle.

## STL or 3DM?

| | `.3dm` | `.stl` |
|---|---|---|
| Stones | Real geometry in their own layer | **None**: an STL is the piece before the machine |
| Stone data | Read exactly from the layer | Only the seats are there; the app measures them |
| Reliability | High | An estimate |

Give the app **`.3dm`** where stone position, size and count matter. STL is fine too, but stone
information is then inferred.

For `.3dm`: put stones on a layer whose name contains `gem` or `stone`, keep seat cutters on a
`Cutter` / `Cutting Objects` layer (they are subtracted from the weight), and keep notes and names
away from the piece. The rules live in `src/catalog_organizer/cad/stone_extraction.py`.

### Placing stones in the seats of a stoneless STL

The Render tab has a checkbox, **Taşları yuvalara yerleştir** ("Place stones in the seats"). The
detector finds seats that are round, taper like a cone and are open to the sky, and puts a standard
brilliant in each. Rules:

* **Off by default.** It is on by default only when the product's catalogue record says it has
  stones; a product recorded as stoneless never gets stones unless you tick the box.
* The choice is **yours, per product, and stored**; the PDF catalogue and PNG export use the same answer.
* Pavé channels, perforated bands and gallery holes are *not* seats and get nothing; seats at a steep
  angle may be missed. **Check the picture.** A stone in the wrong place → untick the box.
* The first open of a large STL scans the mesh in six directions and can take a minute or two.

## Data and backups

Everything lives under the project folder. `data/` (catalogue database, master JSONL, audit log) is
precious: back it up. `cache/` (snapshots, renders, 3D scenes) can be deleted at any time and is rebuilt.

## Troubleshooting

* **"3D oynatıcı için PyQt6-WebEngine gerekli"**: `pip install -e .` again.
* **Blank Render tab / "görüntüleyici hazır olmadı"**: the render engine needs WebGL 2. Update the
  graphics driver; remote desktops and some VMs do not have it.
* **Model not reachable**: is Ollama running (`ollama list`) and the model pulled? Check
  **Settings → VLM Provider** (default `127.0.0.1:11434`).
* **Weights differ from the scale**: they are estimates; overlapping solids, thin walls and STL files
  deviate most.

## Accuracy

Weights were calibrated against workshop-scale measurements on a small set of real files (typically
within a few percent, some files further off). Stone data is exact only for `.3dm`; classification
quality depends on the model you choose, and low-confidence records are flagged for review. Tested on
Windows 11 only.
