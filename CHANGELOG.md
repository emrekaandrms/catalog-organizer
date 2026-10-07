# Changelog

All notable changes. The format follows [Keep a Changelog](https://keepachangelog.com).

## [0.1.0] - first public release

### Added
* Scanner, snapshot generator, vision-model classifier (Ollama, OpenAI, MiniMax, GLM, OpenCode Zen,
  OpenAI-compatible endpoints), geometric extractors (dimensions, ring bore, metal weights for eight
  alloys, stones, sprue, scale check) and a searchable SQLite catalogue.
* Publish tab: selection lists, listing copy, deterministic pricing, Etsy and WooCommerce CSV export,
  PDF catalogue with two views per page.
* **Render engine**: three.js in an embedded Chromium. Live KARSIDAN / CAPRAZ players, instant metal and
  stone colours, saved camera angles, PNG export; the same engine draws the PDF images. Analytic facet
  ray tracing for stones, per-vertex occlusion for metal, progressive accumulation, a scene cache.
  A VTK renderer remains as a fallback (`CATALOG_RENDER_ENGINE=vtk`).
* Stone placement in the seats of a stoneless STL: opt-in per product, defaulting to the catalogue
  record; scans six directions with filters for round, tapering, open seats.
* A rights-free demo ring (`examples/make_demo_ring.py`).
* Documentation in English and Turkish (user guide / kullanim kilavuzu, install, architecture, render
  engine).

### Notes
* Tested on Windows 11 and Python 3.12 only.
