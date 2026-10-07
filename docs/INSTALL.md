# Installation

Tested on **Windows 11** with **Python 3.12**. Python 3.11 is supported by the package metadata.
Other operating systems are untested.

## Requirements

| | |
|---|---|
| Python | 3.11 or 3.12 (64-bit) |
| GPU | Anything with **WebGL 2**. The render engine draws in an embedded Chromium on the GPU. |
| RAM | 16 GB recommended; production files can have millions of triangles |
| Optional | [Ollama](https://ollama.com) for local AI classification; your own licensed Rhino 5–8 for exact cutter volumes |

## Install

```powershell
git clone https://github.com/emrekaandrms/catalog-organizer.git
cd catalog-organizer
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

`pip install -e .` (an *editable* install) is required: the application looks for `config/`, `data/` and
`cache/` next to the source tree, so run it from the checkout.

Start the app:

```powershell
python -m catalog_organizer.app
```

### Optional pieces

**AI classification (local).** Install [Ollama](https://ollama.com), then:

```powershell
ollama pull qwen3.5:9b-q4_K_M
```

Leave Ollama running. The default endpoint is `127.0.0.1:11434`; change it, or switch provider, in
**Settings → VLM Provider**.

**AI classification (cloud).** In **Settings → VLM Provider** pick OpenAI, MiniMax, GLM, OpenCode Zen
or an OpenAI-compatible endpoint and paste the API key. Keys are stored in the operating system's
credential store (via `keyring`), never in `config/`. Images are then sent to that provider.

**Exact cutter volumes with Rhino.** If you own Rhino 5–8:

```powershell
pip install -e ".[rhino]"
```

Without it the app uses a documented heuristic for stone-seat cutters.

**A showroom HDR for stones (optional).** The stone shader looks things up in an environment map. The
project uses a built-in procedural studio by default. If you have a rights-cleared equirectangular
HDR you prefer, drop it into `assets/hdr/` as `env-gem-1.hdr` (and `env-metal-6.hdr` for metal).
None is shipped.

## Check that it works

1. `python -m catalog_organizer.app` opens the window. A start-up check looks at the model provider
   (Ollama, if you chose it) and at free disk space and tells you what is missing.
2. **Render** tab → *Dosyadan aç…* → [`examples/demo_ring.stl`](../examples/README.md). Two live views
   of a ring appear. Tick **Taşları yuvalara yerleştir** and the seat gets a stone.
3. **Analyze** tab → *Analyze file…* → the same file: weights and dimensions appear (no AI needed).

## Run the tests

```powershell
pip install -e ".[dev]"
pytest
```

Tests that draw in the embedded Chromium need a GPU. Skip them with `set CATALOG_ORGANIZER_NO_GPU=1`.
Tests that need private CAD files skip themselves when the files are absent.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: DLL load failed ... rhino3dm` | Use the pinned version: `pip install "rhino3dm>=8.9,<8.18"` (newer builds crashed on import in testing). |
| `No module named PyQt6.QtWebEngineWidgets` | `pip install "PyQt6-WebEngine>=6.7,<6.8"` (the minor version must match PyQt6). |
| Render tab blank, "görüntüleyici hazır olmadı" | WebGL 2 is missing. Update the graphics driver; remote desktops and some VMs lack it. |
| `vtk` import errors on a new Python | Stay on Python 3.11 / 3.12; the VTK 9.3 wheels target them. |
| Nothing is found / config errors | Run the command from the project folder with the virtual environment active. |

More: [Kullanım kılavuzu §11](KULLANIM_KILAVUZU.md#11-sorun-giderme).
