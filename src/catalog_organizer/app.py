"""
Entry point.

GUI:        python -m catalog_organizer.app
Analyze:    python -m catalog_organizer.app analyze FILE [FILE ...] [--csv out.csv]
            (VLM-free: dimensions, weights, stones, sprue — geometry only)
Pilot:      python -m catalog_organizer.app run-pilot --n 5 [--roots PATH]
Validate:   python -m catalog_organizer.app validate --truth path/to/truth.csv [--out report.md]
Web view:   python -m catalog_organizer.app webview JCAD-... [--selection NAME] [--all] [--serve]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        sys.exit(_run_analyze_cli(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "run-pilot":
        sys.exit(_run_pilot_cli(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        sys.exit(_run_validate_cli(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "webview":
        from catalog_organizer.webview.cli import run as _run_webview  # noqa: PLC0415
        sys.exit(_run_webview(sys.argv[2:]))
    _run_gui()


def _run_analyze_cli(argv: list[str]) -> int:
    """Geometric analysis (no VLM / Ollama needed).

    Usage: analyze FILE [FILE ...] [--csv out.csv] [--golden test_case/expected.json]
           analyze --folder DIR [--csv out.csv]

    --golden captures the results as the expected-values file used by
    tests/test_golden_files.py (regression pinning for real workshop files).
    """
    import csv as _csv  # noqa: PLC0415
    import io  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    # Windows console defaults to a legacy codepage; reports contain ° etc.
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from catalog_organizer.orchestrator.analyze import (  # noqa: PLC0415
        analyze_file, render_text, to_flat_dict,
    )

    files: list[Path] = []
    csv_out: Path | None = None
    golden_out: Path | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--csv" and i + 1 < len(argv):
            csv_out = Path(argv[i + 1]); i += 2
        elif tok == "--golden" and i + 1 < len(argv):
            golden_out = Path(argv[i + 1]); i += 2
        elif tok == "--folder" and i + 1 < len(argv):
            folder = Path(argv[i + 1]); i += 2
            files.extend(sorted(
                p for p in folder.rglob("*") if p.suffix.lower() in (".3dm", ".stl")
            ))
        else:
            files.append(Path(tok)); i += 1

    files = [f for f in files if f.exists()]
    if not files:
        print("No input files. Usage: analyze FILE [FILE ...] [--csv out.csv]")
        return 2

    rows = []
    golden: dict[str, dict] = {}
    any_error = False
    for f in files:
        result = analyze_file(f)
        print(render_text(result))
        print()
        flat = to_flat_dict(result)
        rows.append(flat)
        golden[str(f)] = flat
        any_error = any_error or bool(result.error)

    if golden_out is not None:
        golden_out.parent.mkdir(parents=True, exist_ok=True)
        golden_out.write_text(
            _json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"Golden expected-values written: {golden_out}")

    if csv_out is not None and rows:
        # Union of keys across rows, stable order from first occurrence.
        fields: list[str] = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with csv_out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written: {csv_out}")

    return 1 if any_error else 0


def _prepare_web_engine() -> bool:
    """The render engine is the web viewer in the app's own Chromium. Qt wants it announced BEFORE
    the QApplication exists (it shares one GL context between the visible player and the hidden
    one that draws catalogue images). False where PyQt6-WebEngine is not installed."""
    try:
        from PyQt6.QtCore import QCoreApplication, Qt  # noqa: PLC0415

        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        import PyQt6.QtWebEngineWidgets  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _run_gui() -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    from catalog_organizer.gui.startup_dialog import run_startup_check  # noqa: PLC0415
    from catalog_organizer.gui.theme import apply_dark_theme  # noqa: PLC0415

    web_engine = _prepare_web_engine()
    app = QApplication(sys.argv)
    if web_engine:
        from catalog_organizer.webview import service  # noqa: PLC0415
        service.install(app)
    app.setApplicationName("Catalog Organizer")
    app.setOrganizationName("JewelryTools")
    _apply_app_icon(app)

    accent = _read_accent_setting()
    apply_dark_theme(app, accent=accent)

    vlm = _read_vlm_settings()
    if not run_startup_check(
        ollama_host=vlm["host"],
        ollama_port=vlm["port"],
        vlm_model=vlm["model"],
        check_ollama=vlm["provider"] == "ollama",
    ):
        sys.exit(0)

    from catalog_organizer.gui.main_window import MainWindow  # noqa: PLC0415
    window = MainWindow()
    window.show()
    code = app.exec()
    if web_engine:
        service.uninstall()          # stops the loopback server before Qt tears Chromium down
    sys.exit(code)


def _run_pilot_cli(argv: list[str]) -> int:
    """Headless pilot runner.

    Usage: run-pilot [--n N] [--roots PATH ...] [--reprocess]

    --reprocess bypasses the catalog's normal skip-already-done behaviour
    (BatchRunner's skip_filter) — use it to force files that already have
    a catalog record through the pipeline again, e.g. after a rendering or
    classification bug fix invalidates previously-computed results.
    """
    n = 5
    roots: list[Path] = []
    reprocess = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--n" and i + 1 < len(argv):
            n = int(argv[i + 1])
            i += 2
        elif tok == "--reprocess":
            reprocess = True
            i += 1
        elif tok == "--roots":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                roots.append(Path(argv[i]))
                i += 1
        else:
            i += 1

    if not roots:
        roots = [Path(__file__).resolve().parents[2] / "tests" / "fixtures"]

    return _run_pilot(roots=roots, n=n, reprocess=reprocess)


def _select_pilot_entries(manifest: dict, roots: list[Path], n: int) -> list:
    """Pick up to `n` manifest entries that actually live under `roots`.

    `scan()` returns the FULL persistent manifest (every root ever scanned
    across every past session), not just this call's roots — a naive
    `list(manifest.values())[:n]` silently grabs whatever happens to be
    first on disk, which is very likely unrelated leftover entries from an
    earlier session (2026-07-24 regression: a pilot run against a fresh
    folder selected old, already-"missing" test fixture entries instead
    and produced success=0 fail=0 with no error).
    """
    root_strs = [str(r.resolve()) for r in roots]
    return [
        e for e in manifest.values()
        if any(e.source_path.startswith(rs) for rs in root_strs)
    ][:n]


def _run_pilot(roots: list[Path], n: int, reprocess: bool = False) -> int:
    from catalog_organizer.catalog.index import CatalogIndex  # noqa: PLC0415
    from catalog_organizer.catalog.tag_validator import TagValidator  # noqa: PLC0415
    from catalog_organizer.catalog.writer import CatalogWriter  # noqa: PLC0415
    from catalog_organizer.core.audit import AuditWriter  # noqa: PLC0415
    from catalog_organizer.core.config import (  # noqa: PLC0415
        load_metal_densities,
        load_pipeline_settings,
    )
    from catalog_organizer.core.ids import get_allocator  # noqa: PLC0415
    from catalog_organizer.core.paths import data_dir  # noqa: PLC0415
    from catalog_organizer.orchestrator.batch import BatchRunner  # noqa: PLC0415
    from catalog_organizer.orchestrator.pipeline import PipelineDeps  # noqa: PLC0415
    from catalog_organizer.scanner.manifest import scan  # noqa: PLC0415

    print(f"Scanning {len(roots)} root(s)…")
    manifest = scan(roots, batch_name="pilot", allocator=get_allocator())
    entries = _select_pilot_entries(manifest, roots, n)
    print(f"Selected {len(entries)} files for pilot.")

    settings = load_pipeline_settings()
    densities = load_metal_densities()
    vlm_cfg = settings.get("vlm", {})

    from catalog_organizer.vlm.providers import create_vlm_provider  # noqa: PLC0415
    deps = PipelineDeps(
        vlm_client=create_vlm_provider(vlm_cfg),
        tag_validator=TagValidator(),
        metal_densities=densities,
        scale_thresholds=settings.get("scale_thresholds", {}),
        snapshot_resolution=settings.get("snapshot", {}).get("resolution", 1024),
        thumbnail_size=settings.get("snapshot", {}).get("thumbnail_size", 256),
    )
    writer = CatalogWriter()
    index = CatalogIndex()
    audit = AuditWriter(data_dir() / "audit_log.jsonl")

    parallel_workers = int(vlm_cfg.get("parallel_workers", 2))
    print(
        f"VLM parallel workers: {parallel_workers}  "
        f"(set OLLAMA_NUM_PARALLEL={parallel_workers} on Ollama for actual parallelism)"
    )
    runner = BatchRunner(deps, writer, index, audit, parallel_workers=parallel_workers)
    runner.progressChanged.connect(
        lambda done, total: print(f"  [{done}/{total}]")
    )
    runner.fileFailed.connect(
        lambda fid, code: print(f"  FAIL {fid}: {code}")
    )

    success, fail = runner.run(entries, batch_name="pilot", skip_filter=reprocess)
    writer.close()
    audit.close()
    print(f"Done. success={success} fail={fail}")
    return 0 if fail == 0 else 1


def _run_validate_cli(argv: list[str]) -> int:
    """Validate predictions against a truth CSV.

    Usage: validate --truth path [--catalog path] [--out path]
    """
    from catalog_organizer.core.paths import data_dir  # noqa: PLC0415
    from catalog_organizer.validation.report import run_validation  # noqa: PLC0415

    truth: Path | None = None
    catalog: Path | None = None
    out: Path | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--truth" and i + 1 < len(argv):
            truth = Path(argv[i + 1]); i += 2
        elif tok == "--catalog" and i + 1 < len(argv):
            catalog = Path(argv[i + 1]); i += 2
        elif tok == "--out" and i + 1 < len(argv):
            out = Path(argv[i + 1]); i += 2
        else:
            i += 1

    if truth is None or not truth.exists():
        print("Error: --truth path/to/truth.csv is required and must exist.")
        return 2

    if catalog is None:
        catalog = data_dir() / "catalog_master.jsonl"
    if out is None:
        out = data_dir() / "validation_report.md"

    report, markdown = run_validation(truth, catalog, out_md=out)
    print(markdown)
    print(f"\nWrote {out}")
    return 0 if report.main_accuracy >= 0.95 else 1


def _read_accent_setting(default: str = "#1e88e5") -> str:
    try:
        import yaml  # noqa: PLC0415
        config_path = Path(__file__).resolve().parents[2] / "config" / "app_settings.yaml"
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            return data.get("theme", {}).get("accent_color", default)
    except Exception:
        pass
    return default


APP_USER_MODEL_ID = "JewelryTools.CatalogOrganizer"


def _apply_app_icon(app) -> None:
    """Set the window/taskbar icon, best-effort.

    On Windows the taskbar groups by AppUserModelID, and a Python process
    inherits the interpreter's. Without setting our own, Windows shows the
    generic Python icon on the taskbar button no matter what icon the
    window carries — so the ID is set first, before any window exists.
    """
    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID)
    except Exception:
        pass  # not Windows, or the call is unavailable — icon still works
    try:
        from PyQt6.QtGui import QIcon  # noqa: PLC0415

        icon_path = Path(__file__).resolve().parents[2] / "assets" / "app_icon.ico"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass


def _read_vlm_settings() -> dict:
    """Provider + Ollama endpoint the startup check should actually probe.

    Reads `vlm.ollama.{host,port,model}` — the nested provider-block shape
    written by Settings. An earlier version read `vlm.model`, a key that
    does not exist in the current config, so it silently fell back to the
    hard-coded default: picking a different model (or a remote Ollama
    host) in Settings had no effect on the launch check, and the dialog
    would validate a model the app was never going to call.

    Legacy flat configs (pre-2026-05-21, host/port/model directly under
    `vlm:`) are still honoured — same compatibility rule the provider
    factory applies.
    """
    out = {
        "provider": "ollama",
        "host": "127.0.0.1",
        "port": 11434,
        "model": "qwen3.5:9b-q4_K_M",
    }
    try:
        import yaml  # noqa: PLC0415
        config_path = Path(__file__).resolve().parents[2] / "config" / "pipeline_settings.yaml"
        if not config_path.exists():
            return out
        vlm = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("vlm", {})
        out["provider"] = str(vlm.get("provider") or "ollama").lower()
        block = vlm.get("ollama") or vlm          # nested, else legacy flat
        out["host"] = str(block.get("host", out["host"]))
        out["port"] = int(block.get("port", out["port"]))
        out["model"] = str(block.get("model", out["model"]))
    except Exception:
        pass
    return out


if __name__ == "__main__":
    main()
