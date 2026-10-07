"""Command line: export catalogue products as a web viewer.

    python -m catalog_organizer.app webview JCAD-000000003 JCAD-000000009 [--serve]
    python -m catalog_organizer.app webview --selection "Kis 2026" --serve
    python -m catalog_organizer.app webview --all --out D:/site

Writes a static folder (default: <data>/exports/web) and, with --serve, opens it in the browser.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def _resolve(args) -> list[tuple[str, Path, str, str]]:
    """(file_id, source_path, category, label) for what was asked for."""
    from catalog_organizer.db.connection import Database
    from catalog_organizer.db.products import get_products
    from catalog_organizer.db.selections import list_selections, selection_file_ids

    db = Database()
    try:
        ids: list[str] = list(args.ids)
        if args.all:
            ids = [r[0] for r in db.conn.execute("SELECT file_id FROM products ORDER BY file_id")]
        if args.selection:
            match = [s for s in list_selections(db.conn) if s.name == args.selection]
            if not match:
                names = ", ".join(s.name for s in list_selections(db.conn)) or "(yok)"
                raise SystemExit(f"'{args.selection}' adli liste yok. Listeler: {names}")
            ids += selection_file_ids(db.conn, match[0].selection_id)
        ids = list(dict.fromkeys(ids))                      # keep order, drop repeats
        by_id = {r.file_id: r for r in get_products(db.conn, ids)}
    finally:
        db.close()
    items = []
    for fid in ids:
        rec = by_id.get(fid)
        if rec is None:
            print(f"! {fid}: katalogda yok, atlandi")
            continue
        items.append((fid, Path(rec.source_path), rec.main_category,
                      f"{fid} {rec.main_category}/{rec.subcategory}"))
    return items


def run(argv: list[str]) -> int:
    import argparse

    from catalog_organizer.core.paths import data_dir
    from catalog_organizer.webview import export
    from catalog_organizer.webview import server as web_server

    ap = argparse.ArgumentParser(prog="webview", description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*", help="urun kimlikleri (JCAD-...)")
    ap.add_argument("--all", action="store_true", help="katalogdaki tum urunler")
    ap.add_argument("--selection", help="bir secim listesinin adi")
    ap.add_argument("--out", type=Path, default=None, help="cikti klasoru")
    ap.add_argument("--serve", action="store_true", help="bitince yerel sunucuda ac")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tris", type=int, default=export.MAX_TRIANGLES)
    args = ap.parse_args(argv)

    if not (args.ids or args.all or args.selection):
        ap.print_usage()
        print("En az bir urun, --selection veya --all verin.")
        return 2
    out = args.out or (data_dir() / "exports" / "web")
    items = _resolve(args)
    if not items:
        print("Disa aktarilacak urun yok.")
        return 2

    started = time.time()

    def progress(done: int, total: int, file_id: str) -> None:
        print(f"[{done}/{total}] {file_id}  ({time.time() - started:.0f} sn)", flush=True)

    results, errors = export.export_many(items, out, progress=progress, max_triangles=args.max_tris)
    print(f"\n{len(results)} urun yazildi: {out}")
    for r in results:
        note = f"  (sadelestirildi: {r.decimated_from} -> {r.triangles})" if r.decimated_from else ""
        print(f"  {r.file_id}  {r.bytes / 1e6:5.1f} MB  {r.stones} tas{note}")
    for fid, msg in errors.items():
        print(f"  HATA {fid}: {msg}")

    if args.serve and results:
        server = web_server.serve(out, args.port)
        print(f"\nhttp://127.0.0.1:{args.port}/index.html   (Ctrl+C ile kapat)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
    return 1 if errors and not results else 0
