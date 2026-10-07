"""Serve an exported web-viewer folder on localhost.

    python -m catalog_organizer.webview.server <folder> [port]

Responses are never cached. A stale file served from cache once cost an hour on this project
(a retuned render looked like it had changed nothing); an exported viewer is regenerated
often enough that the same trap is waiting here.
"""
from __future__ import annotations

import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".glb": "model/gltf-binary",
                      ".f16": "application/octet-stream", ".json": "application/json"}

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt, *args) -> None:        # quiet
        pass


def make_server(directory: Path, port: int = 8765) -> ThreadingHTTPServer:
    handler = partial(_Handler, directory=str(directory))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(directory: Path, port: int = 8765, *, open_browser: bool = True,
          page: str = "index.html") -> ThreadingHTTPServer:
    """Start serving in a background thread and return the server (call .shutdown() to stop)."""
    server = make_server(Path(directory), port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_address[1]}/{page}")
    return server


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    chosen = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    srv = make_server(folder, chosen)
    print(f"http://127.0.0.1:{chosen}/index.html   (Ctrl+C ile kapat)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
