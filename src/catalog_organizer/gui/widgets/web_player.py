"""The live 3D player: the web viewer, embedded.

A thin wrapper around QWebEngineView that knows the viewer's host API (`window.catalogViewer`,
see webview/assets/viewer.js). The user orbits with the mouse; the Render tab drives materials and
the two catalogue views through it.

The wrapper queues commands until the page reports ready, so callers never have to wait for the
shader compile: ask for a metal right after `load()` and it is applied the moment the page can.
"""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

READY_TIMEOUT_MS = 60_000
_POLL_MS = 50


class WebPlayer(QWidget):
    ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        self._view = QWebEngineView(self)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._is_ready = False
        self._queued: list[str] = []
        self._waited = 0
        self._console: list[str] = []
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._view.loadFinished.connect(self._on_loaded)
        self._view.page().javaScriptConsoleMessage = (
            lambda _lvl, msg, line, src: self._console.append(f"{msg} ({src}:{line})"))

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ── loading ──────────────────────────────────────────────────────────────────────────────

    def load(self, url: str, then: list[str] | None = None) -> None:
        """Open a scene. `then` = JavaScript to run once the viewer is up (first thing)."""
        self._is_ready = False
        self._queued = list(then or [])
        self._console.clear()
        self._timer.stop()
        self._view.load(QUrl(url))

    def clear(self) -> None:
        self._is_ready = False
        self._queued.clear()
        self._timer.stop()
        self._view.setHtml("")

    def _on_loaded(self, ok: bool) -> None:
        if not ok:
            self.failed.emit("görüntüleyici sayfası yüklenemedi")
            return
        self._waited = 0
        self._timer.start()

    def _poll(self) -> None:
        self._view.page().runJavaScript(
            "!!(window.catalogViewer && window.catalogViewer.ready)", self._on_poll)

    def _on_poll(self, ready) -> None:
        if self._is_ready or not self._timer.isActive():
            return
        if ready:
            self._timer.stop()
            self._is_ready = True
            for js in self._queued:
                self._view.page().runJavaScript(js)
            self._queued.clear()
            self.ready.emit()
            return
        self._waited += _POLL_MS
        if self._waited >= READY_TIMEOUT_MS:
            self._timer.stop()
            self.failed.emit("görüntüleyici hazır olmadı: " + " | ".join(self._console[-4:]))

    # ── commands ─────────────────────────────────────────────────────────────────────────────

    def run(self, js: str) -> None:
        if self._is_ready:
            self._view.page().runJavaScript(js)
        else:
            self._queued.append(js)

    def set_metal(self, key: str) -> None:
        self.run(f"window.catalogViewer.setMetal({json.dumps(key)})")

    def set_stone(self, key: str) -> None:
        self.run(f"window.catalogViewer.setStone({json.dumps(key)})")

    def go_to_view(self, name: str) -> None:
        self.run(f"window.catalogViewer.goToView({json.dumps(name)})")

    def set_camera(self, direction, up) -> None:
        spec = {"dir": [float(x) for x in direction], "up": [float(x) for x in up]}
        self.run(f"window.catalogViewer.setCamera({json.dumps(spec)})")

    def camera_state(self, callback) -> None:
        """callback({"dir": [...], "up": [...]}) with the camera as the user has orbited it."""
        if not self._is_ready:
            return
        self._view.page().runJavaScript(
            "JSON.stringify(window.catalogViewer.cameraState())",
            lambda text: callback(json.loads(text)) if text else None)
