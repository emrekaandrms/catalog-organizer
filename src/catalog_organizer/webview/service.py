"""The render engine, running inside the app: the web viewer in a hidden Chromium.

`WebRenderService` lives in the GUI thread (Chromium must) and does three things:

  * serves the cached scenes (`scene_cache`) to the app's own browser views on 127.0.0.1,
  * keeps ONE hidden, never-visible QWebEngineView that draws catalogue images on demand,
  * lets any thread ask for those images -- the Render tab's worker, the PDF catalogue -- and
    wait for them, so callers keep the blocking `render(...) -> files` shape they always had.

Nothing is shown to the user by the hidden view and nothing is written outside the cache.

A hidden Chromium page gets no animation frames, so the viewer's `capture()` draws its frames
synchronously inside one JavaScript call instead of waiting for requestAnimationFrame.
"""
from __future__ import annotations

import base64
import json
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from PyQt6.QtCore import QEventLoop, QObject, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtWidgets import QApplication

from catalog_organizer.webview import scene_cache, server
from catalog_organizer.webview.scene_cache import Scene

PAGE_READY_TIMEOUT_S = 60.0         # first load compiles the stone shader for the whole cut
CAPTURE_TIMEOUT_S = 120.0
_POLL_MS = 40


class WebRenderError(RuntimeError):
    pass


@dataclass
class _Job:
    scene: Scene
    script: str
    future: Future = field(default_factory=Future)


def _q(text: str) -> str:
    return json.dumps(text)


class WebRenderService(QObject):
    _submit = pyqtSignal(object)            # worker thread -> GUI thread

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server = None
        self._port = 0
        self._view = None                   # the hidden QWebEngineView, created on first use
        self._queue: deque[_Job] = deque()
        self._busy = False
        self._loaded: tuple[str, str] | None = None     # (file_id, scene key) the hidden page shows
        self._console: deque[str] = deque(maxlen=20)
        self._submit.connect(self._enqueue, Qt.ConnectionType.QueuedConnection)

    # ── the loopback server (shared with the visible player) ─────────────────────────────────

    def base_url(self) -> str:
        if self._server is None:
            site = scene_cache.ensure_site()
            self._server = server.make_server(site, 0)
            self._port = self._server.server_address[1]
            threading.Thread(target=self._server.serve_forever, daemon=True,
                             name="web-render-server").start()
        return f"http://127.0.0.1:{self._port}"

    def viewer_url(self, scene: Scene, *, hosted: bool = True) -> str:
        return (f"{self.base_url()}/viewer.html?p={quote(scene.file_id)}"
                f"&k={scene.key}{'&host=1' if hosted else ''}")

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._view is not None:
            self._view.deleteLater()
            self._view = None

    # ── the API any thread calls ─────────────────────────────────────────────────────────────

    def capture(self, scene: Scene, *, metal_key: str, stone_key: str, size: int,
                views: dict, frames: int = 16, background: str | None = "default") -> dict[str, bytes]:
        """PNG bytes per view. `views` = {name: (direction, up)} in the scene's canonical frame.

        background: None = transparent, "default" = the viewer's own flat colour, or a hex colour.
        """
        spec = {"size": int(size), "frames": int(frames),
                "views": {n: {"dir": [float(x) for x in d], "up": [float(x) for x in u]}
                          for n, (d, u) in views.items()}}
        if background != "default":
            spec["bg"] = background
        script = (f"(() => {{ try {{ const v = window.catalogViewer;"
                  f" v.setMetal({_q(metal_key)}); v.setStone({_q(stone_key)});"
                  f" return JSON.stringify(v.capture({json.dumps(spec)}));"
                  f" }} catch (e) {{ return JSON.stringify({{__error: String((e && e.stack) || e)}}); }} }})()")
        job = _Job(scene, script)
        self._submit.emit(job)
        encoded = self._wait(job.future)
        return {name: base64.b64decode(url.split(",", 1)[1]) for name, url in encoded.items()}

    def _wait(self, future: Future):
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            # Called on the GUI thread (a test, a script): keep the event loop turning ourselves.
            loop = QEventLoop()
            timer = QTimer()
            timer.setInterval(25)
            timer.timeout.connect(lambda: loop.quit() if future.done() else None)
            timer.start()
            while not future.done():
                loop.exec()
            timer.stop()
        return future.result(timeout=CAPTURE_TIMEOUT_S + PAGE_READY_TIMEOUT_S)

    # ── GUI thread: one job at a time ────────────────────────────────────────────────────────

    def _enqueue(self, job: _Job) -> None:
        self._queue.append(job)
        self._pump()

    def _pump(self) -> None:
        if self._busy or not self._queue:
            return
        self._busy = True
        job = self._queue.popleft()
        try:
            self._ensure_view()
            if self._loaded == (job.scene.file_id, job.scene.key):
                self._run(job)
            else:
                self._loaded = None
                self._view.loadFinished.connect(lambda ok, j=job: self._on_loaded(ok, j))
                self._view.load(QUrl(self.viewer_url(job.scene)))
        except Exception as exc:                          # noqa: BLE001 - handed to the caller
            self._finish(job, exc)

    def _ensure_view(self) -> None:
        if self._view is not None:
            return
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        view = QWebEngineView()
        view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        view.resize(320, 320)
        view.page().javaScriptConsoleMessage = (
            lambda _lvl, msg, line, src: self._console.append(f"{msg} ({src}:{line})"))
        view.show()          # never drawn on screen, but Chromium only runs a "visible" page
        self._view = view

    def _on_loaded(self, ok: bool, job: _Job) -> None:
        try:
            self._view.loadFinished.disconnect()
        except TypeError:
            pass
        if not ok:
            self._finish(job, WebRenderError("görüntüleyici sayfası yüklenemedi"))
            return
        self._loaded = (job.scene.file_id, job.scene.key)
        self._poll_ready(job, waited_ms=0)

    def _poll_ready(self, job: _Job, waited_ms: int) -> None:
        def got(ready):
            if ready:
                self._run(job)
            elif waited_ms >= PAGE_READY_TIMEOUT_S * 1000:
                self._loaded = None
                self._finish(job, WebRenderError(
                    "görüntüleyici hazır olmadı: " + " | ".join(list(self._console)[-4:])))
            else:
                QTimer.singleShot(_POLL_MS, lambda: self._poll_ready(job, waited_ms + _POLL_MS))
        self._view.page().runJavaScript("!!(window.catalogViewer && window.catalogViewer.ready)", got)

    def _run(self, job: _Job) -> None:
        def got(result):
            if not result:
                self._loaded = None
                self._finish(job, WebRenderError(
                    "çizim sonuç vermedi: " + " | ".join(list(self._console)[-4:])))
                return
            data = json.loads(result)
            if "__error" in data:
                self._finish(job, WebRenderError("çizim hatası: " + data["__error"][:600]))
                return
            self._finish(job, data)
        self._view.page().runJavaScript(job.script, got)

    def _finish(self, job: _Job, outcome) -> None:
        if isinstance(outcome, BaseException):
            job.future.set_exception(outcome)
        else:
            job.future.set_result(outcome)
        self._busy = False
        QTimer.singleShot(0, self._pump)


# One service per process: the Render tab and the PDF catalogue share it.
_service: WebRenderService | None = None


def install(parent: QObject | None = None) -> WebRenderService:
    global _service
    if _service is None:
        _service = WebRenderService(parent)
    return _service


def current() -> WebRenderService | None:
    return _service


def uninstall() -> None:
    global _service
    if _service is not None:
        _service.shutdown()
        _service = None
