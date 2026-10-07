"""The Ollama model probe must survive the form that started it.

Regression. Settings builds a form that asks Ollama for its model list on a background thread. The
thread was a child of the form, so closing Settings (or quitting) while Ollama was slow to answer
destroyed the form with its thread still running, and Qt treats "QThread destroyed while the thread
is still running" as fatal: the whole process died with no Python traceback. It showed up on a CI
machine where nothing listens on Ollama's port (a refused connection is slow on Windows) and never on
a developer machine that runs Ollama (an instant answer). Here the answer is made slow on purpose.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")


def test_closing_the_form_while_the_probe_is_waiting_does_not_kill_the_process(qtbot, monkeypatch):
    from PyQt6 import sip

    from catalog_organizer.gui.panels import settings
    from catalog_organizer.vlm import ollama_models as om

    started = []

    def slow_list(host, port, timeout_s=5.0):
        started.append(time.time())
        time.sleep(0.5)                       # Ollama is slow to answer (or not there at all)
        return []

    monkeypatch.setattr(om, "list_ollama_models", slow_list)
    form = settings._OllamaForm()
    qtbot.waitUntil(lambda: bool(started), timeout=3000)       # the thread is inside the slow call
    assert settings._LIVE_PROBES, "a running probe is kept alive by the module, not by the form"

    sip.delete(form)                                            # the form goes while the probe waits
    qtbot.waitUntil(lambda: not settings._LIVE_PROBES, timeout=5000)   # ... and the probe ends cleanly


def test_the_app_can_wait_for_probes_on_the_way_out(qtbot, monkeypatch):
    from catalog_organizer.gui.panels import settings
    from catalog_organizer.vlm import ollama_models as om

    monkeypatch.setattr(om, "list_ollama_models", lambda host, port, timeout_s=5.0: time.sleep(0.3) or [])
    form = settings._OllamaForm()
    qtbot.addWidget(form)
    settings.wait_for_probes(3000)
    qtbot.waitUntil(lambda: not settings._LIVE_PROBES, timeout=3000)
