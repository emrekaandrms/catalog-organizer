"""Settings panel — tabbed YAML round-trip for Paths, Pipeline, VLM, Theme.

The VLM tab is the most elaborate: it picks the active provider
(Ollama / OpenAI / MiniMax) and edits per-provider settings + API key.
API keys are stored in the OS credential manager via `core.secrets`,
never in pipeline_settings.yaml — that file ships in the repo and could
leak.
"""
from __future__ import annotations

import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.paths import config_dir
from catalog_organizer.core.secrets import (
    delete_api_key,
    get_api_key,
    is_backend_available,
    set_api_key,
)
from catalog_organizer.vlm.cost import (
    estimate_call_cost_usd,
    get_pricing,
    list_selectable_models,
)


def _load(name: str) -> dict:
    p = config_dir() / name
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _save(name: str, data: dict) -> None:
    (config_dir() / name).write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


# ── Paths / Pipeline / Theme tabs (unchanged from before) ────────────────────

class _PathsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(10)
        cfg = _load("app_settings.yaml").get("paths", {})
        self._output = QLineEdit(cfg.get("output_root", ""))
        self._cache  = QLineEdit(cfg.get("cache_root", ""))
        self._data   = QLineEdit(cfg.get("data_root", ""))
        form.addRow("Output Root:", self._output)
        form.addRow("Cache Root (blank = default):", self._cache)
        form.addRow("Data Root (blank = default):", self._data)

    def save(self) -> None:
        cfg = _load("app_settings.yaml")
        cfg.setdefault("paths", {})
        cfg["paths"]["output_root"] = self._output.text()
        cfg["paths"]["cache_root"]  = self._cache.text()
        cfg["paths"]["data_root"]   = self._data.text()
        _save("app_settings.yaml", cfg)


class _PipelineTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(10)
        cfg  = _load("pipeline_settings.yaml")
        snap = cfg.get("snapshot", {})
        vlm  = cfg.get("vlm", {})

        self._resolution = QSpinBox()
        self._resolution.setRange(128, 2048)
        self._resolution.setSingleStep(128)
        self._resolution.setValue(snap.get("resolution", 1024))

        self._thumb = QSpinBox()
        self._thumb.setRange(64, 512)
        self._thumb.setValue(snap.get("thumbnail_size", 256))

        self._parallel = QSpinBox()
        self._parallel.setRange(1, 3)
        self._parallel.setValue(int(vlm.get("parallel_workers", 2)))

        form.addRow("Snapshot Resolution (px):", self._resolution)
        form.addRow("Thumbnail Size (px):", self._thumb)
        form.addRow(
            "VLM Parallel Workers (set OLLAMA_NUM_PARALLEL to match):",
            self._parallel,
        )

    def save(self) -> None:
        cfg = _load("pipeline_settings.yaml")
        cfg.setdefault("snapshot", {})
        cfg["snapshot"]["resolution"]     = self._resolution.value()
        cfg["snapshot"]["thumbnail_size"] = self._thumb.value()
        cfg.setdefault("vlm", {})
        cfg["vlm"]["parallel_workers"]    = self._parallel.value()
        _save("pipeline_settings.yaml", cfg)


class _ThemeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(10)
        cfg = _load("app_settings.yaml").get("theme", {})
        self._accent = QLineEdit(cfg.get("accent_color", "#1e88e5"))
        form.addRow("Accent Color (#rrggbb):", self._accent)
        form.addRow(QLabel("Restart required for theme changes to take effect."))

    def save(self) -> None:
        cfg = _load("app_settings.yaml")
        cfg.setdefault("theme", {})
        cfg["theme"]["accent_color"] = self._accent.text()
        _save("app_settings.yaml", cfg)


# ── VLM provider tab ─────────────────────────────────────────────────────────

class _OllamaProbeWorker(QThread):
    """Fetch the installed model list off the GUI thread.

    A hung or firewalled Ollama host blocks for the full request timeout;
    doing that inline would freeze the whole window while the user is
    typing a hostname.
    """
    done = pyqtSignal(object, str)     # (list[OllamaModel] | None, error_text)

    def __init__(self, host: str, port: int, parent=None) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port

    def run(self) -> None:  # type: ignore[override]
        from catalog_organizer.vlm.ollama_models import (  # noqa: PLC0415
            OllamaUnreachable,
            list_ollama_models,
        )
        try:
            self.done.emit(list_ollama_models(self._host, self._port), "")
        except OllamaUnreachable as exc:
            self.done.emit(None, str(exc))
        except Exception as exc:                       # never kill the thread
            self.done.emit(None, f"Unexpected error: {exc}")


class _OllamaForm(QWidget):
    """Per-provider sub-form for local/remote Ollama.

    The model field is a live dropdown populated from the server's
    ``/api/tags``, not free text: typing a tag by hand is easy to get
    subtly wrong (``qwen3.5:9b`` vs ``qwen3.5:9b-q4_K_M``) and the
    mistake only surfaces as a 404 once a long batch is already running.

    By default the list is filtered to vision-capable models. A text-only
    model will happily accept this pipeline's request and answer from the
    prompt text alone — never seeing the jewelry — which yields a catalog
    of confident, entirely invented classifications. Making that
    unselectable is the single most valuable guard-rail here.
    """

    PROVIDER = "ollama"

    def __init__(self) -> None:
        super().__init__()
        cfg = _load("pipeline_settings.yaml").get("vlm", {}).get("ollama", {})
        self._configured_model = cfg.get("model", "qwen3.5:9b-q4_K_M")
        self._worker: _OllamaProbeWorker | None = None
        self._all_models: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(8)

        self._host  = QLineEdit(cfg.get("host", "127.0.0.1"))
        self._port  = QSpinBox(); self._port.setRange(1, 65535)
        self._port.setValue(int(cfg.get("port", 11434)))

        # Editable so a model that isn't pulled yet (or a server that's
        # currently down) can still be configured by hand.
        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setCurrentText(self._configured_model)
        self._model.setMinimumWidth(320)

        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedWidth(36)
        self._refresh_btn.setToolTip("Reload the model list from this Ollama server")
        self._refresh_btn.clicked.connect(self.refresh_models)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self._model, stretch=1)
        model_row.addWidget(self._refresh_btn)
        model_row_w = QWidget(); model_row_w.setLayout(model_row)

        self._vision_only = QCheckBox("Only show vision-capable models")
        self._vision_only.setChecked(True)
        self._vision_only.setToolTip(
            "A text-only model cannot see the snapshots and will invent "
            "classifications. Leave this on unless you know what you're doing."
        )
        self._vision_only.toggled.connect(self._repopulate_model_combo)

        self._num_ctx = QSpinBox(); self._num_ctx.setRange(512, 131072)
        self._num_ctx.setValue(int(cfg.get("num_ctx", 8192)))
        self._keep_alive = QLineEdit(cfg.get("keep_alive", "10m"))

        form.addRow("Host:",       self._host)
        form.addRow("Port:",       self._port)
        form.addRow("Model:",      model_row_w)
        form.addRow("",            self._vision_only)
        form.addRow("num_ctx:",    self._num_ctx)
        form.addRow("keep_alive:", self._keep_alive)
        form_w = QWidget(); form_w.setLayout(form)
        root.addWidget(form_w)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        root.addWidget(self._status)

        # Re-probe when the user points at a different server.
        self._host.editingFinished.connect(self.refresh_models)
        self._port.editingFinished.connect(self.refresh_models)

        self.refresh_models()

    # ── live model discovery ────────────────────────────────────────────────

    def refresh_models(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_status("Connecting to Ollama…", "#9e9e9e")
        self._refresh_btn.setEnabled(False)
        self._worker = _OllamaProbeWorker(
            self._host.text().strip() or "127.0.0.1", self._port.value(), parent=self,
        )
        self._worker.done.connect(self._on_models)
        self._worker.start()

    def _on_models(self, models, error: str) -> None:
        self._refresh_btn.setEnabled(True)
        if models is None:
            self._all_models = []
            self._set_status(f"✗ {error}", "#f44336")
            return
        self._all_models = models
        self._repopulate_model_combo()

    def _repopulate_model_combo(self) -> None:
        """Rebuild the dropdown, preserving whatever the user currently has
        selected even when it isn't installed on this server."""
        from catalog_organizer.vlm.ollama_models import (  # noqa: PLC0415
            filter_vision_models,
        )
        current = self._model.currentText().strip() or self._configured_model
        shown = (
            filter_vision_models(self._all_models)
            if self._vision_only.isChecked() else self._all_models
        )

        self._model.blockSignals(True)
        self._model.clear()
        for m in shown:
            self._model.addItem(m.display_label(), userData=m.name)
        # Keep an out-of-list selection addressable rather than silently
        # switching the user's configured model to something else.
        if current and all(m.name != current for m in shown):
            self._model.addItem(f"{current}  (not installed here)", userData=current)
        idx = self._model.findData(current)
        if idx >= 0:
            self._model.setCurrentIndex(idx)
        else:
            self._model.setCurrentText(current)
        self._model.blockSignals(False)

        total = len(self._all_models)
        hidden = total - len(shown)
        msg = f"✓ Connected — {total} model(s) installed"
        if hidden:
            msg += f", {hidden} hidden (no vision support)"
        installed = any(m.name == current for m in self._all_models)
        colour = "#4caf50"
        if not installed:
            msg += f"\n⚠ Selected model {current!r} is not pulled on this server."
            colour = "#ff9800"
        elif self._vision_only.isChecked():
            sel = next((m for m in self._all_models if m.name == current), None)
            if sel is not None and sel.vision is None:
                msg += ("\n⚠ This Ollama build doesn't report capabilities, so "
                        "vision support couldn't be verified.")
                colour = "#ff9800"
        self._set_status(msg, colour)

    def _set_status(self, text: str, colour: str) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {colour}; font-size: 11px;")

    # ── persistence ─────────────────────────────────────────────────────────

    def collect(self) -> dict:
        return {
            "host":       self._host.text(),
            "port":       self._port.value(),
            "model":      self.get_model(),
            "num_ctx":    self._num_ctx.value(),
            "keep_alive": self._keep_alive.text(),
        }

    def get_model(self) -> str:
        """The bare model tag — never the decorated dropdown label.

        Items carry the real tag in userData because the visible text has
        the size (and sometimes a '(not installed here)' marker) appended;
        saving that decorated string would produce an unusable config.

        The combo is editable, so the text can also be something the user
        typed that matches no item. Only trust userData when the visible
        text still equals the selected item's text — otherwise the user
        has hand-edited it and the typed value is what they mean.
        """
        text = self._model.currentText().strip()
        idx = self._model.currentIndex()
        if idx >= 0 and self._model.itemText(idx) == text:
            data = self._model.itemData(idx)
            if data:
                return str(data)
        return text


class _CloudProviderForm(QWidget):
    """Shared sub-form layout for cloud providers (OpenAI, MiniMax).
    Differences are in: keyring service name, default model list,
    default base URL, optional extra fields (org_id / group_id)."""

    PROVIDER: str = ""

    def __init__(
        self,
        provider_key: str,
        default_model: str,
        default_base_url: str,
        extra_field_label: str | None = None,
        extra_field_key: str | None = None,
        base_url_hint: str | None = None,
        show_json_mode: bool = False,
        default_json_mode: bool = True,
    ) -> None:
        super().__init__()
        self.PROVIDER = provider_key
        self._provider_key = provider_key
        self._extra_field_key = extra_field_key

        form = QFormLayout(self)
        form.setSpacing(8)
        cfg = _load("pipeline_settings.yaml").get("vlm", {}).get(provider_key, {})

        # Model — combo of known models plus free-text via editable. Uses
        # list_selectable_models (not list_priced_models) so backends we
        # can't cost, like GLM, still get a populated picker.
        self._model = QComboBox()
        self._model.setEditable(True)
        for m in list_selectable_models(provider_key):
            self._model.addItem(m)
        active_model = cfg.get("model", default_model)
        if active_model and self._model.findText(active_model) < 0:
            self._model.addItem(active_model)
        self._model.setCurrentText(active_model)

        self._base_url = QLineEdit(cfg.get("base_url", default_base_url))
        if base_url_hint:
            self._base_url.setPlaceholderText(base_url_hint)

        # JSON mode — an OpenAI extension not every compatible gateway
        # implements; some reject the whole request when it's present.
        self._json_mode: QCheckBox | None = None
        if show_json_mode:
            self._json_mode = QCheckBox("Request strict JSON mode (response_format)")
            self._json_mode.setChecked(bool(cfg.get("use_json_mode", default_json_mode)))
            self._json_mode.setToolTip(
                "Turn off if the endpoint rejects 'response_format'. The "
                "response parser already tolerates prose and code fences, "
                "so this is an optimisation rather than a requirement."
            )

        # API key — pre-populate from keyring (masked); allow paste.
        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        existing = get_api_key(provider_key) or ""
        if existing:
            self._api_key.setPlaceholderText(f"(stored, {len(existing)} chars)")
        else:
            self._api_key.setPlaceholderText("Paste your API key here")

        self._show_key_btn = QPushButton("Show")
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.toggled.connect(self._toggle_key_echo)

        self._clear_key_btn = QPushButton("Forget key")
        self._clear_key_btn.clicked.connect(self._forget_key)

        key_row = QHBoxLayout()
        key_row.addWidget(self._api_key, stretch=1)
        key_row.addWidget(self._show_key_btn)
        key_row.addWidget(self._clear_key_btn)
        key_row_w = QWidget(); key_row_w.setLayout(key_row)

        self._extra: QLineEdit | None = None
        if extra_field_label and extra_field_key:
            self._extra = QLineEdit(cfg.get(extra_field_key, ""))

        # Per-call cost estimate label, refreshes when the model changes.
        self._cost_label = QLabel()
        self._cost_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        self._model.currentTextChanged.connect(self._refresh_cost)
        self._refresh_cost(active_model)

        form.addRow("Model:",     self._model)
        form.addRow("Base URL:",  self._base_url)
        form.addRow("API key:",   key_row_w)
        if self._extra and extra_field_label:
            form.addRow(f"{extra_field_label}:", self._extra)
        if self._json_mode is not None:
            form.addRow("", self._json_mode)
        form.addRow("Estimated cost / call:", self._cost_label)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _toggle_key_echo(self, on: bool) -> None:
        self._api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
        )
        self._show_key_btn.setText("Hide" if on else "Show")

    def _forget_key(self) -> None:
        delete_api_key(self._provider_key)
        self._api_key.clear()
        self._api_key.setPlaceholderText("(removed — paste new key to save)")
        QMessageBox.information(self, "API key", "API key removed from keyring.")

    def _refresh_cost(self, model: str) -> None:
        usd = estimate_call_cost_usd(self._provider_key, model, n_images=4)
        if usd <= 0:
            self._cost_label.setText("(not in pricing table)")
            return
        per_300k = usd * 300_000
        info = get_pricing(self._provider_key, model)
        line = f"≈ ${usd:.4f} / file   ·   ${per_300k:,.0f} per 300k files"
        if info:
            line += (
                f"\n   in: ${info.input_usd_per_1m:.2f}/1M  ·  "
                f"out: ${info.output_usd_per_1m:.2f}/1M"
            )
        self._cost_label.setText(line)

    def collect(self) -> dict:
        out = {
            "model":    self._model.currentText().strip(),
            "base_url": self._base_url.text().strip(),
        }
        if self._extra is not None and self._extra_field_key:
            out[self._extra_field_key] = self._extra.text().strip()
        if self._json_mode is not None:
            out["use_json_mode"] = bool(self._json_mode.isChecked())
        return out

    def commit_api_key(self) -> None:
        """Save the typed key to keyring. If the field is empty AND the
        placeholder shows we already have one stored, do nothing."""
        typed = self._api_key.text().strip()
        if typed:
            set_api_key(self._provider_key, typed)
            self._api_key.clear()
            self._api_key.setPlaceholderText(f"(stored, {len(typed)} chars)")

    def get_model(self) -> str:
        return self._model.currentText().strip()


class _VlmTab(QWidget):
    """Provider picker + active sub-form + shared knobs + Test Connection."""

    PROVIDERS: tuple[tuple[str, str], ...] = (
        ("ollama",            "Ollama  (local or remote, no API charge)"),
        ("openai",            "OpenAI  (gpt-4o vision)"),
        ("minimax",           "MiniMax  (MiniMax-VL-01)"),
        ("glm",               "GLM / Zhipu AI  (glm-4.5v)"),
        ("opencode",          "OpenCode Zen  (GPT · Claude · Gemini, one key)"),
        ("openai_compatible", "OpenAI-compatible endpoint  (OpenRouter, vLLM, …)"),
    )

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        vlm = _load("pipeline_settings.yaml").get("vlm", {})
        active = (vlm.get("provider") or "ollama").lower()

        # ── Provider picker ────────────────────────────────────────────────
        self._provider = QComboBox()
        for key, label in self.PROVIDERS:
            self._provider.addItem(label, userData=key)
        idx = max(0, [k for k, _ in self.PROVIDERS].index(active)) \
              if active in [k for k, _ in self.PROVIDERS] else 0
        self._provider.setCurrentIndex(idx)
        self._provider.currentIndexChanged.connect(self._on_provider_change)

        picker_row = QFormLayout()
        picker_row.addRow("Active provider:", self._provider)
        picker_w = QWidget(); picker_w.setLayout(picker_row)
        root.addWidget(picker_w)

        if not is_backend_available():
            warn = QLabel(
                "⚠ OS credential manager unavailable. Install the `keyring` "
                "package to store cloud API keys safely."
            )
            warn.setStyleSheet("color: #f0b132;")
            root.addWidget(warn)

        # ── Per-provider sub-form (stacked) ────────────────────────────────
        self._ollama_form  = _OllamaForm()
        self._openai_form  = _CloudProviderForm(
            provider_key="openai",
            default_model="gpt-4o",
            default_base_url="https://api.openai.com/v1",
            extra_field_label="OpenAI Organization (optional)",
            extra_field_key="org_id",
        )
        self._minimax_form = _CloudProviderForm(
            provider_key="minimax",
            default_model="MiniMax-VL-01",
            default_base_url="https://api.minimaxi.com/v1",
            extra_field_label="MiniMax group_id (optional)",
            extra_field_key="group_id",
        )
        self._glm_form = _CloudProviderForm(
            provider_key="glm",
            default_model="glm-4.5v",
            default_base_url="https://api.z.ai/api/paas/v4",
            base_url_hint="https://api.z.ai/api/paas/v4  (mainland: "
                          "https://open.bigmodel.cn/api/paas/v4)",
            show_json_mode=True,
            default_json_mode=True,
        )
        self._opencode_form = _CloudProviderForm(
            provider_key="opencode",
            default_model="claude-sonnet-5",
            default_base_url="https://opencode.ai/zen/v1",
            base_url_hint="https://opencode.ai/zen/v1",
            show_json_mode=True,
            default_json_mode=True,
        )
        self._compat_form = _CloudProviderForm(
            provider_key="openai_compatible",
            default_model="",
            default_base_url="",
            base_url_hint="https://openrouter.ai/api/v1   ·   "
                          "http://localhost:8000/v1",
            show_json_mode=True,
            default_json_mode=False,
        )

        # Stack order MUST match PROVIDERS order — _on_provider_change maps
        # the combo index straight onto the stack index.
        self._forms_in_order = (
            self._ollama_form, self._openai_form, self._minimax_form,
            self._glm_form, self._opencode_form, self._compat_form,
        )
        self._stack = QStackedWidget()
        for f in self._forms_in_order:
            self._stack.addWidget(f)
        self._stack.setCurrentIndex(idx)
        root.addWidget(self._stack)

        # ── Shared knobs (timeout / temperature) ───────────────────────────
        shared_form = QFormLayout()
        shared_form.setSpacing(8)
        self._timeout = QDoubleSpinBox()
        self._timeout.setRange(5.0, 600.0); self._timeout.setSingleStep(5.0)
        self._timeout.setValue(float(vlm.get("timeout_s", 300)))
        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 2.0); self._temperature.setSingleStep(0.05)
        self._temperature.setValue(float(vlm.get("temperature", 0.1)))
        shared_form.addRow("Request timeout (s):", self._timeout)
        shared_form.addRow("Temperature:", self._temperature)
        shared_w = QWidget(); shared_w.setLayout(shared_form)
        root.addWidget(shared_w)

        # ── Test Connection ────────────────────────────────────────────────
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.setFixedHeight(32)
        self._test_btn.clicked.connect(self._on_test)
        self._test_result = QLabel("")
        self._test_result.setStyleSheet("color: #9e9e9e;")
        self._test_result.setWordWrap(True)
        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_result, stretch=1)
        root.addLayout(test_row)

        root.addStretch()

    # ── slots ──────────────────────────────────────────────────────────────

    def _on_provider_change(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        self._test_result.setText("")

    def _active_form(self):
        return self._stack.currentWidget()

    def _active_key(self) -> str:
        return self._provider.currentData()

    def _on_test(self) -> None:
        """Run one classify() call against a 4-PNG fixture if available,
        otherwise hit the provider's health endpoint. Reports
        success/latency or the exception message — never raises."""
        self._test_result.setStyleSheet("color: #9e9e9e;")
        self._test_result.setText("testing…")

        provider_key = self._active_key()
        form = self._active_form()
        # Persist any newly-typed key first so the factory sees it.
        if hasattr(form, "commit_api_key"):
            form.commit_api_key()

        # Build an in-memory settings dict that mirrors the YAML structure.
        settings = {
            "provider":     provider_key,
            "timeout_s":    float(self._timeout.value()),
            "temperature":  float(self._temperature.value()),
            provider_key:   form.collect(),
        }
        try:
            from catalog_organizer.vlm.providers import (  # noqa: PLC0415
                ProviderConfigurationError,
                create_vlm_provider,
            )
            client = create_vlm_provider(settings)
        except ProviderConfigurationError as exc:
            self._test_result.setStyleSheet("color: #f44336;")
            self._test_result.setText(f"Configuration error: {exc}")
            return
        except Exception as exc:
            self._test_result.setStyleSheet("color: #f44336;")
            self._test_result.setText(f"Setup failed: {exc}")
            return

        # We don't have a snapshot fixture lying around at runtime — instead,
        # probe the provider's *reachability* via a quick HTTP request.
        ok, msg = _probe_provider(client, provider_key, settings)
        if ok:
            self._test_result.setStyleSheet("color: #4caf50;")
            self._test_result.setText(f"✓ {msg}")
        else:
            self._test_result.setStyleSheet("color: #f44336;")
            self._test_result.setText(f"✗ {msg}")

    # ── persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        cfg = _load("pipeline_settings.yaml")
        vlm = cfg.setdefault("vlm", {})
        vlm["provider"]    = self._active_key()
        vlm["timeout_s"]   = float(self._timeout.value())
        vlm["temperature"] = float(self._temperature.value())

        # Every sub-form writes its own provider block; this way switching
        # providers later doesn't wipe the previously-saved values.
        for form in self._forms_in_order:
            vlm[form.PROVIDER] = form.collect()

        # Keys go to the OS credential manager, not the YAML.
        for form in self._forms_in_order:
            if hasattr(form, "commit_api_key"):
                form.commit_api_key()

        _save("pipeline_settings.yaml", cfg)


# ── Connection-probe helper (used by Test Connection) ────────────────────────

def _probe_provider(client, provider_key: str, settings: dict) -> tuple[bool, str]:
    """Cheap reachability test per provider — does the endpoint accept a
    minimal HEAD/GET request with our credentials? We avoid a real
    classify() call because that would burn ~$0.015 per click on cloud
    providers."""
    import requests  # noqa: PLC0415

    if provider_key == "ollama":
        from catalog_organizer.vlm.ollama_models import (  # noqa: PLC0415
            OllamaUnreachable,
            list_ollama_models,
        )
        block = settings["ollama"]
        try:
            models = list_ollama_models(block["host"], int(block["port"]))
        except OllamaUnreachable as exc:
            return False, str(exc)

        wanted = block.get("model", "")
        match = next((m for m in models if m.name == wanted), None)
        if wanted and match is None:
            sample = ", ".join(m.name for m in models[:6]) or "(none installed)"
            return False, (
                f"Reached Ollama, but model {wanted!r} is NOT pulled. "
                f"Installed: {sample}"
            )
        # Reachable + model present, but a text-only model would silently
        # produce invented classifications — that's worse than a hard
        # failure, so report it as a failed test.
        if match is not None and match.vision is False:
            return False, (
                f"Model {wanted!r} has no vision capability — it cannot see "
                "the snapshots and would invent classifications. Pick a "
                "vision-capable model."
            )
        note = f"Reached Ollama ({len(models)} model(s) installed)."
        if match is not None and match.vision is None:
            note += (" This Ollama build doesn't report capabilities, so "
                     "vision support could not be verified.")
        return True, note

    # OpenAI and every OpenAI-compatible backend (GLM, OpenRouter, vLLM…)
    # expose the same GET /models listing endpoint.
    if provider_key in ("openai", "glm", "opencode", "openai_compatible"):
        base = (settings[provider_key].get("base_url") or "").strip().rstrip("/")
        if not base:
            return False, "No Base URL configured for this endpoint."
        url = f"{base}/models"
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {client.api_key}"},
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", payload if isinstance(payload, list) else [])
            return True, (
                f"{client.display_vendor} reachable "
                f"({len(data)} model(s) visible to this key)."
            )
        except Exception as exc:
            return False, f"{getattr(client, 'display_vendor', provider_key)} test failed: {exc}"

    if provider_key == "minimax":
        # MiniMax doesn't have a public /models endpoint that's stable;
        # instead, do a 1-token text-only completion to verify auth.
        url = client.chat_url
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {client.api_key}"},
                json={
                    "model": settings["minimax"].get("model", "MiniMax-VL-01"),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=10,
            )
            r.raise_for_status()
            return True, "MiniMax reachable (auth accepted)."
        except Exception as exc:
            return False, f"MiniMax test failed: {exc}"

    return False, "Unknown provider"


# ── Top-level panel ──────────────────────────────────────────────────────────


# ── Pricing tab ──────────────────────────────────────────────────────────────

class _PricingTab(QWidget):
    """Listing price constants.

    These are live metal rates, not code: 2.9 (Etsy silver), 110 (Woo silver)
    and 7000 (pure gold per gram) go stale within months. Editing them here
    and re-running "Recalculate prices" in the Publish tab re-prices every
    listing without regenerating a single line of copy.

    The gold gram source is deliberately one field for all four karats — the
    workshop prices by milyem, not by per-karat density.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = _load("pricing.yaml")
        etsy = (cfg.get("etsy") or {})
        etsy_silver = (etsy.get("silver") or {})
        woo = (cfg.get("woocommerce") or {})
        woo_silver = (woo.get("silver") or {})
        woo_gold = (woo.get("gold") or {})
        karats = (woo_gold.get("karats") or {})
        woo_stl = (woo.get("stl") or {})

        form = QFormLayout(self)

        self._etsy_who = QComboBox()
        self._etsy_who.addItems(["i_did", "collective", "someone_else"])
        who = str(etsy.get("who_made", "i_did"))
        if self._etsy_who.findText(who) >= 0:
            self._etsy_who.setCurrentText(who)
        self._etsy_who.setToolTip(
            "Etsy requires this. Pick 'collective' if anyone else in the "
            "workshop makes the pieces.")
        form.addRow("Etsy · who made it", self._etsy_who)

        self._etsy_a = QDoubleSpinBox()
        self._etsy_b = QDoubleSpinBox()
        self._etsy_rate = QDoubleSpinBox()
        self._etsy_mult = QDoubleSpinBox()
        for w, val, mx in (
            (self._etsy_a, etsy_silver.get("factor_a", 1.2), 100),
            (self._etsy_b, etsy_silver.get("factor_b", 1.3), 100),
            (self._etsy_rate, etsy_silver.get("metal_rate", 2.9), 100000),
            (self._etsy_mult, etsy_silver.get("final_multiplier", 2), 100),
        ):
            w.setDecimals(4)
            w.setMaximum(mx)
            w.setValue(float(val))
        self._etsy_add = QLineEdit(
            ", ".join(str(x) for x in etsy_silver.get("additions", [20, 5, 5])))
        self._etsy_add.setToolTip("Comma-separated fixed additions, e.g. 20, 5, 5")
        form.addRow("Etsy · factor A", self._etsy_a)
        form.addRow("Etsy · factor B", self._etsy_b)
        form.addRow("Etsy · silver rate", self._etsy_rate)
        form.addRow("Etsy · additions", self._etsy_add)
        form.addRow("Etsy · final multiplier", self._etsy_mult)

        self._woo_ag_factor = QDoubleSpinBox()
        self._woo_ag_rate = QDoubleSpinBox()
        self._woo_markup = QDoubleSpinBox()
        self._woo_gold_rate = QDoubleSpinBox()
        self._woo_stl = QDoubleSpinBox()
        for w, val, mx in (
            (self._woo_ag_factor, woo_silver.get("factor", 2.7), 1000),
            (self._woo_ag_rate, woo_silver.get("metal_rate", 110), 1_000_000),
            (self._woo_markup, woo_gold.get("markup", 1.4), 100),
            (self._woo_gold_rate, woo_gold.get("pure_gold_gram_price", 7000),
             10_000_000),
            (self._woo_stl, woo_stl.get("flat_price", 50), 1_000_000),
        ):
            w.setDecimals(4)
            w.setMaximum(mx)
            w.setValue(float(val))
        form.addRow("Woo · silver factor", self._woo_ag_factor)
        form.addRow("Woo · silver rate (TL/g)", self._woo_ag_rate)
        form.addRow("Woo · gold markup", self._woo_markup)
        form.addRow("Woo · pure gold (TL/g)", self._woo_gold_rate)
        form.addRow("Woo · STL flat price (TL)", self._woo_stl)

        self._karats: dict[int, QDoubleSpinBox] = {}
        for karat, default in ((8, 0.333), (10, 0.417), (14, 0.585), (18, 0.750)):
            box = QDoubleSpinBox()
            box.setDecimals(4)
            box.setMaximum(1.0)
            box.setValue(float(karats.get(karat, karats.get(str(karat), default))))
            self._karats[karat] = box
            form.addRow(f"Milyem · {karat} ayar", box)

        self._gold_source = QComboBox()
        self._gold_source.addItems([
            "gold_14k_yellow_g", "gold_10k_yellow_g", "gold_18k_yellow_g",
            "silver_925_g",
        ])
        source = str(woo_gold.get("gram_source", "gold_14k_yellow_g"))
        if self._gold_source.findText(source) >= 0:
            self._gold_source.setCurrentText(source)
        self._gold_source.setToolTip(
            "One gram feeds all four karats; the karat difference comes only "
            "from milyem. Changing this re-prices every gold variant.")
        form.addRow("Gold gram source", self._gold_source)

    def _additions(self) -> list[float]:
        out: list[float] = []
        for chunk in self._etsy_add.text().split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(float(chunk))
            except ValueError as exc:
                raise ValueError(
                    f"Etsy additions must be numbers separated by commas; "
                    f"got {chunk!r}"
                ) from exc
        return out

    def save(self) -> None:
        cfg = _load("pricing.yaml")
        cfg.setdefault("etsy", {}).setdefault("silver", {})
        cfg["etsy"]["currency"] = cfg["etsy"].get("currency", "USD")
        cfg["etsy"]["who_made"] = self._etsy_who.currentText()
        cfg["etsy"]["when_made"] = cfg["etsy"].get("when_made", "made_to_order")
        cfg["etsy"]["silver"].update({
            "factor_a": self._etsy_a.value(),
            "factor_b": self._etsy_b.value(),
            "metal_rate": self._etsy_rate.value(),
            "additions": self._additions(),
            "final_multiplier": self._etsy_mult.value(),
        })

        woo = cfg.setdefault("woocommerce", {})
        woo["currency"] = woo.get("currency", "TRY")
        woo.setdefault("silver", {}).update({
            "factor": self._woo_ag_factor.value(),
            "metal_rate": self._woo_ag_rate.value(),
        })
        woo.setdefault("gold", {}).update({
            "gram_source": self._gold_source.currentText(),
            "markup": self._woo_markup.value(),
            "pure_gold_gram_price": self._woo_gold_rate.value(),
            "karats": {k: box.value() for k, box in self._karats.items()},
        })
        woo.setdefault("stl", {})["flat_price"] = self._woo_stl.value()
        _save("pricing.yaml", cfg)


class SettingsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        from catalog_organizer.gui.widgets.components import (  # noqa: PLC0415
            PageHeader,
            page_layout,
        )

        root = page_layout(self)
        root.addWidget(PageHeader(
            "Settings",
            "Paths, pipeline tuning, and which model does the classifying.",
        ))

        self._tabs = QTabWidget()
        self._paths_tab    = _PathsTab()
        self._pipeline_tab = _PipelineTab()
        self._vlm_tab      = _VlmTab()
        self._pricing_tab  = _PricingTab()
        self._theme_tab    = _ThemeTab()
        self._tabs.addTab(self._paths_tab,    "Paths")
        self._tabs.addTab(self._pipeline_tab, "Pipeline")
        self._tabs.addTab(self._vlm_tab,      "VLM Provider")
        self._tabs.addTab(self._pricing_tab,  "Pricing")
        self._tabs.addTab(self._theme_tab,    "Theme")
        root.addWidget(self._tabs, stretch=1)

        # Right-aligned rather than a full-width bar: a stretched button
        # reads as a page-level primary action, but this only commits the
        # form above it.
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_btn = QPushButton("Save all settings")
        save_btn.setProperty("variant", "primary")
        save_btn.setMinimumHeight(34)
        save_btn.clicked.connect(self._save_all)
        save_row.addWidget(save_btn)
        root.addLayout(save_row)

    def _save_all(self) -> None:
        try:
            self._paths_tab.save()
            self._pipeline_tab.save()
            self._vlm_tab.save()
            self._pricing_tab.save()
            self._theme_tab.save()
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings:\n{exc}")
