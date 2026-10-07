"""Render tab — the live 3D player, and the colours to try on a product, immediately.

The picture is the web engine's: the same three.js viewer (in the app's own Chromium) that draws
the catalogue PDF's images, so what is judged here is what gets printed. The two views the catalogue
page prints, KARŞIDAN and ÇAPRAZ, are shown side by side, each a live player: orbit either with the
mouse, change metal and stone colours instantly, and save the angle you like as that view.

Nothing is exported: a product's scene is prepared on first use and cached (webview.scene_cache).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.db.camera_overrides import (
    clear_camera_override,
    get_camera_overrides,
    set_camera_override,
)
from catalog_organizer.db.connection import Database
from catalog_organizer.db.products import search_products
from catalog_organizer.db.render_options import (
    set_place_stones,
    wants_placed_stones,
)
from catalog_organizer.gui import theme
from catalog_organizer.gui.widgets.components import (
    Badge, PageHeader, caption, field_label, page_layout,
)
from catalog_organizer.render import materials as mat

# PNG sizes for "PNG kaydet" (the live player has no size: it fills the window).
QUALITY = {"Hızlı (700 px)": 700, "Önizleme (900 px)": 900,
           "Katalog (1400 px)": 1400, "Yüksek (2000 px)": 2000}
DEFAULT_QUALITY = "Katalog (1400 px)"

_LIST_LIMIT = 400


def _swatch(hex_colour: str, size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(QColor(hex_colour))
    return QIcon(pix)


class SceneWorker(QThread):
    """Prepares a product's scene off the GUI thread. First time per product is seconds (the
    occlusion bake); after that it is a cache hit."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source: Path, file_id: str, category: str | None,
                 force: bool = False, place_stones: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._args = (source, file_id, category, force, place_stones)

    def run(self) -> None:  # type: ignore[override]
        from catalog_organizer.webview import scene_cache

        source, file_id, category, force, place_stones = self._args
        try:
            scene = scene_cache.ensure_scene(source, file_id, category=category, force=force,
                                             place_stones=place_stones)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done.emit(scene)


class RenderWorker(QThread):
    """PNG files of a product's two catalogue views (for "PNG kaydet")."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source: Path, file_id: str, metal_key: str,
                 stone_key: str, resolution: int,
                 camera: dict | None = None, parent=None,
                 category: str | None = None, place_stones: bool = False) -> None:
        super().__init__(parent)
        self._args = (source, file_id, metal_key, stone_key, resolution,
                      camera, category, place_stones)

    def run(self) -> None:  # type: ignore[override]
        from catalog_organizer.render.engine import load_and_render

        (source, file_id, metal_key, stone_key, resolution, camera,
         category, place_stones) = self._args
        try:
            result = load_and_render(source, file_id=file_id,
                                     metal_key=metal_key, stone_key=stone_key,
                                     resolution=resolution, camera=camera,
                                     category=category, place_stones=place_stones)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done.emit(result)


def _create_player(parent: QWidget):
    """The live viewer, or None where QtWebEngine is not installed."""
    try:
        from catalog_organizer.gui.widgets.web_player import WebPlayer
        return WebPlayer(parent)
    except ImportError:
        return None


class RenderPanel(QWidget):
    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._records: list[CatalogRecord] = []
        self._external: Path | None = None
        self._worker: SceneWorker | None = None       # preparing a scene
        self._save_worker: RenderWorker | None = None  # writing PNGs
        self._pending: bool = False                   # a product was picked while one was preparing
        self._scene = None
        self._players: dict[str, object] = {}         # "front" / "iso" -> live player
        self._build_ui()
        self.reload_products()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = page_layout(self)
        root.addWidget(PageHeader(
            "Render",
            "Canlı 3D görünüm: fareyle döndür, maden ve taş rengini anında dene — katalog üretmeden.",
        ))

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("ürün ara — ID, kategori, açıklama")
        self._filter.setClearButtonEnabled(True)
        self._filter.returnPressed.connect(self.reload_products)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_product_changed)

        b_browse = QPushButton("Dosyadan aç…")
        b_browse.setProperty("variant", "ghost")
        b_browse.setToolTip("Katalogda olmayan bir .3dm / .stl dosyasını dene.")
        b_browse.clicked.connect(self._on_browse)

        left = QVBoxLayout()
        left.setSpacing(theme.SP_2)
        left.addWidget(field_label("Ürün"))
        left.addWidget(self._filter)
        left.addWidget(self._list, 1)
        left.addWidget(b_browse)

        self._metal = QComboBox()
        for key, material in mat.METALS.items():
            self._metal.addItem(_swatch(material.color), material.label_tr,
                                userData=key)
        self._stone = QComboBox()
        for key, material in mat.STONES.items():
            self._stone.addItem(_swatch(material.color), material.label_tr,
                                userData=key)
        self._set_current(self._metal, mat.DEFAULT_METAL)
        self._set_current(self._stone, mat.DEFAULT_STONE)
        self._quality = QComboBox()
        self._quality.addItems(list(QUALITY))
        self._quality.setCurrentText(DEFAULT_QUALITY)
        self._quality.setToolTip("Yalnızca 'PNG kaydet' için: dosyanın kenar uzunluğu.")

        # A colour change applies to the live player at once: this screen exists to be
        # flipped through, and nothing needs to be drawn again.
        self._metal.currentIndexChanged.connect(self._on_colours_changed)
        self._stone.currentIndexChanged.connect(self._on_colours_changed)

        # Stones in the seats of a stoneless STL. The program cannot tell a seat from any other hole,
        # so the product's record decides the default and this switch is the user's say, kept per
        # product (the PDF reads the same answer).
        self.place_box = QCheckBox("Taşları yuvalara yerleştir")
        self.place_box.setToolTip(
            "Yalnızca taşsız STL'ler için: dosyada taş yok, ama yuvalar var. Açıkken program yuvaları "
            "bulup taşları oraya koyar. Varsayılan, ürün kaydının 'taşlı' olup olmadığıdır; seçimin "
            "bu ürün için saklanır ve PDF katalog da aynısını kullanır.")
        self.place_box.setEnabled(False)
        self.place_box.toggled.connect(self._on_place_toggled)

        self.render_button = QPushButton("Yeniden çiz")
        self.render_button.setProperty("variant", "primary")
        self.render_button.setToolTip("Sahneyi baştan hazırla (dosya değiştiyse).")
        self.render_button.clicked.connect(lambda: self.load_product(force=True))
        self.save_button = QPushButton("PNG kaydet…")
        self.save_button.setProperty("variant", "ghost")
        self.save_button.clicked.connect(self._on_save)
        self.save_button.setEnabled(False)

        # Replaces the old orbit dialog: each player IS an orbit, so saving its angle is one click.
        self.save_angle_buttons: dict[str, QPushButton] = {}
        for view in ("front", "iso"):
            button = QPushButton("Bu açıyı kaydet")
            button.setProperty("variant", "ghost")
            button.setToolTip(
                "Otomatik seçilen açıyı beğenmediysen: parçayı fareyle döndür, sonra bu açıyı "
                "kataloğun bu görünümü olarak kaydet. PDF ve PNG'ler onu kullanır.")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, v=view: self._save_angle(v))
            self.save_angle_buttons[view] = button
        self.reset_angle_button = QPushButton("Otomatiğe sıfırla")
        self.reset_angle_button.setProperty("variant", "ghost")
        self.reset_angle_button.setToolTip(
            "Bu ürün için kaydedilmiş elle ayarlanmış açıları sil.")
        self.reset_angle_button.clicked.connect(self._on_reset_angles)
        self.reset_angle_button.setEnabled(False)

        controls = QHBoxLayout()
        controls.setSpacing(theme.SP_2)
        controls.addWidget(field_label("Maden"))
        controls.addWidget(self._metal)
        controls.addWidget(field_label("Taş"))
        controls.addWidget(self._stone)
        controls.addSpacing(theme.SP_3)
        controls.addWidget(self.place_box)
        controls.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(theme.SP_2)
        actions.addWidget(self.reset_angle_button)
        actions.addStretch(1)
        actions.addWidget(field_label("PNG boyutu"))
        actions.addWidget(self._quality)
        actions.addWidget(self.save_button)
        actions.addWidget(self.render_button)

        pane_style = (f"background: {theme.BG_SUNKEN};"
                      f" border: 1px solid {theme.BORDER};"
                      f" border-radius: {theme.RADIUS_MD}px;"
                      f" color: {theme.TEXT_MUTED};")
        views = QHBoxLayout()
        views.setSpacing(theme.SP_3)
        for view, title in (("front", "KARŞIDAN"), ("iso", "ÇAPRAZ")):
            player = _create_player(self)
            if player is not None:
                self._players[view] = player
                player.failed.connect(self._on_player_failed)
                stage: QWidget = player
            else:
                stage = QLabel("3D oynatıcı için PyQt6-WebEngine gerekli:\npip install PyQt6-WebEngine")
                stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stage.setMinimumSize(260, 320)
            stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            stage.setStyleSheet(pane_style)
            head = QHBoxLayout()
            head.addWidget(field_label(title))
            head.addStretch(1)
            head.addWidget(self.save_angle_buttons[view])
            column = QVBoxLayout()
            column.setSpacing(theme.SP_1)
            column.addLayout(head)
            column.addWidget(stage, 1)
            views.addLayout(column, 1)

        self._status = caption("Soldan bir ürün seç.")
        self._badges = QHBoxLayout()
        self._badges.setSpacing(theme.SP_2)
        self._badges.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(theme.SP_2)
        right.addLayout(controls)
        right.addLayout(views, 1)
        right.addLayout(actions)
        right.addWidget(self._status)
        right.addLayout(self._badges)

        body = QHBoxLayout()
        body.setSpacing(theme.SP_4)
        body.addLayout(left, 1)
        body.addLayout(right, 3)
        root.addLayout(body, 1)

    @staticmethod
    def _set_current(combo: QComboBox, key: str) -> None:
        idx = combo.findData(key)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ── products ─────────────────────────────────────────────────────────

    def reload_products(self) -> None:
        text = self._filter.text().strip() or None
        self._records = search_products(self._db.conn, free_text=text,
                                        limit=_LIST_LIMIT)
        self._list.blockSignals(True)
        self._list.clear()
        for rec in self._records:
            label = f"{rec.file_id}   {rec.main_category}/{rec.subcategory}"
            if rec.stone_summary.status == "stone":
                label += "  ◆"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, rec.file_id)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._status.setText(f"{len(self._records)} ürün listelendi.")

    def product_ids(self) -> list[str]:
        return [r.file_id for r in self._records]

    def current_record(self) -> CatalogRecord | None:
        item = self._list.currentItem()
        if item is None:
            return None
        file_id = item.data(Qt.ItemDataRole.UserRole)
        return next((r for r in self._records if r.file_id == file_id), None)

    def select_product(self, file_id: str) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.ItemDataRole.UserRole) == file_id:
                self._list.setCurrentRow(i)
                return

    def _on_product_changed(self, current, _previous) -> None:
        if current is None:
            return
        self._external = None
        self.load_product()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "CAD dosyası seç", "", "CAD dosyaları (*.3dm *.stl)")
        if not path:
            return
        self._external = Path(path)
        self._list.setCurrentItem(None)
        self.load_product()

    # ── the live player ──────────────────────────────────────────────────

    def current_colours(self) -> tuple[str, str]:
        return str(self._metal.currentData()), str(self._stone.currentData())

    def _source(self) -> tuple[Path, str] | None:
        if self._external is not None:
            return self._external, f"PREVIEW-{self._external.stem[:40]}"
        rec = self.current_record()
        if rec is None:
            return None
        return Path(rec.source_path), rec.file_id

    def _camera_overrides_for(self, file_id: str) -> dict | None:
        overrides = get_camera_overrides(self._db.conn, file_id)
        if not overrides:
            return None
        return {view: (o.direction, o.up) for view, o in overrides.items()}

    def load_product(self, force: bool = False) -> None:
        """Prepare the scene for the selected product (cached), then show it in the player."""
        source = self._source()
        if source is None:
            return
        path, file_id = source
        if not path.exists():
            self._status.setText(f"Kaynak dosya bulunamadı: {path}")
            self._scene = None
            self._clear_player()
            self._sync_buttons()
            return
        if self._worker is not None and self._worker.isRunning():
            # A second pick while one is preparing must not queue threads, and must not be
            # dropped either: remember it and run it when this one lands.
            self._pending = True
            self._status.setText(f"{file_id} sırada…")
            return
        self._pending = False
        place = self._sync_place_box(path)
        self._set_busy(True)
        self._status.setText(
            f"{file_id} hazırlanıyor… (ilk açılış saniyeler sürer; taşsız STL'lerde taş yuvaları taranır, "
            "bir-iki dakikayı bulabilir. Sonra anında açılır)")
        record = self.current_record()
        worker = SceneWorker(path, file_id,
                             record.main_category if record and self._external is None else None,
                             force=force, place_stones=place, parent=self)
        self._worker = worker
        worker.done.connect(self._on_scene)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        worker.start()

    def _wants_stones(self, path: Path) -> bool:
        """Whether this product's scene should get stones put in its seats (STL only)."""
        if path.suffix.lower() != ".stl":
            return False
        if self._external is not None:
            return self.place_box.isChecked()            # a loose file has no record: the switch decides
        record = self.current_record()
        return record is not None and wants_placed_stones(self._db.conn, record)

    def _sync_place_box(self, path: Path) -> bool:
        place = self._wants_stones(path)
        self.place_box.blockSignals(True)
        self.place_box.setChecked(place)
        self.place_box.setEnabled(path.suffix.lower() == ".stl")
        self.place_box.blockSignals(False)
        return place

    def _on_place_toggled(self, checked: bool) -> None:
        source = self._source()
        if source is None:
            return
        _path, file_id = source
        if self._external is None:
            set_place_stones(self._db.conn, file_id, checked)
        self.load_product()

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        if self._pending:
            self._pending = False
            self.load_product()

    def _set_busy(self, busy: bool) -> None:
        self.render_button.setEnabled(not busy)

    def _clear_player(self) -> None:
        for player in self._players.values():
            player.clear()

    def _on_scene(self, scene) -> None:
        if self._pending:
            return              # the user already moved on; the next load replaces this one
        self._scene = scene
        metal_key, stone_key = self.current_colours()
        if self._players:
            from catalog_organizer.webview import service
            svc = service.current() or service.install(self)
            url = f"{svc.viewer_url(scene)}&metal={metal_key}&stone={stone_key}"
            for player in self._players.values():
                player.load(url)
            self._apply_views()
        self._sync_buttons()
        self._show_badges()
        self._status.setText(self._status_text())

    def _status_text(self) -> str:
        metal_key, stone_key = self.current_colours()
        file_id = self._scene.file_id if self._scene is not None else ""
        return (f"{file_id} · {mat.metal(metal_key).label_tr} / "
                f"{mat.stone(stone_key).label_tr} · iki görünümü de fareyle döndürebilirsin")

    def _show_badges(self) -> None:
        while self._badges.count() > 1:
            item = self._badges.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self._scene is None:
            return
        rec = self.current_record()
        if self._scene.stones > 0 and self._scene.placed:
            self._badges.insertWidget(0, Badge(
                f"{self._scene.stones} taş yuvalara yerleştirildi", "info"))
        elif self._scene.stones > 0:
            self._badges.insertWidget(0, Badge("taş uygulandı", "success"))
        elif rec is not None and rec.stone_summary.status == "stone":
            # The analysis says stones, the file gave none to colour — say so
            # rather than let the user wonder why the stone colour did nothing.
            self._badges.insertWidget(
                0, Badge("taş geometrisi yok — taş rengi etkisiz", "warning"))
        else:
            self._badges.insertWidget(0, Badge("taşsız", "neutral"))
        if self._camera_overrides_for(self._scene.file_id):
            self._badges.insertWidget(0, Badge("elle ayarlı açı", "info"))

    def _sync_buttons(self) -> None:
        has_scene = self._scene is not None
        self.save_button.setEnabled(has_scene)
        for view, button in self.save_angle_buttons.items():
            button.setEnabled(has_scene and view in self._players)
        source = self._source()
        has_override = bool(source and self._camera_overrides_for(source[1]))
        self.reset_angle_button.setEnabled(has_override)

    def _on_colours_changed(self) -> None:
        metal_key, stone_key = self.current_colours()
        if self._players and self._scene is not None:
            for player in self._players.values():
                player.set_metal(metal_key)
                player.set_stone(stone_key)
            self._status.setText(self._status_text())

    def _apply_views(self) -> None:
        """Put each player's camera on its catalogue view: the saved angle if there is one."""
        if not self._players or self._scene is None:
            return
        overrides = get_camera_overrides(self._db.conn, self._scene.file_id)
        for name, player in self._players.items():
            if name in overrides:
                o = overrides[name]
                player.set_camera(self._scene.canonical(o.direction),
                                  self._scene.canonical(o.up))
            else:
                player.go_to_view(name)

    def _on_failed(self, message: str) -> None:
        self._scene = None
        self._clear_player()
        self._sync_buttons()
        self._status.setText(message)

    def _on_player_failed(self, message: str) -> None:
        self._status.setText(message)

    # ── saved angles ─────────────────────────────────────────────────────

    def _save_angle(self, view: str) -> None:
        if self._scene is None or view not in self._players:
            return
        scene = self._scene

        def store(state: dict) -> None:
            # canonical frame -> the file's own frame: the rotation is orthonormal, so transpose
            direction = tuple(float(x) for x in scene.rotation.T @ state["dir"])
            up = tuple(float(x) for x in scene.rotation.T @ state["up"])
            set_camera_override(self._db.conn, scene.file_id, view, direction, up)
            self._sync_buttons()
            self._show_badges()
            label = "karşıdan" if view == "front" else "çapraz"
            self._status.setText(
                f"{scene.file_id}: bu açı '{label}' görünümü olarak kaydedildi — "
                "katalog ve PNG'ler bunu kullanır.")

        self._players[view].camera_state(store)

    def _on_reset_angles(self) -> None:
        source = self._source()
        if source is None:
            return
        _path, file_id = source
        n = clear_camera_override(self._db.conn, file_id)
        if n:
            self._status.setText(f"{file_id}: {n} özel açı silindi, otomatiğe döndü.")
            self._sync_buttons()
            self._show_badges()
            self._apply_views()

    # ── PNG files ────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        source = self._source()
        if source is None or self._scene is None:
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            return
        directory = QFileDialog.getExistingDirectory(self, "Klasör seç")
        if not directory:
            return
        path, file_id = source
        metal_key, stone_key = self.current_colours()
        record = self.current_record()
        worker = RenderWorker(
            path, file_id, metal_key, stone_key, QUALITY[self._quality.currentText()],
            camera=self._camera_overrides_for(file_id), parent=self,
            category=record.main_category if record and self._external is None else None,
            place_stones=self._wants_stones(path))
        self._save_worker = worker
        self.save_button.setEnabled(False)
        self._status.setText(f"{file_id} PNG'leri hazırlanıyor…")
        worker.done.connect(lambda result, d=directory: self._write_files(result, Path(d)))
        worker.failed.connect(self._on_save_failed)
        worker.finished.connect(self._on_save_finished)
        worker.start()

    def _on_save_finished(self) -> None:
        self._save_worker = None
        self.save_button.setEnabled(self._scene is not None)

    def _on_save_failed(self, message: str) -> None:
        self._status.setText(message)

    def _write_files(self, result, target: Path) -> None:
        import shutil
        written = []
        for view, src in result.views.items():
            dest = target / f"{result.file_id}_{result.metal_key}_" \
                            f"{result.stone_key}_{view}.png"
            shutil.copyfile(src, dest)
            written.append(dest.name)
        self._status.setText(f"{result.file_id}: {len(written)} dosya kaydedildi.")
        QMessageBox.information(self, "Kaydedildi", "\n".join(written))
