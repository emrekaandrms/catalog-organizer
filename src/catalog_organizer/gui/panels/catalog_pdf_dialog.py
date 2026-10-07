"""PDF catalogue dialog + background worker.

Rendering is the slow part — roughly 2-4 seconds per product for two views —
so a 40-item catalogue takes minutes. That has to run off the UI thread or the
window freezes and Windows paints it "not responding" halfway through a job
the user cannot tell is still working.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QLineEdit, QProgressDialog, QSpinBox, QVBoxLayout, QWidget,
)

from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.render import materials as mat

# Render size per view. 1400 px across an ~85 mm printed box is ~420 DPI, past
# what print needs; 900 is still comfortably above 300 DPI and renders roughly
# twice as fast.
QUALITY_PRESETS = {
    "Hızlı (900 px)": 900,
    "Standart (1400 px)": 1400,
    "Yüksek (2000 px)": 2000,
}
DEFAULT_QUALITY = "Standart (1400 px)"


class CatalogOptions:
    def __init__(self, metal_key: str, stone_key: str, title: str,
                 resolution: int, cover: bool, out_path: Path) -> None:
        self.metal_key = metal_key
        self.stone_key = stone_key
        self.title = title
        self.resolution = resolution
        self.cover = cover
        self.out_path = out_path


class CatalogPdfDialog(QDialog):
    """Asks for stone colour and metal colour before rendering."""

    def __init__(self, *, default_title: str, item_count: int,
                 out_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PDF Katalog")
        self._out_path = out_path

        self._title = QLineEdit(default_title)
        self._metal = QComboBox()
        for key, material in mat.METALS.items():
            self._metal.addItem(material.label_tr, userData=key)
        self._stone = QComboBox()
        for key, material in mat.STONES.items():
            self._stone.addItem(material.label_tr, userData=key)
        self._set_current(self._metal, mat.DEFAULT_METAL)
        self._set_current(self._stone, mat.DEFAULT_STONE)

        self._quality = QComboBox()
        self._quality.addItems(list(QUALITY_PRESETS))
        self._quality.setCurrentText(DEFAULT_QUALITY)

        self._cover = QCheckBox("Kapak sayfası ekle")
        self._cover.setChecked(True)

        self._start = QSpinBox()
        self._start.setRange(1, 10_000)
        self._start.setValue(1)
        self._limit = QSpinBox()
        self._limit.setRange(1, 10_000)
        self._limit.setValue(item_count)
        self._limit.setToolTip(
            "Kaç ürün alınsın. Renk denemesi için birkaç üründe çalıştırıp "
            "sonucu görmek, 40 ürünü baştan render etmekten hızlıdır.")

        form = QFormLayout()
        form.addRow("Katalog adı", self._title)
        form.addRow("Maden rengi", self._metal)
        form.addRow("Taş rengi", self._stone)
        form.addRow("Görsel kalitesi", self._quality)
        form.addRow("Başlangıç sırası", self._start)
        form.addRow("Ürün sayısı", self._limit)
        form.addRow("", self._cover)

        note = QLabel(
            f"{item_count} ürün seçili. Her ürün için iki görüntü üretilecek; "
            "ürün başına yaklaşık 2-4 saniye sürer.")
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    @staticmethod
    def _set_current(combo: QComboBox, key: str) -> None:
        idx = combo.findData(key)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def options(self) -> CatalogOptions:
        return CatalogOptions(
            metal_key=str(self._metal.currentData()),
            stone_key=str(self._stone.currentData()),
            title=self._title.text().strip() or "Ürün Kataloğu",
            resolution=QUALITY_PRESETS[self._quality.currentText()],
            cover=self._cover.isChecked(),
            out_path=self._out_path,
        )

    def slice_bounds(self) -> tuple[int, int]:
        """(start index, count) — 1-based start as shown to the user."""
        return self._start.value() - 1, self._limit.value()


class CatalogWorker(QThread):
    """Renders and writes the PDF off the UI thread."""

    progressed = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object, dict)
    failed = pyqtSignal(str)

    def __init__(self, records: list[CatalogRecord], options: CatalogOptions,
                 db_conn=None, parent=None) -> None:
        super().__init__(parent)
        self._records = records
        self._options = options
        self._db_conn = db_conn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # type: ignore[override]
        from catalog_organizer.export.catalog_pdf import render_and_export

        def _progress(done: int, total: int, file_id: str) -> None:
            if self._cancelled:
                raise InterruptedError("kullanıcı iptal etti")
            self.progressed.emit(done, total, file_id)

        try:
            path, errors = render_and_export(
                self._records, self._options.out_path,
                metal_key=self._options.metal_key,
                stone_key=self._options.stone_key,
                title=self._options.title,
                resolution=self._options.resolution,
                cover=self._options.cover,
                progress=_progress,
                db_conn=self._db_conn,
            )
        except InterruptedError:
            self.failed.emit("İptal edildi.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(path, errors)


def make_progress_dialog(total: int, parent: QWidget | None) -> QProgressDialog:
    dialog = QProgressDialog("Görseller üretiliyor…", "İptal", 0, total, parent)
    dialog.setWindowTitle("PDF Katalog")
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    return dialog
