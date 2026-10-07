"""Stone carat / mass estimator backed by CSV lookup tables.

Two table flavours coexist:

  * **CZ** (cubic zirconia, default for this workshop) — per-shape CSVs
    `cz_round.csv`, `cz_oval.csv`, `cz_pear.csv`, … carrying *both* the
    CZ carat weight and the per-stone gram mass directly from the
    supplier table the user provided. CZ is what their production
    actually uses; values agree with their suppliers within tolerance.

  * **Diamond** (`round.csv` + `_shape_factors.yaml`) — kept for
    Diamond-equivalent carat estimation when needed (the original
    table is still there). Not used by default any more.

Both round (single mm value) and non-round (`LONGxSHORT` mm pair)
sizes are handled. Linear interpolation between adjacent rows; clamp at
endpoints. Returns both `carat` and `grams` so the pipeline can compute
the final product weight (metal + stones) directly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from catalog_organizer.core.config import load_stone_shape_factors
from catalog_organizer.core.paths import config_dir


@dataclass(frozen=True)
class _RoundRow:
    size_mm: float
    carat_avg: float
    grams_per_stone: float | None = None      # only populated for CZ rows
    carat_min: float | None = None             # only populated for diamond rows
    carat_max: float | None = None


@dataclass(frozen=True)
class _PairRow:
    """A row for non-round shapes keyed by (long, short) mm."""
    long_mm: float
    short_mm: float
    carat: float
    grams_per_stone: float


_SHAPE_FILES_CZ = {
    "round":    "cz_round.csv",
    "oval":     "cz_oval.csv",
    "pear":     "cz_pear.csv",
    "heart":    "cz_heart.csv",
    "marquise": "cz_marquise.csv",
    "princess": "cz_princess.csv",
    "trillion": "cz_trillion.csv",
    "emerald":  "cz_emerald.csv",
    "triangle": "cz_triangle.csv",
    "radiant":  "cz_radiant.csv",
}


def _load_cz_round_table(path: Path) -> list[_RoundRow]:
    rows: list[_RoundRow] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(_RoundRow(
                size_mm=float(r["size_mm"]),
                carat_avg=float(r["carat_cz"]),
                grams_per_stone=float(r["grams_per_stone"]),
            ))
    rows.sort(key=lambda x: x.size_mm)
    return rows


def _load_cz_pair_table(path: Path) -> list[_PairRow]:
    rows: list[_PairRow] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(_PairRow(
                long_mm=float(r["long_mm"]),
                short_mm=float(r["short_mm"]),
                carat=float(r["carat_cz"]),
                grams_per_stone=float(r["grams_per_stone"]),
            ))
    rows.sort(key=lambda x: (x.long_mm, x.short_mm))
    return rows


def _load_diamond_round_table(path: Path) -> list[_RoundRow]:
    """Legacy diamond round table (size_mm,carat_avg,carat_min,carat_max)."""
    rows: list[_RoundRow] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(_RoundRow(
                size_mm=float(r["size_mm"]),
                carat_avg=float(r["carat_avg"]),
                grams_per_stone=float(r["carat_avg"]) * 0.2,   # 1 ct = 0.2 g
                carat_min=float(r["carat_min"]),
                carat_max=float(r["carat_max"]),
            ))
    rows.sort(key=lambda x: x.size_mm)
    return rows


def _interp_round(rows: list[_RoundRow], size_mm: float) -> _RoundRow:
    if not rows:
        return _RoundRow(0.0, 0.0, 0.0)
    if size_mm <= rows[0].size_mm:
        return rows[0]
    if size_mm >= rows[-1].size_mm:
        return rows[-1]
    for i in range(len(rows) - 1):
        lo, hi = rows[i], rows[i + 1]
        if lo.size_mm <= size_mm < hi.size_mm:
            t = (size_mm - lo.size_mm) / (hi.size_mm - lo.size_mm)
            return _RoundRow(
                size_mm=size_mm,
                carat_avg=lo.carat_avg + t * (hi.carat_avg - lo.carat_avg),
                grams_per_stone=(
                    (lo.grams_per_stone or 0) + t * ((hi.grams_per_stone or 0) - (lo.grams_per_stone or 0))
                ),
            )
    return rows[-1]


def _interp_pair(rows: list[_PairRow], long_mm: float, short_mm: float) -> _PairRow:
    """Nearest-row match in (long, short) space by sum of squared mm error.
    The CZ tables are spaced finely enough that nearest-neighbour is within
    ~5% of true. Linear interpolation in two dimensions adds complexity
    for marginal gain."""
    if not rows:
        return _PairRow(0.0, 0.0, 0.0, 0.0)
    best = rows[0]
    best_d = (best.long_mm - long_mm) ** 2 + (best.short_mm - short_mm) ** 2
    for r in rows[1:]:
        d = (r.long_mm - long_mm) ** 2 + (r.short_mm - short_mm) ** 2
        if d < best_d:
            best = r
            best_d = d
    return best


class StoneWeightTable:
    """Estimate carat + mass per stone from physical size.

    Default behaviour reads from the CZ-flavoured CSVs (`cz_*.csv`). Pass
    `material="diamond"` to fall back to the original diamond round
    table + shape factors (legacy path).
    """

    def __init__(
        self,
        material: Literal["cz", "diamond"] = "cz",
        round_csv: Path | None = None,
        shape_factors: dict[str, float] | None = None,
        cz_table_dir: Path | None = None,
    ) -> None:
        self._material = material
        if material == "cz":
            d = cz_table_dir or (config_dir() / "stone_weight_tables")
            self._cz_round = _load_cz_round_table(d / _SHAPE_FILES_CZ["round"])
            self._cz_pairs: dict[str, list[_PairRow]] = {}
            for shape, fname in _SHAPE_FILES_CZ.items():
                if shape == "round":
                    continue
                p = d / fname
                if p.exists():
                    self._cz_pairs[shape] = _load_cz_pair_table(p)
        else:
            round_csv = round_csv or (config_dir() / "stone_weight_tables" / "round.csv")
            self._diamond_round = _load_diamond_round_table(round_csv)
            self._factors = (
                shape_factors if shape_factors is not None
                else load_stone_shape_factors()
            )

    # ── Round-stone helpers (kept for backward compat with tests) ──────────

    def round_carat(self, size_mm: float) -> float:
        """Carat for a round stone of given mm. Default = CZ table."""
        if self._material == "cz":
            return _interp_round(self._cz_round, size_mm).carat_avg
        return _interp_round(self._diamond_round, size_mm).carat_avg

    # ── Primary API ─────────────────────────────────────────────────────────

    def estimate_carat(
        self,
        shape: str,
        size_mm: float | tuple[float, float],
    ) -> float:
        """Return carat only. Kept for callers that don't need mass."""
        return self.estimate(shape, size_mm)[0]

    def estimate(
        self,
        shape: str,
        size_mm: float | tuple[float, float],
    ) -> tuple[float, float]:
        """Return (carat, grams) for one stone of `shape` and `size_mm`.

        `size_mm` is a single value for round-like shapes or a (long, short)
        tuple for non-round. Unknown shape → fall back to round table at the
        short-side mm (an under-estimate but never zero).
        """
        if self._material == "cz":
            return self._estimate_cz(shape, size_mm)
        return self._estimate_diamond(shape, size_mm)

    # ── CZ path ─────────────────────────────────────────────────────────────

    def _estimate_cz(
        self,
        shape: str,
        size_mm: float | tuple[float, float],
    ) -> tuple[float, float]:
        if shape == "round":
            s = float(size_mm if not isinstance(size_mm, tuple) else size_mm[0])
            r = _interp_round(self._cz_round, s)
            return r.carat_avg, (r.grams_per_stone or r.carat_avg * 0.2)

        pair_rows = self._cz_pairs.get(shape)
        if pair_rows is None:
            # Unknown shape — use the round table on the short side as a
            # conservative fall-back.
            short = float(size_mm[1] if isinstance(size_mm, tuple) else size_mm)
            r = _interp_round(self._cz_round, short)
            return r.carat_avg, (r.grams_per_stone or r.carat_avg * 0.2)

        if isinstance(size_mm, tuple):
            long, short = max(size_mm), min(size_mm)
        else:
            long = short = float(size_mm)
        row = _interp_pair(pair_rows, long, short)
        return row.carat, row.grams_per_stone

    # ── Diamond legacy path ─────────────────────────────────────────────────

    def _estimate_diamond(
        self,
        shape: str,
        size_mm: float | tuple[float, float],
    ) -> tuple[float, float]:
        if shape == "round":
            s = float(size_mm if not isinstance(size_mm, tuple) else size_mm[0])
            r = _interp_round(self._diamond_round, s)
            return r.carat_avg, r.carat_avg * 0.2

        if isinstance(size_mm, tuple):
            short, long = sorted(size_mm)
        else:
            short = long = float(size_mm)
        factor = float(self._factors.get(shape, 1.0))
        ratio = (long / short) if short > 0 else 1.0
        ct = self.round_carat(short) * factor * ratio
        return ct, ct * 0.2
