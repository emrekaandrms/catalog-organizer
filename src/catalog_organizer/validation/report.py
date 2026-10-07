"""Pilot validation — compare predictions in catalog_master.jsonl against a
human-labelled truth CSV and emit a Markdown accuracy report.

Truth CSV schema (header row required):
    file_id,main_category,subcategory,polished_status,tags
    JCAD-000000001,ring,solitaire,polished,"ring;band;polished;round_form;closed_form"

`tags` is `;`-separated, may be empty. Unknown columns are ignored. Missing
optional columns are skipped during their metric.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from catalog_organizer.catalog.writer import read_all_records
from catalog_organizer.core.schemas import CatalogRecord


# ── Truth data ────────────────────────────────────────────────────────────────

@dataclass
class TruthRow:
    file_id: str
    main_category: str | None = None
    subcategory: str | None = None
    polished_status: str | None = None
    tags: list[str] = field(default_factory=list)


def load_truth_csv(path: Path) -> dict[str, TruthRow]:
    """Read truth CSV; return {file_id: TruthRow}. Empty cells become None."""
    out: dict[str, TruthRow] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            fid = (raw.get("file_id") or "").strip()
            if not fid:
                continue
            tags_cell = (raw.get("tags") or "").strip()
            out[fid] = TruthRow(
                file_id=fid,
                main_category=_clean(raw.get("main_category")),
                subcategory=_clean(raw.get("subcategory")),
                polished_status=_clean(raw.get("polished_status")),
                tags=[t.strip() for t in tags_cell.split(";") if t.strip()],
            )
    return out


def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    return s or None


# ── Comparison ───────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    total_compared:           int = 0
    truth_only_missing:       int = 0       # in truth, missing from catalog
    catalog_only_extra:       int = 0       # in catalog, missing from truth

    main_correct:             int = 0
    sub_correct:              int = 0       # only counted when main is correct
    sub_eligible:             int = 0       # records where main is correct AND truth.subcategory != None
    polished_correct:         int = 0
    polished_eligible:        int = 0

    tag_relevance_sum:        float = 0.0
    tag_eligible:             int = 0

    confusion: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    mispredictions: list[tuple[str, str, str]] = field(default_factory=list)
    # (file_id, predicted "main/sub", expected "main/sub")

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def main_accuracy(self) -> float:
        return self.main_correct / self.total_compared if self.total_compared else 0.0

    @property
    def sub_accuracy(self) -> float:
        return self.sub_correct / self.sub_eligible if self.sub_eligible else 0.0

    @property
    def polished_accuracy(self) -> float:
        return self.polished_correct / self.polished_eligible if self.polished_eligible else 0.0

    @property
    def avg_tag_relevance(self) -> float:
        return self.tag_relevance_sum / self.tag_eligible if self.tag_eligible else 0.0


def compare(
    catalog_records: list[CatalogRecord],
    truth: dict[str, TruthRow],
) -> ValidationReport:
    report = ValidationReport()
    catalog_by_id = {r.file_id: r for r in catalog_records}

    # Records present in truth but not in catalog
    for fid in truth:
        if fid not in catalog_by_id:
            report.truth_only_missing += 1

    # Records in catalog but not in truth
    for fid in catalog_by_id:
        if fid not in truth:
            report.catalog_only_extra += 1

    for fid, t in truth.items():
        rec = catalog_by_id.get(fid)
        if rec is None:
            continue
        report.total_compared += 1

        # Main category
        main_ok = (t.main_category is not None and rec.main_category == t.main_category)
        if t.main_category is not None:
            report.confusion[t.main_category][rec.main_category] += 1
        if main_ok:
            report.main_correct += 1
        elif t.main_category is not None:
            report.mispredictions.append((
                fid,
                f"{rec.main_category}/{rec.subcategory}",
                f"{t.main_category}/{t.subcategory or '?'}",
            ))

        # Subcategory (only when main matches)
        if main_ok and t.subcategory is not None:
            report.sub_eligible += 1
            if rec.subcategory == t.subcategory:
                report.sub_correct += 1

        # Polished status
        if t.polished_status is not None:
            report.polished_eligible += 1
            if rec.polished_status == t.polished_status:
                report.polished_correct += 1

        # Tag relevance: |predicted ∩ truth| / |truth|
        if t.tags:
            report.tag_eligible += 1
            truth_set = set(t.tags)
            pred_set  = set(rec.controlled_tags)
            overlap   = truth_set & pred_set
            report.tag_relevance_sum += len(overlap) / len(truth_set)

    return report


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_markdown(report: ValidationReport, top_n_mispred: int = 10) -> str:
    lines: list[str] = []
    lines.append("# Pilot Validation Report\n")
    lines.append(f"- Total records compared: **{report.total_compared}**")
    if report.truth_only_missing:
        lines.append(f"- In truth but missing from catalog: {report.truth_only_missing}")
    if report.catalog_only_extra:
        lines.append(f"- In catalog but missing from truth: {report.catalog_only_extra}")
    lines.append("")
    lines.append(f"- **Main-category accuracy:** {report.main_accuracy:.1%}  "
                 f"({report.main_correct}/{report.total_compared})")
    if report.sub_eligible:
        lines.append(f"- **Subcategory accuracy** (when main correct): "
                     f"{report.sub_accuracy:.1%}  ({report.sub_correct}/{report.sub_eligible})")
    if report.polished_eligible:
        lines.append(f"- **Polished accuracy:** {report.polished_accuracy:.1%}  "
                     f"({report.polished_correct}/{report.polished_eligible})")
    if report.tag_eligible:
        lines.append(f"- **Avg tag relevance:** {report.avg_tag_relevance:.2f}  "
                     f"(over {report.tag_eligible} records)")
    lines.append("")

    if report.confusion:
        lines.append("## Confusion matrix — main category")
        preds = sorted({p for c in report.confusion.values() for p in c.keys()})
        truths = sorted(report.confusion.keys())
        header = "| truth \\ pred | " + " | ".join(preds) + " |"
        sep    = "|" + " --- |" * (len(preds) + 1)
        lines.append(header)
        lines.append(sep)
        for t in truths:
            row = "| " + t + " | " + " | ".join(
                str(report.confusion[t].get(p, 0)) for p in preds
            ) + " |"
            lines.append(row)
        lines.append("")

    if report.mispredictions:
        lines.append(f"## Top {top_n_mispred} mispredictions")
        for fid, pred, exp in report.mispredictions[:top_n_mispred]:
            lines.append(f"- `{fid}`: predicted **{pred}**, expected **{exp}**")
        lines.append("")

    return "\n".join(lines)


# ── Convenience entry ────────────────────────────────────────────────────────

def run_validation(
    truth_csv: Path,
    catalog_jsonl: Path,
    out_md: Path | None = None,
) -> tuple[ValidationReport, str]:
    truth = load_truth_csv(truth_csv)
    records = list(read_all_records(catalog_jsonl))
    report = compare(records, truth)
    markdown = render_markdown(report)
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    return report, markdown
