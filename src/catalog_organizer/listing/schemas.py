"""LLM'in üreteceği ilan metinlerinin şeması.

Sınırlar platformların GERÇEK kuralları — Etsy başlığı 140 karakteri, tag'i
20 karakteri aşarsa ilan reddedilir. Bu yüzden sınırları burada, üretim
anında zorluyoruz; dışa aktarımda değil.

Sayısal alanlar (ağırlık, ölçü, karat, fiyat) bu şemada YOK ve olmayacak —
onlar DB'den basılır. Bir dil modeline "bu yüzük 4,2 gram" dedirtmiyoruz.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator

ETSY_TITLE_MAX = 140
ETSY_TAG_MAX_COUNT = 13
ETSY_TAG_MAX_LEN = 20
ETSY_MATERIAL_MAX_COUNT = 13
ETSY_MATERIAL_MAX_LEN = 45
META_DESCRIPTION_MAX = 160


def _clean_list(values: list[str], max_len: int) -> list[str]:
    """Boşları at, kırp, tekrarları kaldır (sırayı koru), uzunları ele."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or len(value) > max_len:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


class EtsyListingDraft(BaseModel):
    """Etsy ilanı — İngilizce, fiziksel gümüş."""

    title: str
    description: str
    tags: list[str] = []
    materials: list[str] = []
    taxonomy_hint: str = ""
    sustainability_note: str = ""

    @field_validator("title")
    @classmethod
    def _title_fits(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title boş olamaz")
        if len(v) > ETSY_TITLE_MAX:
            raise ValueError(
                f"Etsy başlığı en fazla {ETSY_TITLE_MAX} karakter, {len(v)} geldi"
            )
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_fit(cls, v):
        tags = _clean_list(list(v or []), ETSY_TAG_MAX_LEN)
        return tags[:ETSY_TAG_MAX_COUNT]

    @field_validator("materials", mode="before")
    @classmethod
    def _materials_fit(cls, v):
        mats = _clean_list(list(v or []), ETSY_MATERIAL_MAX_LEN)
        return mats[:ETSY_MATERIAL_MAX_COUNT]


class WooListingDraft(BaseModel):
    """WooCommerce ilanı — Türkçe, varyasyonlu."""

    title: str
    short_description: str = ""
    description: str
    tags: list[str] = []
    categories: list[str] = []
    slug: str = ""
    meta_description: str = ""
    sustainability_note: str = ""

    @field_validator("title")
    @classmethod
    def _title_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title boş olamaz")
        return v

    @field_validator("tags", "categories", mode="before")
    @classmethod
    def _clean(cls, v):
        return _clean_list(list(v or []), 60)

    @field_validator("meta_description")
    @classmethod
    def _meta_fits(cls, v: str) -> str:
        v = v.strip()
        return v[:META_DESCRIPTION_MAX]


def slugify(text: str) -> str:
    """Türkçe karakterleri koruyan değil, ASCII'ye indiren slug — WooCommerce
    URL'lerinde ş/ğ/ı yüzde-kodlanır ve okunmaz hale gelir."""
    table = str.maketrans({
        "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "İ": "i",
        "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
    })
    ascii_text = text.translate(table).lower()
    out: list[str] = []
    for ch in ascii_text:
        if ch.isalnum() and ch.isascii():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")
