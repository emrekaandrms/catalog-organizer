"""Türkçe-doğru metin normalizasyonu ve FTS5 sorgu üretimi.

FTS5'in unicode61 tokenizer'ı Türkçe bilmiyor: 'İ' küçültüldüğünde
'i' + U+0307 (birleşen nokta) üretiyor, 'i' değil. Çözüm tokenizer'ı
değiştirmek değil — indekse yazılan metni de sorguyu da AYNI fonksiyondan
geçirmek. Böylece tokenizer'ın Türkçe bilmesine gerek kalmıyor.
"""
from __future__ import annotations

_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def tr_lower(text: str) -> str:
    return text.translate(_TR_LOWER_MAP).lower()


def normalise_for_index(text: str) -> str:
    return tr_lower(text).strip()


def to_fts_query(free_text: str) -> str:
    """Serbest metni FTS5 MATCH ifadesine çevirir.

    Her kelime tırnaklanır — FTS5'in AND/OR/NOT/NEAR operatörleri ve tırnak
    karakterleri kullanıcı metninde geçerse sözdizimi hatası verirdi — ve
    AND ile birleştirilir (mevcut `search_records`'ın "her kelime geçmeli"
    davranışıyla aynı).
    """
    cleaned = normalise_for_index(free_text)
    for ch in '"*():^-':
        cleaned = cleaned.replace(ch, " ")
    words = [w for w in cleaned.split() if w]
    return " AND ".join(f'"{w}"' for w in words)
