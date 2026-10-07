"""Varyant fiyatlaması — deterministik, LLM içermez.

Formüller `config/pricing.yaml`'de; kullanıcıdan alındı (2026-09-09):

    Etsy  gümüş : (g × 1.2 × 1.3 × 2.9 + 20 + 5 + 5) × 2   → USD
    Woo   gümüş : g × 2.7 × 110                             → TL
    Woo   altın : g × milyem × 1.4 × 7000                   → TL
    Woo   STL   : 50 sabit                                  → TL

"g" tanımı:
  * Etsy ve Woo gümüş → `metal_weights.silver_925_g`
  * Woo altın (dört ayarın hepsi) → `metal_weights.gold_14k_yellow_g`

Ayar başına ayrı yoğunluk KULLANILMIYOR (kullanıcı kararı). Fiziksel olarak
18 ayar aynı hacimde daha ağırdır, yani tek gram kullanmak 18 ayarı gerçek
ağırlığına göre ~%16 ucuza, 8 ayarı ~%20 pahalıya yazar. Kullanıcıya
söylendi, bilerek kabul edildi.

Ağırlık `pipeline.py` tarafından koçan hacmi düşülerek hesaplanıyor, yani
buraya gelen gram zaten net — koçansız — değer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catalog_organizer.core.schemas import CatalogRecord

CHANNELS = ("etsy", "woocommerce")

SILVER = "silver_925"
STL = "stl_digital"
GOLD_KARATS = (8, 10, 14, 18)

ETSY_VARIANTS = (SILVER,)
WOO_VARIANTS = (SILVER, "gold_8k", "gold_10k", "gold_14k", "gold_18k", STL)

_SKU_SUFFIX = {
    SILVER: "AG",
    "gold_8k": "8K",
    "gold_10k": "10K",
    "gold_14k": "14K",
    "gold_18k": "18K",
    STL: "STL",
}

VARIANT_LABEL_TR = {
    SILVER: "Gümüş",
    "gold_8k": "8 Ayar Altın",
    "gold_10k": "10 Ayar Altın",
    "gold_14k": "14 Ayar Altın",
    "gold_18k": "18 Ayar Altın",
    STL: "STL Dosyası",
}

VARIANT_LABEL_EN = {SILVER: "Sterling Silver 925"}


class PricingError(ValueError):
    """Fiyat hesaplanamıyor — tahmin üretmek yerine hata veriyoruz."""


@dataclass(frozen=True)
class PricingConfig:
    raw: dict[str, Any]

    # --- Etsy ---
    @property
    def etsy_currency(self) -> str:
        return str(self.raw["etsy"]["currency"])

    @property
    def etsy_who_made(self) -> str:
        return str(self.raw["etsy"].get("who_made", "i_did"))

    @property
    def etsy_when_made(self) -> str:
        return str(self.raw["etsy"].get("when_made", "made_to_order"))

    # --- Woo ---
    @property
    def woo_currency(self) -> str:
        return str(self.raw["woocommerce"]["currency"])

    @property
    def gold_gram_source(self) -> str:
        return str(self.raw["woocommerce"]["gold"]["gram_source"])

    def milyem(self, karat: int) -> float:
        # YAML anahtarları int gelir, ama JSON turundan (pricing_snapshot)
        # dönerse str olur — ikisini de kabul et.
        karats = self.raw["woocommerce"]["gold"]["karats"]
        for key in (karat, str(karat)):
            if key in karats:
                return float(karats[key])
        raise PricingError(f"{karat} ayar için milyem tanımlı değil")

    def snapshot(self) -> dict[str, Any]:
        """Bir fiyatın hangi sabitlerle hesaplandığının kaydı."""
        etsy = self.raw["etsy"]["silver"]
        woo = self.raw["woocommerce"]
        return {
            "etsy_silver": dict(etsy),
            "woo_silver": dict(woo["silver"]),
            "woo_gold": {
                "gram_source": woo["gold"]["gram_source"],
                "markup": woo["gold"]["markup"],
                "pure_gold_gram_price": woo["gold"]["pure_gold_gram_price"],
                # Anahtarlar str'e çevriliyor: snapshot JSON olarak
                # saklanıyor ve orjson int anahtarlı dict'i reddediyor.
                "karats": {str(k): v for k, v in woo["gold"]["karats"].items()},
            },
            "woo_stl_flat": woo["stl"]["flat_price"],
        }


@dataclass(frozen=True)
class VariantPrice:
    variant_key: str
    sku: str
    price: float
    currency: str
    weight_g: float | None
    is_digital: bool


def load_pricing_config() -> PricingConfig:
    from catalog_organizer.core.config import load_pricing_settings

    return PricingConfig(load_pricing_settings())


def make_sku(file_id: str, variant_key: str) -> str:
    """Deterministik ve kalıcı — WooCommerce içe aktarıcısı bundan eşleştirip
    mevcut ürünü günceller, kopya açmaz."""
    suffix = _SKU_SUFFIX.get(variant_key)
    if suffix is None:
        raise PricingError(f"bilinmeyen varyant: {variant_key}")
    return f"{file_id}-{suffix}"


def etsy_silver_price(gram: float, cfg: PricingConfig) -> float:
    s = cfg.raw["etsy"]["silver"]
    base = gram * float(s["factor_a"]) * float(s["factor_b"]) * float(s["metal_rate"])
    base += sum(float(a) for a in s["additions"])
    return round(base * float(s["final_multiplier"]), 2)


def woo_silver_price(gram: float, cfg: PricingConfig) -> float:
    s = cfg.raw["woocommerce"]["silver"]
    return round(gram * float(s["factor"]) * float(s["metal_rate"]), 2)


def woo_gold_price(gram: float, karat: int, cfg: PricingConfig) -> float:
    g = cfg.raw["woocommerce"]["gold"]
    return round(
        gram * cfg.milyem(karat) * float(g["markup"]) * float(g["pure_gold_gram_price"]),
        2,
    )


def woo_stl_price(cfg: PricingConfig) -> float:
    return round(float(cfg.raw["woocommerce"]["stl"]["flat_price"]), 2)


def _silver_gram(rec: CatalogRecord) -> float | None:
    mw = rec.metal_weights
    if mw is None or mw.silver_925_g <= 0:
        return None
    return mw.silver_925_g


def _gold_gram(rec: CatalogRecord, cfg: PricingConfig) -> float | None:
    mw = rec.metal_weights
    if mw is None:
        return None
    value = getattr(mw, cfg.gold_gram_source, 0.0)
    return value if value > 0 else None


def price_variants(
    rec: CatalogRecord, channel: str, cfg: PricingConfig
) -> list[VariantPrice]:
    """Bir ürünün bir kanaldaki tüm varyant fiyatları.

    Ağırlık yoksa fiziksel varyant HİÇ üretilmez — sıfır ya da tahmini fiyat
    döndürmek, siteye 0 TL'lik altın yüzük koymak demek olurdu. STL sabit
    fiyatlı olduğu için ağırlıktan bağımsız üretilir.
    """
    if channel not in CHANNELS:
        raise PricingError(f"bilinmeyen kanal: {channel}")

    out: list[VariantPrice] = []
    silver_g = _silver_gram(rec)

    if channel == "etsy":
        if silver_g is not None:
            out.append(VariantPrice(
                variant_key=SILVER,
                sku=make_sku(rec.file_id, SILVER),
                price=etsy_silver_price(silver_g, cfg),
                currency=cfg.etsy_currency,
                weight_g=round(silver_g, 3),
                is_digital=False,
            ))
        return out

    currency = cfg.woo_currency
    if silver_g is not None:
        out.append(VariantPrice(
            variant_key=SILVER,
            sku=make_sku(rec.file_id, SILVER),
            price=woo_silver_price(silver_g, cfg),
            currency=currency,
            weight_g=round(silver_g, 3),
            is_digital=False,
        ))

    gold_g = _gold_gram(rec, cfg)
    if gold_g is not None:
        for karat in GOLD_KARATS:
            key = f"gold_{karat}k"
            out.append(VariantPrice(
                variant_key=key,
                sku=make_sku(rec.file_id, key),
                price=woo_gold_price(gold_g, karat, cfg),
                currency=currency,
                weight_g=round(gold_g, 3),
                is_digital=False,
            ))

    out.append(VariantPrice(
        variant_key=STL,
        sku=make_sku(rec.file_id, STL),
        price=woo_stl_price(cfg),
        currency=currency,
        weight_g=None,
        is_digital=True,
    ))
    return out


def has_physical_variants(variants: list[VariantPrice]) -> bool:
    return any(not v.is_digital for v in variants)
