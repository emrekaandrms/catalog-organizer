"""Fiyat formülleri — beklenen değerler ELLE hesaplandı.

Kullanıcının verdiği formüller (2026-09-09):
    Etsy  gümüş : (g × 1.2 × 1.3 × 2.9 + 20 + 5 + 5) × 2
    Woo   gümüş : g × 2.7 × 110
    Woo   altın : g × milyem × 1.4 × 7000
    Woo   STL   : 50 sabit

5 gram için elle:
    Etsy  : 5 × 1.2 = 6 → × 1.3 = 7.8 → × 2.9 = 22.62 → + 30 = 52.62 → × 2 = 105.24
    Woo Ag: 5 × 2.7 = 13.5 → × 110 = 1485.00
    Woo 8 : 5 × 0.333 = 1.665 → × 1.4 = 2.331 → × 7000 = 16317.00
    Woo 10: 5 × 0.417 = 2.085 → × 1.4 = 2.919 → × 7000 = 20433.00
    Woo 14: 5 × 0.585 = 2.925 → × 1.4 = 4.095 → × 7000 = 28665.00
    Woo 18: 5 × 0.750 = 3.750 → × 1.4 = 5.250 → × 7000 = 36750.00

Bu sayılar kayarsa test kırılır — sitede yanlış fiyat çıkmasının önündeki
tek engel bu.
"""
from __future__ import annotations

import pytest

from catalog_organizer.core.schemas import MetalWeights
from catalog_organizer.listing.pricing import (
    ETSY_VARIANTS, WOO_VARIANTS, PricingError, etsy_silver_price,
    has_physical_variants, load_pricing_config, make_sku, price_variants,
    woo_gold_price, woo_silver_price, woo_stl_price,
)


@pytest.fixture(scope="module")
def cfg():
    return load_pricing_config()


def test_config_loads_from_real_yaml(cfg):
    assert cfg.etsy_currency == "USD"
    assert cfg.woo_currency == "TRY"
    assert cfg.gold_gram_source == "gold_14k_yellow_g"


def test_milyem_values(cfg):
    assert cfg.milyem(8) == 0.333
    assert cfg.milyem(10) == 0.417
    assert cfg.milyem(14) == 0.585
    assert cfg.milyem(18) == 0.750


def test_unknown_karat_raises(cfg):
    with pytest.raises(PricingError):
        cfg.milyem(22)


def test_etsy_silver_5g(cfg):
    assert etsy_silver_price(5.0, cfg) == 105.24


def test_woo_silver_5g(cfg):
    assert woo_silver_price(5.0, cfg) == 1485.00


@pytest.mark.parametrize("karat,expected", [
    (8, 16317.00), (10, 20433.00), (14, 28665.00), (18, 36750.00),
])
def test_woo_gold_5g(cfg, karat, expected):
    assert woo_gold_price(5.0, karat, cfg) == expected


def test_woo_stl_is_flat(cfg):
    assert woo_stl_price(cfg) == 50.00


def test_zero_gram_etsy_still_has_fixed_additions(cfg):
    """(0 + 30) × 2 = 60 — formülün kendisi böyle. Ürün fiyatlamasında sıfır
    gram zaten `price_variants` tarafından elenir (ağırlık yok sayılır)."""
    assert etsy_silver_price(0.0, cfg) == 60.00


def test_sku_format():
    assert make_sku("JCAD-000000017", "silver_925") == "JCAD-000000017-AG"
    assert make_sku("JCAD-000000017", "gold_8k") == "JCAD-000000017-8K"
    assert make_sku("JCAD-000000017", "gold_18k") == "JCAD-000000017-18K"
    assert make_sku("JCAD-000000017", "stl_digital") == "JCAD-000000017-STL"


def test_sku_rejects_unknown_variant():
    with pytest.raises(PricingError):
        make_sku("JCAD-1", "gold_22k")


def test_etsy_produces_only_silver(cfg, record_factory):
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=5.0, gold_14k_yellow_g=6.3))
    variants = price_variants(rec, "etsy", cfg)
    assert [v.variant_key for v in variants] == list(ETSY_VARIANTS)
    assert variants[0].price == 105.24
    assert variants[0].currency == "USD"
    assert variants[0].sku == "A-AG"


def test_woo_produces_six_variants_in_order(cfg, record_factory):
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=5.0, gold_14k_yellow_g=5.0))
    variants = price_variants(rec, "woocommerce", cfg)
    assert [v.variant_key for v in variants] == list(WOO_VARIANTS)
    prices = {v.variant_key: v.price for v in variants}
    assert prices == {
        "silver_925": 1485.00,
        "gold_8k": 16317.00,
        "gold_10k": 20433.00,
        "gold_14k": 28665.00,
        "gold_18k": 36750.00,
        "stl_digital": 50.00,
    }
    assert all(v.currency == "TRY" for v in variants)


def test_all_gold_karats_share_one_gram(cfg, record_factory):
    """Kullanıcı kararı: ayar başına yoğunluk yok. Dört ayar da AYNI gramı
    taşımalı; fark yalnızca milyemden gelmeli."""
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=5.0, gold_10k_yellow_g=4.0,
        gold_14k_yellow_g=5.0, gold_18k_yellow_g=6.0))
    variants = {v.variant_key: v for v in price_variants(rec, "woocommerce", cfg)}
    golds = [variants[f"gold_{k}k"].weight_g for k in (8, 10, 14, 18)]
    assert golds == [5.0, 5.0, 5.0, 5.0]


def test_stl_is_digital_and_weightless(cfg, record_factory):
    rec = record_factory(file_id="A", metal_weights=MetalWeights(
        silver_925_g=5.0, gold_14k_yellow_g=5.0))
    stl = [v for v in price_variants(rec, "woocommerce", cfg)
           if v.variant_key == "stl_digital"][0]
    assert stl.is_digital is True
    assert stl.weight_g is None


def test_missing_weights_produce_no_physical_variants(cfg, record_factory):
    """Ağırlık yoksa 0 TL'lik altın yüzük üretmek yerine fiziksel varyant
    hiç çıkarılmaz. STL sabit fiyatlı olduğu için kalır."""
    rec = record_factory(file_id="A", metal_weights=None,
                         weight_source="unavailable",
                         weight_confidence="unavailable")
    assert price_variants(rec, "etsy", cfg) == []
    woo = price_variants(rec, "woocommerce", cfg)
    assert [v.variant_key for v in woo] == ["stl_digital"]
    assert has_physical_variants(woo) is False


def test_zero_weight_treated_as_missing(cfg, record_factory):
    rec = record_factory(file_id="A", metal_weights=MetalWeights())
    assert price_variants(rec, "etsy", cfg) == []
    assert [v.variant_key for v in price_variants(rec, "woocommerce", cfg)] \
        == ["stl_digital"]


def test_unknown_channel_raises(cfg, record_factory):
    with pytest.raises(PricingError):
        price_variants(record_factory(file_id="A"), "amazon", cfg)


def test_snapshot_records_the_constants_used(cfg):
    """Kur değiştiğinde hangi ilanların bayat kaldığı bundan anlaşılacak."""
    snap = cfg.snapshot()
    assert snap["woo_gold"]["pure_gold_gram_price"] == 7000
    assert snap["woo_gold"]["karats"] == {"8": 0.333, "10": 0.417, "14": 0.585, "18": 0.75}
    assert snap["woo_silver"]["metal_rate"] == 110
    assert snap["etsy_silver"]["metal_rate"] == 2.9
    assert snap["woo_stl_flat"] == 50
