from __future__ import annotations

from pathlib import Path

import pytest

from catalog_organizer.core.config import load_listing_prompt_templates
from catalog_organizer.core.schemas import MetalWeights, StoneEntry, StoneSummary
from catalog_organizer.db.connection import Database
from catalog_organizer.db.listings import get_listing
from catalog_organizer.listing.generator import (
    ListingGenerationError, build_facts, generate_listing, persist, reprice,
)
from catalog_organizer.listing.pricing import load_pricing_config, price_variants
from catalog_organizer.vlm.text_provider import (
    TextProviderError, _parse_json_or_raise, create_text_provider,
)
from catalog_organizer.vlm.providers import ProviderConfigurationError


class FakeProvider:
    """Sabit cevap döndüren sağlayıcı — ağ yok, testler deterministik."""

    def __init__(self, payload: dict, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise TextProviderError("bozuk cevap")
        return self.payload

    @property
    def display_name(self) -> str:
        return "Fake (test-model)"

    @property
    def model_name(self) -> str:
        return "test-model"


ETSY_PAYLOAD = {
    "title": "Sterling Silver Heart Pendant Handmade Made to Order",
    "description": "A heart pendant.\n\nCast to order.\n\nArrives boxed.",
    "tags": [f"tag {i}" for i in range(13)],
    "materials": ["sterling silver", "recycled silver"],
    "taxonomy_hint": "Jewelry > Necklaces > Pendant Necklaces",
    "sustainability_note": "Cast to order from recycled silver.",
}

WOO_PAYLOAD = {
    "title": "Kalp Kolye Ucu",
    "short_description": "Zarif kalp kolye ucu.",
    "description": "<p>Zarif kalp kolye ucu.</p>",
    "tags": ["kalp", "kolye ucu", "gümüş"],
    "categories": ["Kolye", "Kolye Ucu"],
    "slug": "",
    "meta_description": "Zarif kalp kolye ucu.",
    "sustainability_note": "Siparişe özel dökülür, stok fazlası oluşmaz.",
}


@pytest.fixture(scope="module")
def cfg():
    return load_pricing_config()


@pytest.fixture(scope="module")
def templates():
    return load_listing_prompt_templates()


@pytest.fixture
def db(tmp_path: Path):
    d = Database(tmp_path / "c.db")
    yield d
    d.close()


@pytest.fixture
def rec(record_factory):
    return record_factory(
        file_id="JCAD-000000017",
        main_category="pendant",
        subcategory="heart",
        rich_description="ince kalp formlu kolye ucu",
        metal_weights=MetalWeights(silver_925_g=5.0, gold_14k_yellow_g=6.3),
        stone_summary=StoneSummary(
            status="stone",
            center_stone=StoneEntry(shape="round", size_mm="2.0", quantity=1,
                                    estimated_carat_each=0.03,
                                    estimated_total_carat=0.03,
                                    source="geometry", confidence=0.9),
            side_stones=[StoneEntry(shape="round", size_mm="1.0", quantity=8,
                                    estimated_carat_each=0.005,
                                    estimated_total_carat=0.04,
                                    source="geometry", confidence=0.9)],
            total_estimated_carat=0.07,
        ),
    )


# ---------------------------------------------------------------- facts

def test_facts_include_measured_data(rec, cfg):
    facts = build_facts(rec, "etsy", price_variants(rec, "etsy", cfg))
    assert "category: Pendant" in facts
    assert "20.0 x 22.0 x 8.0 mm" in facts
    assert "stone_seat_count: 9" in facts
    assert "silver_weight_g: 5.00" in facts


def test_facts_are_turkish_for_woo(rec, cfg):
    facts = build_facts(rec, "woocommerce", price_variants(rec, "woocommerce", cfg))
    assert "category: Kolye Ucu" in facts
    assert "satis_secenekleri:" in facts
    assert "8 Ayar Altın" in facts


def test_facts_never_contain_price(rec, cfg):
    """Fiyat modele verilmiyor — metinde fiyat geçmesini istemiyoruz,
    fiyat varyant satırlarından gelir."""
    for channel in ("etsy", "woocommerce"):
        facts = build_facts(rec, channel, price_variants(rec, channel, cfg))
        assert "1485" not in facts and "105.24" not in facts
        assert "28665" not in facts


# ---------------------------------------------------------------- üretim

def test_generate_etsy_listing(rec, cfg, templates):
    provider = FakeProvider(ETSY_PAYLOAD)
    result = generate_listing(rec, "etsy", provider, templates, cfg)
    assert result.language == "en"
    assert result.fields["title"].startswith("Sterling Silver")
    assert len(result.fields["tags"]) == 13
    assert result.fields["who_made"] == cfg.etsy_who_made
    assert result.fields["when_made"] == "made_to_order"
    assert [v.variant_key for v in result.variants] == ["silver_925"]
    assert result.prompt_version == "listing-v1"
    assert result.model == "test-model"


def test_generate_woo_listing_has_six_variants(rec, cfg, templates):
    provider = FakeProvider(WOO_PAYLOAD)
    result = generate_listing(rec, "woocommerce", provider, templates, cfg)
    assert result.language == "tr"
    assert len(result.variants) == 6
    assert result.fields["category_path"] == "Kolye > Kolye Ucu"


def test_woo_slug_generated_when_model_omits_it(rec, cfg, templates):
    provider = FakeProvider(WOO_PAYLOAD)
    result = generate_listing(rec, "woocommerce", provider, templates, cfg)
    assert result.fields["slug"] == "kalp-kolye-ucu"


def test_facts_reach_the_prompt(rec, cfg, templates):
    provider = FakeProvider(ETSY_PAYLOAD)
    generate_listing(rec, "etsy", provider, templates, cfg)
    _system, user = provider.calls[0]
    assert "20.0 x 22.0 x 8.0 mm" in user


def test_overlong_title_from_model_is_rejected(rec, cfg, templates):
    """Model 140 karakteri aşarsa ilan Etsy'de reddedilirdi — burada
    yakalıyoruz."""
    provider = FakeProvider({**ETSY_PAYLOAD, "title": "x" * 200})
    with pytest.raises(ListingGenerationError, match="şemaya uymadı"):
        generate_listing(rec, "etsy", provider, templates, cfg)


def test_too_many_tags_are_trimmed_not_rejected(rec, cfg, templates):
    provider = FakeProvider({**ETSY_PAYLOAD,
                             "tags": [f"tag {i}" for i in range(30)]})
    result = generate_listing(rec, "etsy", provider, templates, cfg)
    assert len(result.fields["tags"]) == 13


def test_unknown_channel_rejected(rec, cfg, templates):
    with pytest.raises(ListingGenerationError):
        generate_listing(rec, "amazon", FakeProvider(ETSY_PAYLOAD), templates, cfg)


def test_taxonomy_id_is_none_without_mapping(rec, cfg, templates):
    """Etsy taksonomi ID'si UYDURULMUYOR — eşleme dosyası yoksa boş kalır."""
    result = generate_listing(rec, "etsy", FakeProvider(ETSY_PAYLOAD), templates, cfg)
    assert result.fields["etsy_taxonomy_id"] is None


def test_missing_weight_still_generates_woo_stl_only(record_factory, cfg, templates):
    rec = record_factory(file_id="A", metal_weights=None,
                         weight_source="unavailable",
                         weight_confidence="unavailable")
    result = generate_listing(rec, "woocommerce", FakeProvider(WOO_PAYLOAD),
                              templates, cfg)
    assert [v.variant_key for v in result.variants] == ["stl_digital"]


# ---------------------------------------------------------------- kalıcılık

def test_persist_writes_listing_and_variants(db, rec, cfg, templates):
    result = generate_listing(rec, "woocommerce", FakeProvider(WOO_PAYLOAD),
                              templates, cfg)
    lid = persist(db.conn, result, cfg)
    listing = get_listing(db.conn, rec.file_id, "woocommerce")
    assert listing.listing_id == lid
    assert listing.status == "draft"
    assert len(listing.variants) == 6
    assert listing.generated_model == "test-model"
    assert listing.prompt_version == "listing-v1"


def test_reprice_updates_prices_without_touching_text(db, rec, cfg, templates):
    result = generate_listing(rec, "woocommerce", FakeProvider(WOO_PAYLOAD),
                              templates, cfg)
    persist(db.conn, result, cfg)
    before = get_listing(db.conn, rec.file_id, "woocommerce")

    import copy
    raised = copy.deepcopy(cfg.raw)
    raised["woocommerce"]["gold"]["pure_gold_gram_price"] = 9000
    from catalog_organizer.listing.pricing import PricingConfig
    new_cfg = PricingConfig(raised)

    assert reprice(db.conn, rec, "woocommerce", new_cfg) == 6
    after = get_listing(db.conn, rec.file_id, "woocommerce")
    assert after.title == before.title
    old = {v.variant_key: v.price for v in before.variants}
    new = {v.variant_key: v.price for v in after.variants}
    assert new["gold_14k"] > old["gold_14k"]
    assert new["silver_925"] == old["silver_925"]
    assert new["stl_digital"] == 50.0


def test_reprice_on_missing_listing_is_noop(db, rec, cfg):
    assert reprice(db.conn, rec, "etsy", cfg) == 0


# ---------------------------------------------------------------- sağlayıcı

def test_parse_json_tolerates_fenced_and_prose():
    assert _parse_json_or_raise('```json\n{"a": 1}\n```', "x") == {"a": 1}
    assert _parse_json_or_raise('Sure! {"a": 1} hope that helps', "x") == {"a": 1}


def test_parse_json_raises_on_garbage():
    with pytest.raises(TextProviderError, match="JSON döndürmedi"):
        _parse_json_or_raise("no json here", "x")


def test_create_text_provider_ollama_defaults():
    provider = create_text_provider({"provider": "ollama",
                                     "ollama": {"model": "qwen3.5:9b-q4_K_M"}})
    assert provider.model_name == "qwen3.5:9b-q4_K_M"
    assert "Ollama" in provider.display_name


def test_create_text_provider_rejects_minimax():
    """Denenmemiş uç noktaya sessizce istek göndermektense açıkça reddet."""
    with pytest.raises(ProviderConfigurationError, match="desteklenmiyor"):
        create_text_provider({"provider": "minimax"})


def test_create_text_provider_requires_key(monkeypatch):
    import catalog_organizer.vlm.text_provider as tp
    monkeypatch.setattr(tp, "get_api_key", lambda name: None)
    with pytest.raises(ProviderConfigurationError, match="anahtar yok"):
        create_text_provider({"provider": "glm"})


def test_create_text_provider_glm_with_key(monkeypatch):
    import catalog_organizer.vlm.text_provider as tp
    monkeypatch.setattr(tp, "get_api_key", lambda name: "test-key")
    provider = create_text_provider({"provider": "glm", "glm": {"model": "glm-4.5v"}})
    assert provider.model_name == "glm-4.5v"
    assert "GLM" in provider.display_name


def test_openai_compatible_needs_base_url(monkeypatch):
    import catalog_organizer.vlm.text_provider as tp
    monkeypatch.setattr(tp, "get_api_key", lambda name: "test-key")
    with pytest.raises(ProviderConfigurationError, match="Base URL"):
        create_text_provider({"provider": "openai_compatible",
                              "openai_compatible": {"base_url": ""}})
