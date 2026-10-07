from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from catalog_organizer.core.paths import config_dir


class ConfigError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _load_yaml(name: str) -> dict[str, Any]:
    path = config_dir() / name
    if not path.exists():
        raise ConfigError("config.missing", f"Missing config file: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError("config.invalid", f"{name} must be a YAML mapping")
        return data
    except yaml.YAMLError as exc:
        raise ConfigError("config.invalid", f"YAML parse error in {name}: {exc}") from exc


def load_app_settings() -> dict[str, Any]:
    return _load_yaml("app_settings.yaml")


def load_pipeline_settings() -> dict[str, Any]:
    return _load_yaml("pipeline_settings.yaml")


def load_categories() -> dict[str, Any]:
    return _load_yaml("categories.yaml")


def load_tag_dictionary() -> list[str]:
    data = _load_yaml("tag_dictionary.yaml")
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ConfigError("config.invalid", "tag_dictionary.yaml must have a 'tags' list")
    return [str(t) for t in tags]


def load_fallback_tags() -> dict[str, list[str]]:
    data = _load_yaml("tag_dictionary.yaml")
    fb = data.get("fallback_tags", {})
    if not isinstance(fb, dict):
        raise ConfigError("config.invalid", "tag_dictionary.yaml fallback_tags must be a mapping")
    return {k: [str(t) for t in v] for k, v in fb.items()}


def load_metal_densities() -> dict[str, dict[str, Any]]:
    data = _load_yaml("metal_density_table.yaml")
    metals = data.get("metals", {})
    if not isinstance(metals, dict):
        raise ConfigError("config.invalid", "metal_density_table.yaml must have a 'metals' mapping")
    return metals


def load_vlm_prompt_templates() -> dict[str, str]:
    return _load_yaml("vlm_prompt_templates.yaml")


def load_brand_names() -> list[str]:
    data = _load_yaml("brand_names.yaml")
    brands = data.get("brands", [])
    if not isinstance(brands, list):
        raise ConfigError("config.invalid", "brand_names.yaml must have a 'brands' list")
    return [str(b) for b in brands]


def load_stone_shape_factors() -> dict[str, float]:
    path = config_dir() / "stone_weight_tables" / "_shape_factors.yaml"
    if not path.exists():
        raise ConfigError("config.missing", f"Missing: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return {k: float(v) for k, v in data.get("shape_factors", {}).items()}
    except yaml.YAMLError as exc:
        raise ConfigError("config.invalid", f"YAML parse error in _shape_factors.yaml: {exc}") from exc


def load_pricing_settings() -> dict[str, Any]:
    """Fiyat sabitleri (config/pricing.yaml). Etsy/Woo formül katsayıları ve
    metal kurları burada — koda gömülmez, kur değişince tek yerden güncellenir."""
    data = _load_yaml("pricing.yaml")
    for key in ("etsy", "woocommerce"):
        if key not in data:
            raise ConfigError("config.invalid", f"pricing.yaml must have a '{key}' block")
    return data


def load_listing_prompt_templates() -> dict[str, Any]:
    return _load_yaml("listing_prompt_templates.yaml")
