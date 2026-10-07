from __future__ import annotations

from catalog_organizer.core.schemas import MetalWeights


def compute_metal_weights(volume_mm3: float, densities: dict) -> MetalWeights:
    """
    Convert volume → mass for each tracked metal/alloy variant.

    1 mm³ = 1e-3 cm³; mass [g] = volume_cm3 × density_g_cm3.

    `densities` is the dict returned by `load_metal_densities()`. Yellow
    and white gold are stored as separate keys per metal_density_table.yaml
    because their densities differ by ~5%.
    """
    vol_cm3 = max(0.0, float(volume_mm3)) / 1000.0

    def d(key: str) -> float:
        if key not in densities:
            raise KeyError(f"missing density for metal '{key}' in densities config")
        return float(densities[key]["density_g_cm3"])

    return MetalWeights(
        silver_925_g=vol_cm3 * d("silver_925"),
        gold_10k_yellow_g=vol_cm3 * d("gold_10k_yellow"),
        gold_10k_white_g=vol_cm3 * d("gold_10k_white"),
        gold_14k_yellow_g=vol_cm3 * d("gold_14k_yellow"),
        gold_14k_white_g=vol_cm3 * d("gold_14k_white"),
        gold_18k_yellow_g=vol_cm3 * d("gold_18k_yellow"),
        gold_18k_white_g=vol_cm3 * d("gold_18k_white"),
        platinum_g=vol_cm3 * d("platinum_950"),
    )
