"""Tag validation per §C.4: strip invalid, apply deterministic fallback, ensure >=5."""
from __future__ import annotations

from catalog_organizer.core.config import load_fallback_tags, load_tag_dictionary

_MIN_TAGS = 5


class TagValidator:
    def __init__(
        self,
        dictionary: list[str] | None = None,
        fallbacks: dict[str, list[str]] | None = None,
    ) -> None:
        self._dictionary: set[str] = set(dictionary if dictionary is not None else load_tag_dictionary())
        self._fallbacks: dict[str, list[str]] = (
            fallbacks if fallbacks is not None else load_fallback_tags()
        )

    def validate(
        self,
        tags: list[str],
        main_category: str,
        stone_presence: str = "no_stone",
        estimated_visible_stone_count: int = 0,
    ) -> list[str]:
        # 1–2. Keep only dictionary-valid tags, preserve order, dedupe.
        seen: set[str] = set()
        kept: list[str] = []
        for t in tags:
            if t in self._dictionary and t not in seen:
                kept.append(t)
                seen.add(t)

        # 3–4. Fill from fallback list if under threshold.
        if len(kept) < _MIN_TAGS:
            fallback = list(self._fallbacks.get(main_category, self._fallbacks.get("unknown", [])))
            # §C.4: if stone_presence == "stone", replace last fallback
            # with "stone" (or "center_stone" if visible count == 1).
            if stone_presence == "stone" and fallback:
                replacement = "center_stone" if estimated_visible_stone_count == 1 else "stone"
                if replacement in self._dictionary:
                    fallback[-1] = replacement

            for t in fallback:
                if len(kept) >= _MIN_TAGS:
                    break
                if t in self._dictionary and t not in seen:
                    kept.append(t)
                    seen.add(t)

        return kept
