from __future__ import annotations

import base64
from pathlib import Path

import yaml

from catalog_organizer.core.config import (
    load_brand_names,
    load_categories,
    load_tag_dictionary,
    load_vlm_prompt_templates,
)


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_messages(
    snapshot_paths: list[Path],
    second_pass: dict | None = None,
) -> list[dict]:
    """
    Build Ollama `/api/chat` messages list with system + user content,
    images attached as base64. If `second_pass` is given (with keys
    prev_category, prev_subcategory, prev_main_conf, prev_sub_conf),
    the addendum from the template is appended to the user message.
    """
    tpl = load_vlm_prompt_templates()
    categories = load_categories()
    tag_dictionary = load_tag_dictionary()
    brand_names = load_brand_names()

    taxonomy_yaml = yaml.safe_dump(categories, sort_keys=False)
    tag_dict_yaml = yaml.safe_dump({"tags": tag_dictionary}, sort_keys=False)
    brand_names_yaml = yaml.safe_dump({"brands": brand_names}, sort_keys=False)

    # NOTE: `.format()` cannot be used here — the template embeds a literal
    # JSON object as the expected output shape, and `{`/`}` would be parsed
    # as format placeholders. Use plain string replacement instead.
    user_text = (
        tpl["user_template"]
        .replace("{taxonomy_yaml}", taxonomy_yaml)
        .replace("{tag_dictionary_yaml}", tag_dict_yaml)
        .replace("{brand_names_yaml}", brand_names_yaml)
    )

    if second_pass:
        addendum = tpl["second_pass_addendum"]
        for key, val in second_pass.items():
            addendum = addendum.replace("{" + key + "}", str(val))
        user_text += "\n\n" + addendum

    images_b64 = [_encode_image(p) for p in snapshot_paths]

    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": user_text, "images": images_b64},
    ]
