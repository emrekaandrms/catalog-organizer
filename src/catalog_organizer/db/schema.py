"""SQLite şeması ve sürümleme.

Sınır kuralı: `products` JSONL'in aynasıdır ve her senkronda üzerine
yazılır. `product_overrides` ve aşağıdaki diğer tabloların sahibi DB'dir —
senkron onlara dokunmaz.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS products (
  file_id             TEXT PRIMARY KEY,
  source_path         TEXT NOT NULL,
  file_extension      TEXT NOT NULL,
  sha256              TEXT NOT NULL,
  main_category       TEXT,
  subcategory         TEXT,
  brand               TEXT,
  controlled_tags     TEXT,
  rich_description    TEXT,
  stone_status        TEXT,
  total_carat         REAL,
  stone_count         INTEGER,
  sprue_detected      INTEGER,
  sprue_volume_mm3    REAL,
  silver_925_g        REAL,
  gold_10k_yellow_g   REAL,
  gold_14k_yellow_g   REAL,
  gold_18k_yellow_g   REAL,
  platinum_g          REAL,
  bbox_width_mm       REAL,
  bbox_height_mm      REAL,
  bbox_depth_mm       REAL,
  volume_mm3          REAL,
  design_complete     INTEGER,
  sellability         TEXT,
  state               TEXT,
  needs_manual_review INTEGER,
  snapshot_dir        TEXT,
  processed_at        TEXT,
  synced_at           TEXT,
  record_json         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(main_category);
CREATE INDEX IF NOT EXISTS idx_products_subcat   ON products(subcategory);
CREATE INDEX IF NOT EXISTS idx_products_brand    ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_stone    ON products(stone_status);
CREATE INDEX IF NOT EXISTS idx_products_sellable ON products(sellability);
CREATE INDEX IF NOT EXISTS idx_products_complete ON products(design_complete);
CREATE INDEX IF NOT EXISTS idx_products_state    ON products(state);

CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
  file_id UNINDEXED,
  search_text
);

CREATE TABLE IF NOT EXISTS product_overrides (
  file_id    TEXT NOT NULL,
  field      TEXT NOT NULL,
  value      TEXT,
  note       TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (file_id, field)
);

CREATE TABLE IF NOT EXISTS selections (
  selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT NOT NULL UNIQUE,
  note         TEXT,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS selection_items (
  selection_id INTEGER NOT NULL
                 REFERENCES selections(selection_id) ON DELETE CASCADE,
  file_id      TEXT NOT NULL,
  position     INTEGER NOT NULL DEFAULT 0,
  added_at     TEXT NOT NULL,
  PRIMARY KEY (selection_id, file_id)
);

CREATE TABLE IF NOT EXISTS listings (
  listing_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id             TEXT NOT NULL,
  channel             TEXT NOT NULL CHECK (channel IN ('etsy','woocommerce')),
  selection_id        INTEGER REFERENCES selections(selection_id) ON DELETE SET NULL,
  language            TEXT NOT NULL,
  status              TEXT NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft','approved','exported')),
  title               TEXT,
  short_description   TEXT,
  description         TEXT,
  tags                TEXT,
  materials           TEXT,
  category_path       TEXT,
  etsy_taxonomy_id    INTEGER,
  slug                TEXT,
  meta_description    TEXT,
  who_made            TEXT,
  when_made           TEXT,
  is_supply           INTEGER DEFAULT 0,
  made_to_order       INTEGER DEFAULT 1,
  sustainability_note TEXT,
  generated_provider  TEXT,
  generated_model     TEXT,
  prompt_version      TEXT,
  generated_at        TEXT,
  human_edited        INTEGER DEFAULT 0,
  edited_at           TEXT,
  UNIQUE (file_id, channel)
);

CREATE TABLE IF NOT EXISTS listing_variants (
  variant_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  listing_id       INTEGER NOT NULL
                     REFERENCES listings(listing_id) ON DELETE CASCADE,
  variant_key      TEXT NOT NULL,
  sku              TEXT NOT NULL UNIQUE,
  price            REAL NOT NULL,
  currency         TEXT NOT NULL,
  weight_g         REAL,
  is_digital       INTEGER NOT NULL DEFAULT 0,
  pricing_snapshot TEXT NOT NULL,
  computed_at      TEXT NOT NULL,
  UNIQUE (listing_id, variant_key)
);

CREATE TABLE IF NOT EXISTS exports (
  export_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  channel       TEXT NOT NULL,
  selection_id  INTEGER,
  file_path     TEXT NOT NULL,
  row_count     INTEGER NOT NULL,
  skipped_count INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL);

-- v2: hand-set camera angles from the Render tab's manual-orbit dialog.
-- camera_axes() gets a ring's bore-vs-side call wrong often enough (see
-- render/product_render.py) that the user needs a way to override it per
-- piece, per view — this is that override, keyed the same way a render is
-- cached (file_id + view name).
CREATE TABLE IF NOT EXISTS camera_overrides (
  file_id    TEXT NOT NULL,
  view       TEXT NOT NULL CHECK (view IN ('front','iso')),
  dir_x      REAL NOT NULL,
  dir_y      REAL NOT NULL,
  dir_z      REAL NOT NULL,
  up_x       REAL NOT NULL,
  up_y       REAL NOT NULL,
  up_z       REAL NOT NULL,
  zoom       REAL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (file_id, view)
);

-- Choices the user makes by hand about how a piece is drawn. Today: whether stones are put in the
-- seats of a stoneless STL. No row = no choice: the product's own stone status decides.
CREATE TABLE IF NOT EXISTS render_options (
  file_id      TEXT PRIMARY KEY,
  place_stones INTEGER NOT NULL CHECK (place_stones IN (0, 1)),
  updated_at   TEXT NOT NULL
);
"""


def apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotent — uygulama her açılışta çağırır."""
    conn.executescript(_DDL)
    if conn.execute("SELECT COUNT(*) FROM schema_meta").fetchone()[0] == 0:
        conn.execute("INSERT INTO schema_meta (version) VALUES (?)", (SCHEMA_VERSION,))
    else:
        conn.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
    conn.commit()


def current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT version FROM schema_meta").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0
