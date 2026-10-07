from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from catalog_organizer.app import _select_pilot_entries
from catalog_organizer.core.schemas import ManifestEntry

_NOW = datetime.now(tz=timezone.utc)


def _entry(source_path: str, state: str = "new") -> ManifestEntry:
    return ManifestEntry(
        file_id=f"JCAD-{abs(hash(source_path)) % 10**9:09d}",
        source_path=source_path,
        file_extension=".3dm",
        file_size_bytes=1024,
        created_at=_NOW,
        modified_at=_NOW,
        sha256="a" * 64,
        duplicate_of=None,
        state=state,
        processing_batch="pilot",
        scan_timestamp=_NOW,
    )


def test_select_pilot_entries_only_returns_entries_under_the_given_roots(tmp_path: Path):
    """Regression (2026-07-24): scan() returns the FULL cross-session
    manifest, not just this call's roots. A naive list(manifest.values())
    [:n] grabbed unrelated leftover entries from an old root and produced
    success=0/fail=0 with no error on a real pilot run against a fresh
    folder. _select_pilot_entries must restrict to `roots`."""
    old_root = tmp_path / "old_customer_folder"
    new_root = tmp_path / "PilotBatch"

    manifest = {
        str(old_root / "a.3dm"): _entry(str(old_root / "a.3dm"), state="missing"),
        str(old_root / "b.3dm"): _entry(str(old_root / "b.3dm"), state="new"),
        str(new_root / "c.3dm"): _entry(str(new_root / "c.3dm"), state="new"),
        str(new_root / "d.3dm"): _entry(str(new_root / "d.3dm"), state="new"),
    }

    selected = _select_pilot_entries(manifest, [new_root], n=10)

    assert {e.source_path for e in selected} == {
        str(new_root / "c.3dm"), str(new_root / "d.3dm"),
    }


def _write_config(tmp_path: Path, monkeypatch, vlm_block: dict) -> None:
    """Point app.py's config reader at an isolated pipeline_settings.yaml."""
    import yaml

    from catalog_organizer import app as app_mod

    cfg_root = tmp_path / "fake_pkg" / "src" / "catalog_organizer"
    (cfg_root.parent.parent / "config").mkdir(parents=True, exist_ok=True)
    (cfg_root.parent.parent / "config" / "pipeline_settings.yaml").write_text(
        yaml.dump({"vlm": vlm_block}), encoding="utf-8",
    )
    monkeypatch.setattr(app_mod, "__file__", str(cfg_root / "app.py"))


def test_read_vlm_settings_uses_nested_provider_block(tmp_path: Path, monkeypatch):
    """Regression (2026-07-28): the reader looked up `vlm.model`, a key the
    current config never contains (the real path is `vlm.ollama.model`).
    It therefore always returned the hard-coded default, so choosing a
    different model — or a remote Ollama host — in Settings had no effect
    on the launch check: the dialog validated a model the app would never
    call, and probed 127.0.0.1 no matter what was configured.
    """
    from catalog_organizer.app import _read_vlm_settings

    _write_config(tmp_path, monkeypatch, {
        "provider": "ollama",
        "ollama": {"host": "10.0.0.9", "port": 12345, "model": "custom-vlm:latest"},
    })
    got = _read_vlm_settings()
    assert got["model"] == "custom-vlm:latest"
    assert got["host"] == "10.0.0.9"
    assert got["port"] == 12345
    assert got["provider"] == "ollama"


def test_read_vlm_settings_reports_cloud_provider(tmp_path: Path, monkeypatch):
    """Used to decide whether to probe Ollama at all — a user on GLM must
    not be blocked at launch by a local service they never installed."""
    from catalog_organizer.app import _read_vlm_settings

    _write_config(tmp_path, monkeypatch, {"provider": "glm", "glm": {"model": "glm-4.5v"}})
    assert _read_vlm_settings()["provider"] == "glm"


def test_read_vlm_settings_honours_legacy_flat_config(tmp_path: Path, monkeypatch):
    """Pre-2026-05-21 configs put host/port/model directly under `vlm:`."""
    from catalog_organizer.app import _read_vlm_settings

    _write_config(tmp_path, monkeypatch, {
        "host": "192.168.1.5", "port": 9999, "model": "legacy:tag",
    })
    got = _read_vlm_settings()
    assert (got["host"], got["port"], got["model"]) == ("192.168.1.5", 9999, "legacy:tag")


def test_read_vlm_settings_falls_back_when_config_missing(tmp_path: Path, monkeypatch):
    from catalog_organizer import app as app_mod
    from catalog_organizer.app import _read_vlm_settings

    monkeypatch.setattr(app_mod, "__file__", str(tmp_path / "nope" / "a" / "b" / "app.py"))
    got = _read_vlm_settings()
    assert got["provider"] == "ollama"
    assert got["model"] == "qwen3.5:9b-q4_K_M"


def test_startup_check_can_skip_ollama_probe():
    """`check_ollama=False` must remove the Ollama rows entirely, not just
    downgrade them — otherwise a cloud-provider user still sees a red
    'Ollama not running' failure."""
    from catalog_organizer.core.startup_check import run_all_checks

    with_ollama = run_all_checks(check_ollama=True)
    without = run_all_checks(check_ollama=False)
    names_with = [r.name for r in with_ollama.results]
    names_without = [r.name for r in without.results]
    assert any("ollama" in n.lower() for n in names_with)
    assert not any("ollama" in n.lower() for n in names_without)


def test_select_pilot_entries_respects_n_limit(tmp_path: Path):
    root = tmp_path / "PilotBatch"
    manifest = {
        str(root / f"{i}.3dm"): _entry(str(root / f"{i}.3dm"))
        for i in range(5)
    }
    selected = _select_pilot_entries(manifest, [root], n=2)
    assert len(selected) == 2
