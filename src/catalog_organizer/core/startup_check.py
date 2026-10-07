"""
Startup requirements checker.

Run once at app launch (before the main window appears) and once on demand
from the Settings panel. Produces a list of CheckResult items that the GUI
displays; critical failures block startup, warnings are surfaced in the
status bar.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import requests


# ── Result types ──────────────────────────────────────────────────────────────

class Status(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    hint: str = ""          # shown as a tooltip / "how to fix" note


@dataclass
class StartupReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status != Status.CRITICAL for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == Status.WARNING for r in self.results)

    def critical_messages(self) -> list[str]:
        return [r.message for r in self.results if r.status == Status.CRITICAL]


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 12):
        return CheckResult("Python version", Status.OK, f"Python {major}.{minor}")
    return CheckResult(
        "Python version", Status.CRITICAL,
        f"Python {major}.{minor} detected — 3.12+ required.",
        hint="Download Python 3.12 from https://www.python.org/downloads/",
    )


# Packages that MUST be importable for the app to function at all.
_CRITICAL_PACKAGES: list[tuple[str, str]] = [
    ("PyQt6", "PyQt6"),
    ("pyvista", "PyVista"),
    ("trimesh", "trimesh"),
    ("rhino3dm", "rhino3dm"),
    ("pydantic", "pydantic"),
    ("orjson", "orjson"),
    ("polars", "polars"),
    ("yaml", "PyYAML"),
    ("structlog", "structlog"),
    ("PIL", "Pillow"),
    ("requests", "requests"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
]

# Packages that are used lazily — warn if missing but don't block startup.
_OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    ("open3d", "open3d"),
]


def _check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for import_name, display_name in _CRITICAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            results.append(CheckResult(
                f"Package: {display_name}", Status.OK,
                f"{display_name} {version} installed",
            ))
        except ImportError:
            results.append(CheckResult(
                f"Package: {display_name}", Status.CRITICAL,
                f"{display_name} is not installed.",
                hint=f"Run: pip install {display_name}",
            ))
    for import_name, display_name in _OPTIONAL_PACKAGES:
        try:
            importlib.import_module(import_name)
            results.append(CheckResult(
                f"Package: {display_name}", Status.OK,
                f"{display_name} installed",
            ))
        except ImportError:
            results.append(CheckResult(
                f"Package: {display_name}", Status.WARNING,
                f"{display_name} not installed — some geometry features may be limited.",
                hint=f"Run: pip install {display_name}",
            ))
    return results


def _check_ollama(
    host: str = "127.0.0.1",
    port: int = 11434,
    model: str = "qwen3.5:9b-q4_K_M",
    timeout: float = 4.0,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    base_url = f"http://{host}:{port}"

    # 1. Is Ollama reachable?
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.ConnectionError:
        results.append(CheckResult(
            "Ollama service", Status.CRITICAL,
            f"Ollama is not reachable at {base_url}.",
            hint="Start Ollama: open a terminal and run `ollama serve`",
        ))
        return results
    except Exception as exc:
        results.append(CheckResult(
            "Ollama service", Status.CRITICAL,
            f"Ollama returned an unexpected error: {exc}",
        ))
        return results

    results.append(CheckResult("Ollama service", Status.OK, f"Ollama reachable at {base_url}"))

    # 2. Is the required model pulled?
    try:
        tags_data = resp.json()
        pulled_models: list[str] = [m.get("name", "") for m in tags_data.get("models", [])]
    except Exception:
        pulled_models = []

    model_found = any(
        m == model or m.startswith(model.split(":")[0])
        for m in pulled_models
    )
    if model_found:
        results.append(CheckResult(
            f"VLM model: {model}", Status.OK,
            f"Model '{model}' is available in Ollama",
        ))
    else:
        results.append(CheckResult(
            f"VLM model: {model}", Status.CRITICAL,
            f"Model '{model}' is NOT pulled in Ollama.",
            hint=f"Run: ollama pull {model}",
        ))
        return results

    # 3. Is the model vision-capable? Generate a fresh small PNG.
    # NOTE: must be ≥ 32×32 — Ollama's qwen3-vl image processor panics with
    # "height:H or width:W must be larger than factor:32" on smaller inputs.
    # Use 64×64 to leave headroom.
    import base64 as _b64
    import io as _io
    from PIL import Image as _PILImage
    _buf = _io.BytesIO()
    _PILImage.new("RGB", (64, 64), (255, 255, 255)).save(_buf, "PNG")
    _TINY_PNG_B64 = _b64.b64encode(_buf.getvalue()).decode("ascii")
    probe_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with the single word: OK",
                "images": [_TINY_PNG_B64],
            }
        ],
        "stream": False,
        "options": {"num_predict": 5, "temperature": 0},
    }
    try:
        probe_resp = requests.post(
            f"{base_url}/api/chat",
            json=probe_payload,
            timeout=30.0,
        )
        probe_data = probe_resp.json()
        if "error" in probe_data:
            err = probe_data["error"]
            if "multimodal" in err.lower() or "image" in err.lower() or "vision" in err.lower():
                results.append(CheckResult(
                    "VLM vision capability", Status.CRITICAL,
                    f"Model '{model}' does not support images (text-only model).",
                    hint="Switch to the project's locked VLM: 'qwen3.5:9b-q4_K_M' (fallback only: 'qwen3-vl:8b-thinking')",
                ))
            else:
                results.append(CheckResult(
                    "VLM vision capability", Status.WARNING,
                    f"Model probe returned an error: {err}",
                    hint="The model may still work; check Ollama logs for details.",
                ))
        else:
            results.append(CheckResult(
                "VLM vision capability", Status.OK,
                f"Model '{model}' accepted an image input successfully",
            ))
    except Exception as exc:
        results.append(CheckResult(
            "VLM vision capability", Status.WARNING,
            f"Could not probe vision capability: {exc}",
            hint="Ollama may be loading the model; retry after a few seconds.",
        ))

    return results


def _check_disk_space(paths: Optional[list[Path]] = None, min_gb: float = 10.0) -> list[CheckResult]:
    if paths is None:
        paths = [Path.home()]
    results: list[CheckResult] = []
    checked: set[str] = set()
    for path in paths:
        drive = Path(path).anchor or str(path)
        if drive in checked:
            continue
        checked.add(drive)
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= min_gb:
                results.append(CheckResult(
                    f"Disk space ({drive})", Status.OK,
                    f"{free_gb:.1f} GB free on {drive}",
                ))
            elif free_gb >= 2.0:
                results.append(CheckResult(
                    f"Disk space ({drive})", Status.WARNING,
                    f"Only {free_gb:.1f} GB free on {drive} — cache may fill quickly.",
                    hint=f"Free at least {min_gb:.0f} GB for comfortable operation.",
                ))
            else:
                results.append(CheckResult(
                    f"Disk space ({drive})", Status.CRITICAL,
                    f"Only {free_gb:.1f} GB free on {drive} — cannot safely proceed.",
                    hint="Free disk space before running the app.",
                ))
        except Exception as exc:
            results.append(CheckResult(
                f"Disk space ({drive})", Status.WARNING,
                f"Could not check disk usage: {exc}",
            ))
    return results


def _check_config_files(config_dir: Optional[Path] = None) -> list[CheckResult]:
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[3] / "config"

    required = [
        "app_settings.yaml",
        "pipeline_settings.yaml",
        "categories.yaml",
        "tag_dictionary.yaml",
        "metal_density_table.yaml",
        "vlm_prompt_templates.yaml",
    ]
    results: list[CheckResult] = []
    for fname in required:
        fpath = config_dir / fname
        if fpath.exists():
            results.append(CheckResult(
                f"Config: {fname}", Status.OK,
                f"{fname} found",
            ))
        else:
            results.append(CheckResult(
                f"Config: {fname}", Status.CRITICAL,
                f"Missing config file: {fpath}",
                hint="Run Session D.1 to generate all config files.",
            ))
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_checks(
    ollama_host: str = "127.0.0.1",
    ollama_port: int = 11434,
    vlm_model: str = "qwen3.5:9b-q4_K_M",
    config_dir: Optional[Path] = None,
    cache_paths: Optional[list[Path]] = None,
    min_disk_gb: float = 10.0,
    check_ollama: bool = True,
) -> StartupReport:
    """Run every launch-time requirement check.

    `check_ollama=False` skips the local-model probe entirely — set it
    when the configured VLM provider is a cloud backend (OpenAI, GLM,
    MiniMax, a custom OpenAI-compatible endpoint). Reporting "Ollama is
    not running" as a CRITICAL failure to someone who deliberately chose
    an API provider would block launch over a service they never intend
    to install.
    """
    report = StartupReport()

    report.results.append(_check_python_version())
    report.results.extend(_check_packages())
    if check_ollama:
        report.results.extend(_check_ollama(
            host=ollama_host,
            port=ollama_port,
            model=vlm_model,
        ))
    report.results.extend(_check_disk_space(
        paths=cache_paths,
        min_gb=min_disk_gb,
    ))
    report.results.extend(_check_config_files(config_dir=config_dir))

    return report
