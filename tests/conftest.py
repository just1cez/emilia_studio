"""Root pytest configuration for Emilia Studio."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP = ROOT / "app"
if (APP / "comfy").is_dir() and str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def pytest_ignore_collect(collection_path: Path) -> bool:
    """Skip comfy_core runtime tests when ComfyUI app/ or torch is absent."""
    if "comfy_core" in collection_path.parts:
        if not (APP / "comfy").is_dir() or importlib.util.find_spec("torch") is None:
            return True
    return False
