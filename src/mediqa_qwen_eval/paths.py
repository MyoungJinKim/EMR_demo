from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_configs() -> dict[str, dict[str, Any]]:
    return load_json(CONFIG_DIR / "models.json")


def data_manifest() -> dict[str, Any]:
    return load_json(CONFIG_DIR / "data_manifest.json")
