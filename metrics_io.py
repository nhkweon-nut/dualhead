# JSON helpers for eval_viz (not hivt/utils.py — that defines TemporalData).
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
