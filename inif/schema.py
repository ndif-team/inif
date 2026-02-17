from __future__ import annotations

import json
from pathlib import Path

from inif.models import InifDocument


def get_schema() -> dict:
    return InifDocument.model_json_schema()


def validate(data: dict) -> None:
    InifDocument.model_validate(data)


def write_schema(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(get_schema(), f, indent=2)
        f.write("\n")
