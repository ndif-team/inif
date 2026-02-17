from __future__ import annotations

import gzip
import json
from pathlib import Path

from inif.models import InifDocument


def to_dict(doc: InifDocument, compact: bool = True) -> dict:
    if compact:
        return doc.model_dump(exclude_none=True, exclude_defaults=True)
    return doc.model_dump()


def from_dict(data: dict) -> InifDocument:
    return InifDocument.model_validate(data)


def save(
    doc: InifDocument,
    path: str | Path,
    compress: bool | None = None,
    compact: bool = True,
) -> None:
    path = Path(path)
    data = to_dict(doc, compact=compact)
    json_str = json.dumps(data, ensure_ascii=False)

    suffixes = (".gz", ".inif")
    use_gzip = compress if compress is not None else str(path).endswith(suffixes)
    if use_gzip:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)


def load(path: str | Path) -> InifDocument:
    path = Path(path)
    if str(path).endswith((".gz", ".inif")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return from_dict(data)
