from __future__ import annotations

import gzip
import json
from pathlib import Path

from inif.models import InifDocument

# A path is considered gzipped (and gets gzip read/write) when its last suffix
# is one of these. ``.inif`` is the convention for a fully-compressed bundle.
_GZIP_SUFFIXES = frozenset({".gz", ".inif"})


def _is_gzipped_path(path: Path) -> bool:
    return path.suffix in _GZIP_SUFFIXES


def to_dict(doc: InifDocument, compact: bool = True) -> dict:
    """Convert ``doc`` to a JSON-ready dict.

    Uses pydantic's ``mode="json"`` so datetimes serialize as ISO-8601 strings.
    """
    if compact:
        return doc.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    return doc.model_dump(mode="json")


def from_dict(data: dict) -> InifDocument:
    return InifDocument.model_validate(data)


def save(
    doc: InifDocument,
    path: str | Path,
    compress: bool | None = None,
    compact: bool = True,
) -> None:
    """Save ``doc`` to ``path``.

    ``compress`` controls gzip compression: ``True`` forces gzip, ``False``
    forces plain text, ``None`` (default) infers from the path suffix —
    gzipped iff the suffix is ``.gz`` or ``.inif``.
    """
    path = Path(path)
    data = to_dict(doc, compact=compact)
    json_str = json.dumps(data, ensure_ascii=False)

    use_gzip = compress if compress is not None else _is_gzipped_path(path)
    if use_gzip:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)


def load(path: str | Path, compress: bool | None = None) -> InifDocument:
    """Load an inif document from ``path``.

    ``compress`` mirrors :func:`save`: ``True``/``False`` force gzip on or off,
    ``None`` (default) infers from the path suffix using the same rule.
    """
    path = Path(path)
    use_gzip = compress if compress is not None else _is_gzipped_path(path)
    if use_gzip:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return from_dict(data)
