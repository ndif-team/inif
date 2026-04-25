from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from inif.models import InifDocument

# A path is considered gzipped (and gets gzip read/write) when its last suffix
# is one of these. ``.inif`` is the convention for a fully-compressed bundle.
_GZIP_SUFFIXES = frozenset({".gz", ".inif"})


def _is_gzipped_path(path: Path) -> bool:
    return path.suffix in _GZIP_SUFFIXES


def to_dict(doc: InifDocument, compact: bool = True) -> dict:
    """Convert ``doc`` to a JSON-ready dict.

    Uses pydantic's ``mode="json"`` so datetimes serialize as ISO-8601 strings,
    and ``by_alias=True`` so ``Token.sequence_id`` is emitted as ``"seq_id"``.
    """
    if compact:
        return doc.model_dump(
            mode="json", exclude_none=True, exclude_defaults=True, by_alias=True
        )
    return doc.model_dump(mode="json", by_alias=True)


def from_dict(data: dict) -> InifDocument:
    return InifDocument.model_validate(data)


def _is_token_list(value: list) -> bool:
    """A list-of-Token-dicts: each item is a dict with an integer ``id``."""
    if not value:
        return False
    return all(
        isinstance(v, dict) and "id" in v and isinstance(v["id"], int) for v in value
    )


def _format_token_list(tokens: list[dict], pad: str) -> list[str]:
    """Render each token dict on a single line with the ``"id"`` column aligned.

    The left side of each line contains every key except ``id`` (as a
    comma-terminated JSON fragment); the right side is ``"id": N}``. Left
    fragments are padded so the ``"id"`` keys line up across the list.
    """
    lefts: list[str] = []
    rights: list[str] = []
    for tok in tokens:
        non_id = [(k, v) for k, v in tok.items() if k != "id"]
        left = ", ".join(
            f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
            for k, v in non_id
        )
        if non_id:
            left += ","
        lefts.append(left)
        rights.append(f'"id": {json.dumps(tok["id"])}')

    max_left = max(len(left) for left in lefts)
    lines: list[str] = []
    for left, right in zip(lefts, rights):
        if max_left > 0:
            padded = left.ljust(max_left)
            lines.append(f"{pad}{{{padded} {right}}}")
        else:
            lines.append(f"{pad}{{{right}}}")
    return lines


def _dumps_pretty(value: Any, indent: int = 4, level: int = 0) -> str:
    """Custom JSON encoder: ``indent`` spaces for containers, token lists
    rendered compactly one-per-line with aligned columns.
    """
    pad = " " * (indent * level)
    inner_pad = " " * (indent * (level + 1))

    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [
            f"{inner_pad}{json.dumps(k, ensure_ascii=False)}: "
            f"{_dumps_pretty(v, indent, level + 1)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

    if isinstance(value, list):
        if not value:
            return "[]"
        if _is_token_list(value):
            lines = _format_token_list(value, inner_pad)
            return "[\n" + ",\n".join(lines) + "\n" + pad + "]"
        parts = [f"{inner_pad}{_dumps_pretty(v, indent, level + 1)}" for v in value]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"

    return json.dumps(value, ensure_ascii=False)


def save(
    doc: InifDocument,
    path: str | Path,
    compress: bool | None = None,
    compact: bool = True,
    indent: int | None = 4,
) -> None:
    """Save ``doc`` to ``path``.

    ``compress`` controls gzip compression: ``True`` forces gzip, ``False``
    forces plain text, ``None`` (default) infers from the path suffix —
    gzipped iff the suffix is ``.gz`` or ``.inif``.

    ``indent`` controls pretty-printing: a positive int uses the custom inif
    formatter (tokens rendered one-per-line with aligned ``"id"`` columns);
    ``None`` produces a single-line dump via ``json.dumps``.
    """
    path = Path(path)
    data = to_dict(doc, compact=compact)
    if indent is None or indent <= 0:
        json_str = json.dumps(data, ensure_ascii=False)
    else:
        json_str = _dumps_pretty(data, indent=indent)

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
