from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inif.models import InifDocument

_INDEXED_SUFFIXES = frozenset({".inif"})
_UNSUPPORTED_COMPRESSED_SUFFIXES = frozenset({".gz"})
_UNSUPPORTED_INDEXED_SUFFIXES = frozenset({".inifx"})


def _is_indexed_path(path: Path) -> bool:
    return path.suffix in _INDEXED_SUFFIXES


def _assert_json_or_indexed(path: Path, compress: bool | None) -> None:
    assert compress is None, (
        "`compress` is no longer supported. Use `.inif.json` for plain JSON "
        "or `.inif` for the indexed compressed archive."
    )
    assert path.suffix not in _UNSUPPORTED_COMPRESSED_SUFFIXES, (
        "gzip `.gz` INIF files are no longer supported. Use `.inif.json` for "
        "plain JSON or `.inif` for the indexed compressed archive."
    )
    assert path.suffix not in _UNSUPPORTED_INDEXED_SUFFIXES, (
        "`.inifx` is no longer supported. Use `.inif` for the indexed "
        "compressed archive."
    )


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
    """Build an :class:`InifDocument` from a JSON-shaped dict.

    The inverse of :func:`to_dict`. Runs the full Pydantic validation pass —
    out-of-range annotation ranges, missing required fields, and Token
    sentinel violations all raise here.

    Args:
        data: A dict matching the INIF JSON schema.

    Returns:
        The parsed :class:`InifDocument`.
    """
    return InifDocument.model_validate(data)


def _is_token_list(value: list) -> bool:
    """A list-of-Token-dicts: each item is a dict with an integer ``id``."""
    if not value:
        return False
    return all(
        isinstance(v, dict) and "id" in v and isinstance(v["id"], int) for v in value
    )


_TOKEN_CORE_KEYS = {"id", "token", "sequence_id"}


def _is_token_without_extras(tok: dict) -> bool:
    return all(k in _TOKEN_CORE_KEYS for k in tok.keys())


def _format_token_inline(tok: dict) -> str:
    parts = [
        f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
        for k, v in tok.items()
    ]
    return "{" + ", ".join(parts) + "}"


def _dumps_pretty(value: Any, indent: int = 4, level: int = 0) -> str:
    """Custom JSON encoder: ``indent`` spaces for containers; token dicts
    without extras (only ``id`` / ``token`` / ``sequence_id``) render on a
    single line, token dicts with extras render multi-line like any other dict.
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
            lines = []
            for tok in value:
                if _is_token_without_extras(tok):
                    lines.append(f"{inner_pad}{_format_token_inline(tok)}")
                else:
                    lines.append(f"{inner_pad}{_dumps_pretty(tok, indent, level + 1)}")
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

    ``.inif`` paths are written as indexed compressed archives. JSON paths are
    written as plain JSON. The old gzip ``compress`` override is no longer
    supported.

    ``indent`` controls pretty-printing: a positive int uses the custom inif
    formatter (tokens rendered one-per-line with aligned ``"id"`` columns);
    ``None`` produces a single-line dump via ``json.dumps``.
    """
    path = Path(path)
    _assert_json_or_indexed(path, compress)
    if _is_indexed_path(path):
        from inif.indexed import save_indexed

        save_indexed(doc, path, compact=compact)
        return

    data = to_dict(doc, compact=compact)
    if indent is None or indent <= 0:
        json_str = json.dumps(data, ensure_ascii=False)
    else:
        json_str = _dumps_pretty(data, indent=indent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load(path: str | Path, compress: bool | None = None) -> InifDocument:
    """Load an inif document from ``path``.

    ``.inif`` paths are read as indexed archives. JSON paths are read as plain
    JSON. The old gzip ``compress`` override is no longer supported.
    """
    path = Path(path)
    _assert_json_or_indexed(path, compress)
    if _is_indexed_path(path):
        from inif.indexed import load_indexed

        return load_indexed(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data)
