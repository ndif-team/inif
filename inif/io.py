from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inif.models import InifDocument, Metadata, Sample

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

    Uses pydantic's ``mode="json"`` so datetimes serialize as ISO-8601
    strings. With ``compact=True`` (default), default-valued and ``None``
    fields are stripped — for example, sequence-ref tokens (``id is None``)
    serialize to a single-key ``{"token": "<seq_id>"}`` dict.
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
    """A list-of-token dicts: every item carries a ``token`` string.

    Distinguishes token lists from spans / annotations / texts (which use
    ``name``) and other lists. Sequence refs serialize without ``id`` (it's
    ``None`` and stripped as a default), so ``"token"`` is the discriminator.
    """
    if not value:
        return False
    return all(
        isinstance(v, dict)
        and "token" in v
        and ("id" not in v or v["id"] is None or isinstance(v["id"], int))
        for v in value
    )


_TOKEN_CORE_KEYS = {"id", "token"}


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
    without extras (only ``id`` / ``token``) render on a single line, token
    dicts with extras render multi-line like any other dict.
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


# ---------------------------------------------------------------------------
# Unified read API
#
# These helpers abstract over the on-disk format: passing a ``.inif`` path
# uses the indexed-archive readers (random access, header-only reads),
# while a ``.inif.json`` path falls back to a full load and operates over
# the in-memory document. Same return shapes either way — callers don't
# need to branch on suffix.
# ---------------------------------------------------------------------------


class DocumentInfo(BaseModel):
    """Header-only view of an INIF document — metadata plus per-sample summaries.

    Returned by :func:`read_info`. Carries the document's :class:`Metadata`
    and a compact summary dict per sample (``id``, ``n_tokens``, ``n_texts``,
    ``n_spans``, ``scores``, ``text_preview``, plus optional ``target`` /
    ``error`` / ``input_tokens`` / ``output_tokens`` when present).

    The full sequence list and full token streams are NOT included, so this
    is cheap to inspect even for very large documents.
    """

    metadata: Metadata
    samples: list[dict[str, Any]] = Field(default_factory=list)


def iter_samples(path: str | Path) -> Iterator[Sample]:
    """Stream :class:`Sample` objects from ``path`` (any supported format).

    For ``.inif`` archives, samples are inflated lazily in manifest order.
    For ``.inif.json`` files, the full document is parsed once and its
    samples are then iterated. Either way the iterator yields the same
    :class:`Sample` objects.

    Args:
        path: Path to a ``.inif`` indexed archive or a ``.inif.json`` file.

    Yields:
        One :class:`Sample` at a time, in document order.
    """
    path = Path(path)
    _assert_json_or_indexed(path, None)
    if _is_indexed_path(path):
        from inif.indexed import iter_indexed_samples

        yield from iter_indexed_samples(path)
        return
    doc = load(path)
    yield from doc.samples


def read_samples(
    path: str | Path,
    sample_ids: str | int | Iterable[str | int],
) -> list[Sample]:
    """Read a selection of samples from ``path`` by id.

    Works on both ``.inif`` archives (random access) and ``.inif.json`` files
    (the full document is loaded then filtered). Passing a single id as a
    bare string / int is supported and is equivalent to passing ``[id]``;
    the return type is always a list.

    Args:
        path: Path to a ``.inif`` or ``.inif.json`` file.
        sample_ids: A single sample id or an iterable of ids. Ids must be
            unique within the call.

    Returns:
        Samples in the requested order.

    Raises:
        KeyError: If any requested id is not present in the file.
    """
    if isinstance(sample_ids, (str, int)):
        sample_ids = [sample_ids]
    requested = [str(sid) for sid in sample_ids]
    assert len(requested) == len(set(requested)), "Requested sample ids must be unique"
    path = Path(path)
    _assert_json_or_indexed(path, None)
    if _is_indexed_path(path):
        from inif.indexed import read_indexed_samples

        return read_indexed_samples(path, requested)

    doc = load(path)
    by_id = {s.id: s for s in doc.samples}
    missing = [sid for sid in requested if sid not in by_id]
    if missing:
        raise KeyError(f"Samples not found: {missing!r}")
    return [by_id[sid] for sid in requested]


def read_info(path: str | Path) -> DocumentInfo:
    """Read a header view of ``path`` — metadata plus per-sample summaries.

    Works on both ``.inif`` archives (only the manifest is parsed) and
    ``.inif.json`` files (the full document is loaded but only the summary
    dicts are materialised). The summary shape matches what the indexed
    archive stores in its manifest, so callers can rely on the same fields
    regardless of the file's on-disk format.

    Args:
        path: Path to a ``.inif`` or ``.inif.json`` file.

    Returns:
        A :class:`DocumentInfo` with :attr:`Metadata` and a list of summary
        dicts (one per sample). Sequences are not included.
    """
    path = Path(path)
    _assert_json_or_indexed(path, None)
    if _is_indexed_path(path):
        from inif.indexed import (
            read_indexed_header,
            read_indexed_sample_summaries,
        )

        header = read_indexed_header(path)
        summaries = read_indexed_sample_summaries(path)
    else:
        from inif.indexed import _DEFAULT_TEXT_PREVIEW_CHARS, _sample_summary

        doc = load(path)
        header = doc
        summaries = [
            _sample_summary(s, _DEFAULT_TEXT_PREVIEW_CHARS) for s in doc.samples
        ]
    # Drop archive-only bookkeeping fields so the returned summary shape is
    # identical between formats.
    for s in summaries:
        s.pop("path", None)
        s.pop("index", None)
    return DocumentInfo(metadata=header.metadata, samples=summaries)
