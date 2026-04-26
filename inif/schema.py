from __future__ import annotations

import json
from pathlib import Path

from inif.models import InifDocument, TokenExtras


def get_schema() -> dict:
    """Return the inif JSON schema, with conventional Token extras documented.

    The ``Token`` model permits arbitrary extra fields. ``TokenExtras`` is
    embedded under ``$defs`` purely as documentation of well-known names
    (``logprob``, ``logit_lens``) so external validators, UIs, and viewers know
    what to expect. Repeated token labels live in ``Sample.annotations``.
    """
    schema = InifDocument.model_json_schema()
    defs = schema.setdefault("$defs", {})
    defs["TokenExtras"] = TokenExtras.model_json_schema()
    return schema


def validate(data: dict) -> None:
    """Validate a JSON-shaped dict against the INIF schema.

    Runs the same Pydantic validators as :func:`load` but discards the
    parsed object. Raises ``pydantic.ValidationError`` on failure.

    Args:
        data: The dict to validate.
    """
    InifDocument.model_validate(data)


def write_schema(path: str | Path) -> None:
    """Write the INIF JSON schema to ``path``.

    Parent directories are created if they don't exist. The file is written
    with two-space indentation and a trailing newline.

    Args:
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(get_schema(), f, indent=2)
        f.write("\n")
