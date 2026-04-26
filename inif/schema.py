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
    InifDocument.model_validate(data)


def write_schema(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(get_schema(), f, indent=2)
        f.write("\n")
