from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from inif.models import InifDocument, Sample, Token


@dataclass
class TokenSelection:
    sample_id: str | int
    tokens: list[Token] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


def select_by_position(
    sample: Sample, positions: int | list[int] | slice
) -> TokenSelection:
    n = len(sample.tokens)
    if isinstance(positions, int):
        positions = [positions]
    elif isinstance(positions, slice):
        positions = list(range(*positions.indices(n)))
    pos_set = set(positions)
    tokens = [t for i, t in enumerate(sample.tokens) if i in pos_set]
    actual_positions = [i for i in range(n) if i in pos_set]
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=actual_positions,
    )


def select_by_tag(sample: Sample, tag: str) -> TokenSelection:
    tokens = []
    positions = []
    for i, t in enumerate(sample.tokens):
        token_tags = getattr(t, "tags", [])
        if tag in token_tags:
            tokens.append(t)
            positions.append(i)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=positions,
    )


def select_by_sequence_id(sample: Sample, seq_id: str) -> TokenSelection:
    tokens = []
    positions = []
    for i, t in enumerate(sample.tokens):
        if t.sequence_id == seq_id:
            tokens.append(t)
            positions.append(i)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=positions,
    )


def select_by_span(sample: Sample, span_name: str) -> TokenSelection:
    position_set: set[int] = set()
    for span in sample.spans:
        if span.name == span_name:
            position_set.update(span.positions)
    tokens = [t for i, t in enumerate(sample.tokens) if i in position_set]
    actual_positions = sorted(i for i in range(len(sample.tokens)) if i in position_set)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=actual_positions,
    )


def select_by_score(
    doc: InifDocument,
    scorer: str,
    predicate: Callable[[str | int | float | bool | list | dict], bool],
) -> list[TokenSelection]:
    results = []
    for sample in doc.samples:
        for score in sample.scores:
            if score.scorer == scorer and predicate(score.value):
                results.append(
                    TokenSelection(
                        sample_id=sample.id,
                        tokens=list(sample.tokens),
                        positions=list(range(len(sample.tokens))),
                    )
                )
                break
    return results
