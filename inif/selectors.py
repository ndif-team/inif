from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from inif.models import InifDocument, Sample, TokenOrSeqRef


@dataclass
class TokenSelection:
    sample_id: str | int
    tokens: list[TokenOrSeqRef] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


def _select_by_position(
    sample: Sample, positions: int | list[int] | slice
) -> TokenSelection:
    """Implementation backing :meth:`Sample.select_by_position`."""
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


def _select_by_annotation(sample: Sample, annotation_name: str) -> TokenSelection:
    """Implementation backing :meth:`Sample.select_by_annotation`."""
    positions = sample.annotation_positions(annotation_name)
    tokens = sample.get_tokens_by_positions(positions)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=positions,
    )


def _select_by_sequence_id(sample: Sample, seq_id: str) -> TokenSelection:
    """Implementation backing :meth:`Sample.select_by_sequence_id`.

    A sequence ref is identified by ``id is None`` and carries the target
    :class:`Sequence` id in its ``token`` field. Useful for finding *where*
    a shared run is referenced in a sample without expanding it.
    """
    tokens = []
    positions = []
    for i, t in enumerate(sample.tokens):
        if t.is_sequence_ref and t.token == seq_id:
            tokens.append(t)
            positions.append(i)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=positions,
    )


def _select_by_span(sample: Sample, span_name: str) -> TokenSelection:
    """Implementation backing :meth:`Sample.select_by_span`."""
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


def _filter_samples_by_score(
    doc: InifDocument,
    scorer: str,
    predicate: Callable[[str | int | float | bool | list | dict], bool],
) -> list[Sample]:
    """Implementation backing :meth:`InifDocument.filter_samples_by_score`.

    Returns samples whose ``scorer`` value satisfies ``predicate``.
    """
    results: list[Sample] = []
    for sample in doc.samples:
        for score in sample.scores:
            if score.scorer == scorer and predicate(score.value):
                results.append(sample)
                break
    return results
