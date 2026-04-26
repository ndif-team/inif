from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from inif.models import InifDocument, Sample, TokenOrSeqRef


@dataclass
class TokenSelection:
    """A subset of tokens from a single sample, returned by every selector.

    ``positions`` always reflects the actual sample-local indices (sorted,
    deduplicated). ``tokens`` is the parallel list of :class:`Token`
    objects pulled from those positions.
    """

    sample_id: str | int
    tokens: list[TokenOrSeqRef] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


def select_by_position(
    sample: Sample, positions: int | list[int] | slice
) -> TokenSelection:
    """Select tokens by sample-local position.

    Args:
        sample: The sample to select from.
        positions: A single index, a list of indices, or a ``slice``. Slices
            are resolved against ``len(sample.tokens)`` via ``slice.indices``,
            so negative starts behave like Python's normal slicing.

    Returns:
        A :class:`TokenSelection` whose ``positions`` are sorted and unique.
    """
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


def select_by_annotation(sample: Sample, annotation_name: str) -> TokenSelection:
    """Select every token covered by an annotation with the given name.

    Positions are flattened across all the ranges of every matching
    annotation, so this works whether the annotation has one range or many.

    Args:
        sample: The sample to select from.
        annotation_name: The annotation name to match.

    Returns:
        A :class:`TokenSelection` with positions sorted ascending.
    """
    positions = sample.annotation_positions(annotation_name)
    tokens = sample.get_tokens_by_positions(positions)
    return TokenSelection(
        sample_id=sample.id,
        tokens=tokens,
        positions=positions,
    )


def select_by_sequence_id(sample: Sample, seq_id: str) -> TokenSelection:
    """Select sequence-ref tokens that point at the given sequence id.

    A sequence ref is identified by ``id is None`` and carries the target
    :class:`Sequence` id in its ``token`` field. Useful for finding *where*
    a shared run is referenced in a sample without expanding it.

    Args:
        sample: The sample to select from.
        seq_id: The :class:`Sequence` id to match (e.g. ``"seq_0"``).

    Returns:
        A :class:`TokenSelection` containing the ref tokens.
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


def select_by_span(sample: Sample, span_name: str) -> TokenSelection:
    """Select tokens covered by any :class:`Span` with the given name.

    Positions are unioned across every matching span on the sample.

    Args:
        sample: The sample to select from.
        span_name: The span name to match.

    Returns:
        A :class:`TokenSelection` with positions sorted ascending.
    """
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


def filter_samples_by_score(
    doc: InifDocument,
    scorer: str,
    predicate: Callable[[str | int | float | bool | list | dict], bool],
) -> list[Sample]:
    """Return samples whose ``scorer`` value satisfies ``predicate``.

    Compose with the position selectors above (``select`` etc.) on
    each returned sample to drill down to specific tokens.
    """
    results: list[Sample] = []
    for sample in doc.samples:
        for score in sample.scores:
            if score.scorer == scorer and predicate(score.value):
                results.append(sample)
                break
    return results
