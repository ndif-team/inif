from __future__ import annotations

import re
from collections.abc import Callable

from inif.models import Sample, Sequence, TokenOrSeqRef

CompiledRegexTag = tuple[re.Pattern[str], str]
PredicateTag = tuple[Callable[[TokenOrSeqRef], bool], str]


def sample_has_sequence_refs(sample: Sample) -> bool:
    return any(tok.is_sequence_ref for tok in sample.tokens)


def compile_regex_tags(
    regex_tags: list[tuple[str | re.Pattern[str], str]],
) -> list[CompiledRegexTag]:
    """Compile regex tag specs into the representation used by token passes.

    This deliberately small boundary is where a future Rust backend can slot in:
    callers provide token strings plus compiled/tag specs, and the backend returns
    match decisions without knowing about Pydantic models.
    """
    return [
        (re.compile(pattern) if isinstance(pattern, str) else pattern, tag)
        for pattern, tag in regex_tags
    ]


def _regex_matches(
    token: TokenOrSeqRef, regex_tags: list[CompiledRegexTag]
) -> list[str]:
    # Sequence refs carry the target Sequence id in ``token.token``; matching a
    # regex against that string would tag spurious positions, so skip refs.
    if token.is_sequence_ref:
        return []
    text = token.token
    if text is None:
        return []
    return [tag for pattern, tag in regex_tags if pattern.search(text)]


def _predicate_matches(
    token: TokenOrSeqRef, predicate_tags: list[PredicateTag]
) -> list[str]:
    return [tag for predicate, tag in predicate_tags if predicate(token)]


def _remap_annotations(sample: Sample, old_to_new: list[tuple[int, int]]) -> None:
    for annotation in sample.annotations:
        ranges: list[tuple[int, int]] = []
        for start, end in annotation.ranges:
            ranges.append((old_to_new[start][0], old_to_new[end - 1][1]))
        annotation.ranges = ranges


def apply_regex_tags(
    sample: Sample,
    regex_tags: list[CompiledRegexTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply regex tags, materializing sequence refs only when needed."""
    if not regex_tags:
        return
    positions_by_tag: dict[str, list[int]] = {tag: [] for _, tag in regex_tags}
    if not sequences or not sample_has_sequence_refs(sample):
        for pos, token in enumerate(sample.tokens):
            for tag in _regex_matches(token, regex_tags):
                positions_by_tag[tag].append(pos)
        for tag, positions in positions_by_tag.items():
            sample.annotate_positions(tag, positions)
        return

    seq_map = {seq.id: seq for seq in sequences}
    new_tokens: list[TokenOrSeqRef] = []
    old_to_new: list[tuple[int, int]] = []
    for token in sample.tokens:
        new_start = len(new_tokens)
        if not token.is_sequence_ref:
            pos = len(new_tokens)
            for tag in _regex_matches(token, regex_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(token)
            old_to_new.append((new_start, len(new_tokens)))
            continue

        assert token.token in seq_map, f"Sequence '{token.token}' not found"
        seq = seq_map[token.token]
        should_materialize = any(
            seq_token.token is not None
            and any(pattern.search(seq_token.token) for pattern, _ in regex_tags)
            for seq_token in seq.tokens
        )
        if not should_materialize:
            pos = len(new_tokens)
            for tag in _regex_matches(token, regex_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(token)
            old_to_new.append((new_start, len(new_tokens)))
            continue

        for seq_token in seq.tokens:
            pos = len(new_tokens)
            real_token = TokenOrSeqRef(id=seq_token.id, token=seq_token.token)
            for tag in _regex_matches(real_token, regex_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(real_token)
        old_to_new.append((new_start, len(new_tokens)))

    _remap_annotations(sample, old_to_new)
    sample.tokens = new_tokens
    for tag, positions in positions_by_tag.items():
        sample.annotate_positions(tag, positions)


def apply_predicate_tags(
    sample: Sample,
    predicate_tags: list[PredicateTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply predicate tags, materializing sequence refs only when needed."""
    if not predicate_tags:
        return
    positions_by_tag: dict[str, list[int]] = {tag: [] for _, tag in predicate_tags}
    if not sequences or not sample_has_sequence_refs(sample):
        for pos, token in enumerate(sample.tokens):
            for tag in _predicate_matches(token, predicate_tags):
                positions_by_tag[tag].append(pos)
        for tag, positions in positions_by_tag.items():
            sample.annotate_positions(tag, positions)
        return

    seq_map = {seq.id: seq for seq in sequences}
    new_tokens: list[TokenOrSeqRef] = []
    old_to_new: list[tuple[int, int]] = []
    for token in sample.tokens:
        new_start = len(new_tokens)
        if not token.is_sequence_ref:
            pos = len(new_tokens)
            for tag in _predicate_matches(token, predicate_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(token)
            old_to_new.append((new_start, len(new_tokens)))
            continue

        assert token.token in seq_map, f"Sequence '{token.token}' not found"
        seq = seq_map[token.token]
        should_materialize = any(
            any(predicate(seq_token) for predicate, _ in predicate_tags)
            for seq_token in seq.tokens
        )
        if not should_materialize:
            pos = len(new_tokens)
            for tag in _predicate_matches(token, predicate_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(token)
            old_to_new.append((new_start, len(new_tokens)))
            continue

        for seq_token in seq.tokens:
            pos = len(new_tokens)
            real_token = TokenOrSeqRef(id=seq_token.id, token=seq_token.token)
            for tag in _predicate_matches(real_token, predicate_tags):
                positions_by_tag[tag].append(pos)
            new_tokens.append(real_token)
        old_to_new.append((new_start, len(new_tokens)))

    _remap_annotations(sample, old_to_new)
    sample.tokens = new_tokens
    for tag, positions in positions_by_tag.items():
        sample.annotate_positions(tag, positions)
