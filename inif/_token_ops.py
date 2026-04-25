from __future__ import annotations

import re
from collections.abc import Callable

from inif.models import Sample, Sequence, Token

CompiledRegexTag = tuple[re.Pattern[str], str]
PredicateTag = tuple[Callable[[Token], bool], str]


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


def _apply_regex_tags(token: Token, regex_tags: list[CompiledRegexTag]) -> None:
    text = token.token
    if text is None:
        return
    for pattern, tag in regex_tags:
        if pattern.search(text):
            token.add_tag(tag)


def _apply_predicate_tags(token: Token, predicate_tags: list[PredicateTag]) -> None:
    for predicate, tag in predicate_tags:
        if predicate(token):
            token.add_tag(tag)


def apply_regex_tags(
    sample: Sample,
    regex_tags: list[CompiledRegexTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply regex tags, materializing sequence refs only when needed."""
    if not regex_tags:
        return
    if not sequences or not sample_has_sequence_refs(sample):
        for token in sample.tokens:
            _apply_regex_tags(token, regex_tags)
        return

    seq_map = {seq.id: seq for seq in sequences}
    new_tokens: list[Token] = []
    changed = False
    for token in sample.tokens:
        if not token.is_sequence_ref:
            _apply_regex_tags(token, regex_tags)
            new_tokens.append(token)
            continue

        assert token.sequence_id is not None, "Sequence ref token must have sequence_id"
        assert token.sequence_id in seq_map, f"Sequence '{token.sequence_id}' not found"
        seq = seq_map[token.sequence_id]
        should_materialize = any(
            seq_token.token is not None
            and any(pattern.search(seq_token.token) for pattern, _ in regex_tags)
            for seq_token in seq.tokens
        )
        if not should_materialize:
            new_tokens.append(token)
            continue

        changed = True
        for seq_token in seq.tokens:
            real_token = Token(
                id=seq_token.id,
                token=seq_token.token,
                sequence_id=seq.id,
            )
            _apply_regex_tags(real_token, regex_tags)
            new_tokens.append(real_token)

    if changed:
        sample.tokens = new_tokens


def apply_predicate_tags(
    sample: Sample,
    predicate_tags: list[PredicateTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply predicate tags, materializing sequence refs only when needed."""
    if not predicate_tags:
        return
    if not sequences or not sample_has_sequence_refs(sample):
        for token in sample.tokens:
            _apply_predicate_tags(token, predicate_tags)
        return

    seq_map = {seq.id: seq for seq in sequences}
    new_tokens: list[Token] = []
    changed = False
    for token in sample.tokens:
        if not token.is_sequence_ref:
            _apply_predicate_tags(token, predicate_tags)
            new_tokens.append(token)
            continue

        assert token.sequence_id is not None, "Sequence ref token must have sequence_id"
        assert token.sequence_id in seq_map, f"Sequence '{token.sequence_id}' not found"
        seq = seq_map[token.sequence_id]
        should_materialize = any(
            any(predicate(seq_token) for predicate, _ in predicate_tags)
            for seq_token in seq.tokens
        )
        if not should_materialize:
            new_tokens.append(token)
            continue

        changed = True
        for seq_token in seq.tokens:
            real_token = Token(
                id=seq_token.id,
                token=seq_token.token,
                sequence_id=seq.id,
            )
            _apply_predicate_tags(real_token, predicate_tags)
            new_tokens.append(real_token)

    if changed:
        sample.tokens = new_tokens
