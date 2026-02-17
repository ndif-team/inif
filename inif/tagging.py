from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable

from inif.models import InifDocument, Sample, Sequence, Span, Token


class TextTagMode(str, Enum):
    """Which tokens to tag when a text-level regex match spans multiple tokens."""

    ALL = "all"
    FIRST = "first"
    LAST = "last"


def _get_tags(token: Token) -> list[str]:
    """Get tags from token extra field."""
    return getattr(token, "tags", [])


def _add_tag(token: Token, tag: str) -> None:
    """Add a tag to a token's extra field."""
    tags = _get_tags(token)
    if tag not in tags:
        tags = list(tags) + [tag]
        token.__dict__["tags"] = tags
        # Also update pydantic's extra fields
        if token.model_extra is not None:
            token.model_extra["tags"] = tags


def _remove_tag_from_token(token: Token, tag: str) -> None:
    """Remove a tag from a token's extra field."""
    tags = _get_tags(token)
    if tag in tags:
        tags = [t for t in tags if t != tag]
        if tags:
            token.__dict__["tags"] = tags
            if token.model_extra is not None:
                token.model_extra["tags"] = tags
        else:
            token.__dict__.pop("tags", None)
            if token.model_extra is not None:
                token.model_extra.pop("tags", None)


def tag_by_regex(
    sample: Sample,
    pattern: str,
    tag: str,
    sequences: list[Sequence] | None = None,
) -> None:
    compiled = re.compile(pattern)
    if sequences:
        expanded = sample.get_expanded_tokens(sequences)
        # Map expanded tokens back to original indices
        orig_idx = 0
        exp_idx = 0
        for orig_token in sample.tokens:
            if orig_token.is_sequence_ref:
                # This token expands to multiple tokens
                assert orig_token.sequence_id is not None
                seq_map = {s.id: s for s in sequences}
                seq = seq_map[orig_token.sequence_id]
                n_expanded = seq.n_tokens
                for ei in range(n_expanded):
                    et = expanded[exp_idx + ei]
                    if et.token is not None and compiled.search(et.token):
                        _add_tag(orig_token, tag)
                        break
                exp_idx += n_expanded
            else:
                if orig_token.token is not None and compiled.search(orig_token.token):
                    _add_tag(orig_token, tag)
                exp_idx += 1
            orig_idx += 1
    else:
        for token in sample.tokens:
            if token.token is not None and compiled.search(token.token):
                _add_tag(token, tag)


def tag_by_regex_all(doc: InifDocument, pattern: str, tag: str) -> None:
    for sample in doc.samples:
        tag_by_regex(sample, pattern, tag, doc.sequences)


def tag_by_text_regex(
    sample: Sample,
    pattern: str,
    tag: str,
    mode: TextTagMode = TextTagMode.ALL,
) -> None:
    """Tag tokens whose concatenated text matches a regex.

    Joins all token strings, finds regex matches in the joined text,
    then maps character spans back to token indices. This handles
    BPE subword splits (e.g. "Eiffel" split into [" E", "iff", "el"]).

    Args:
        sample: The sample to tag.
        pattern: Regex pattern to match against the concatenated text.
        tag: Tag name to add.
        mode: Which tokens to tag per match:
            ALL — tag all tokens overlapping the match (default)
            FIRST — tag only the first token of each match
            LAST — tag only the last token of each match
    """
    mode = TextTagMode(mode)
    compiled = re.compile(pattern)
    # Build character-to-token mapping
    token_strings: list[str] = []
    for t in sample.tokens:
        token_strings.append(t.token if t.token is not None else "")
    joined = "".join(token_strings)

    # Compute character offset for each token
    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in token_strings:
        offsets.append((pos, pos + len(s)))
        pos += len(s)

    # Find all matches and map back to token indices
    for m in compiled.finditer(joined):
        m_start, m_end = m.start(), m.end()
        matching_indices = [
            i
            for i, (t_start, t_end) in enumerate(offsets)
            if t_end > m_start and t_start < m_end
        ]
        if not matching_indices:
            continue
        if mode is TextTagMode.FIRST:
            _add_tag(sample.tokens[matching_indices[0]], tag)
        elif mode is TextTagMode.LAST:
            _add_tag(sample.tokens[matching_indices[-1]], tag)
        else:
            for i in matching_indices:
                _add_tag(sample.tokens[i], tag)


def tag_by_text_regex_all(
    doc: InifDocument,
    pattern: str,
    tag: str,
    mode: TextTagMode = TextTagMode.ALL,
) -> None:
    """Apply text-based regex tagging across all samples."""
    for sample in doc.samples:
        tag_by_text_regex(sample, pattern, tag, mode=mode)


def tag_by_predicate(
    sample: Sample, predicate: Callable[[Token], bool], tag: str
) -> None:
    for token in sample.tokens:
        if predicate(token) and tag not in _get_tags(token):
            _add_tag(token, tag)


def tag_positions(sample: Sample, positions: list[int], tag: str) -> None:
    pos_set = set(positions)
    for i, token in enumerate(sample.tokens):
        if i in pos_set and tag not in _get_tags(token):
            _add_tag(token, tag)


def remove_tag(sample: Sample, tag: str) -> None:
    for token in sample.tokens:
        _remove_tag_from_token(token, tag)


def remove_tag_all(doc: InifDocument, tag: str) -> None:
    for sample in doc.samples:
        remove_tag(sample, tag)


def create_span_from_tag(sample: Sample, tag: str, span_name: str) -> Span:
    positions = [i for i, t in enumerate(sample.tokens) if tag in _get_tags(t)]
    span = Span(name=span_name, positions=positions, tags=[tag])
    sample.spans.append(span)
    return span


def tag_chat_roles(
    sample: Sample,
    messages: list[dict[str, str]],
    tokenizer: Any,
    sequences: list[Sequence] | None = None,
) -> None:
    """Tag tokens with their chat template role via character-span matching.

    Works with any HuggingFace chat template. Content tokens get the role of
    their enclosing message; everything else (delimiters, role names,
    auto-generated text) is tagged as ``"template"``.

    Operates on expanded token views but writes roles only to non-ref tokens
    in ``sample.tokens``. Must be called AFTER sequence deduplication.

    Args:
        sample: The sample to tag.
        messages: List of message dicts with ``"role"`` and ``"content"`` keys.
        tokenizer: A tokenizer with ``apply_chat_template`` support.
        sequences: Sequences for expanding refs (from ``doc.sequences``).
    """
    expanded = (
        sample.get_expanded_tokens(sequences) if sequences else list(sample.tokens)
    )

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    decoded = [t.token or "" for t in expanded]

    # Build character offsets for each token
    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in decoded:
        offsets.append((pos, pos + len(s)))
        pos += len(s)

    concatenated = "".join(decoded)
    assert concatenated == formatted, (
        f"Decoded tokens don't reconstruct the formatted string "
        f"({len(concatenated)} vs {len(formatted)} chars)"
    )

    # Find each message's content as a character span
    content_spans: list[tuple[int, int, str]] = []
    search_from = 0
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        idx = concatenated.find(content, search_from)
        if idx >= 0:
            content_spans.append((idx, idx + len(content), msg["role"]))
            search_from = idx + len(content)

    # Assign a role to every expanded position
    roles: list[str] = []
    for i in range(len(expanded)):
        tok_start, tok_end = offsets[i]
        role = "template"
        for span_start, span_end, span_role in content_spans:
            if tok_start >= span_start and tok_end <= span_end:
                role = span_role
                break
        roles.append(role)

    # Write roles to non-ref tokens in sample.tokens, skipping ref tokens
    exp_idx = 0
    seq_map = {s.id: s for s in sequences} if sequences else {}
    for tok in sample.tokens:
        if tok.is_sequence_ref:
            assert tok.sequence_id is not None
            n = seq_map[tok.sequence_id].n_tokens
            exp_idx += n
        else:
            tok.__dict__["role"] = roles[exp_idx]
            if tok.model_extra is not None:
                tok.model_extra["role"] = roles[exp_idx]
            exp_idx += 1


def tag_chat_roles_doc(
    doc: InifDocument,
    messages_per_sample: list[list[dict[str, str]]],
    tokenizer: Any,
) -> None:
    """Tag chat roles for all samples in a document.

    Args:
        doc: The document whose samples to tag.
        messages_per_sample: One message list per sample, same order as ``doc.samples``.
        tokenizer: A tokenizer with ``apply_chat_template`` support.
    """
    assert len(messages_per_sample) == len(doc.samples), (
        f"Expected {len(doc.samples)} message lists, got {len(messages_per_sample)}"
    )
    for sample, messages in zip(doc.samples, messages_per_sample):
        tag_chat_roles(sample, messages, tokenizer, doc.sequences or None)


def tag_special_tokens(sample: Sample, tokenizer, tag: str = "special") -> None:
    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)
    for token in sample.tokens:
        if (
            not token.is_sequence_ref
            and token.id in special_ids
            and tag not in _get_tags(token)
        ):
            _add_tag(token, tag)
