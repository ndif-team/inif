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
    """Backwards-compatible accessor; prefer ``token.tags``."""
    return token.tags


def tag_by_regex(
    sample: Sample,
    pattern: str,
    tag: str,
    sequences: list[Sequence] | None = None,
) -> None:
    """Tag every token whose string matches ``pattern``.

    When ``sequences`` is provided, the search runs over the *expanded* view
    of the sample so tokens currently compressed inside a sequence ref are
    inspected too. Matches inside a ref cause the containing ref to be
    materialized in this sample (per-token information attaches to real
    Tokens; other refs and other samples are untouched).
    """
    compiled = re.compile(pattern)
    if not sequences:
        for token in sample.tokens:
            if token.token is not None and compiled.search(token.token):
                token.add_tag(tag)
        return

    expanded = sample.get_expanded_tokens(sequences)
    matching_positions = [
        i
        for i, t in enumerate(expanded)
        if t.token is not None and compiled.search(t.token)
    ]
    for pos in matching_positions:
        _, real_token = sample.materialize_position(pos, sequences)
        real_token.add_tag(tag)


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

    Operates directly on ``sample.tokens``; sequence refs are not expanded
    here. To match inside refs, call ``sample.materialize_position`` first
    or apply this on a fully expanded view.

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
    token_strings = [t.token if t.token is not None else "" for t in sample.tokens]
    joined = "".join(token_strings)

    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in token_strings:
        offsets.append((pos, pos + len(s)))
        pos += len(s)

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
            sample.tokens[matching_indices[0]].add_tag(tag)
        elif mode is TextTagMode.LAST:
            sample.tokens[matching_indices[-1]].add_tag(tag)
        else:
            for i in matching_indices:
                sample.tokens[i].add_tag(tag)


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
        if predicate(token):
            token.add_tag(tag)


def tag_positions(sample: Sample, positions: list[int], tag: str) -> None:
    pos_set = set(positions)
    for i, token in enumerate(sample.tokens):
        if i in pos_set:
            token.add_tag(tag)


def remove_tag(sample: Sample, tag: str) -> None:
    for token in sample.tokens:
        token.remove_tag(tag)


def remove_tag_all(doc: InifDocument, tag: str) -> None:
    for sample in doc.samples:
        remove_tag(sample, tag)


def create_span_from_tag(sample: Sample, tag: str, span_name: str) -> Span:
    positions = [i for i, t in enumerate(sample.tokens) if t.has_tag(tag)]
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

    roles: list[str] = []
    for i in range(len(expanded)):
        tok_start, tok_end = offsets[i]
        role = "template"
        for span_start, span_end, span_role in content_spans:
            if tok_start >= span_start and tok_end <= span_end:
                role = span_role
                break
        roles.append(role)

    # Write roles to non-ref tokens; ref tokens stay collapsed because their
    # contents share a single role within a deduplicated chat template.
    exp_idx = 0
    seq_map = {s.id: s for s in sequences} if sequences else {}
    for tok in sample.tokens:
        if tok.is_sequence_ref:
            assert tok.sequence_id is not None
            n = seq_map[tok.sequence_id].n_tokens
            exp_idx += n
        else:
            tok.set_extra("role", roles[exp_idx])
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
        if not token.is_sequence_ref and token.id in special_ids:
            token.add_tag(tag)
