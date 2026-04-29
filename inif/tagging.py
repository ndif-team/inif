from __future__ import annotations

import re
from bisect import bisect_right
from enum import Enum
from typing import Any, Callable

from inif._token_ops import (
    PredicateTag,
    apply_predicate_tags,
    apply_regex_tags,
    compile_regex_tags,
)
from inif.models import InifDocument, Sample, Sequence, Span, TokenOrSeqRef


class TextTagMode(str, Enum):
    """Which tokens to tag when a text-level regex match spans multiple tokens."""

    ALL = "all"
    FIRST = "first"
    LAST = "last"


def _tag_by_regexes(
    sample: Sample,
    regex_tags: list[tuple[str | re.Pattern[str], str]],
    sequences: list[Sequence] | None = None,
) -> None:
    """Implementation backing :meth:`Sample.tag_by_regexes`.

    Apply multiple regex taggers in one token pass. If ``sequences`` is
    provided, sequence refs are materialized only when at least one expanded
    token actually matches.
    """
    apply_regex_tags(sample, compile_regex_tags(regex_tags), sequences=sequences)


def _tag_by_regexes_doc(
    doc: InifDocument,
    regex_tags: list[tuple[str | re.Pattern[str], str]],
) -> None:
    """Implementation backing :meth:`InifDocument.tag_by_regexes`."""
    compiled = compile_regex_tags(regex_tags)
    for sample in doc.samples:
        apply_regex_tags(sample, compiled, doc.sequences or None)


def _tag_by_text_regex(
    sample: Sample,
    pattern: str,
    tag: str,
    mode: TextTagMode = TextTagMode.ALL,
) -> None:
    """Implementation backing :meth:`Sample.tag_by_text_regex`.

    Joins all token strings, finds regex matches in the joined text, then maps
    character spans back to token indices. Handles BPE subword splits (e.g.
    ``"Eiffel"`` split into ``[" E", "iff", "el"]``).

    Operates directly on ``sample.tokens``; sequence refs are not expanded
    here. To match inside refs, call :meth:`Sample.materialize_position` first
    or apply this on a fully expanded view.

    ``mode`` selects which tokens to tag per match:

    - ``ALL`` — tag all tokens overlapping the match (default)
    - ``FIRST`` — tag only the first token of each match
    - ``LAST`` — tag only the last token of each match
    """
    mode = TextTagMode(mode)
    compiled = re.compile(pattern)
    token_strings = [t.token if t.token is not None else "" for t in sample.tokens]
    joined = "".join(token_strings)

    offsets: list[tuple[int, int]] = []
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for s in token_strings:
        start, end = pos, pos + len(s)
        offsets.append((start, end))
        starts.append(start)
        ends.append(end)
        pos = end

    for m in compiled.finditer(joined):
        m_start, m_end = m.start(), m.end()
        if m_start == m_end:
            continue
        first = bisect_right(ends, m_start)
        matching_indices: list[int] = []
        i = first
        while i < len(offsets) and starts[i] < m_end:
            matching_indices.append(i)
            i += 1
        if not matching_indices:
            continue
        if mode is TextTagMode.FIRST:
            sample.annotate_positions(tag, [matching_indices[0]])
        elif mode is TextTagMode.LAST:
            sample.annotate_positions(tag, [matching_indices[-1]])
        else:
            sample.annotate_positions(tag, matching_indices)


def _tag_by_text_regex_doc(
    doc: InifDocument,
    pattern: str,
    tag: str,
    mode: TextTagMode = TextTagMode.ALL,
) -> None:
    """Implementation backing :meth:`InifDocument.tag_by_text_regex`."""
    for sample in doc.samples:
        _tag_by_text_regex(sample, pattern, tag, mode=mode)


def _tag_by_predicates(
    sample: Sample,
    predicate_tags: list[PredicateTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Implementation backing :meth:`Sample.tag_by_predicates`."""
    apply_predicate_tags(sample, predicate_tags, sequences=sequences)


def _tag_by_predicates_doc(
    doc: InifDocument,
    predicate_tags: list[PredicateTag],
) -> None:
    """Implementation backing :meth:`InifDocument.tag_by_predicates`."""
    for sample in doc.samples:
        apply_predicate_tags(sample, predicate_tags, doc.sequences or None)


def _create_span_from_tag(sample: Sample, tag: str, span_name: str) -> Span:
    """Implementation backing :meth:`Sample.create_span_from_tag`."""
    positions = sample.annotation_positions(tag)
    span = Span(name=span_name, positions=positions, tags=[tag])
    sample.spans.append(span)
    return span


def _tag_chat_roles(
    sample: Sample,
    messages: list[dict[str, str]],
    tokenizer: Any,
    sequences: list[Sequence] | None = None,
) -> None:
    """Implementation backing :meth:`Sample.tag_chat_roles`.

    Tags tokens with their chat-template role via character-span matching.
    Works with any HuggingFace chat template. Content tokens get the role of
    their enclosing message; everything else (delimiters, role names,
    auto-generated text) is tagged as ``"template"``.

    Operates on expanded token views but writes roles only to non-ref tokens
    in ``sample.tokens``. Must be called AFTER sequence deduplication.
    """
    expanded = (
        sample.get_expanded_tokens(sequences) if sequences else list(sample.tokens)
    )

    from inif.converters._tokenize import render_chat_template

    formatted = render_chat_template(tokenizer, messages)
    decoded = [t.token or "" for t in expanded]

    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in decoded:
        offsets.append((pos, pos + len(s)))
        pos += len(s)

    concatenated = "".join(decoded)
    # Best-effort: some tokenizers are lossy for specific code points (e.g.
    # Qwen replaces U+2028 with U+FFFD during encode/decode). When the
    # decoded stream diverges from the formatted chat template, character
    # offsets for ``content`` lookups stop lining up, so we skip role tagging
    # for this sample rather than raise — mirroring ``_annotate_response_tokens``.
    if concatenated != formatted:
        return

    # For each message, locate (in order) its reasoning_content text and its
    # content text in the formatted output. Both belong to the message's role
    # — reasoning is content the model emitted, just rendered inside the chat
    # template's reasoning slot rather than the main content slot.
    content_spans: list[tuple[int, int, str]] = []
    search_from = 0
    for msg in messages:
        role = msg["role"]
        for field in ("reasoning_content", "reasoning", "content"):
            text = msg.get(field) or ""
            if not text:
                continue
            idx = concatenated.find(text, search_from)
            if idx < 0:
                continue
            content_spans.append((idx, idx + len(text), role))
            search_from = idx + len(text)

    roles: list[str] = []
    for i in range(len(expanded)):
        tok_start, tok_end = offsets[i]
        role = "template"
        for span_start, span_end, span_role in content_spans:
            if tok_start >= span_start and tok_end <= span_end:
                role = span_role
                break
        roles.append(role)

    # Materialize sequence refs whose expanded positions span multiple roles
    # — the alternative (skipping mixed-role refs) leaves them untagged AND
    # blocks downstream taggers from assigning role-derived annotations.
    from inif.converters._tokenize import _materialize_inconsistent_refs

    role_per_exp = dict(enumerate(roles))
    _materialize_inconsistent_refs(sample, sequences, role_per_exp)

    seq_map = {s.id: s for s in sequences} if sequences else {}
    positions_by_role: dict[str, list[int]] = {}
    exp_idx = 0
    for pos, tok in enumerate(sample.tokens):
        if tok.is_sequence_ref:
            n = seq_map[tok.token].n_tokens
            # All expanded positions in this ref now share a role (post-
            # materialization), so the first one represents the whole ref.
            positions_by_role.setdefault(roles[exp_idx], []).append(pos)
            exp_idx += n
        else:
            positions_by_role.setdefault(roles[exp_idx], []).append(pos)
            exp_idx += 1
    for role, positions in positions_by_role.items():
        sample.annotate_positions(role, positions, metadata={"source": "message_role"})


def _tag_chat_roles_doc(
    doc: InifDocument,
    messages_per_sample: list[list[dict[str, str]]],
    tokenizer: Any,
) -> None:
    """Implementation backing :meth:`InifDocument.tag_chat_roles`."""
    assert len(messages_per_sample) == len(doc.samples), (
        f"Expected {len(doc.samples)} message lists, got {len(messages_per_sample)}"
    )
    for sample, messages in zip(doc.samples, messages_per_sample):
        _tag_chat_roles(sample, messages, tokenizer, doc.sequences or None)


def _tag_special_tokens(sample: Sample, tokenizer, tag: str = "special") -> None:
    """Implementation backing :meth:`Sample.tag_special_tokens`."""
    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)
    positions: list[int] = []
    for i, token in enumerate(sample.tokens):
        if not token.is_sequence_ref and token.id in special_ids:
            positions.append(i)
    sample.annotate_positions(tag, positions)


# Type re-export — callers can import :class:`PredicateTag` from ``inif.tagging``
# without dipping into the private ``_token_ops`` module.
PredicateTag = PredicateTag

# Tagging callables that accept either a ``str`` regex or an already-compiled
# pattern. Re-exported here for callers building up batched tagger lists.
RegexTag = tuple[str | re.Pattern[str], str]
TokenPredicate = Callable[[TokenOrSeqRef], bool]
