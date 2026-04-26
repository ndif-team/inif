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
    tag_by_regexes(sample, [(pattern, tag)], sequences=sequences)


def tag_by_regexes(
    sample: Sample,
    regex_tags: list[tuple[str | re.Pattern[str], str]],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply multiple regex taggers in one token pass.

    This is the preferred API for large documents when several regex-based
    strategies are known up front. If ``sequences`` is provided, sequence refs
    are materialized only when at least one expanded token actually matches.
    """
    apply_regex_tags(sample, compile_regex_tags(regex_tags), sequences=sequences)


def tag_by_regex_all(doc: InifDocument, pattern: str, tag: str) -> None:
    """Apply a single regex tagger across every sample in ``doc``.

    Args:
        doc: The document whose samples to tag.
        pattern: Regex pattern matched against each token's string.
        tag: Annotation name to add to matching positions.
    """
    tag_by_regexes_all(doc, [(pattern, tag)])


def tag_by_regexes_all(
    doc: InifDocument,
    regex_tags: list[tuple[str | re.Pattern[str], str]],
) -> None:
    """Apply multiple regex taggers across all samples in one pass per sample."""
    compiled = compile_regex_tags(regex_tags)
    for sample in doc.samples:
        apply_regex_tags(sample, compiled, doc.sequences or None)


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
    sample: Sample, predicate: Callable[[TokenOrSeqRef], bool], tag: str
) -> None:
    """Tag every token for which ``predicate(token)`` returns ``True``.

    Args:
        sample: The sample to tag.
        predicate: Callable receiving the :class:`Token` (not just the
            string), returning a truthy value for matches.
        tag: Annotation name to add.
    """
    tag_by_predicates(sample, [(predicate, tag)])


def tag_by_predicates(
    sample: Sample,
    predicate_tags: list[PredicateTag],
    sequences: list[Sequence] | None = None,
) -> None:
    """Apply multiple Python predicate taggers in one token pass."""
    apply_predicate_tags(sample, predicate_tags, sequences=sequences)


def tag_by_predicates_all(
    doc: InifDocument,
    predicate_tags: list[PredicateTag],
) -> None:
    """Apply multiple Python predicate taggers across all samples."""
    for sample in doc.samples:
        apply_predicate_tags(sample, predicate_tags, doc.sequences or None)


def tag_positions(sample: Sample, positions: list[int], tag: str) -> None:
    """Add an annotation covering exactly the given positions.

    Adjacent positions are coalesced into ranges by
    :meth:`Sample.annotate_positions`.

    Args:
        sample: The sample to tag.
        positions: Sample-local token positions to annotate.
        tag: Annotation name to add.
    """
    sample.annotate_positions(tag, positions)


def remove_tag(sample: Sample, tag: str) -> None:
    """Drop every annotation with the given name from ``sample``.

    Annotations that share the name but differ only in metadata are also
    removed — the helper does not preserve metadata-distinguished duplicates.

    Args:
        sample: The sample to modify.
        tag: Annotation name to drop.
    """
    sample.remove_annotation(tag)


def remove_tag_all(doc: InifDocument, tag: str) -> None:
    """Drop every annotation with the given name from every sample in ``doc``.

    Args:
        doc: The document whose samples to modify.
        tag: Annotation name to drop.
    """
    for sample in doc.samples:
        remove_tag(sample, tag)


def create_span_from_tag(sample: Sample, tag: str, span_name: str) -> Span:
    """Convert an annotation into a :class:`Span` on the same sample.

    The new span carries the original tag in its ``tags`` list and the
    flattened position list across every range with that name.

    Args:
        sample: The sample to add the span to.
        tag: Annotation name to convert.
        span_name: Name to give the resulting span.

    Returns:
        The newly created :class:`Span`, also appended to ``sample.spans``.
    """
    positions = sample.annotation_positions(tag)
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
    # Best-effort: some tokenizers are lossy for specific code points (e.g.
    # Qwen replaces U+2028 with U+FFFD during encode/decode). When the
    # decoded stream diverges from the formatted chat template, character
    # offsets for ``content`` lookups stop lining up, so we skip role tagging
    # for this sample rather than raise — mirroring ``_annotate_response_tokens``.
    if concatenated != formatted:
        return

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

    exp_idx = 0
    seq_map = {s.id: s for s in sequences} if sequences else {}
    positions_by_role: dict[str, list[int]] = {}
    for pos, tok in enumerate(sample.tokens):
        if tok.is_sequence_ref:
            # Sequence-ref target id lives in ``tok.token`` after dropping
            # the dedicated ``sequence_id`` field.
            n = seq_map[tok.token].n_tokens
            ref_roles = set(roles[exp_idx : exp_idx + n])
            if len(ref_roles) == 1:
                positions_by_role.setdefault(ref_roles.pop(), []).append(pos)
            exp_idx += n
        else:
            positions_by_role.setdefault(roles[exp_idx], []).append(pos)
            exp_idx += 1
    for role, positions in positions_by_role.items():
        sample.annotate_positions(role, positions, metadata={"source": "message_role"})


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


def tag_special_tokens(sample: Sample, tokenizer: Any, tag: str = "special") -> None:
    """Tag every special token (BOS, EOS, etc.) in ``sample``.

    Special token ids are read from the tokenizer's ``all_special_ids``
    attribute (any tokenizer without that attribute results in no tags).
    Sequence-ref tokens are skipped.

    Args:
        sample: The sample to tag.
        tokenizer: A HuggingFace tokenizer exposing ``all_special_ids``.
        tag: Annotation name to add. Defaults to ``"special"``.
    """
    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)
    positions: list[int] = []
    for i, token in enumerate(sample.tokens):
        if not token.is_sequence_ref and token.id in special_ids:
            positions.append(i)
    sample.annotate_positions(tag, positions)
