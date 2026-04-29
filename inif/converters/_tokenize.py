"""Shared helpers for converting chat-style message dicts into INIF tokens.

Used by multiple eval-format converters (Inspect AI, evaleval). The logic here
is framework-agnostic: it consumes a list of ``{"role": ..., "content": ...}``
dicts and a HuggingFace tokenizer.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Callable

from inif.converters._decode import (
    ByteSliceDecoder,
    decode_token_ids,
    offset_mapping_decode_text,
)
from inif.models import Sample, Sequence, Text, TokenOrSeqRef


def render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool = False,
) -> Any:
    """Render the chat template with kwargs that maximally preserve information.

    Templates that have history-stripping behavior (e.g. Kimi's
    ``preserve_thinking=False`` default that drops reasoning from every
    message except the LAST non-tool-call assistant) make INIF lose data we
    care about for analysis. Passing ``preserve_thinking=True`` keeps the
    full reasoning trace; templates that don't recognize the kwarg silently
    ignore it.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=False,
        return_dict=False,
        preserve_thinking=True,
    )


def _build_message_children(
    reasoning: str | None,
    content: str,
    tool_calls: list[dict] | None,
) -> list[Text]:
    """Build the per-section sub-Texts for an assistant message.

    Returns ``[]`` when the message has only plain content (no reasoning,
    no tool calls) — in that case the parent ``Text.value`` carries the
    content directly. Otherwise emits one child per non-empty section so
    the viewer can render them as labeled boxes.
    """
    if not reasoning and not tool_calls:
        return []
    children: list[Text] = []
    if reasoning:
        children.append(Text(name="reasoning", value=reasoning))
    if content:
        children.append(Text(name="content", value=content))
    if tool_calls:
        call_children: list[Text] = []
        for call in tool_calls:
            fn = call.get("function") or {}
            fn_name = fn.get("name") or call.get("name") or "tool_call"
            args = fn.get("arguments") if "function" in call else call.get("arguments")
            if args is None:
                args = {}
            value = (
                args
                if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False, default=str)
            )
            md: dict[str, Any] = {}
            if "id" in call:
                md["id"] = call["id"]
            if "type" in call:
                md["type"] = call["type"]
            call_children.append(Text(name=fn_name, value=value, metadata=md))
        children.append(Text(name="tool_calls", value="", children=call_children))
    return children


def name_messages(messages: list[dict[str, Any]]) -> list[Text]:
    """Convert chat message dicts into role-named :class:`Text` segments.

    Each message becomes one ``Text`` whose ``name`` is its role suffixed
    with a per-role index (``"system_0"``, ``"user_0"``, ``"assistant_0"``,
    ``"user_1"``, …) and whose ``value`` is the message content. The
    system prompt, when present, is included.

    Assistant turns that issued a tool call (or pure reasoning) typically
    have ``content == ""`` — the actual payload sits on sibling fields. To
    keep that information visible, those sub-sections are lifted into
    :attr:`Text.children` (one child per ``reasoning`` / ``content`` /
    ``tool_calls`` block) and the parent's own ``value`` is left empty.
    Per-child token offsets are filled in later by
    :func:`_populate_text_offsets`.
    """
    counters: dict[str, int] = {}
    out: list[Text] = []
    for msg in messages:
        role = msg.get("role") or "text"
        idx = counters.get(role, 0)
        counters[role] = idx + 1
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or None
        raw_tool_calls = msg.get("tool_calls")
        tool_calls = list(raw_tool_calls) if raw_tool_calls else None
        content = msg.get("content", "") or ""
        children = _build_message_children(reasoning, content, tool_calls)
        # When sub-sections are lifted into children the parent's value is
        # cleared so the viewer doesn't render content twice.
        parent_value = "" if children else content
        out.append(Text(name=f"{role}_{idx}", value=parent_value, children=children))
    return out


def _detect_message_terminator(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    formatted: str,
) -> str | None:
    """Find the chat template's per-message terminator (``<|im_end|>`` etc.).

    Picks the special token from ``tokenizer.all_special_tokens`` whose
    occurrence count in ``formatted`` matches the number of messages —
    that's the one the template emits exactly once per message, i.e. the
    per-message close. When several specials match (Qwen-style templates
    use both ``<|im_start|>`` and ``<|im_end|>``, each appearing
    ``n_messages`` times), the one whose LAST occurrence sits closest to
    the end of ``formatted`` wins, since the terminator is what the
    template emits AFTER the last message's content while the role marker
    sits BEFORE it.

    Returns ``None`` when the tokenizer doesn't expose
    ``all_special_tokens`` or when no candidate matches the count
    heuristic — callers fall back to content-anchored splits.
    """
    specials = getattr(tokenizer, "all_special_tokens", None)
    if not specials:
        return None
    n_msgs = len(messages)
    if n_msgs <= 0:
        return None
    candidates: list[str] = []
    for tok in specials:
        if not isinstance(tok, str) or not tok:
            continue
        if formatted.count(tok) == n_msgs:
            candidates.append(tok)
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-formatted.rfind(t), -len(t)))
    return candidates[0]


def compute_message_token_ranges(
    messages: list[dict[str, Any]],
    tokens: list[TokenOrSeqRef],
    formatted: str,
    terminator: str | None = None,
) -> list[tuple[int, int] | None]:
    """For each message, return ``(start, end)`` token offsets that partition
    ``tokens`` across messages (or ``None`` per message when the partition
    cannot be reconstructed).

    The split point between adjacent messages is the chat template's
    per-message terminator (``<|im_end|>`` for Kimi/Qwen, ``<end_of_turn>``
    for Gemma, etc.) when one is provided: every token up to and including
    that terminator belongs to the message it closes; the next message
    starts immediately after, taking ownership of any leading delimiters
    (``<|im_start|>assistant\\n`` etc.) the template emits before its
    content.

    When ``terminator`` is ``None`` (or the template doesn't use a fixed
    terminator), we fall back to anchoring on each message's content
    start: msg ``i+1`` begins at the first token whose char start sits at
    or after msg ``i+1``'s content start, so trailing template chars get
    attributed to the message before — adequate but coarser than a real
    terminator split.

    Either way the first message starts at 0 and the last ends at
    ``len(tokens)``. Messages with no findable content / terminator inherit
    the previous boundary (empty range). Returns ``[None, ...]`` when
    ``"".join(token.token) != formatted`` so callers can skip the
    misattribution risk.
    """
    n_msgs = len(messages)
    if n_msgs == 0:
        return []
    n_tokens = len(tokens)
    decoded = [t.token or "" for t in tokens]
    if "".join(decoded) != formatted:
        return [None] * n_msgs

    # Token start / end char offsets.
    token_starts: list[int] = []
    token_ends: list[int] = []
    pos = 0
    for s in decoded:
        token_starts.append(pos)
        pos += len(s)
        token_ends.append(pos)

    boundaries: list[int] = [0]

    if terminator:
        # Walk the formatted string finding each terminator occurrence; the
        # first N-1 of them mark the boundaries between consecutive messages
        # (the trailing terminator that closes the last message is not a
        # boundary because there is no message after it).
        term_ends: list[int] = []
        cursor = 0
        while True:
            idx = formatted.find(terminator, cursor)
            if idx < 0:
                break
            term_ends.append(idx + len(terminator))
            cursor = idx + len(terminator)
        # Convert each terminator end-char to a token boundary: first token
        # whose char start sits at or after the terminator's end. That
        # token is the start of the next message.
        for ce in term_ends[: n_msgs - 1]:
            k = next(
                (j for j in range(n_tokens) if token_starts[j] >= ce),
                n_tokens,
            )
            boundaries.append(k)
        # If the template emitted fewer terminators than messages-1 (rare —
        # the last assistant turn sometimes lacks one when
        # ``add_generation_prompt`` is on), fall through and let the
        # remaining boundaries default to ``n_tokens``.
    else:
        # No terminator: anchor on content starts. ``search_from`` advances
        # monotonically so messages stay in order even when the same content
        # string appears more than once.
        content_starts: list[int | None] = []
        search_from = 0
        for msg in messages:
            m_start: int | None = None
            m_end = search_from
            for field in ("reasoning_content", "reasoning", "content"):
                text = msg.get(field) or ""
                if not isinstance(text, str) or not text:
                    continue
                idx = formatted.find(text, search_from)
                if idx < 0:
                    continue
                end = idx + len(text)
                if m_start is None or idx < m_start:
                    m_start = idx
                if end > m_end:
                    m_end = end
                search_from = end
            content_starts.append(m_start)

        for i in range(1, n_msgs):
            cs = content_starts[i]
            if cs is None:
                boundaries.append(boundaries[-1])
                continue
            k = next(
                (j for j in range(n_tokens) if token_starts[j] >= cs),
                n_tokens,
            )
            boundaries.append(k)

    while len(boundaries) < n_msgs:
        boundaries.append(boundaries[-1])
    boundaries.append(n_tokens)

    return list(zip(boundaries[:-1], boundaries[1:]))


def _field_char_spans(
    msg_dicts: list[dict[str, Any]],
    tokenizer: Any,
    formatted: str,
) -> dict[str, dict[int, tuple[int, int]]]:
    """For each field in :data:`_FIELD_ANNOTATIONS`, return a
    ``msg_idx → (char_start, char_end)`` map covering its rendered span in
    ``formatted``.

    The detection strategy mirrors :func:`tag_template_field_renderings`:
    render the chat template once with the field stripped from EVERY
    message and use :func:`_multi_diff_chunks` to find all missing
    regions. Each chunk is attributed to its source message by order.
    Falls back to per-message rendering when the chunk count doesn't
    match.
    """
    out: dict[str, dict[int, tuple[int, int]]] = {}
    for field_name in _FIELD_ANNOTATIONS:
        msgs_with_field = [
            (idx, msg) for idx, msg in enumerate(msg_dicts) if msg.get(field_name)
        ]
        if not msgs_with_field:
            continue
        batched_variant = [
            _strip_field(m, field_name) if m.get(field_name) else m for m in msg_dicts
        ]
        try:
            formatted_batched = render_chat_template(tokenizer, batched_variant)
        except (TypeError, KeyError, ValueError):
            continue
        chunks = _multi_diff_chunks(formatted, formatted_batched)
        if len(chunks) == len(msgs_with_field):
            spans_by_msg: list[tuple[int, int] | None] = list(chunks)
        else:
            spans_by_msg = []
            for msg_idx, msg in msgs_with_field:
                variant = list(msg_dicts)
                variant[msg_idx] = _strip_field(msg, field_name)
                try:
                    formatted_one = render_chat_template(tokenizer, variant)
                except (TypeError, KeyError, ValueError):
                    spans_by_msg.append(None)
                    continue
                spans_by_msg.append(_diff_span(formatted, formatted_one))
        field_map: dict[int, tuple[int, int]] = {}
        for (msg_idx, _msg), span in zip(msgs_with_field, spans_by_msg):
            if span is not None:
                field_map[msg_idx] = span
        if field_map:
            out[field_name] = field_map
    return out


def _populate_child_offsets(
    text: Text,
    msg_field_spans: dict[str, tuple[int, int]],
    char_to_token: Callable[[int], int],
) -> None:
    """Assign ``start`` / ``end`` to a message Text's children from its field
    char spans.

    ``reasoning`` and ``tool_calls`` children get their token range directly
    from the diff-derived spans; the ``content`` child takes the leftover
    range between them inside the parent's span (or the whole parent span
    when there are no other sub-blocks to displace it).
    """
    if not text.children or text.start is None or text.end is None:
        return
    msg_start, msg_end = text.start, text.end
    child_ranges: dict[str, tuple[int, int]] = {}
    for field_name, span in msg_field_spans.items():
        cs, ce = span
        ts = max(char_to_token(cs), msg_start)
        te = min(char_to_token(ce), msg_end)
        if ts < te:
            child_ranges[field_name] = (ts, te)

    reasoning_range = child_ranges.get("reasoning")
    tool_calls_range = child_ranges.get("tool_calls")

    for child in text.children:
        if child.name == "reasoning" and reasoning_range is not None:
            child.start, child.end = reasoning_range
        elif child.name == "tool_calls" and tool_calls_range is not None:
            child.start, child.end = tool_calls_range
        elif child.name == "content":
            cs = reasoning_range[1] if reasoning_range else msg_start
            ce = tool_calls_range[0] if tool_calls_range else msg_end
            if cs < ce:
                child.start, child.end = cs, ce


def _populate_text_offsets(
    texts: list[Text],
    messages: list[dict[str, Any]],
    tokens: list[TokenOrSeqRef],
    tokenizer: Any,
) -> None:
    """Set ``start`` / ``end`` on each :class:`Text` (and its children) from
    the message-token partition, when one can be computed."""
    if not texts or len(texts) != len(messages):
        return
    if not tokens or not hasattr(tokenizer, "apply_chat_template"):
        return
    try:
        formatted = render_chat_template(tokenizer, messages, tokenize=False)
    except (TypeError, KeyError, ValueError):
        return
    if not isinstance(formatted, str):
        return
    terminator = _detect_message_terminator(tokenizer, messages, formatted)
    ranges = compute_message_token_ranges(
        messages, tokens, formatted, terminator=terminator
    )
    for text, rng in zip(texts, ranges):
        if rng is None:
            continue
        text.start, text.end = rng

    needs_children = any(text.children for text in texts)
    if not needs_children:
        return

    decoded = [t.token or "" for t in tokens]
    if "".join(decoded) != formatted:
        return
    n_tokens = len(tokens)
    token_starts: list[int] = []
    pos = 0
    for s in decoded:
        token_starts.append(pos)
        pos += len(s)

    def char_to_token(char_pos: int) -> int:
        return next(
            (j for j in range(n_tokens) if token_starts[j] >= char_pos),
            n_tokens,
        )

    field_spans = _field_char_spans(messages, tokenizer, formatted)
    for msg_idx, text in enumerate(texts):
        if not text.children:
            continue
        msg_field_spans = {
            field_name: spans[msg_idx]
            for field_name, spans in field_spans.items()
            if msg_idx in spans
        }
        _populate_child_offsets(text, msg_field_spans, char_to_token)


def offset_mapping_decode(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> tuple[list[int], list[str]] | None:
    """Decode via ``return_offsets_mapping`` if the tokenizer supports it.

    Fast tokenizers (most HF-fast SentencePiece + BPE variants) expose byte
    offsets from a single ``tokenizer(text, return_offsets_mapping=True)``
    call, so we can slice per-token strings directly from the formatted chat
    template — bulletproof even when the tokenizer's decode path is lossy.

    Returns ``(ids, pieces)`` on success (``"".join(pieces) == formatted``)
    or ``None`` if offsets aren't supported / the re-tokenized ids don't
    match ``apply_chat_template(tokenize=True)`` (template-internal
    normalization). The caller should try the next tier on ``None``.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        ids_from_template = render_chat_template(tokenizer, messages, tokenize=True)
        formatted = render_chat_template(tokenizer, messages, tokenize=False)
    except (TypeError, KeyError, ValueError):
        return None

    pieces = offset_mapping_decode_text(formatted, list(ids_from_template), tokenizer)
    if pieces is None:
        return None
    return list(ids_from_template), pieces


def messages_to_tokens(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
    decode_cache: dict[int, str] | None = None,
    byte_decoder: "ByteSliceDecoder | None" = None,
    use_offset_mapping: bool = True,
) -> tuple[list[Text], list[TokenOrSeqRef]]:
    """Convert message dicts to role-named :class:`Text` segments and Tokens.

    When the tokenizer supports ``apply_chat_template``, uses it to produce the
    full token sequence including template delimiters. Otherwise falls back to
    per-message ``tokenizer.encode()``.

    Role tagging is NOT done here — it is a separate post-dedup step via
    ``tag_chat_roles_doc``.

    Returns (texts, tokens).

    Three-tier decode strategy (in order of preference):

    1. **offset_mapping** (HF fast tokenizers that support it — most fast
       SentencePiece + BPE): per-token strings are exact slices of the
       formatted chat template, so ``"".join(token.token) == formatted``
       for every sample, no matter which chars appear. Tried first; on
       mismatch with template-internal ids we move on.

    2. **byte-slice** (byte-level BPE without offset support, e.g. Kimi's
       tiktoken variant): reconstruct per-token bytes via
       ``tokenizer.byte_decoder`` + ``encoder`` (no decode calls), decode
       the whole byte stream once, attribute each char to the token whose
       byte range contains the char's last byte. Also bulletproof.

    3. **per-token decode with cache** (fallback for tokenizers with
       neither of the above — mostly old SentencePiece slow tokenizers):
       ``tokenizer.decode([tid])`` returns U+FFFD when a multi-byte char
       is split across BPE tokens. ``tag_chat_roles`` silently skips
       those samples.

    Callers supply ``byte_decoder`` (shared across samples of one archive)
    and / or ``decode_cache`` (same) to amortize per-id work.
    """
    texts = name_messages(messages)
    tokens: list[TokenOrSeqRef] = []

    if tokenizer is None:
        return texts, tokens

    if hasattr(tokenizer, "apply_chat_template"):
        if use_offset_mapping:
            result = offset_mapping_decode(messages, tokenizer)
            if result is not None:
                ids, pieces = result
                for tid, piece in zip(ids, pieces):
                    tokens.append(TokenOrSeqRef(id=tid, token=piece))
                _populate_text_offsets(texts, messages, tokens, tokenizer)
                return texts, tokens

        token_ids = render_chat_template(tokenizer, messages, tokenize=True)
        try:
            pieces = decode_token_ids(
                list(token_ids),
                tokenizer,
                decode_cache=decode_cache,
                byte_decoder=byte_decoder,
                strict_roundtrip=True,
            )
        except ValueError as e:
            raise ValueError(
                "Per-token decode does not round-trip to the full decode: "
                f"{type(tokenizer).__name__} does not support "
                "``return_offsets_mapping`` and does not expose byte-level "
                "BPE internals (``encoder`` + ``byte_decoder``), so INIF "
                "cannot produce reliable per-token strings for content "
                "containing multi-byte UTF-8 characters split across tokens. "
                "Use a tokenizer that supports either mechanism, or strip the "
                "problematic characters before tokenizing."
            ) from e
        for tid, piece in zip(token_ids, pieces):
            tokens.append(TokenOrSeqRef(id=tid, token=piece))
    else:
        for i, msg in enumerate(messages):
            text = msg["content"]
            if not text:
                continue
            start_idx = len(tokens)
            encoded = tokenizer.encode(text, add_special_tokens=False)
            for tid in encoded:
                token_str = tokenizer.decode([tid])
                tokens.append(TokenOrSeqRef(id=tid, token=token_str))
            end_idx = len(tokens)
            if start_idx != end_idx:
                texts[i].start = start_idx
                texts[i].end = end_idx

    if hasattr(tokenizer, "apply_chat_template"):
        _populate_text_offsets(texts, messages, tokens, tokenizer)

    return texts, tokens


def find_message_token_range(
    tokens: list[TokenOrSeqRef],
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
    role: str = "assistant",
    which: str = "last",
) -> tuple[int, int] | None:
    """Locate the [start, end) token indices covering a message's content.

    ``which`` selects between the first / last occurrence of the role in
    ``msg_dicts``. Returns ``None`` when the formatted string can't be
    reconstructed from the token sequence (e.g. a tokenizer that drops bytes),
    or when the content can't be located unambiguously.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    assert which in ("first", "last"), f"which must be 'first' or 'last', got {which!r}"
    iterator = reversed(msg_dicts) if which == "last" else iter(msg_dicts)
    content: str | None = None
    for msg in iterator:
        if msg.get("role") == role:
            c = msg.get("content")
            if c:
                content = c
                break
    if content is None:
        return None

    formatted = render_chat_template(tokenizer, msg_dicts)
    decoded = [t.token or "" for t in tokens]
    if "".join(decoded) != formatted:
        return None

    char_start = (
        formatted.rfind(content) if which == "last" else formatted.find(content)
    )
    if char_start < 0:
        return None
    char_end = char_start + len(content)

    pos = 0
    tok_start: int | None = None
    tok_end: int | None = None
    for i, s in enumerate(decoded):
        s_start, s_end = pos, pos + len(s)
        if tok_start is None and s_end > char_start:
            tok_start = i
        if s_start < char_end:
            tok_end = i + 1
        pos = s_end
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


def tag_char_span(
    sample: Sample,
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
    text: str,
    tag: str,
) -> bool:
    """Tag every token whose decoded piece overlaps ``text`` inside the
    tokenizer's ``apply_chat_template`` rendering of ``msg_dicts``.

    Returns True on success, False when the token stream doesn't round-trip to
    the formatted string or ``text`` isn't found.
    """
    if not text or not hasattr(tokenizer, "apply_chat_template"):
        return False
    formatted = render_chat_template(tokenizer, msg_dicts)
    decoded = [t.token or "" for t in sample.tokens]
    if "".join(decoded) != formatted:
        return False
    char_start = formatted.find(text)
    if char_start < 0:
        return False
    char_end = char_start + len(text)
    positions: list[int] = []
    pos = 0
    for i, s in enumerate(decoded):
        s_start, s_end = pos, pos + len(s)
        if s_end > char_start and s_start < char_end:
            positions.append(i)
        pos = s_end
    sample.annotate_positions(tag, positions)
    return True


# ---------------------------------------------------------------------------
# Reasoning-block tagging
# ---------------------------------------------------------------------------


def _expanded_token_offsets(
    sample: Sample, sequences: list[Sequence] | None
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(decoded_strings, char_offsets)`` for every expanded token."""
    expanded = (
        sample.get_expanded_tokens(sequences) if sequences else list(sample.tokens)
    )
    decoded = [t.token or "" for t in expanded]
    offsets: list[tuple[int, int]] = []
    pos = 0
    for s in decoded:
        offsets.append((pos, pos + len(s)))
        pos += len(s)
    return decoded, offsets


def _expanded_to_native_positions(
    sample: Sample,
    sequences: list[Sequence] | None,
    expanded_positions: set[int],
) -> list[int]:
    """Map a set of expanded-token positions back to ``sample.tokens`` indices.

    A sequence-ref token in ``sample.tokens`` corresponds to multiple expanded
    tokens; the ref is included in the result whenever ANY of its expanded
    children fall inside ``expanded_positions``. (We accept the imprecision —
    the alternative would be to materialize every reasoning-touched ref, which
    defeats the point of dedup.)
    """
    if not expanded_positions:
        return []
    if not sequences:
        return sorted(p for p in expanded_positions if 0 <= p < len(sample.tokens))
    seq_map = {s.id: s for s in sequences}
    out: list[int] = []
    exp_idx = 0
    for native_idx, tok in enumerate(sample.tokens):
        if tok.is_sequence_ref:
            n = seq_map[tok.token].n_tokens
            if any(p in expanded_positions for p in range(exp_idx, exp_idx + n)):
                out.append(native_idx)
            exp_idx += n
        else:
            if exp_idx in expanded_positions:
                out.append(native_idx)
            exp_idx += 1
    return out


def _materialize_inconsistent_refs(
    sample: Sample,
    sequences: list[Sequence] | None,
    sig_per_expanded_pos: dict[int, Any],
) -> None:
    """Expand sequence refs in ``sample.tokens`` whose expanded positions
    don't all share the same ``sig``.

    A token annotation is per-position by design, but sequence refs collapse
    multiple positions into a single native index — so a ref that straddles
    a tag boundary (e.g. the end of a reasoning block and the start of a
    tool-call section) cannot be tagged correctly without expanding it
    first. This helper finds such refs and replaces them in-place with
    their expanded tokens, preserving the rest of the document's dedup.

    ``sig_per_expanded_pos`` maps expanded-position index to a hashable
    signature (e.g. a frozenset of (annotation_name, role) tuples, or a
    role name); a ref is materialized when any two of its expanded
    positions have different sigs (positions absent from the dict are
    treated as ``None`` and contribute to the comparison).
    """
    if not sequences:
        return
    seq_map = {s.id: s for s in sequences}

    refs_to_materialize: list[int] = []
    exp_idx = 0
    for native_idx, tok in enumerate(sample.tokens):
        if tok.is_sequence_ref:
            n = seq_map[tok.token].n_tokens
            sigs = {sig_per_expanded_pos.get(p) for p in range(exp_idx, exp_idx + n)}
            if len(sigs) > 1:
                refs_to_materialize.append(native_idx)
            exp_idx += n
        else:
            exp_idx += 1

    # Materialize from the highest index downward so earlier indices stay valid.
    for native_idx in reversed(refs_to_materialize):
        ref_tok = sample.tokens[native_idx]
        seq = seq_map[ref_tok.token]
        sample._replace_token_with_tokens(
            native_idx,
            [TokenOrSeqRef(id=t.id, token=t.token) for t in seq.tokens],
        )


# Fields whose chat-template rendering we want to surface as INIF annotations,
# mapped to the annotation name. Adding a new entry here is enough to make
# ``tag_template_field_renderings`` annotate that field's rendered tokens too.
_FIELD_ANNOTATIONS: dict[str, str] = {
    "reasoning": "reasoning",
    "tool_calls": "tool_call",
}

# Aliases that the tokenizer chat template might read for the same logical
# field — when stripping a field for the diff probe we drop every alias so the
# template can't fall back to one of them.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "reasoning": ("reasoning", "reasoning_content"),
    "tool_calls": ("tool_calls",),
}


def _strip_field(msg: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Return a shallow copy of ``msg`` with ``field_name`` (and its aliases)
    removed."""
    aliases = _FIELD_ALIASES.get(field_name, (field_name,))
    return {k: v for k, v in msg.items() if k not in aliases}


def _multi_diff_chunks(
    full: str, stripped: str, anchor_len: int = 32
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` ranges in ``full`` that are missing from
    ``stripped`` (the equivalent rendering with one or more field values
    removed). One chunk per missing region.

    Walks both strings forward. At every divergence point, takes the next
    ``anchor_len`` chars from ``stripped`` as a re-alignment anchor and
    searches forward in ``full`` for it — those intermediate chars are the
    chunk. O(n) on typical inputs; degrades to O(n × anchor_len) only when
    every divergence requires a long forward search.

    The anchor length needs to be longer than any plausible run of chars that
    coincidentally repeats inside a field rendering AND also appears at the
    next static-template segment. 32 chars covers the typical separators
    between field renderings (chat-template role markers,
    ``<|im_end|><|im_assistant|>...`` etc.).
    """
    if full == stripped:
        return []
    spans: list[tuple[int, int]] = []
    i, j = 0, 0
    n_full, n_strip = len(full), len(stripped)
    while i < n_full and j < n_strip:
        if full[i] == stripped[j]:
            i += 1
            j += 1
            continue
        chunk_start = i
        # Use up to anchor_len chars from stripped for re-alignment. If less
        # is available (we're near the end of stripped), use whatever's left;
        # the find call will still uniquely pin down the next alignment as
        # long as that suffix doesn't coincidentally occur earlier in full.
        remaining = n_strip - j
        anchor = stripped[j : j + min(anchor_len, remaining)]
        if not anchor:
            spans.append((chunk_start, n_full))
            i = n_full
            break
        next_full = full.find(anchor, i)
        if next_full < 0:
            spans.append((chunk_start, n_full))
            i = n_full
            break
        spans.append((chunk_start, next_full))
        i = next_full  # don't advance j — we matched
    if i < n_full:
        spans.append((i, n_full))
    return spans


def _diff_span(formatted_full: str, formatted_without: str) -> tuple[int, int] | None:
    """Single-bounding-box diff (longest-common-prefix + suffix).

    Used as a fallback when the batched multi-diff produces a chunk count
    that doesn't match the number of messages with the field — happens when
    two field renderings happen to be adjacent in the formatted output, or
    when an anchor coincidentally matches inside a field value.
    """
    if formatted_full == formatted_without:
        return None
    n_full, n_without = len(formatted_full), len(formatted_without)
    n_min = min(n_full, n_without)

    n_pref = 0
    while n_pref < n_min and formatted_full[n_pref] == formatted_without[n_pref]:
        n_pref += 1

    n_suf = 0
    while (
        n_suf < n_min - n_pref
        and formatted_full[n_full - 1 - n_suf]
        == formatted_without[n_without - 1 - n_suf]
    ):
        n_suf += 1

    diff_start = n_pref
    diff_end = n_full - n_suf
    if diff_start >= diff_end:
        return None
    return diff_start, diff_end


def tag_template_field_renderings(
    sample: Sample,
    msg_dicts: list[dict[str, Any]],
    tokenizer: Any,
    sequences: list[Sequence] | None = None,
) -> int:
    """Annotate the rendered chars of each per-message structured field.

    For each field in :data:`_FIELD_ANNOTATIONS`, the helper renders the
    chat template ONCE with that field stripped from EVERY message, then
    uses :func:`_multi_diff_chunks` to find all the missing regions in one
    pass. Each chunk is attributed to its source message by order (chunk
    ``i`` corresponds to the ``i``-th message that had the field). When the
    chunk count doesn't match the number of messages with the field
    (e.g. two adjacent renderings collapse into one chunk, or an anchor
    coincidentally matches inside a value), the field falls back to
    per-message rendering for safety.

    Total renders per sample: ``1 + len(fields_present)`` in the happy
    path, instead of ``len(messages_with_fields)`` of the previous design
    — for a 65-message agentic trace with reasoning + tool_calls, that's
    ~3 renders instead of ~33.

    Tokens in each chunk get:

    - The field's annotation name (``reasoning`` / ``tool_call``).
    - The message's role (``assistant``) — moved out of ``template`` if
      ``tag_chat_roles`` already labelled the chars there.

    Fully model-agnostic: the chat template itself decides where the field
    renders and what wrapper (if any) it adds. For ``tool_calls`` the diff
    naturally captures the entire
    ``<|tool_calls_section_begin|>…<|tool_calls_section_end|>`` block
    because the wrapper only emits when the field is present. For
    ``reasoning`` (Kimi/Qwen-style ``<think>{rc}</think>`` always-on
    wrapper) the diff is just the reasoning content — the wrapper markers
    are static and stay tagged as ``template``.

    Returns the number of (msg, field) renderings successfully annotated.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return 0

    try:
        formatted = render_chat_template(tokenizer, msg_dicts)
    except (TypeError, KeyError, ValueError):
        return 0

    decoded, offsets = _expanded_token_offsets(sample, sequences)
    if "".join(decoded) != formatted:
        return 0

    n_tagged = 0
    # Per-expanded-position tag bag: each entry is a set of (ann_name, role)
    # tuples. The signature of a position (frozenset of these tuples) is used
    # both to drive sequence-ref materialization decisions AND to apply the
    # tags consistently at the native level afterward.
    tags_per_pos: dict[int, set[tuple[str, str | None]]] = {}

    for field_name, ann_name in _FIELD_ANNOTATIONS.items():
        msgs_with_field = [
            (idx, msg) for idx, msg in enumerate(msg_dicts) if msg.get(field_name)
        ]
        if not msgs_with_field:
            continue

        # Batched render: strip THIS field from every message at once.
        batched_variant = [
            _strip_field(m, field_name) if m.get(field_name) else m for m in msg_dicts
        ]
        try:
            formatted_batched = render_chat_template(tokenizer, batched_variant)
        except (TypeError, KeyError, ValueError):
            continue

        chunks = _multi_diff_chunks(formatted, formatted_batched)
        spans_by_msg: list[tuple[int, int] | None]
        if len(chunks) == len(msgs_with_field):
            spans_by_msg = list(chunks)
        else:
            # Chunk count mismatch — re-render per message rather than
            # misattribute a chunk to the wrong sender.
            spans_by_msg = []
            for msg_idx, msg in msgs_with_field:
                variant = list(msg_dicts)
                variant[msg_idx] = _strip_field(msg, field_name)
                try:
                    formatted_one = render_chat_template(tokenizer, variant)
                except (TypeError, KeyError, ValueError):
                    spans_by_msg.append(None)
                    continue
                spans_by_msg.append(_diff_span(formatted, formatted_one))

        for (msg_idx, msg), span in zip(msgs_with_field, spans_by_msg):
            if span is None:
                continue
            start, end = span
            role = msg.get("role")
            for i, (s, e) in enumerate(offsets):
                if s >= end or e <= start:
                    continue
                tags_per_pos.setdefault(i, set()).add((ann_name, role))
            n_tagged += 1

    if n_tagged == 0:
        return 0

    # Sequence refs whose expanded positions have inconsistent tag signatures
    # must be materialized — otherwise the entire ref would be over-tagged
    # with the union of every annotation that touches any of its positions
    # (e.g. a ref straddling the end of a reasoning block and the start of a
    # tool-call section would receive both `reasoning` AND `tool_call`).
    sigs_per_pos = {p: frozenset(tags) for p, tags in tags_per_pos.items()}
    _materialize_inconsistent_refs(sample, sequences, sigs_per_pos)

    # After materialization the native indices may have shifted; build a
    # fresh native_idx -> first_expanded_pos map. Every remaining ref now has
    # uniform tags across its expanded positions, so the first one's tag bag
    # correctly represents the whole native token.
    seq_map = {s.id: s for s in sequences} if sequences else {}
    native_to_first_exp: dict[int, int] = {}
    exp_idx = 0
    for native_idx, tok in enumerate(sample.tokens):
        native_to_first_exp[native_idx] = exp_idx
        if tok.is_sequence_ref:
            exp_idx += seq_map[tok.token].n_tokens
        else:
            exp_idx += 1

    field_positions: dict[str, set[int]] = {}
    role_positions: dict[str, set[int]] = {}
    for native_idx, first_exp in native_to_first_exp.items():
        for ann_name, role in tags_per_pos.get(first_exp, ()):
            field_positions.setdefault(ann_name, set()).add(native_idx)
            if role:
                role_positions.setdefault(role, set()).add(native_idx)

    for ann_name, positions in field_positions.items():
        if positions:
            sample.annotate_positions(
                ann_name, sorted(positions), metadata={"source": ann_name}
            )

    for role, positions in role_positions.items():
        if positions:
            _move_positions_to_role(sample, role, sorted(positions))

    return n_tagged


def _move_positions_to_role(
    sample: Sample,
    role: str,
    positions: list[int],
) -> None:
    """Add ``positions`` to the ``role`` chat-role annotation, removing them
    from any other role/template annotation that previously claimed them.

    The chat-role annotations are mutually exclusive (a token belongs to
    exactly one role); the template-rendered chars for fields like
    ``tool_calls`` start out tagged as ``template`` (since they sit outside
    the message ``content`` text), but conceptually belong to the message's
    role. This helper migrates them.
    """
    if not positions:
        return
    pos_set = set(positions)
    role_md = {"source": "message_role"}

    # Strip the positions from any other chat-role annotation that owns them.
    for ann in list(sample.annotations):
        if ann.metadata != role_md or ann.name == role:
            continue
        new_ranges: list[tuple[int, int]] = []
        changed = False
        for start, end in ann.ranges:
            for p in range(start, end):
                if p in pos_set:
                    changed = True
                    continue
                new_ranges.append((p, p + 1))
        if not changed:
            continue
        if new_ranges:
            from inif.models import _merge_ranges  # late import: avoid cycle

            ann.ranges = _merge_ranges(new_ranges)
        else:
            sample.annotations.remove(ann)

    sample.annotate_positions(role, positions, metadata=role_md)


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------

# Routing prefixes some eval frameworks prepend to model ids (e.g. Inspect AI's
# ``together/moonshotai/Kimi-K2.5``). Stripped before handing the id to
# ``AutoTokenizer.from_pretrained`` so the underlying HF id resolves cleanly.
# Keyed on a small allowlist to avoid chopping a legitimate org from a HF id
# (``meta-llama/Llama-3.1-8B`` must NOT lose its first component).
_KNOWN_PROVIDERS = frozenset(
    {
        "azureml",
        "anthropic",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks",
        "google",
        "groq",
        "hf",
        "huggingface",
        "mistral",
        "mockllm",
        "ollama",
        "openai",
        "perplexity",
        "replicate",
        "together",
        "vertex",
        "vllm",
    }
)


def _strip_provider_prefix(model_id: str) -> str:
    """Strip a routing prefix like ``together/`` from a model id.

    Only strips when the head matches the provider allowlist AND the tail
    still contains a ``/``, so HF-format ids like ``meta-llama/Llama-3.1-8B``
    are left untouched.
    """
    if "/" not in model_id:
        return model_id
    head, _, rest = model_id.partition("/")
    if head.lower() in _KNOWN_PROVIDERS and "/" in rest:
        return rest
    return model_id


def _ids_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return _strip_provider_prefix(a) == _strip_provider_prefix(b)


def _warn_tokenizer_mismatch(tokenizer: Any, source_model_id: str | None) -> None:
    if not source_model_id:
        return
    tokenizer_id = getattr(tokenizer, "name_or_path", None)
    if not tokenizer_id or _ids_match(tokenizer_id, source_model_id):
        return
    warnings.warn(
        f"Tokenizer mismatch: tokenizer ({tokenizer_id!r}) differs from the "
        f"source eval's model id ({source_model_id!r}). Token-level data "
        "will reflect the supplied tokenizer, not the original model. Pass "
        '``tokenizer="auto"`` to auto-load the matching tokenizer, or '
        "ignore this warning if the mismatch is intentional.",
        UserWarning,
        stacklevel=3,
    )


def resolve_tokenizer(tokenizer: Any, source_model_id: str | None) -> Any:
    """Resolve the tokenizer used for INIF conversion.

    Tokens are a load-bearing invariant of every INIF sample, so this never
    returns ``None``: callers always get back a usable tokenizer or a
    ``ValueError`` explaining why one couldn't be obtained.

    - ``tokenizer="auto"`` (the default for converter entry points) and
      ``tokenizer=None`` (treated as ``"auto"`` for ergonomics):
      ``AutoTokenizer.from_pretrained(source_model_id)`` with the routing
      prefix stripped (``together/`` etc.). Raises ``ValueError`` if
      ``source_model_id`` is missing or the load fails (closed-source ids
      like ``openai/gpt-4`` will hit this path).
    - ``tokenizer`` is a string other than ``"auto"``: load it via
      ``AutoTokenizer.from_pretrained``, then warn (but don't fail) if its
      id disagrees with ``source_model_id``.
    - ``tokenizer`` is a tokenizer instance: use as-is, warn (but don't fail)
      if its ``name_or_path`` disagrees with ``source_model_id``.
    """
    if tokenizer is None or (isinstance(tokenizer, str) and tokenizer == "auto"):
        if not source_model_id:
            raise ValueError(
                "Cannot auto-load a tokenizer: no model id is available in "
                "the source. Pass ``tokenizer=<id-or-instance>`` explicitly."
            )
        from transformers import AutoTokenizer

        canonical_id = _strip_provider_prefix(source_model_id)
        try:
            return AutoTokenizer.from_pretrained(canonical_id, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f"Could not auto-load a tokenizer for model id "
                f"{source_model_id!r} (canonical: {canonical_id!r}). This is "
                "common for closed-source models (OpenAI, Anthropic, …). "
                "Pass ``tokenizer=<hf-id-or-instance>`` with an HF tokenizer "
                "that approximates the source model — token-level data will "
                "then be approximate but tokens will still be produced. "
                f"Underlying error: {e}"
            ) from e

    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        _warn_tokenizer_mismatch(loaded, source_model_id)
        return loaded

    _warn_tokenizer_mismatch(tokenizer, source_model_id)
    return tokenizer
