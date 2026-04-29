from __future__ import annotations

from inif.models import InifDocument, Sample, Sequence, TokenOrSeqRef


def _ngram_positions(tokens: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    """Map each length-n contiguous n-gram in ``tokens`` to its start positions."""
    positions: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(tokens) - n + 1):
        positions.setdefault(tuple(tokens[i : i + n]), []).append(i)
    return positions


def _find_common_contiguous(
    token_lists: list[list[str]], min_length: int
) -> list[list[str]]:
    """Find contiguous token subsequences (length ≥ ``min_length``) common to
    ALL ``token_lists``.

    Algorithm: index only length-``min_length`` n-grams per sample (O(n)
    memory per sample, not O(n^2)), intersect the keys across samples to get
    candidate anchors, then greedily extend each sample-0 occurrence rightward
    while every other sample has an occurrence of the current n-gram whose
    next token matches. Candidate positions are
    filtered as we extend, so total work is bounded by the matched length
    summed across candidates.
    """
    if not token_lists or min_length < 1:
        return []
    if any(len(tl) < min_length for tl in token_lists):
        return []

    k = len(token_lists)
    per_sample = [_ngram_positions(tl, min_length) for tl in token_lists]

    common_keys = set(per_sample[0].keys())
    for pm in per_sample[1:]:
        common_keys &= pm.keys()
        if not common_keys:
            return []

    tl0 = token_lists[0]
    results: set[tuple[str, ...]] = set()
    for ngram in common_keys:
        for start0 in per_sample[0][ngram]:
            # Current matching positions for samples 1..k-1. Sample 0 is anchored
            # at ``start0`` — we only need the other samples' cursors.
            others = [list(per_sample[i][ngram]) for i in range(1, k)]
            length = min_length
            while start0 + length < len(tl0):
                next_tok = tl0[start0 + length]
                new_others: list[list[int]] = []
                all_extend = True
                for i, positions in enumerate(others, start=1):
                    tl = token_lists[i]
                    kept = [
                        p
                        for p in positions
                        if p + length < len(tl) and tl[p + length] == next_tok
                    ]
                    if not kept:
                        all_extend = False
                        break
                    new_others.append(kept)
                if not all_extend:
                    break
                others = new_others
                length += 1
            results.add(tuple(tl0[start0 : start0 + length]))

    return [list(s) for s in results]


def _filter_maximal_sequences(seqs: list[list[str]]) -> list[list[str]]:
    """Keep only the longest non-overlapping sequences (joined substring check)."""
    if not seqs:
        return []
    # Sort by length descending
    sorted_seqs = sorted(seqs, key=lambda s: len(s), reverse=True)
    kept: list[list[str]] = []
    for seq in sorted_seqs:
        joined = "\x00".join(seq)
        is_sub = False
        for existing in kept:
            existing_joined = "\x00".join(existing)
            if joined in existing_joined:
                is_sub = True
                break
        if not is_sub:
            kept.append(seq)
    return kept


def _has_extra_fields(token: TokenOrSeqRef) -> bool:
    """Check if a token has extra fields (interpretability data attached)."""
    return bool(token.model_extra)


def _clean_pairs(
    sample: Sample,
    sample_key: object = None,
) -> list[tuple[str, int]]:
    """Project a sample's tokens to (string, id) pairs, dropping refs and tokens
    with extras. This matches the projection used by ``_find_common_contiguous``.

    When ``sample_key`` is not ``None``, a unique-per-sample marker is
    inserted BEFORE the first token of every chat message (every position
    in ``sample.texts`` whose ``start > 0``). The marker breaks contiguity
    in the projection: an n-gram that would have crossed a message
    boundary now contains the marker, and because the marker carries
    ``sample_key`` it can never appear in any other sample's projection
    — so the cross-boundary n-gram drops out of the all-samples
    intersection that powers dedup. Within-message runs (and runs that
    *start* at a boundary token) don't include the marker and dedup
    normally.
    """
    annotated_positions: set[int] = set()
    for annotation in sample.annotations:
        for start, end in annotation.ranges:
            annotated_positions.update(range(start, end))
    boundary_positions: set[int] = set()
    if sample_key is not None:
        for text in sample.texts:
            if text.start is not None and text.start > 0:
                boundary_positions.add(text.start)
    pairs: list[tuple[str, int]] = []
    for i, t in enumerate(sample.tokens):
        if i in boundary_positions:
            pairs.append((f"\x01BOUNDARY-{sample_key}-{i}\x01", -2))
        if i in annotated_positions or t.is_sequence_ref or _has_extra_fields(t):
            continue
        # Vocab tokens always have an int id (guaranteed by ``is_sequence_ref``
        # being False above); the assertion is for the type checker.
        assert t.id is not None
        pairs.append((t.token or "", t.id))
    return pairs


def _capture_ids_for_sequence(sample: Sample, seq_strs: list[str]) -> list[int] | None:
    """Find the first contiguous (clean) window in ``sample`` matching
    ``seq_strs`` and return the corresponding token ids. Returns ``None`` if no
    match (which shouldn't happen for sequences produced by the dedup
    intersection algorithm)."""
    pairs = _clean_pairs(sample)
    n = len(seq_strs)
    for start in range(len(pairs) - n + 1):
        window = pairs[start : start + n]
        if [s for s, _ in window] == seq_strs:
            return [i for _, i in window]
    return None


def _replace_sequences_in_tokens(
    tokens: list[TokenOrSeqRef],
    sequences: list[Sequence],
    blocked_positions: set[int] | None = None,
    boundary_positions: set[int] | None = None,
) -> tuple[list[TokenOrSeqRef], list[tuple[int, int]]]:
    """Greedy longest-match replacement of token sequences with refs.

    A window is only replaced when both the token strings AND ids match the
    sequence — this guards against (rare) cases where two samples share a
    string subsequence but have different ids for those positions.

    ``boundary_positions`` (typically the ``Text.start`` positions of every
    chat message after the first) gates matches: a candidate window
    ``[i, i+L)`` is rejected when any *internal* position
    ``i+1 .. i+L-1`` is a boundary, so a single replacement run can never
    span a message boundary. Matches that *start* at a boundary token
    are still allowed (the run begins with the next message's leading
    delimiters and stays inside that message).

    Optimized: index sequences by their first (token, id) pair so the common
    case (no match) is O(1) per token instead of O(n_sequences).
    """
    if not sequences:
        return list(tokens), [(i, i + 1) for i in range(len(tokens))]

    blocked_positions = blocked_positions or set()
    boundary_positions = boundary_positions or set()

    # Sort descending by length so longer matches win over shorter ones that
    # share a prefix.
    sorted_seqs = sorted(sequences, key=lambda s: s.n_tokens, reverse=True)

    # Pre-extract per-sequence parallel token-string and id arrays for fast
    # comparison (avoids attribute lookups in the hot loop). Sequence tokens
    # are always vocab tokens (id is int), so the id arrays carry no None.
    seq_info: list[tuple[Sequence, list[str], list[int | None], int]] = [
        (s, [t.token for t in s.tokens], [t.id for t in s.tokens], s.n_tokens)
        for s in sorted_seqs
    ]

    # Index by (first_token, first_id) so we only enter the match loop when
    # the current position could possibly start one of our sequences.
    first_index: dict[tuple[str, int | None], list[int]] = {}
    for idx, (_, toks, ids, _) in enumerate(seq_info):
        first_index.setdefault((toks[0], ids[0]), []).append(idx)

    new_tokens: list[TokenOrSeqRef] = []
    old_to_new: list[tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        # Ref tokens and extras-carrying tokens never start a match.
        if i in blocked_positions or tok.is_sequence_ref or _has_extra_fields(tok):
            new_pos = len(new_tokens)
            new_tokens.append(tok)
            old_to_new.append((new_pos, new_pos + 1))
            i += 1
            continue
        key = (tok.token, tok.id)
        candidates = first_index.get(key)
        if not candidates:
            new_pos = len(new_tokens)
            new_tokens.append(tok)
            old_to_new.append((new_pos, new_pos + 1))
            i += 1
            continue

        matched = False
        for cand_idx in candidates:
            seq, seq_toks, seq_ids, seq_len = seq_info[cand_idx]
            if i + seq_len > n:
                continue
            # A run that crosses a message boundary internally would put
            # role markers from the next message inside the previous
            # message's range; skip such candidates.
            if any(p in boundary_positions for p in range(i + 1, i + seq_len)):
                continue
            # Walk the window, bailing out early on mismatch / disqualifier.
            ok = True
            for j in range(seq_len):
                w = tokens[i + j]
                if (
                    (i + j) in blocked_positions
                    or w.is_sequence_ref
                    or _has_extra_fields(w)
                ):
                    ok = False
                    break
                if w.token != seq_toks[j] or w.id != seq_ids[j]:
                    ok = False
                    break
            if ok:
                new_pos = len(new_tokens)
                new_tokens.append(TokenOrSeqRef(id=None, token=seq.id))
                for _ in range(seq_len):
                    old_to_new.append((new_pos, new_pos + 1))
                i += seq_len
                matched = True
                break
        if not matched:
            new_pos = len(new_tokens)
            new_tokens.append(tok)
            old_to_new.append((new_pos, new_pos + 1))
            i += 1
    return new_tokens, old_to_new


def _annotation_positions(sample: Sample) -> set[int]:
    positions: set[int] = set()
    for annotation in sample.annotations:
        for start, end in annotation.ranges:
            positions.update(range(start, end))
    return positions


def _remap_sample_positions(
    sample: Sample,
    old_to_new: list[tuple[int, int]],
) -> None:
    """Remap every position-bearing field on ``sample`` after a token-list
    rewrite.

    ``old_to_new[i] = (new_start, new_end)`` is the new position range
    occupied by the token that was at index ``i`` in the old layout. We use
    it to translate annotations (token-range labels) and the per-text
    ``[start, end)`` boundaries so they keep referring to the same
    underlying chars after dedup / expand / materialization.

    Per-text offsets keep their adjacency invariant after dedup: when the
    boundary between two adjacent texts falls inside a sequence ref (the
    ref now spans both messages), the ref is attributed to the FIRST text
    and the next text starts after it. Otherwise the partition would
    overlap, which breaks the per-message viewer's eye-toggle range.
    """
    for annotation in sample.annotations:
        ranges: list[tuple[int, int]] = []
        for start, end in annotation.ranges:
            ranges.append((old_to_new[start][0], old_to_new[end - 1][1]))
        annotation.ranges = ranges
    if not old_to_new:
        return
    new_n = old_to_new[-1][1]
    prev_end = 0
    for text in sample.texts:
        if text.start is None or text.end is None:
            continue
        if text.start < len(old_to_new):
            new_start = old_to_new[text.start][0]
        else:
            new_start = new_n
        if text.end == 0:
            new_end = 0
        elif text.end - 1 < len(old_to_new):
            new_end = old_to_new[text.end - 1][1]
        else:
            new_end = new_n
        new_start = max(new_start, prev_end)
        new_end = max(new_end, new_start)
        text.start = new_start
        text.end = new_end
        prev_end = new_end


def _deduplicate_sequences(
    doc: InifDocument,
    min_length: int = 5,
) -> InifDocument:
    """Implementation backing :meth:`InifDocument.deduplicate_sequences`.

    Finds token sequences common to ALL samples (intersection) and replaces
    them with refs.

    Skips tokens with extra fields (have interpretability data attached).
    Captured Sequences store both the token strings and the original token ids
    so a downstream expansion (e.g. when interpretability outputs are attached
    to ref-internal positions) can restore the exact ids.

    Returns a new document; the input is not modified, and mutable token state
    is not shared with the returned document.
    """
    doc = doc.model_copy(deep=True)

    if not doc.samples:
        # Nothing to do, but return a new doc to preserve the "no mutation"
        # contract callers rely on.
        return doc

    token_lists = [
        [s for s, _ in _clean_pairs(sample, sample_key=i)]
        for i, sample in enumerate(doc.samples)
    ]

    common = _find_common_contiguous(token_lists, min_length)
    if not common:
        return doc

    maximal = _filter_maximal_sequences(common)
    if not maximal:
        return doc

    existing_ids = {s.id for s in doc.sequences}
    new_sequences: list[Sequence] = []
    next_idx = 0
    for seq_tokens in maximal:
        seq_ids = _capture_ids_for_sequence(doc.samples[0], seq_tokens)
        if seq_ids is None:
            # Should be unreachable: maximal sequences are present in every
            # sample's clean projection by construction.
            continue

        while f"seq_{next_idx}" in existing_ids:
            next_idx += 1
        seq_id = f"seq_{next_idx}"
        existing_ids.add(seq_id)

        new_sequences.append(
            Sequence(
                id=seq_id,
                n_tokens=len(seq_tokens),
                tokens=[
                    TokenOrSeqRef(id=tid, token=tstr)
                    for tid, tstr in zip(seq_ids, seq_tokens)
                ],
            )
        )
        next_idx += 1

    all_sequences = list(doc.sequences) + new_sequences

    new_samples = []
    for sample in doc.samples:
        boundary_positions = {
            text.start
            for text in sample.texts
            if text.start is not None and text.start > 0
        }
        tokens, old_to_new = _replace_sequences_in_tokens(
            sample.tokens,
            all_sequences,
            blocked_positions=_annotation_positions(sample),
            boundary_positions=boundary_positions,
        )
        new_sample = sample.model_copy(update={"tokens": tokens}, deep=True)
        _remap_sample_positions(new_sample, old_to_new)
        new_samples.append(new_sample)

    return doc.model_copy(update={"samples": new_samples, "sequences": all_sequences})


def _expand_sequences(doc: InifDocument) -> InifDocument:
    """Implementation backing :meth:`InifDocument.expand_sequences`.

    Returns a new document whose tokens are independent from ``doc``. The
    returned document drops the sequence list and emits each materialised
    token as a plain vocab token — running
    :meth:`InifDocument.deduplicate_sequences` again will rediscover the same
    shared runs.
    """
    doc = doc.model_copy(deep=True)
    seq_map = doc.sequence_map

    new_samples: list[Sample] = []
    for sample in doc.samples:
        new_tokens: list[TokenOrSeqRef] = []
        old_to_new: list[tuple[int, int]] = []
        for token in sample.tokens:
            new_start = len(new_tokens)
            if not token.is_sequence_ref:
                new_tokens.append(token)
                old_to_new.append((new_start, len(new_tokens)))
                continue
            assert token.token in seq_map, f"Sequence '{token.token}' not found"
            seq = seq_map[token.token]
            for t in seq.tokens:
                new_tokens.append(TokenOrSeqRef(id=t.id, token=t.token))
            old_to_new.append((new_start, len(new_tokens)))
        new_sample = sample.model_copy(update={"tokens": new_tokens}, deep=True)
        _remap_sample_positions(new_sample, old_to_new)
        new_samples.append(new_sample)

    return doc.model_copy(update={"samples": new_samples, "sequences": []})
