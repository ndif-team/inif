from __future__ import annotations

from inif.models import InifDocument, Sample, Sequence, Token


def _get_contiguous_sequences(
    tokens: list[str], min_length: int
) -> set[tuple[str, ...]]:
    """Extract all contiguous token subsequences of at least min_length."""
    seqs: set[tuple[str, ...]] = set()
    for length in range(min_length, len(tokens) + 1):
        for start in range(len(tokens) - length + 1):
            seqs.add(tuple(tokens[start : start + length]))
    return seqs


def _find_common_contiguous(
    token_lists: list[list[str]], min_length: int
) -> list[list[str]]:
    """Find contiguous token sequences common to ALL token lists (intersection)."""
    if not token_lists:
        return []
    common = _get_contiguous_sequences(token_lists[0], min_length)
    for tl in token_lists[1:]:
        common &= _get_contiguous_sequences(tl, min_length)
        if not common:
            return []
    return [list(s) for s in common]


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


def _has_extra_fields(token: Token) -> bool:
    """Check if a token has extra fields (interpretability data attached)."""
    return bool(token.model_extra)


def _clean_pairs(sample: Sample) -> list[tuple[str, int]]:
    """Project a sample's tokens to (string, id) pairs, dropping refs and tokens
    with extras. This matches the projection used by ``_find_common_contiguous``."""
    return [
        (t.token or "", t.id)
        for t in sample.tokens
        if not t.is_sequence_ref and not _has_extra_fields(t)
    ]


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
    tokens: list[Token],
    sequences: list[Sequence],
) -> list[Token]:
    """Greedy longest-match replacement of token sequences with refs.

    A window is only replaced when both the token strings AND ids match the
    sequence — this guards against (rare) cases where two samples share a
    string subsequence but have different ids for those positions.
    """
    sorted_seqs = sorted(sequences, key=lambda s: s.n_tokens, reverse=True)

    new_tokens: list[Token] = []
    i = 0
    while i < len(tokens):
        matched = False
        for seq in sorted_seqs:
            seq_len = seq.n_tokens
            if i + seq_len > len(tokens):
                continue
            window = tokens[i : i + seq_len]
            if any(_has_extra_fields(t) or t.is_sequence_ref for t in window):
                continue
            if [t.token for t in window] != [s.token for s in seq.tokens]:
                continue
            if [t.id for t in window] != [s.id for s in seq.tokens]:
                continue
            new_tokens.append(Token(id=-1, sequence_id=seq.id))
            i += seq_len
            matched = True
            break
        if not matched:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def deduplicate_sequences(
    doc: InifDocument,
    min_length: int = 3,
) -> InifDocument:
    """Find sequences common to ALL samples (intersection), replace with refs.

    Skips tokens with extra fields (have interpretability data attached).
    Captured Sequences store both the token strings and the original token ids
    so a downstream expansion (e.g. when interpretability outputs are attached
    to ref-internal positions) can restore the exact ids.
    """
    doc = doc.model_copy(deep=True)

    if not doc.samples:
        return doc

    token_lists = [[s for s, _ in _clean_pairs(sample)] for sample in doc.samples]

    common = _find_common_contiguous(token_lists, min_length)
    if not common:
        return doc

    maximal = _filter_maximal_sequences(common)
    if not maximal:
        return doc

    existing_ids = {s.id for s in doc.sequences}
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

        seq = Sequence(
            id=seq_id,
            n_tokens=len(seq_tokens),
            tokens=[
                Token(id=tid, token=tstr)
                for tid, tstr in zip(seq_ids, seq_tokens)
            ],
        )
        doc.sequences.append(seq)
        next_idx += 1

    for sample in doc.samples:
        sample.tokens = _replace_sequences_in_tokens(sample.tokens, doc.sequences)

    return doc


def expand_sequences(doc: InifDocument) -> InifDocument:
    """Expand all sequence references back to flat tokens with original ids."""
    doc = doc.model_copy(deep=True)
    seq_map = doc.sequence_map
    for sample in doc.samples:
        new_tokens: list[Token] = []
        for token in sample.tokens:
            if token.is_sequence_ref:
                assert token.sequence_id is not None, (
                    "Sequence ref token must have sequence_id"
                )
                assert token.sequence_id in seq_map, (
                    f"Sequence '{token.sequence_id}' not found"
                )
                seq = seq_map[token.sequence_id]
                for t in seq.tokens:
                    new_tokens.append(
                        Token(id=t.id, token=t.token, sequence_id=seq.id)
                    )
            else:
                new_tokens.append(token)
        sample.tokens = new_tokens
    return doc
