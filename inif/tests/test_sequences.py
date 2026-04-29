from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    TokenOrSeqRef,
)


def test_deduplicate_basic(doc_for_dedup):
    deduped = doc_for_dedup.deduplicate_sequences(min_length=3)
    assert len(deduped.sequences) == 1
    seq = deduped.sequences[0]
    assert [t.token for t in seq.tokens] == ["<|endoftext|>", "This", " is"]
    assert [t.id for t in seq.tokens] == [50256, 1212, 318]
    assert seq.n_tokens == 3
    # Each sample should now start with a ref token
    for sample in deduped.samples:
        assert sample.tokens[0].is_sequence_ref
        assert sample.tokens[0].sequence_id == seq.id


def test_deduplicate_default_min_length_is_five():
    """The default ``min_length=5`` means a 3-token shared prefix is NOT
    captured unless callers opt in to a smaller window."""

    def toks(names):
        return [TokenOrSeqRef(id=ord(name[0]), token=name) for name in names]

    short_doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(id="s0", tokens=toks(["a", "b", "c", "x"])),
            Sample(id="s1", tokens=toks(["a", "b", "c", "y"])),
        ],
    )
    # Length-3 shared run, default min_length=5 → no dedup.
    assert short_doc.deduplicate_sequences().sequences == []

    long_doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(id="s0", tokens=toks(["a", "b", "c", "d", "e", "x"])),
            Sample(id="s1", tokens=toks(["a", "b", "c", "d", "e", "y"])),
        ],
    )
    # Length-5 shared run, default min_length=5 → captured.
    deduped = long_doc.deduplicate_sequences()
    assert len(deduped.sequences) == 1
    assert [t.token for t in deduped.sequences[0].tokens] == ["a", "b", "c", "d", "e"]


def test_deduplicate_extends_later_anchor_occurrence():
    """All sample-0 anchor occurrences must be considered for maximal matches."""
    ids = {"a": 1, "b": 2, "x": 3, "c": 4}

    def toks(names):
        return [TokenOrSeqRef(id=ids[name], token=name) for name in names]

    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(id="s0", tokens=toks(["a", "b", "x", "a", "b", "c"])),
            Sample(id="s1", tokens=toks(["a", "b", "c"])),
        ],
    )

    deduped = doc.deduplicate_sequences(min_length=2)

    assert len(deduped.sequences) == 1
    assert [t.token for t in deduped.sequences[0].tokens] == ["a", "b", "c"]
    assert deduped.samples[0].tokens[-1].is_sequence_ref
    assert deduped.samples[1].tokens[0].is_sequence_ref


def test_expand_basic(doc_for_dedup):
    deduped = doc_for_dedup.deduplicate_sequences(min_length=3)
    expanded = deduped.expand_sequences()

    # After expanding, each sample should have 4 flat tokens again
    for sample in expanded.samples:
        assert len(sample.tokens) == 4
        assert not any(t.is_sequence_ref for t in sample.tokens)


def test_roundtrip_dedup_expand(doc_for_dedup):
    deduped = doc_for_dedup.deduplicate_sequences(min_length=3)
    expanded = deduped.expand_sequences()

    for orig, exp in zip(doc_for_dedup.samples, expanded.samples):
        orig_strs = [t.token for t in orig.tokens]
        exp_strs = [t.token for t in exp.tokens]
        assert orig_strs == exp_strs
        # Original ids must be preserved through dedup → expand round-trip.
        orig_ids = [t.id for t in orig.tokens]
        exp_ids = [t.id for t in exp.tokens]
        assert orig_ids == exp_ids


def test_dedup_preserves_extra_field_tokens():
    """Tokens with extra fields should never be collapsed."""
    tok_with_data = TokenOrSeqRef(id=2, token="b", data={"logit_lens": {"layer_0": {}}})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[
                    TokenOrSeqRef(id=1, token="a"),
                    tok_with_data,
                    TokenOrSeqRef(id=3, token="c"),
                ],
            ),
            Sample(
                id="s1",
                tokens=[
                    TokenOrSeqRef(id=1, token="a"),
                    TokenOrSeqRef(id=2, token="b"),
                    TokenOrSeqRef(id=3, token="c"),
                ],
            ),
        ],
    )
    deduped = doc.deduplicate_sequences(min_length=3)
    # s0 has a token with extra data, so the sequence "a","b","c" can't be
    # fully matched in s0. The intersection algorithm needs the sequence
    # present in ALL samples (without extra fields). Since s0's token list
    # (excluding extra-field tokens) is ["a", "c"], the 3-gram "a","b","c"
    # won't be found in s0's clean list. No dedup.
    assert len(deduped.sequences) == 0


def test_deduplicate_returns_independent_tokens():
    """Mutating a returned doc must not mutate the source doc."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(id="s0", tokens=[TokenOrSeqRef(id=1, token="a")]),
            Sample(id="s1", tokens=[TokenOrSeqRef(id=2, token="b")]),
        ],
    )

    deduped = doc.deduplicate_sequences(min_length=2)
    deduped.samples[0].tokens[0].set_extra("changed", True)

    assert deduped.samples[0].tokens[0].get_extra("changed") is True
    assert not doc.samples[0].tokens[0].has_extra("changed")


def test_dedup_no_duplicates():
    """No sequences should be created if no duplicates exist."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=i + 1, token=f"t{i}") for i in range(3)],
            ),
            Sample(
                id="s1",
                tokens=[TokenOrSeqRef(id=i + 10, token=f"u{i}") for i in range(3)],
            ),
        ],
    )
    deduped = doc.deduplicate_sequences(min_length=3)
    assert len(deduped.sequences) == 0


def test_expand_preserves_non_ref_tokens():
    """Expanding a document with no refs should be a no-op."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=i + 1, token=f"t{i}") for i in range(3)],
            ),
        ],
    )
    expanded = doc.expand_sequences()
    assert len(expanded.samples[0].tokens) == 3
    assert expanded.samples[0].tokens[0].id == 1


def test_dedup_skips_window_with_mismatched_ids():
    """If two samples share the same token strings but disagree on ids
    (rare, but possible across tokenizer quirks), the window with the wrong
    ids must NOT be collapsed — otherwise expansion would corrupt that
    sample's ids."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[
                    TokenOrSeqRef(id=10, token="a"),
                    TokenOrSeqRef(id=20, token="b"),
                    TokenOrSeqRef(id=30, token="c"),
                    TokenOrSeqRef(id=99, token="x"),
                ],
            ),
            Sample(
                id="s1",
                tokens=[
                    TokenOrSeqRef(id=10, token="a"),
                    # Same string "b" but DIFFERENT id; must block replacement.
                    TokenOrSeqRef(id=21, token="b"),
                    TokenOrSeqRef(id=30, token="c"),
                    TokenOrSeqRef(id=88, token="y"),
                ],
            ),
        ],
    )
    deduped = doc.deduplicate_sequences(min_length=3)
    # Sequence is created from sample 0's ids.
    assert len(deduped.sequences) == 1
    assert [t.id for t in deduped.sequences[0].tokens] == [10, 20, 30]
    # Sample 0 collapses cleanly.
    assert deduped.samples[0].tokens[0].is_sequence_ref
    # Sample 1 keeps its original tokens (its middle id doesn't match).
    assert [t.id for t in deduped.samples[1].tokens] == [10, 21, 30, 88]
    assert all(not t.is_sequence_ref for t in deduped.samples[1].tokens)


def test_expanded_tokens_drop_sequence_provenance(doc_for_dedup):
    """expand_sequences should produce plain vocab tokens (no
    ``sequence_id``) and drop the sequences list — running dedup again
    rediscovers the same shared runs."""
    deduped = doc_for_dedup.deduplicate_sequences(min_length=3)
    expanded = deduped.expand_sequences()

    assert expanded.sequences == []
    expected_ids = [50256, 1212, 318]
    for sample in expanded.samples:
        for t, expected_id in zip(sample.tokens[:3], expected_ids):
            assert t.sequence_id is None
            assert t.id == expected_id
        assert sample.tokens[3].sequence_id is None
