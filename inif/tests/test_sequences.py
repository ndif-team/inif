from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    Token,
)
from inif.sequences import deduplicate_sequences, expand_sequences


def test_deduplicate_basic(doc_for_dedup):
    deduped = deduplicate_sequences(doc_for_dedup, min_length=3)
    assert len(deduped.sequences) == 1
    seq = deduped.sequences[0]
    assert seq.tokens == ["<|endoftext|>", "This", " is"]
    assert seq.n_tokens == 3
    # Each sample should now start with a ref token
    for sample in deduped.samples:
        assert sample.tokens[0].is_sequence_ref
        assert sample.tokens[0].sequence_id == seq.id


def test_expand_basic(doc_for_dedup):
    deduped = deduplicate_sequences(doc_for_dedup, min_length=3)
    expanded = expand_sequences(deduped)

    # After expanding, each sample should have 4 flat tokens again
    for sample in expanded.samples:
        assert len(sample.tokens) == 4
        assert not any(t.is_sequence_ref for t in sample.tokens)


def test_roundtrip_dedup_expand(doc_for_dedup):
    deduped = deduplicate_sequences(doc_for_dedup, min_length=3)
    expanded = expand_sequences(deduped)

    for orig, exp in zip(doc_for_dedup.samples, expanded.samples):
        orig_strs = [t.token for t in orig.tokens]
        exp_strs = [t.token for t in exp.tokens]
        assert orig_strs == exp_strs


def test_dedup_preserves_extra_field_tokens():
    """Tokens with extra fields should never be collapsed."""
    tok_with_data = Token(id=2, token="b", data={"logit_lens": {"layer_0": {}}})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[
                    Token(id=1, token="a"),
                    tok_with_data,
                    Token(id=3, token="c"),
                ],
            ),
            Sample(
                id="s1",
                tokens=[
                    Token(id=1, token="a"),
                    Token(id=2, token="b"),
                    Token(id=3, token="c"),
                ],
            ),
        ],
    )
    deduped = deduplicate_sequences(doc, min_length=3)
    # s0 has a token with extra data, so the sequence "a","b","c" can't be
    # fully matched in s0. The intersection algorithm needs the sequence
    # present in ALL samples (without extra fields). Since s0's token list
    # (excluding extra-field tokens) is ["a", "c"], the 3-gram "a","b","c"
    # won't be found in s0's clean list. No dedup.
    assert len(deduped.sequences) == 0


def test_dedup_no_duplicates():
    """No sequences should be created if no duplicates exist."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=i + 1, token=f"t{i}") for i in range(3)],
            ),
            Sample(
                id="s1",
                tokens=[Token(id=i + 10, token=f"u{i}") for i in range(3)],
            ),
        ],
    )
    deduped = deduplicate_sequences(doc, min_length=3)
    assert len(deduped.sequences) == 0


def test_expand_preserves_non_ref_tokens():
    """Expanding a document with no refs should be a no-op."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=i + 1, token=f"t{i}") for i in range(3)],
            ),
        ],
    )
    expanded = expand_sequences(doc)
    assert len(expanded.samples[0].tokens) == 3
    assert expanded.samples[0].tokens[0].id == 1


def test_expanded_tokens_have_sequence_id(doc_for_dedup):
    """Expanded tokens should carry the sequence_id they came from."""
    deduped = deduplicate_sequences(doc_for_dedup, min_length=3)
    expanded = expand_sequences(deduped)

    seq_id = deduped.sequences[0].id
    for sample in expanded.samples:
        # First 3 tokens came from expansion
        for t in sample.tokens[:3]:
            assert t.sequence_id == seq_id
            assert t.id == 0  # expanded tokens get id=0
        # Last token is original, no sequence_id
        assert sample.tokens[3].sequence_id is None
