import pytest
from pydantic import ValidationError

from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    Sequence,
    Span,
    Text,
    TokenAnnotation,
    TokenOrSeqRef,
)


def test_model_info_defaults():
    m = ModelInfo(name="gpt2")
    assert m.name == "gpt2"
    assert m.revision is None
    assert m.generation_config == {}
    assert m.loading_config == {}


def test_sequence_validation_n_tokens():
    with pytest.raises(ValidationError, match="n_tokens must equal"):
        Sequence(
            id="s0",
            tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            n_tokens=1,
        )


def test_sequence_valid():
    s = Sequence(
        id="seq_0",
        n_tokens=2,
        tokens=[
            TokenOrSeqRef(id=1, token="hello"),
            TokenOrSeqRef(id=2, token=" world"),
        ],
    )
    assert len(s.tokens) == 2
    assert [t.id for t in s.tokens] == [1, 2]
    assert s.id == "seq_0"


def test_token_is_sequence_ref():
    t_regular = TokenOrSeqRef(id=100, token="hello")
    assert not t_regular.is_sequence_ref

    t_ref = TokenOrSeqRef(id=None, token="seq_0")
    assert t_ref.is_sequence_ref


def test_token_expanded_tokens(sequences):
    ref = TokenOrSeqRef(id=None, token="seq_0")
    expanded = ref.expanded_tokens(sequences)
    assert len(expanded) == 3
    assert expanded[0].id == 50256
    assert expanded[0].token == "<|endoftext|>"
    assert expanded[1].id == 1212
    assert expanded[1].token == "This"
    assert expanded[2].id == 318
    assert expanded[2].token == " is"
    # Expanded tokens are vocab tokens; the convenience property returns None.
    assert all(t.sequence_id is None for t in expanded)
    assert all(t.id is not None for t in expanded)


def test_token_expanded_regular(sequences):
    t = TokenOrSeqRef(id=100, token="hello")
    expanded = t.expanded_tokens(sequences)
    assert len(expanded) == 1
    assert expanded[0] is t


def test_token_expanded_missing_sequence():
    ref = TokenOrSeqRef(id=None, token="nonexistent")
    with pytest.raises(AssertionError, match="not found"):
        ref.expanded_tokens([])


def test_token_extra_fields():
    t = TokenOrSeqRef(id=1, token="hello", source="user", logprob=-0.5)
    assert t.get_extra("source") == "user"
    assert t.get_extra("logprob") == -0.5


def test_token_extra_roundtrip():
    t = TokenOrSeqRef(id=1, token="hello", custom_data={"key": "value"})
    d = t.model_dump()
    assert d["custom_data"] == {"key": "value"}
    t2 = TokenOrSeqRef.model_validate(d)
    assert t2.get_extra("custom_data") == {"key": "value"}


def test_token_set_get_pop_extra():
    t = TokenOrSeqRef(id=1, token="x")
    assert t.get_extra("logprob") is None
    assert t.get_extra("logprob", 0.0) == 0.0
    assert not t.has_extra("logprob")

    t.set_extra("logprob", -0.25)
    assert t.has_extra("logprob")
    assert t.get_extra("logprob") == -0.25
    # Roundtrips through serialization
    assert t.model_dump()["logprob"] == -0.25

    popped = t.pop_extra("logprob")
    assert popped == -0.25
    assert not t.has_extra("logprob")
    assert "logprob" not in t.model_dump()

    # pop on a missing key returns the supplied default
    assert t.pop_extra("missing") is None
    assert t.pop_extra("missing", "fallback") == "fallback"


def test_sample_annotation_api():
    s = Sample(
        id="x",
        tokens=[
            TokenOrSeqRef(id=1, token="a"),
            TokenOrSeqRef(id=2, token="b"),
            TokenOrSeqRef(id=3, token="c"),
        ],
    )

    s.annotate_positions("foo", [0, 2])
    s.annotate("foo", [(1, 2)])

    assert s.annotations == [TokenAnnotation(name="foo", ranges=[(0, 3)])]
    assert s.annotation_positions("foo") == [0, 1, 2]

    s.remove_annotation("foo")
    assert s.annotations == []


def test_token_extras_property():
    t = TokenOrSeqRef(id=1, token="x", source="user", logprob=-0.5)
    assert t.extras == {"source": "user", "logprob": -0.5}


def test_sample_get_expanded_tokens(sample, sequences):
    expanded = sample.get_expanded_tokens(sequences)
    # seq_0 expands to 3, then 3 regular tokens, then seq_1 expands to 1
    assert len(expanded) == 3 + 3 + 1


def test_sample_get_tokens_by_positions(sample):
    tokens = sample.get_tokens_by_positions([1, 3])
    assert len(tokens) == 2


def test_sample_materialize_position_in_ref(sample, sequences):
    """Materializing a position inside a sequence ref expands that ref in place."""
    # sample.tokens = [ref(seq_0, len 3), " a", " test", ".", ref(seq_1, len 1)]
    # Expanded view positions: 0,1,2 (seq_0), 3 (" a"), 4 (" test"), 5 ("."), 6 (seq_1)
    n_before = len(sample.tokens)
    actual_idx, tok = sample.materialize_position(1, sequences)
    # ref expanded into 3 tokens, so sample.tokens grew by 2
    assert len(sample.tokens) == n_before + 2
    # The materialized token is the second token of seq_0 ("This", id 1212)
    assert tok.id == 1212
    assert tok.token == "This"
    assert actual_idx == 1
    # The first ref is gone, replaced by 3 real tokens with original ids
    assert sample.tokens[0].id == 50256
    assert sample.tokens[1].id == 1212
    assert sample.tokens[2].id == 318
    assert sample.annotation_positions("content") == [4]
    # The other ref (seq_1) is untouched
    assert sample.tokens[-1].is_sequence_ref
    assert sample.tokens[-1].sequence_id == "seq_1"


def test_sample_materialize_position_outside_ref(sample, sequences):
    """Materializing a position outside any ref returns the existing token."""
    n_before = len(sample.tokens)
    # Expanded position 4 = " test" (regular token, sample.tokens[2])
    actual_idx, tok = sample.materialize_position(4, sequences)
    assert len(sample.tokens) == n_before  # no expansion
    assert tok.token == " test"
    assert actual_idx == 2


def test_sample_materialize_position_then_attach(sample, sequences):
    """The motivating use case: enrich a token that started inside a ref."""
    actual_idx, tok = sample.materialize_position(2, sequences)
    tok.set_extra("logit_lens", {"layer_5": "predicted"})
    # The data is now attached to the materialized token in sample.tokens
    assert sample.tokens[actual_idx].get_extra("logit_lens") == {"layer_5": "predicted"}


def test_sample_materialize_position_out_of_range(sample, sequences):
    with pytest.raises(IndexError, match="out of range"):
        sample.materialize_position(999, sequences)


def test_sample_materialize_at_first_ref_boundary(sample, sequences):
    """Position 0 is the first token of the first ref — materialization should
    place a real token at index 0 of self.tokens."""
    actual_idx, tok = sample.materialize_position(0, sequences)
    assert actual_idx == 0
    assert tok.id == 50256
    assert tok.token == "<|endoftext|>"
    assert sample.tokens[0].id == 50256


def test_sample_materialize_at_last_ref_boundary(sample, sequences):
    """The last expanded position falls inside the trailing 1-token ref."""
    # sample expanded view length: 3 (seq_0) + 3 regular + 1 (seq_1) = 7
    actual_idx, tok = sample.materialize_position(6, sequences)
    assert tok.id == 50256
    assert tok.token == "<|endoftext|>"
    # The trailing ref was expanded into a single real token at the end.
    assert sample.tokens[actual_idx] is tok
    assert not sample.tokens[actual_idx].is_sequence_ref


def test_sample_materialize_position_is_idempotent(sample, sequences):
    """Calling materialize_position twice on the same expanded position is safe
    and returns the same Token object the second time."""
    _, tok_first = sample.materialize_position(2, sequences)
    n_after_first = len(sample.tokens)
    actual_idx, tok_second = sample.materialize_position(2, sequences)
    # No further expansion happened.
    assert len(sample.tokens) == n_after_first
    assert tok_second is tok_first
    assert sample.tokens[actual_idx] is tok_first


def test_sample_materialize_two_positions_in_same_ref(sample, sequences):
    """The pattern used by tag_by_regex: enrich two tokens that both fall
    inside one sequence ref. After the first materialization, the second one
    finds the now-real token without further expansion."""
    _, t1 = sample.materialize_position(0, sequences)
    n_after_first = len(sample.tokens)
    _, t2 = sample.materialize_position(2, sequences)
    # Second call did not re-expand.
    assert len(sample.tokens) == n_after_first
    t1.set_extra("kind", "first")
    t2.set_extra("kind", "third")
    # Both extras land on the right tokens.
    assert sample.tokens[0].get_extra("kind") == "first"
    assert sample.tokens[2].get_extra("kind") == "third"
    # The middle token of the ref is still untouched (no extras).
    assert not sample.tokens[1].has_extra("kind")


def test_document_sequence_map(doc):
    seq_map = doc.sequence_map
    assert "seq_0" in seq_map
    assert "seq_1" in seq_map


def test_document_get_sample(doc):
    s = doc.get_sample("sample_0")
    assert s.id == "sample_0"

    with pytest.raises(KeyError):
        doc.get_sample("nonexistent")


def test_document_filter_samples(doc):
    results = doc.filter_samples(lambda s: s.target == "test")
    assert len(results) == 1

    results = doc.filter_samples(lambda s: s.target == "none")
    assert len(results) == 0


def test_document_total_samples_property():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="t")),
        samples=[Sample(id=str(i)) for i in range(4)],
    )
    assert doc.total_samples == 4


def test_document_subset_filters_and_prunes_sequences():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="t")),
        sequences=[
            Sequence(id="s_keep", tokens=[TokenOrSeqRef(id=1, token="a")], n_tokens=1),
            Sequence(id="s_drop", tokens=[TokenOrSeqRef(id=2, token="b")], n_tokens=1),
        ],
        samples=[
            Sample(id="alpha", tokens=[TokenOrSeqRef(id=None, token="s_keep")]),
            Sample(id="beta", tokens=[TokenOrSeqRef(id=None, token="s_drop")]),
        ],
    )
    sub = doc.subset(lambda s: s.id == "alpha")
    assert [s.id for s in sub.samples] == ["alpha"]
    # Only the sequence still referenced is retained.
    assert [s.id for s in sub.sequences] == ["s_keep"]
    # Original document unchanged.
    assert len(doc.samples) == 2
    assert len(doc.sequences) == 2


def test_document_subset_preserves_metadata_and_is_independent():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="t"), inif_version="0.9"),
        samples=[Sample(id="x"), Sample(id="y")],
    )
    sub = doc.subset(lambda s: s.id == "x")
    assert sub.metadata.inif_version == "0.9"
    # Mutating subset doesn't affect original.
    sub.samples[0].metadata["new"] = True
    assert "new" not in doc.samples[0].metadata


def test_document_subset_deep_copies_tokens_and_sequences():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="t")),
        sequences=[
            Sequence(id="seq", tokens=[TokenOrSeqRef(id=1, token="a")], n_tokens=1),
        ],
        samples=[
            Sample(id="x", tokens=[TokenOrSeqRef(id=None, token="seq")]),
        ],
    )

    sub = doc.subset(lambda s: s.id == "x")
    sub.samples[0].tokens[0].set_extra("note", "sample")
    sub.sequences[0].tokens[0].set_extra("note", "sequence")

    assert not doc.samples[0].tokens[0].has_extra("note")
    assert not doc.sequences[0].tokens[0].has_extra("note")


def test_metadata_created_at_is_datetime():
    """``created_at`` strings round-trip into datetime objects via pydantic."""
    from datetime import datetime, timezone

    m = Metadata(model=ModelInfo(name="t"), created_at="2025-01-02T03:04:05Z")
    assert isinstance(m.created_at, datetime)
    assert m.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_token_ref_uses_token_field_for_sequence_id():
    # Sequence-ref tokens have ``id is None``; the target Sequence id lives
    # in ``token``. The ``sequence_id`` convenience property returns it.
    t = TokenOrSeqRef(id=None, token="seq_x")
    assert t.is_sequence_ref
    assert t.token == "seq_x"
    assert t.sequence_id == "seq_x"


def test_token_vocab_has_no_sequence_id():
    t = TokenOrSeqRef(id=42, token="hello")
    assert not t.is_sequence_ref
    assert t.sequence_id is None


def test_token_invariant_token_field_required():
    # ``token`` is a required string — both vocab tokens and sequence refs
    # must supply one. Pydantic rejects the missing field at validation time.
    with pytest.raises(ValidationError):
        TokenOrSeqRef(id=1)
    with pytest.raises(ValidationError):
        TokenOrSeqRef(id=None)


def test_sample_id_is_coerced_to_str():
    s = Sample(id=42)
    assert s.id == "42"
    assert isinstance(s.id, str)


def test_span_positions_validated_against_tokens():
    with pytest.raises(ValidationError, match="out of range"):
        Sample(
            id="x",
            tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            spans=[Span(name="bad", positions=[0, 5])],
        )


def test_span_positions_in_bounds_ok():
    s = Sample(
        id="x",
        tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
        spans=[Span(name="ok", positions=[0, 1])],
    )
    assert s.spans[0].positions == [0, 1]


def test_annotation_ranges_validated_against_tokens():
    with pytest.raises(ValidationError, match="out of range"):
        Sample(
            id="x",
            tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            annotations=[TokenAnnotation(name="bad", ranges=[(0, 3)])],
        )


def test_annotation_ranges_are_merged():
    s = Sample(
        id="x",
        tokens=[
            TokenOrSeqRef(id=1, token="a"),
            TokenOrSeqRef(id=2, token="b"),
            TokenOrSeqRef(id=3, token="c"),
        ],
        annotations=[TokenAnnotation(name="ok", ranges=[(1, 2), (0, 1)])],
    )
    assert s.annotations[0].ranges == [(0, 2)]


def test_duplicate_annotation_names_rejected_at_construction():
    """Two annotation entries sharing a name are forbidden — ``Sample.annotations``
    is keyed by name."""
    with pytest.raises(ValidationError, match="Duplicate annotation name 'role'"):
        Sample(
            id="x",
            tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            annotations=[
                TokenAnnotation(name="role", ranges=[(0, 1)], metadata={"src": "a"}),
                TokenAnnotation(name="role", ranges=[(1, 2)], metadata={"src": "b"}),
            ],
        )


def test_annotate_same_name_merges_into_existing_entry():
    """Subsequent ``annotate`` calls with the same name extend the existing
    entry's ranges instead of creating a new one — metadata of the second
    call is ignored, the existing metadata is preserved."""
    s = Sample(
        id="x",
        tokens=[
            TokenOrSeqRef(id=1, token="a"),
            TokenOrSeqRef(id=2, token="b"),
            TokenOrSeqRef(id=3, token="c"),
        ],
    )
    s.annotate_positions("role", [0], metadata={"src": "a"})
    # Same name, different metadata — merges ranges, keeps original metadata.
    s.annotate_positions("role", [2], metadata={"src": "b"})
    assert len(s.annotations) == 1
    assert s.annotations[0].name == "role"
    assert s.annotations[0].ranges == [(0, 1), (2, 3)]
    assert s.annotations[0].metadata == {"src": "a"}


def test_annotate_overlapping_ranges_are_collapsed():
    """Adding overlapping or adjacent ranges to the same name collapses them
    into a single span (e.g. existing ``[5, 7)`` + new ``[6, 10)`` →
    ``[5, 10)``)."""
    s = Sample(
        id="x",
        tokens=[TokenOrSeqRef(id=i, token=chr(ord("a") + i)) for i in range(12)],
    )
    s.annotate("ann", [(5, 7)])
    s.annotate("ann", [(6, 10)])
    assert s.annotations[0].ranges == [(5, 10)]
    # Adjacent (touching but not overlapping) ranges also collapse.
    s.annotate("ann", [(10, 12)])
    assert s.annotations[0].ranges == [(5, 12)]
    # An island that doesn't touch stays as a separate range.
    s.annotate("ann", [(0, 2)])
    assert s.annotations[0].ranges == [(0, 2), (5, 12)]


def test_sample_defaults():
    s = Sample(id="x")
    assert s.tokens == []
    assert s.texts == []
    assert s.annotations == []
    assert s.spans == []
    assert s.scores == []
    assert s.target is None
    assert s.metadata == {}


def test_sample_texts_are_text_objects():
    s = Sample(
        id="x",
        texts=[
            Text(name="text_0", value="Hello"),
            Text(name="text_1", value="World", metadata={"source": "demo"}),
        ],
    )
    assert [t.name for t in s.texts] == ["text_0", "text_1"]
    assert [t.value for t in s.texts] == ["Hello", "World"]
    assert s.texts[0].metadata == {}
    assert s.texts[1].metadata == {"source": "demo"}


def test_text_start_end_validation():
    # Both unset is fine.
    t = Text(name="x", value="y")
    assert t.start is None and t.end is None

    # Both set, valid range.
    t = Text(name="x", value="y", start=0, end=5)
    assert (t.start, t.end) == (0, 5)

    # Empty range is allowed (a chat message that produced no tokens).
    Text(name="x", value="y", start=3, end=3)

    # Negative start.
    with pytest.raises(ValidationError, match="invalid range"):
        Text(name="x", value="y", start=-1, end=2)
    # End before start.
    with pytest.raises(ValidationError, match="invalid range"):
        Text(name="x", value="y", start=5, end=3)
    # Setting only one of the two raises.
    with pytest.raises(ValidationError, match="must set both"):
        Text(name="x", value="y", start=0)
    with pytest.raises(ValidationError, match="must set both"):
        Text(name="x", value="y", end=5)


def test_replace_token_with_tokens_shifts_text_offsets():
    # Replacing a single ref token with N expanded tokens should push every
    # text whose range starts after the replacement by ``N - 1`` positions
    # and stretch the text containing the replacement.
    s = Sample(
        id="x",
        tokens=[
            TokenOrSeqRef(id=None, token="seq_0"),  # ref at index 0
            TokenOrSeqRef(id=2, token="b"),
            TokenOrSeqRef(id=3, token="c"),
        ],
        texts=[
            Text(name="t0", value="ab", start=0, end=2),
            Text(name="t1", value="c", start=2, end=3),
        ],
    )
    s._replace_token_with_tokens(
        0,
        [TokenOrSeqRef(id=10, token="x"), TokenOrSeqRef(id=11, token="y")],
    )
    assert (s.texts[0].start, s.texts[0].end) == (0, 3)
    assert (s.texts[1].start, s.texts[1].end) == (3, 4)
