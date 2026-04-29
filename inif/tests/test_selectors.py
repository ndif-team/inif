from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    TokenOrSeqRef,
)


def test_select_by_position_int(sample_flat):
    sel = sample_flat.select_by_position(0)
    assert len(sel.tokens) == 1
    assert sel.positions == [0]


def test_select_by_position_list(sample_flat):
    sel = sample_flat.select_by_position([1, 3, 5])
    assert len(sel.tokens) == 3
    assert sel.positions == [1, 3, 5]


def test_select_by_position_slice(sample_flat):
    sel = sample_flat.select_by_position(slice(0, 3))
    assert len(sel.tokens) == 3
    assert sel.positions == [0, 1, 2]


def test_select_by_annotation(sample):
    sel = sample.select_by_annotation("content")
    assert len(sel.tokens) == 1
    assert sel.positions == [2]  # index in sample.tokens list


def test_select_by_annotation_no_match(sample):
    sel = sample.select_by_annotation("nonexistent")
    assert len(sel.tokens) == 0


def test_select_by_sequence_id(sample):
    sel = sample.select_by_sequence_id("seq_0")
    assert len(sel.tokens) == 1
    # Sequence refs have ``id is None`` and carry the target id in ``token``.
    assert sel.tokens[0].is_sequence_ref
    assert sel.tokens[0].id is None
    assert sel.tokens[0].token == "seq_0"


def test_select_by_span(sample):
    sel = sample.select_by_span("answer")
    assert len(sel.tokens) == 2
    assert set(sel.positions) == {2, 3}


def test_select_by_span_no_match(sample):
    sel = sample.select_by_span("nonexistent")
    assert len(sel.tokens) == 0


def test_filter_samples_by_score():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="a")],
                scores=[SampleScore(scorer="acc", value=1.0)],
            ),
            Sample(
                id="s1",
                tokens=[TokenOrSeqRef(id=2, token="b")],
                scores=[SampleScore(scorer="acc", value=0.0)],
            ),
            Sample(
                id="s2",
                tokens=[TokenOrSeqRef(id=3, token="c")],
                scores=[SampleScore(scorer="f1", value=0.5)],
            ),
        ],
    )

    results = doc.filter_samples_by_score("acc", lambda v: v == 1.0)
    assert len(results) == 1
    assert results[0].id == "s0"
    # Returned objects are Samples; composing with position selectors works.
    sel = results[0].select_by_position(0)
    assert sel.tokens[0].token == "a"

    results = doc.filter_samples_by_score("acc", lambda v: v >= 0.0)
    assert [s.id for s in results] == ["s0", "s1"]
