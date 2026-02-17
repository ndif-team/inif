from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Token,
)
from inif.selectors import (
    select_by_position,
    select_by_score,
    select_by_sequence_id,
    select_by_span,
    select_by_tag,
)


def test_select_by_position_int(sample_flat):
    sel = select_by_position(sample_flat, 0)
    assert len(sel.tokens) == 1
    assert sel.positions == [0]


def test_select_by_position_list(sample_flat):
    sel = select_by_position(sample_flat, [1, 3, 5])
    assert len(sel.tokens) == 3
    assert sel.positions == [1, 3, 5]


def test_select_by_position_slice(sample_flat):
    sel = select_by_position(sample_flat, slice(0, 3))
    assert len(sel.tokens) == 3
    assert sel.positions == [0, 1, 2]


def test_select_by_tag(sample):
    sel = select_by_tag(sample, "content")
    assert len(sel.tokens) == 1
    assert sel.positions == [2]  # index in sample.tokens list


def test_select_by_tag_no_match(sample):
    sel = select_by_tag(sample, "nonexistent")
    assert len(sel.tokens) == 0


def test_select_by_sequence_id(sample):
    sel = select_by_sequence_id(sample, "seq_0")
    assert len(sel.tokens) == 1
    assert sel.tokens[0].id == -1  # ref token


def test_select_by_span(sample):
    sel = select_by_span(sample, "answer")
    assert len(sel.tokens) == 2
    assert set(sel.positions) == {2, 3}


def test_select_by_span_no_match(sample):
    sel = select_by_span(sample, "nonexistent")
    assert len(sel.tokens) == 0


def test_select_by_score():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=1, token="a")],
                scores=[SampleScore(scorer="acc", value=1.0)],
            ),
            Sample(
                id="s1",
                tokens=[Token(id=2, token="b")],
                scores=[SampleScore(scorer="acc", value=0.0)],
            ),
            Sample(
                id="s2",
                tokens=[Token(id=3, token="c")],
                scores=[SampleScore(scorer="f1", value=0.5)],
            ),
        ],
    )

    results = select_by_score(doc, "acc", lambda v: v == 1.0)
    assert len(results) == 1
    assert results[0].sample_id == "s0"

    results = select_by_score(doc, "acc", lambda v: v >= 0.0)
    assert len(results) == 2
