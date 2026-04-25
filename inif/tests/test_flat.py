import re

from inif.flat import FlatTokenStore
from inif.models import InifDocument, Metadata, ModelInfo, Sample, Token


def test_flat_store_from_document_expands_sequences(doc):
    store = FlatTokenStore.from_document(doc)

    assert store.sample_ids == ["sample_0"]
    assert store.sample_offsets == [0, 7]
    assert store.token_ids == [50256, 1212, 318, 257, 1332, 764, 50256]
    assert store.token_texts == [
        "<|endoftext|>",
        "This",
        " is",
        " a",
        " test",
        ".",
        "<|endoftext|>",
    ]
    assert store.sequence_ids == [
        "seq_0",
        "seq_0",
        "seq_0",
        None,
        None,
        None,
        "seq_1",
    ]
    assert store.positions_by_tag("content") == [4]


def test_flat_store_regex_search_and_tagging(doc):
    store = FlatTokenStore.from_document(doc)

    matches = store.find_regexes([(r"test", "testy"), (re.compile(r"^<\|"), "bos")])
    assert matches == {"testy": [4], "bos": [0, 6]}
    assert store.positions_by_tag("testy") == []

    store.tag_regexes([(r"test", "testy"), (r"^<\|", "bos")])
    store.tag_regexes([(r"test", "testy")])

    assert store.positions_by_tag("content") == [4]
    assert store.positions_by_tag("testy") == [4]
    assert store.positions_by_tag("bos") == [0, 6]
    assert store.tag_counts() == {"content": 1, "testy": 1, "bos": 2}
    assert store.tokens_by_tag("bos") == [
        (0, 50256, "<|endoftext|>"),
        (6, 50256, "<|endoftext|>"),
    ]


def test_flat_store_sample_position_and_iteration():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(id="s0", tokens=[Token(id=1, token="a"), Token(id=2, token="b")]),
            Sample(id="s1", tokens=[Token(id=3, token="c")]),
        ],
    )
    store = FlatTokenStore.from_document(doc)

    assert store.sample_slice("s0") == slice(0, 2)
    assert store.sample_slice("s1") == slice(2, 3)
    assert store.sample_position(2) == ("s1", 0)
    assert list(store.iter_sample_tokens("s0")) == [(0, 1, "a"), (1, 2, "b")]


def test_flat_store_to_document_preserves_flat_tokens_and_tags(doc):
    store = FlatTokenStore.from_document(doc)
    store.tag_regexes([(r"^This$", "word")])

    materialized = store.to_document(metadata=doc.metadata)

    assert materialized.metadata.model.name == doc.metadata.model.name
    assert materialized.sequences == []
    assert len(materialized.samples) == 1

    sample = materialized.samples[0]
    assert sample.id == "sample_0"
    assert [token.id for token in sample.tokens] == store.token_ids
    assert [token.sequence_id for token in sample.tokens] == store.sequence_ids
    assert sample.tokens[1].has_tag("word")
    assert sample.tokens[4].has_tag("content")
