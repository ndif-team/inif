import re

from inif.flat import FlatTokenStore
from inif.models import InifDocument, Metadata, ModelInfo, Sample, TokenOrSeqRef


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
    assert store.positions("content") == [4]


def test_flat_store_regex_search_and_tagging(doc):
    store = FlatTokenStore.from_document(doc)

    matches = store.find_regexes([(r"test", "testy"), (re.compile(r"^<\|"), "bos")])
    assert matches == {"testy": [4], "bos": [0, 6]}
    assert store.positions("testy") == []

    store.annotate_regexes([(r"test", "testy"), (r"^<\|", "bos")])
    store.annotate_regexes([(r"test", "testy")])

    assert store.positions("content") == [4]
    assert store.positions("testy") == [4]
    assert store.positions("bos") == [0, 6]
    assert store.annotation_counts() == {"content": 1, "testy": 1, "bos": 2}
    assert store.tokens("bos") == [
        (0, 50256, "<|endoftext|>"),
        (6, 50256, "<|endoftext|>"),
    ]


def test_flat_store_sample_position_and_iteration():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            ),
            Sample(id="s1", tokens=[TokenOrSeqRef(id=3, token="c")]),
        ],
    )
    store = FlatTokenStore.from_document(doc)

    assert store.sample_slice("s0") == slice(0, 2)
    assert store.sample_slice("s1") == slice(2, 3)
    assert store.sample_position(2) == ("s1", 0)
    assert list(store.iter_sample_tokens("s0")) == [(0, 1, "a"), (1, 2, "b")]


def test_flat_store_to_document_preserves_flat_tokens_and_annotations(doc):
    store = FlatTokenStore.from_document(doc)
    store.annotate_regexes([(r"^This$", "word")])

    materialized = store.to_document(metadata=doc.metadata)

    assert materialized.metadata.model.name == doc.metadata.model.name
    assert materialized.sequences == []
    assert len(materialized.samples) == 1

    sample = materialized.samples[0]
    assert sample.id == "sample_0"
    assert [token.id for token in sample.tokens] == store.token_ids
    # Sequence-ref provenance lives only on the flat store; materialized
    # tokens are vocab tokens, so ``sequence_id`` is None for every entry.
    assert all(token.sequence_id is None for token in sample.tokens)
    assert sample.annotation_positions("word") == [1]
    assert sample.annotation_positions("content") == [4]
