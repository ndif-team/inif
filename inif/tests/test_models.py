import pytest
from pydantic import ValidationError

from inif.models import (
    ModelInfo,
    Sample,
    Sequence,
    Token,
)


def test_model_info_defaults():
    m = ModelInfo(name="gpt2")
    assert m.name == "gpt2"
    assert m.revision is None
    assert m.generation_config == {}
    assert m.loading_config == {}


def test_sequence_validation():
    with pytest.raises(ValidationError, match="n_tokens must equal"):
        Sequence(id="s0", tokens=["a", "b"], n_tokens=1)


def test_sequence_valid():
    s = Sequence(
        id="seq_0",
        tokens=["hello", " world"],
        n_tokens=2,
    )
    assert len(s.tokens) == 2
    assert s.id == "seq_0"


def test_token_is_sequence_ref():
    t_regular = Token(id=100, token="hello")
    assert not t_regular.is_sequence_ref

    t_ref = Token(id=-1, sequence_id="seq_0")
    assert t_ref.is_sequence_ref


def test_token_expanded_tokens(sequences):
    ref = Token(id=-1, sequence_id="seq_0")
    expanded = ref.expanded_tokens(sequences)
    assert len(expanded) == 3
    assert expanded[0].id == 0
    assert expanded[0].token == "<|endoftext|>"
    assert expanded[1].token == "This"
    assert expanded[2].token == " is"
    assert all(t.sequence_id == "seq_0" for t in expanded)


def test_token_expanded_regular(sequences):
    t = Token(id=100, token="hello")
    expanded = t.expanded_tokens(sequences)
    assert len(expanded) == 1
    assert expanded[0] is t


def test_token_expanded_missing_sequence():
    ref = Token(id=-1, sequence_id="nonexistent")
    with pytest.raises(AssertionError, match="not found"):
        ref.expanded_tokens([])


def test_token_extra_fields():
    t = Token(id=1, token="hello", role="user", logprob=-0.5)
    assert t.model_extra["role"] == "user"
    assert t.model_extra["logprob"] == -0.5


def test_token_extra_roundtrip():
    t = Token(id=1, token="hello", custom_data={"key": "value"})
    d = t.model_dump()
    assert d["custom_data"] == {"key": "value"}
    t2 = Token.model_validate(d)
    assert t2.model_extra["custom_data"] == {"key": "value"}


def test_sample_get_expanded_tokens(sample, sequences):
    expanded = sample.get_expanded_tokens(sequences)
    # seq_0 expands to 3, then 3 regular tokens, then seq_1 expands to 1
    assert len(expanded) == 3 + 3 + 1


def test_sample_get_tokens_by_positions(sample):
    tokens = sample.get_tokens_by_positions([1, 3])
    assert len(tokens) == 2


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


def test_token_defaults():
    t = Token(id=1)
    assert t.token is None
    assert t.sequence_id is None


def test_sample_defaults():
    s = Sample(id="x")
    assert s.tokens == []
    assert s.texts == []
    assert s.spans == []
    assert s.scores == []
    assert s.target is None
    assert s.metadata == {}


def test_sample_texts_are_strings():
    s = Sample(id="x", texts=["Hello", "World"])
    assert s.texts == ["Hello", "World"]
    assert isinstance(s.texts[0], str)
