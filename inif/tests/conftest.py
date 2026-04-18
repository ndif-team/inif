import pytest

from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    SourceEval,
    Span,
    Token,
)


@pytest.fixture
def model_info():
    return ModelInfo(
        name="gpt2",
        huggingface_id="gpt2",
        generation_config={"max_tokens": 100, "temperature": 0.7},
    )


@pytest.fixture
def source_eval():
    return SourceEval(
        framework="inspect_ai",
        framework_version="0.3.0",
        task="mmlu",
        eval_id="eval_123",
    )


@pytest.fixture
def metadata(model_info, source_eval):
    return Metadata(
        model=model_info,
        inif_version="0.1",
        packages={"inif": "0.1.0"},
        source_eval=source_eval,
        created_at="2025-01-01T00:00:00Z",
    )


@pytest.fixture
def sequences():
    return [
        Sequence(
            id="seq_0",
            tokens=["<|endoftext|>", "This", " is"],
            ids=[50256, 1212, 318],
            n_tokens=3,
        ),
        Sequence(
            id="seq_1",
            tokens=["<|endoftext|>"],
            ids=[50256],
            n_tokens=1,
        ),
    ]


@pytest.fixture
def sample_tokens():
    tok_content = Token(id=1332, token=" test")
    tok_content.add_tag("content")

    tok_data = Token(
        id=764,
        token=".",
        logprob=-0.5,
        data={"logit_lens": {"layer_5": {"top_token": "."}}},
    )

    return [
        Token(id=-1, sequence_id="seq_0"),  # ref to seq 0
        Token(id=257, token=" a"),
        tok_content,
        tok_data,
        Token(id=-1, sequence_id="seq_1"),  # ref to seq 1
    ]


@pytest.fixture
def sample(sample_tokens):
    return Sample(
        id="sample_0",
        tokens=sample_tokens,
        texts=["System prompt", " a test."],
        spans=[
            Span(name="answer", positions=[2, 3]),
        ],
        scores=[
            SampleScore(scorer="accuracy", value=1.0, answer="test"),
            SampleScore(scorer="f1", value=0.8),
        ],
        target="test",
        input_tokens=5,
        output_tokens=2,
    )


@pytest.fixture
def sample_flat():
    """Sample with no sequence refs, for tagging tests."""
    return Sample(
        id="flat_0",
        tokens=[
            Token(id=50256, token="<|endoftext|>"),
            Token(id=1212, token="This"),
            Token(id=318, token=" is"),
            Token(id=257, token=" a"),
            Token(id=1332, token=" test"),
            Token(id=764, token="."),
        ],
        texts=["This is a test."],
    )


@pytest.fixture
def doc(metadata, sequences, sample):
    return InifDocument(
        metadata=metadata,
        sequences=sequences,
        samples=[sample],
    )


@pytest.fixture
def doc_for_dedup():
    """Document with duplicate token patterns across samples."""
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="gpt2")),
        samples=[
            Sample(
                id=f"s{i}",
                tokens=[
                    Token(id=50256, token="<|endoftext|>"),
                    Token(id=1212, token="This"),
                    Token(id=318, token=" is"),
                    Token(id=257 + i, token=f" word{i}"),
                ],
            )
            for i in range(3)
        ],
    )
