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
    Text,
    TokenOrSeqRef,
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
            n_tokens=3,
            tokens=[
                TokenOrSeqRef(id=50256, token="<|endoftext|>"),
                TokenOrSeqRef(id=1212, token="This"),
                TokenOrSeqRef(id=318, token=" is"),
            ],
        ),
        Sequence(
            id="seq_1",
            n_tokens=1,
            tokens=[TokenOrSeqRef(id=50256, token="<|endoftext|>")],
        ),
    ]


@pytest.fixture
def sample_tokens():
    tok_data = TokenOrSeqRef(
        id=764,
        token=".",
        logprob=-0.5,
        data={"logit_lens": {"layer_5": {"top_token": "."}}},
    )

    return [
        TokenOrSeqRef(id=None, token="seq_0"),  # ref to seq 0
        TokenOrSeqRef(id=257, token=" a"),
        TokenOrSeqRef(id=1332, token=" test"),
        tok_data,
        TokenOrSeqRef(id=None, token="seq_1"),  # ref to seq 1
    ]


@pytest.fixture
def sample(sample_tokens):
    return Sample(
        id="sample_0",
        tokens=sample_tokens,
        texts=[
            Text(name="system_0", value="System prompt"),
            Text(name="user_0", value=" a test."),
        ],
        annotations=[
            # index in sample.tokens list
            {"name": "content", "ranges": [(2, 3)]},
        ],
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
            TokenOrSeqRef(id=50256, token="<|endoftext|>"),
            TokenOrSeqRef(id=1212, token="This"),
            TokenOrSeqRef(id=318, token=" is"),
            TokenOrSeqRef(id=257, token=" a"),
            TokenOrSeqRef(id=1332, token=" test"),
            TokenOrSeqRef(id=764, token="."),
        ],
        texts=[Text(name="text_0", value="This is a test.")],
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
                    TokenOrSeqRef(id=50256, token="<|endoftext|>"),
                    TokenOrSeqRef(id=1212, token="This"),
                    TokenOrSeqRef(id=318, token=" is"),
                    TokenOrSeqRef(id=257 + i, token=f" word{i}"),
                ],
            )
            for i in range(3)
        ],
    )
