from __future__ import annotations

from typing import Callable, Union

from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    name: str
    revision: str | None = None
    huggingface_id: str | None = None
    generation_config: dict = Field(default_factory=dict)
    loading_config: dict = Field(default_factory=dict)


class SourceEval(BaseModel):
    framework: str
    framework_version: str | None = None
    task: str | None = None
    task_version: str | None = None
    eval_id: str | None = None
    run_id: str | None = None
    extra: dict = Field(default_factory=dict)


class Metadata(BaseModel):
    model: ModelInfo
    inif_version: str = "0.1"
    packages: dict = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    source_eval: SourceEval | None = None
    created_at: str | None = None
    total_samples: int | None = None
    total_time: float | None = None
    extra: dict = Field(default_factory=dict)


class Sequence(BaseModel):
    id: str
    tokens: list[str]
    n_tokens: int
    text: str | None = None

    @model_validator(mode="after")
    def _validate_sequence(self) -> Sequence:
        assert self.n_tokens == len(self.tokens), (
            f"n_tokens must equal len(tokens), "
            f"got {self.n_tokens} and {len(self.tokens)}"
        )
        return self


class Token(BaseModel):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    id: int
    token: str | None = None
    sequence_id: str | None = None

    @property
    def is_sequence_ref(self) -> bool:
        return self.id < 0

    def expanded_tokens(self, sequences: list[Sequence]) -> list[Token]:
        if not self.is_sequence_ref:
            return [self]
        assert self.sequence_id is not None, "Sequence ref token must have sequence_id"
        seq_map = {s.id: s for s in sequences}
        assert self.sequence_id in seq_map, f"Sequence '{self.sequence_id}' not found"
        seq = seq_map[self.sequence_id]
        return [Token(id=0, token=tstr, sequence_id=seq.id) for tstr in seq.tokens]


class Span(BaseModel):
    name: str
    positions: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SampleScore(BaseModel):
    scorer: str
    value: Union[str, int, float, bool, list, dict]
    answer: str | None = None
    explanation: str | None = None
    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    id: Union[str, int]
    tokens: list[Token] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    scores: list[SampleScore] = Field(default_factory=list)
    target: str | None = None
    metadata: dict = Field(default_factory=dict)
    total_time: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def get_expanded_tokens(self, sequences: list[Sequence]) -> list[Token]:
        result = []
        for token in self.tokens:
            result.extend(token.expanded_tokens(sequences))
        return result

    def get_tokens_by_positions(self, positions: list[int]) -> list[Token]:
        pos_set = set(positions)
        return [t for i, t in enumerate(self.tokens) if i in pos_set]


class InifDocument(BaseModel):
    metadata: Metadata
    sequences: list[Sequence] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)

    @property
    def sequence_map(self) -> dict[str, Sequence]:
        return {s.id: s for s in self.sequences}

    def get_sample(self, sample_id: str | int) -> Sample:
        for s in self.samples:
            if s.id == sample_id:
                return s
        raise KeyError(f"Sample {sample_id} not found")

    def filter_samples(self, predicate: Callable[[Sample], bool]) -> list[Sample]:
        return [s for s in self.samples if predicate(s)]
