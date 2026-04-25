from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


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
    created_at: datetime | None = None
    total_time: float | None = None
    extra: dict = Field(default_factory=dict)


class Token(BaseModel):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    # Declaration order drives serialization order: ``token`` / ``seq_id``
    # first, ``id`` last — so a rendered token dict reads left-to-right as
    # ``{"token": "...", "id": N}`` or ``{"seq_id": "...", "id": -1}``.
    token: str | None = None
    sequence_id: str | None = Field(
        default=None,
        serialization_alias="seq_id",
        validation_alias=AliasChoices("seq_id", "sequence_id"),
    )
    id: int

    @model_validator(mode="after")
    def _validate_token(self) -> Token:
        if self.is_sequence_ref:
            assert self.sequence_id is not None, (
                "Sequence ref tokens (id < 0) must have sequence_id"
            )
        else:
            assert self.token is not None, (
                f"Vocabulary tokens (id >= 0) must have a token string, "
                f"got id={self.id}"
            )
        return self

    @property
    def is_sequence_ref(self) -> bool:
        return self.id < 0

    # ------------------------------------------------------------------
    # Extras API
    #
    # ``extra="allow"`` lets pydantic carry arbitrary fields, but the
    # storage is split between ``self.__dict__`` (attribute access) and
    # ``self.model_extra`` (serialization). These helpers keep the two
    # in sync so callers don't have to.
    # ------------------------------------------------------------------

    def _extra(self) -> dict[str, Any] | None:
        # ``model_extra`` is typed as a read-only mapping by some stubs but
        # pydantic v2 returns a real mutable dict at runtime, which the
        # set_/pop_extra helpers below rely on.
        return self.model_extra

    def set_extra(self, key: str, value: Any) -> None:
        """Set an extra field, keeping attribute access and serialization in sync."""
        self.__dict__[key] = value
        extra = self._extra()
        if extra is not None:
            extra[key] = value

    def get_extra(self, key: str, default: Any = None) -> Any:
        extra = self._extra()
        if extra is not None and key in extra:
            return extra[key]
        return default

    def has_extra(self, key: str) -> bool:
        extra = self._extra()
        return extra is not None and key in extra

    def pop_extra(self, key: str, default: Any = None) -> Any:
        val = self.__dict__.pop(key, default)
        extra = self._extra()
        if extra is not None and key in extra:
            val = extra.pop(key)
        return val

    @property
    def extras(self) -> dict[str, Any]:
        extra = self._extra()
        return dict(extra) if extra is not None else {}

    # --- tags are a specific kind of extra ---------------------------

    @property
    def tags(self) -> list[str]:
        return self.get_extra("tags", []) or []

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def add_tag(self, tag: str) -> None:
        tags = list(self.tags)
        if tag in tags:
            return
        tags.append(tag)
        self.set_extra("tags", tags)

    def remove_tag(self, tag: str) -> None:
        tags = list(self.tags)
        if tag not in tags:
            return
        tags.remove(tag)
        if tags:
            self.set_extra("tags", tags)
        else:
            self.pop_extra("tags")

    # ------------------------------------------------------------------
    # Sequence expansion
    # ------------------------------------------------------------------

    def expanded_tokens(self, sequences: list[Sequence]) -> list[Token]:
        if not self.is_sequence_ref:
            return [self]
        assert self.sequence_id is not None, "Sequence ref token must have sequence_id"
        seq_map = {s.id: s for s in sequences}
        assert self.sequence_id in seq_map, f"Sequence '{self.sequence_id}' not found"
        seq = seq_map[self.sequence_id]
        return [Token(id=t.id, token=t.token, sequence_id=seq.id) for t in seq.tokens]


class Sequence(BaseModel):
    id: str
    n_tokens: int
    tokens: list[Token]
    text: str | None = None

    @model_validator(mode="after")
    def _validate_sequence(self) -> Sequence:
        assert self.n_tokens == len(self.tokens), (
            f"n_tokens must equal len(tokens), "
            f"got {self.n_tokens} and {len(self.tokens)}"
        )
        return self


class Span(BaseModel):
    name: str
    positions: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SampleScore(BaseModel):
    scorer: str
    value: str | int | float | bool | list | dict
    answer: str | None = None
    explanation: str | None = None
    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    id: str
    tokens: list[Token] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    scores: list[SampleScore] = Field(default_factory=list)
    target: str | None = None
    metadata: dict = Field(default_factory=dict)
    total_time: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> Any:
        # Inspect AI and other eval frameworks often use int sample ids; coerce
        # to str so downstream lookups are unambiguous.
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        return v

    @model_validator(mode="after")
    def _validate_span_bounds(self) -> Sample:
        n = len(self.tokens)
        for span in self.spans:
            for pos in span.positions:
                assert 0 <= pos < n, (
                    f"Span '{span.name}' position {pos} out of range "
                    f"for sample {self.id!r} (n_tokens={n})"
                )
        return self

    def get_expanded_tokens(self, sequences: list[Sequence]) -> list[Token]:
        # Build the sequence map once per call instead of rebuilding it inside
        # ``Token.expanded_tokens`` for every token (which would make this
        # O(n_tokens × n_sequences)).
        seq_map = {s.id: s for s in sequences}
        result: list[Token] = []
        for token in self.tokens:
            if not token.is_sequence_ref:
                result.append(token)
                continue
            assert token.sequence_id is not None, (
                "Sequence ref token must have sequence_id"
            )
            assert token.sequence_id in seq_map, (
                f"Sequence '{token.sequence_id}' not found"
            )
            seq = seq_map[token.sequence_id]
            for t in seq.tokens:
                result.append(Token(id=t.id, token=t.token, sequence_id=seq.id))
        return result

    def get_tokens_by_positions(self, positions: list[int]) -> list[Token]:
        pos_set = set(positions)
        return [t for i, t in enumerate(self.tokens) if i in pos_set]

    def materialize_position(
        self,
        expanded_position: int,
        sequences: list[Sequence],
    ) -> tuple[int, Token]:
        """Ensure ``expanded_position`` is a real Token in ``self.tokens``.

        If the position falls inside a sequence ref, the entire ref is
        expanded in place into its constituent tokens (with their original
        ids preserved). Other refs are left untouched. This is the
        primitive that lets callers attach interpretability outputs (logit
        lens scores, probe activations, etc.) to specific positions even
        when those positions are currently compressed inside a shared
        sequence.

        Returns ``(actual_index_in_self_tokens, the_token)``.
        """
        seq_map = {s.id: s for s in sequences}
        current = 0
        for i, tok in enumerate(self.tokens):
            if tok.is_sequence_ref:
                assert tok.sequence_id is not None, (
                    "Sequence ref token must have sequence_id"
                )
                assert tok.sequence_id in seq_map, (
                    f"Sequence '{tok.sequence_id}' not found"
                )
                seq = seq_map[tok.sequence_id]
                n = seq.n_tokens
                if current <= expanded_position < current + n:
                    offset = expanded_position - current
                    expanded = [
                        Token(id=t.id, token=t.token, sequence_id=seq.id)
                        for t in seq.tokens
                    ]
                    self.tokens = self.tokens[:i] + expanded + self.tokens[i + 1 :]
                    return i + offset, self.tokens[i + offset]
                current += n
            else:
                if current == expanded_position:
                    return i, tok
                current += 1
        raise IndexError(
            f"Expanded position {expanded_position} out of range "
            f"({current} expanded tokens in sample {self.id!r})"
        )


class TokenExtras(BaseModel):
    """Conventional extra fields recognized by inif tooling.

    Tokens accept arbitrary extras (``model_config["extra"] = "allow"``); this
    model documents the well-known names so external validators, viewers, and
    converters know what to expect. Not used at runtime — it only contributes
    to the JSON schema as a ``$defs`` entry.

    Conventional ``role`` values: ``"system"``, ``"user"``, ``"assistant"``,
    ``"template"``.

    Conventional tags include: ``"generated"`` (model-produced),
    ``"special"`` (special tokenizer ids), ``"number"``, ``"entity"``.
    """

    tags: list[str] = Field(default_factory=list, description="User-assigned labels")
    role: str | None = Field(
        default=None,
        description="Chat-template role: system | user | assistant | template",
    )
    logprob: float | None = Field(
        default=None, description="Per-token log-probability from the model"
    )
    logit_lens: dict | None = Field(
        default=None,
        description="Per-layer logit-lens output keyed by layer name",
    )


class InifDocument(BaseModel):
    metadata: Metadata
    sequences: list[Sequence] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)

    @property
    def sequence_map(self) -> dict[str, Sequence]:
        return {s.id: s for s in self.sequences}

    @property
    def total_samples(self) -> int:
        return len(self.samples)

    def get_sample(self, sample_id: str) -> Sample:
        sample_id = str(sample_id)
        for s in self.samples:
            if s.id == sample_id:
                return s
        raise KeyError(f"Sample {sample_id} not found")

    def filter_samples(self, predicate: Callable[[Sample], bool]) -> list[Sample]:
        return [s for s in self.samples if predicate(s)]

    def subset(self, predicate: Callable[[Sample], bool]) -> InifDocument:
        """Return a new ``InifDocument`` with only samples matching ``predicate``.

        Sequences not referenced by any retained sample are dropped, so the
        output stays self-contained and minimal. Metadata is copied as-is.

        The returned document is independent from this one: samples, sequences,
        metadata, tokens, and nested extras are deep-copied.
        """
        kept_samples = [s.model_copy(deep=True) for s in self.samples if predicate(s)]
        referenced: set[str] = set()
        for sample in kept_samples:
            for tok in sample.tokens:
                if tok.is_sequence_ref and tok.sequence_id is not None:
                    referenced.add(tok.sequence_id)
        kept_sequences = [
            s.model_copy(deep=True) for s in self.sequences if s.id in referenced
        ]
        return InifDocument(
            metadata=self.metadata.model_copy(deep=True),
            sequences=kept_sequences,
            samples=kept_samples,
        )

    def to_dict(self, compact: bool = True) -> dict:
        """See :func:`inif.io.to_dict`."""
        from inif.io import to_dict

        return to_dict(self, compact=compact)

    def save(
        self,
        path: str | Path,
        compress: bool | None = None,
        compact: bool = True,
        indent: int | None = 4,
    ) -> None:
        """See :func:`inif.io.save`."""
        from inif.io import save

        save(self, path, compress=compress, compact=compact, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> InifDocument:
        """See :func:`inif.io.from_dict`."""
        from inif.io import from_dict

        return from_dict(data)

    @classmethod
    def load(cls, path: str | Path, compress: bool | None = None) -> InifDocument:
        """See :func:`inif.io.load`."""
        from inif.io import load

        return load(path, compress=compress)
