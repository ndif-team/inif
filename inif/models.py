from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, Field, field_validator, model_validator


class Text(BaseModel):
    """A named text segment on a :class:`Sample`.

    A sample's ``texts`` list carries one entry per source segment — for
    chat inputs that's one per message (including the system prompt); for
    plain-text inputs it's one per input string. The default naming scheme
    is role-based for chat (``"system_0"``, ``"user_0"``, ``"assistant_0"``,
    ``"user_1"``, …) and index-based for plain text (``"text_0"``,
    ``"text_1"``, …).

    ``metadata`` is a free-form dict for caller-supplied context (e.g.
    turn index, tool-call payload).
    """

    name: str
    value: str
    metadata: dict = Field(default_factory=dict)


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


class TokenOrSeqRef(BaseModel):
    """A token entry — either a vocabulary token or a reference to a Sequence.

    A vocabulary token has an integer ``id`` (the model's vocab id) and a
    ``token`` string (the decoded piece). A sequence reference has
    ``id is None`` and uses the ``token`` field to carry the target
    :class:`Sequence` id (a string). The presence / absence of ``id`` is
    the discriminator — there is no separate ``sequence_id`` field.

    Tokens accept arbitrary extra fields via ``model_config["extra"] =
    "allow"``. Use the :meth:`set_extra` / :meth:`get_extra` helpers (not
    direct attribute assignment) so values stay in sync between
    ``__dict__`` and ``model_extra``.
    """

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    # Declaration order drives serialization order: ``token`` first, ``id``
    # last — so a rendered dict reads left-to-right as
    # ``{"token": "...", "id": N}`` for vocab tokens and
    # ``{"token": "seq_0"}`` (id elided as default) for sequence refs.
    token: str
    id: int | None = None

    @model_validator(mode="after")
    def _validate_token(self) -> TokenOrSeqRef:
        # ``token`` is always required: a vocabulary token uses it as the
        # decoded string; a sequence ref uses it as the target Sequence id.
        assert self.token is not None and self.token != "" or self.id is not None, (
            "TokenOrSeqRef requires a non-empty `token` string "
            "(decoded piece for vocab tokens, sequence id for refs)"
        )
        return self

    @property
    def is_sequence_ref(self) -> bool:
        return self.id is None

    @property
    def sequence_id(self) -> str | None:
        """Convenience accessor returning the target Sequence id for refs.

        Returns the value of ``self.token`` when this is a sequence ref
        (``id is None``); ``None`` for vocabulary tokens. Provided so the
        ``"is this a ref to seq X?"`` check reads naturally without
        callers having to remember the ``id is None`` invariant.
        """
        return self.token if self.is_sequence_ref else None

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

    # ------------------------------------------------------------------
    # Sequence expansion
    # ------------------------------------------------------------------

    def expanded_tokens(self, sequences: list[Sequence]) -> list[TokenOrSeqRef]:
        """Materialize this entry into vocab tokens (no-op for vocab tokens).

        For a sequence ref, ``self.token`` names the target sequence; the
        method looks it up in ``sequences`` and returns fresh vocab tokens
        with the original ids preserved.
        """
        if not self.is_sequence_ref:
            return [self]
        seq_map = {s.id: s for s in sequences}
        assert self.token in seq_map, f"Sequence '{self.token}' not found"
        seq = seq_map[self.token]
        return [TokenOrSeqRef(id=t.id, token=t.token) for t in seq.tokens]


class Sequence(BaseModel):
    id: str
    n_tokens: int
    tokens: list[TokenOrSeqRef]
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


class TokenAnnotation(BaseModel):
    name: str
    ranges: list[tuple[int, int]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ranges(self) -> TokenAnnotation:
        for start, end in self.ranges:
            assert start < end, (
                f"Annotation {self.name!r} has invalid range [{start}, {end})"
            )
        return self


class SampleScore(BaseModel):
    scorer: str
    value: str | int | float | bool | list | dict
    answer: str | None = None
    explanation: str | None = None
    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    id: str
    tokens: list[TokenOrSeqRef] = Field(default_factory=list)
    texts: list[Text] = Field(default_factory=list)
    annotations: list[TokenAnnotation] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    scores: list[SampleScore] = Field(default_factory=list)
    target: str | None = None
    references: list[str] = Field(default_factory=list)
    choices: list[str] | None = None
    interaction_type: str | None = None
    error: str | None = None
    sample_hash: str | None = None
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
    def _validate_position_bounds(self) -> Sample:
        n = len(self.tokens)
        for span in self.spans:
            for pos in span.positions:
                assert 0 <= pos < n, (
                    f"Span '{span.name}' position {pos} out of range "
                    f"for sample {self.id!r} (n_tokens={n})"
                )
        for annotation in self.annotations:
            annotation.ranges = _merge_ranges(annotation.ranges)
            for start, end in annotation.ranges:
                assert 0 <= start < end <= n, (
                    f"Annotation '{annotation.name}' range [{start}, {end}) "
                    f"out of range for sample {self.id!r} (n_tokens={n})"
                )
        return self

    def annotate(
        self,
        name: str,
        ranges: Iterable[tuple[int, int]],
        metadata: dict | None = None,
    ) -> TokenAnnotation:
        merged = _merge_ranges(ranges)
        if not merged:
            return TokenAnnotation(name=name, metadata=dict(metadata or {}))
        n = len(self.tokens)
        for start, end in merged:
            assert 0 <= start < end <= n, (
                f"Annotation '{name}' range [{start}, {end}) out of range "
                f"for sample {self.id!r} (n_tokens={n})"
            )
        md = dict(metadata or {})
        for annotation in self.annotations:
            if annotation.name == name and annotation.metadata == md:
                annotation.ranges = _merge_ranges([*annotation.ranges, *merged])
                return annotation
        annotation = TokenAnnotation(name=name, ranges=merged, metadata=md)
        self.annotations.append(annotation)
        return annotation

    def annotate_positions(
        self,
        name: str,
        positions: Iterable[int],
        metadata: dict | None = None,
    ) -> TokenAnnotation:
        return self.annotate(name, _positions_to_ranges(positions), metadata=metadata)

    def annotation_positions(self, name: str) -> list[int]:
        positions: set[int] = set()
        for annotation in self.annotations:
            if annotation.name != name:
                continue
            for start, end in annotation.ranges:
                positions.update(range(start, end))
        return sorted(positions)

    def remove_annotation(self, name: str) -> None:
        self.annotations = [ann for ann in self.annotations if ann.name != name]

    def _replace_token_with_tokens(
        self, index: int, replacement: list[TokenOrSeqRef]
    ) -> None:
        assert 0 <= index < len(self.tokens), f"Token index {index} out of range"
        assert replacement, "Replacement must contain at least one token"
        delta = len(replacement) - 1
        self.tokens = self.tokens[:index] + replacement + self.tokens[index + 1 :]
        if delta == 0:
            return
        for annotation in self.annotations:
            adjusted: list[tuple[int, int]] = []
            for start, end in annotation.ranges:
                if end <= index:
                    adjusted.append((start, end))
                elif start > index:
                    adjusted.append((start + delta, end + delta))
                else:
                    adjusted.append((start, end + delta))
            annotation.ranges = _merge_ranges(adjusted)

    def get_expanded_tokens(self, sequences: list[Sequence]) -> list[TokenOrSeqRef]:
        # Build the sequence map once per call instead of rebuilding it inside
        # ``TokenOrSeqRef.expanded_tokens`` for every token (which would make
        # this O(n_tokens × n_sequences)).
        seq_map = {s.id: s for s in sequences}
        result: list[TokenOrSeqRef] = []
        for token in self.tokens:
            if not token.is_sequence_ref:
                result.append(token)
                continue
            assert token.token in seq_map, f"Sequence '{token.token}' not found"
            seq = seq_map[token.token]
            for t in seq.tokens:
                result.append(TokenOrSeqRef(id=t.id, token=t.token))
        return result

    def get_tokens_by_positions(self, positions: list[int]) -> list[TokenOrSeqRef]:
        pos_set = set(positions)
        return [t for i, t in enumerate(self.tokens) if i in pos_set]

    def materialize_position(
        self,
        expanded_position: int,
        sequences: list[Sequence],
    ) -> tuple[int, TokenOrSeqRef]:
        """Ensure ``expanded_position`` is a real vocab token in ``self.tokens``.

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
                assert tok.token in seq_map, f"Sequence '{tok.token}' not found"
                seq = seq_map[tok.token]
                n = seq.n_tokens
                if current <= expanded_position < current + n:
                    offset = expanded_position - current
                    expanded = [
                        TokenOrSeqRef(id=t.id, token=t.token) for t in seq.tokens
                    ]
                    self._replace_token_with_tokens(i, expanded)
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

    Repeated token labels such as chat roles, ``"generated"``, and
    ``"reasoning"`` are stored in :class:`TokenAnnotation` ranges on
    :class:`Sample`, not repeated on every token.
    """

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
                if tok.is_sequence_ref:
                    referenced.add(tok.token)
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


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in ranges if start < end)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _positions_to_ranges(positions: Iterable[int]) -> list[tuple[int, int]]:
    sorted_positions = sorted({int(pos) for pos in positions})
    if not sorted_positions:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_positions[0]
    for pos in sorted_positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        ranges.append((start, prev + 1))
        start = prev = pos
    ranges.append((start, prev + 1))
    return ranges
