from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from inif._token_ops import PredicateTag
    from inif.selectors import TokenSelection
    from inif.tagging import TextTagMode


class Text(BaseModel):
    """A named text segment on a :class:`Sample`.

    A sample's ``texts`` list carries one entry per source segment — for
    chat inputs that's one per message (including the system prompt); for
    plain-text inputs it's one per input string. The default naming scheme
    is role-based for chat (``"system_0"``, ``"user_0"``, ``"assistant_0"``,
    ``"user_1"``, …) and index-based for plain text (``"text_0"``,
    ``"text_1"``, …).

    ``start`` and ``end`` are half-open token offsets covering the tokens
    that render this text in the chat-template output (``[start, end)``):
    for chat messages this captures the content plus any chat-template
    delimiters assigned to that message; for plain text it covers the
    whole token stream. They are ``None`` when the converter cannot map
    the text to a token range (e.g. lossy decode round-trip).

    ``children`` lets a message carry sub-segments — used for assistant
    turns whose body splits into reasoning / content / tool-calls (each
    becomes one child :class:`Text` with its own ``value``, ``start`` /
    ``end``, and ``metadata``). When children are present the parent's
    own ``value`` is typically empty and the renderer iterates the
    children. Each child's range must sit inside the parent's range.

    ``metadata`` is a free-form dict for caller-supplied context (e.g.
    turn index, tool-call payload — the tool-call children carry
    ``{"id", "type", "function"}`` here).
    """

    name: str
    value: str = ""
    start: int | None = None
    end: int | None = None
    children: list[Text] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Text:
        if self.start is not None or self.end is not None:
            assert self.start is not None and self.end is not None, (
                f"Text {self.name!r} must set both start and end or neither"
            )
            assert 0 <= self.start <= self.end, (
                f"Text {self.name!r} has invalid range [{self.start}, {self.end})"
            )
        for child in self.children:
            cs, ce = child.start, child.end
            if cs is None and ce is None:
                continue
            assert cs is not None and ce is not None, (
                f"Text child {child.name!r} of {self.name!r} must set both "
                "start and end or neither"
            )
            if self.start is not None and self.end is not None:
                assert self.start <= cs and ce <= self.end, (
                    f"Text child {child.name!r} range [{cs}, {ce}) sits "
                    f"outside parent {self.name!r} range "
                    f"[{self.start}, {self.end})"
                )
        return self


class ModelInfo(BaseModel):
    """Identifying info for the model whose tokens populate this document.

    ``name`` is the only required field. ``revision`` and ``huggingface_id``
    aid reproducibility; ``generation_config`` and ``loading_config`` are
    free-form bags of inference / loading parameters as recorded by the
    upstream framework.
    """

    name: str
    revision: str | None = None
    huggingface_id: str | None = None
    generation_config: dict = Field(default_factory=dict)
    loading_config: dict = Field(default_factory=dict)


class SourceEval(BaseModel):
    """Provenance metadata for a document built from an evaluation framework.

    Set by the Inspect AI / evaleval converters, absent for ad-hoc text
    inputs.
    """

    framework: str
    framework_version: str | None = None
    task: str | None = None
    task_version: str | None = None
    eval_id: str | None = None
    run_id: str | None = None
    extra: dict = Field(default_factory=dict)


class Metadata(BaseModel):
    """Document-level metadata: model, source eval, packages, timestamps.

    Required on every :class:`InifDocument`. ``created_at`` is a real
    ``datetime`` in memory; pydantic auto-parses ISO strings on load and
    re-emits ISO on save.
    """

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
    """A token run shared across every sample in a document.

    Both the token strings and their vocabulary ids are stored, so a
    ``deduplicate_sequences`` → ``expand_sequences`` round-trip preserves
    the exact ids. ``n_tokens`` must equal ``len(tokens)`` (enforced by a
    validator).
    """

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
    """An ad-hoc named position list on a sample.

    Unlike :class:`TokenAnnotation`, spans never auto-merge — adding a
    second span with the same ``name`` produces two ``Span`` objects on
    the sample. Use spans for free-form bookmarks and answer locations
    where the named-range model doesn't fit.
    """

    name: str
    positions: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class TokenAnnotation(BaseModel):
    """A named region on a sample expressed as half-open token ranges.

    ``ranges`` are ``[start, end)``; ``start < end`` is enforced. Use the
    sample-level helpers (:meth:`Sample.annotate`,
    :meth:`Sample.annotate_positions`) to add annotations — they merge
    adjacent / overlapping ranges with identical metadata into a single
    record.
    """

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
    """One scorer's evaluation result for a sample.

    ``value`` is whatever the scorer reports — bool for pass/fail, float
    for graded scorers, dict / list for structured outputs. ``answer`` is
    the extracted answer string (when applicable).
    """

    scorer: str
    value: str | int | float | bool | list | dict
    answer: str | None = None
    explanation: str | None = None
    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    """One self-contained tokenized generation trace.

    The required field is ``id``; everything else defaults to an empty list
    or ``None``. The first-class fields ``target``, ``references``,
    ``choices``, ``interaction_type``, ``error``, and ``sample_hash`` are
    aligned with the every_eval_ever schema so filters and viewers can rely
    on them without reaching into ``metadata``.

    Construction validates that every ``span.positions`` index and every
    ``annotation.ranges`` window lies within ``len(tokens)``; out-of-range
    values raise ``ValidationError``.
    """

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
        seen_names: set[str] = set()
        for annotation in self.annotations:
            assert annotation.name not in seen_names, (
                f"Duplicate annotation name '{annotation.name}' on sample "
                f"{self.id!r}: each name may appear at most once in "
                "Sample.annotations."
            )
            seen_names.add(annotation.name)
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
        """Add ``ranges`` to the annotation called ``name``.

        Each annotation name maps to exactly one entry on
        ``Sample.annotations``. If the entry already exists, the new ranges
        are appended and ``_merge_ranges`` collapses any overlapping or
        adjacent intervals into single half-open spans (so adding ``[6, 10)``
        to existing ``[5, 7)`` yields a single ``[5, 10)``). The existing
        entry's metadata is preserved — passing a different ``metadata`` on a
        subsequent call to the same name is a silent no-op for the metadata
        field; only the ranges are merged in.
        """
        merged = _merge_ranges(ranges)
        if not merged:
            return TokenAnnotation(name=name, metadata=dict(metadata or {}))
        n = len(self.tokens)
        for start, end in merged:
            assert 0 <= start < end <= n, (
                f"Annotation '{name}' range [{start}, {end}) out of range "
                f"for sample {self.id!r} (n_tokens={n})"
            )
        existing = next((a for a in self.annotations if a.name == name), None)
        if existing is not None:
            existing.ranges = _merge_ranges([*existing.ranges, *merged])
            return existing
        annotation = TokenAnnotation(
            name=name, ranges=merged, metadata=dict(metadata or {})
        )
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

        def _shift_text(text: Text) -> None:
            if text.start is None or text.end is None:
                return
            if text.end <= index:
                return
            if text.start > index:
                text.start += delta
                text.end += delta
            else:
                text.end += delta

        def _shift_text_tree(text: Text) -> None:
            _shift_text(text)
            for child in text.children:
                _shift_text_tree(child)

        for text in self.texts:
            _shift_text_tree(text)

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

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_by_position(
        self, positions: int | list[int] | slice
    ) -> "TokenSelection":
        """Select tokens at the given position(s).

        Accepts an ``int``, list of ``int``, or ``slice``. Returns a
        :class:`~inif.selectors.TokenSelection` with the matching tokens and
        their positions in this sample.
        """
        from inif.selectors import _select_by_position

        return _select_by_position(self, positions)

    def select_by_annotation(self, annotation_name: str) -> "TokenSelection":
        """Select tokens covered by ``annotation_name`` on this sample."""
        from inif.selectors import _select_by_annotation

        return _select_by_annotation(self, annotation_name)

    def select_by_sequence_id(self, seq_id: str) -> "TokenSelection":
        """Select sequence-ref tokens that point at ``seq_id``.

        A sequence ref is identified by ``id is None`` and carries the target
        :class:`Sequence` id in its ``token`` field. Useful for finding *where*
        a shared run is referenced in a sample without expanding it.
        """
        from inif.selectors import _select_by_sequence_id

        return _select_by_sequence_id(self, seq_id)

    def select_by_span(self, span_name: str) -> "TokenSelection":
        """Select tokens whose positions fall inside the named span."""
        from inif.selectors import _select_by_span

        return _select_by_span(self, span_name)

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag_by_regex(
        self,
        pattern: str | re.Pattern[str],
        tag: str,
        sequences: list[Sequence] | None = None,
    ) -> None:
        """Tag every token whose string matches ``pattern``.

        When ``sequences`` is provided, the search runs over the *expanded*
        view of the sample so tokens currently compressed inside a sequence
        ref are inspected too. Matches inside a ref cause the containing ref
        to be materialized in this sample (per-token information attaches to
        real Tokens; other refs and other samples are untouched).
        """
        self.tag_by_regexes([(pattern, tag)], sequences=sequences)

    def tag_by_regexes(
        self,
        regex_tags: list[tuple[str | re.Pattern[str], str]],
        sequences: list[Sequence] | None = None,
    ) -> None:
        """Apply multiple regex taggers in one token pass.

        The preferred entry point when several regex strategies are known up
        front. If ``sequences`` is provided, sequence refs are materialized
        only when at least one expanded token actually matches.
        """
        from inif.tagging import _tag_by_regexes

        _tag_by_regexes(self, regex_tags, sequences=sequences)

    def tag_by_text_regex(
        self,
        pattern: str,
        tag: str,
        mode: "TextTagMode | str" = "all",
    ) -> None:
        """Tag tokens whose concatenated text matches a regex.

        See :func:`inif.tagging._tag_by_text_regex` for the matching algorithm
        and the meaning of ``mode``.
        """
        from inif.tagging import TextTagMode, _tag_by_text_regex

        _tag_by_text_regex(self, pattern, tag, mode=TextTagMode(mode))

    def tag_by_predicate(
        self,
        predicate: Callable[["TokenOrSeqRef"], bool],
        tag: str,
    ) -> None:
        """Tag tokens that satisfy ``predicate``."""
        self.tag_by_predicates([(predicate, tag)])

    def tag_by_predicates(
        self,
        predicate_tags: list["PredicateTag"],
        sequences: list[Sequence] | None = None,
    ) -> None:
        """Apply multiple Python predicate taggers in one token pass."""
        from inif.tagging import _tag_by_predicates

        _tag_by_predicates(self, predicate_tags, sequences=sequences)

    def tag_chat_roles(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        sequences: list[Sequence] | None = None,
    ) -> None:
        """Tag tokens with their chat-template role.

        Works with any HuggingFace chat template. Content tokens get the role
        of their enclosing message; everything else (delimiters, role names,
        auto-generated text) is tagged ``"template"``. Must be called AFTER
        :meth:`InifDocument.deduplicate_sequences` if the document was
        deduplicated.
        """
        from inif.tagging import _tag_chat_roles

        _tag_chat_roles(self, messages, tokenizer, sequences=sequences)

    def tag_special_tokens(self, tokenizer: Any, tag: str = "special") -> None:
        """Tag tokens whose id appears in ``tokenizer.all_special_ids``."""
        from inif.tagging import _tag_special_tokens

        _tag_special_tokens(self, tokenizer, tag=tag)

    def create_span_from_tag(self, tag: str, span_name: str) -> Span:
        """Build a :class:`Span` whose positions are everywhere ``tag`` is set.

        The new span is appended to ``self.spans`` and returned. Existing
        spans are left untouched.
        """
        from inif.tagging import _create_span_from_tag

        return _create_span_from_tag(self, tag, span_name)

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
    """The top-level container — metadata, deduplicated sequences, samples.

    ``total_samples`` is a computed property (``len(samples)``); there is
    no stored field for it. Use :meth:`subset` to derive a self-contained
    sub-document with sequences pruned to those referenced by the kept
    samples.
    """

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

    # ------------------------------------------------------------------
    # Serialization & IO
    # ------------------------------------------------------------------

    def to_dict(self, compact: bool = True) -> dict:
        """Convert this document to a JSON-ready dict.

        Uses pydantic's ``mode="json"`` so datetimes serialize as ISO-8601
        strings. With ``compact=True`` (default), default-valued and ``None``
        fields are stripped — for example, sequence-ref tokens
        (``id is None``) serialize to a single-key
        ``{"token": "<seq_id>"}`` dict.
        """
        from inif.io import _to_dict

        return _to_dict(self, compact=compact)

    def save(
        self,
        path: str | Path,
        compress: bool | None = None,
        compact: bool = True,
        indent: int | None = 4,
    ) -> None:
        """Save this document to ``path``.

        ``.inif`` paths are written as indexed compressed archives;
        ``.inif.json`` / ``.json`` paths are written as plain JSON. ``indent``
        controls pretty-printing for plain JSON (``None`` produces a
        single-line dump).
        """
        from inif.io import _save

        _save(self, path, compress=compress, compact=compact, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> InifDocument:
        """Build an :class:`InifDocument` from a JSON-ready dict."""
        from inif.io import _from_dict

        return _from_dict(data)

    @classmethod
    def load(cls, path: str | Path, compress: bool | None = None) -> InifDocument:
        """Load an :class:`InifDocument` from ``path``.

        ``.inif`` paths are read as indexed archives; ``.inif.json`` /
        ``.json`` paths are read as plain JSON.
        """
        from inif.io import _load

        return _load(path, compress=compress)

    # ------------------------------------------------------------------
    # Sequence (de)duplication
    # ------------------------------------------------------------------

    def deduplicate_sequences(self, min_length: int = 5) -> InifDocument:
        """Find token runs common to ALL samples and replace them with refs.

        Returns a new document; this one is not modified. Common runs of
        length ``min_length`` or more are extracted into :class:`Sequence`
        objects and the tokens carrying them in each sample are swapped for
        a sequence-ref :class:`TokenOrSeqRef`.
        """
        from inif.sequences import _deduplicate_sequences

        return _deduplicate_sequences(self, min_length=min_length)

    def expand_sequences(self) -> InifDocument:
        """Expand all sequence references back to flat vocab tokens.

        Returns a new document whose tokens are independent from this one and
        whose ``sequences`` list is empty. Re-running
        :meth:`deduplicate_sequences` rediscovers the same shared runs.
        """
        from inif.sequences import _expand_sequences

        return _expand_sequences(self)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def filter_samples_by_score(
        self,
        scorer: str,
        predicate: Callable[[str | int | float | bool | list | dict], bool],
    ) -> list[Sample]:
        """Return samples whose ``scorer`` value satisfies ``predicate``.

        Compose with the per-sample selection methods (e.g.
        :meth:`Sample.select_by_annotation`) on each returned sample to drill
        down to specific tokens.
        """
        from inif.selectors import _filter_samples_by_score

        return _filter_samples_by_score(self, scorer, predicate)

    # ------------------------------------------------------------------
    # Tagging (document-wide)
    # ------------------------------------------------------------------

    def tag_by_regex(self, pattern: str | re.Pattern[str], tag: str) -> None:
        """Tag every token across all samples whose string matches ``pattern``."""
        self.tag_by_regexes([(pattern, tag)])

    def tag_by_regexes(
        self,
        regex_tags: list[tuple[str | re.Pattern[str], str]],
    ) -> None:
        """Apply multiple regex taggers across all samples in one pass per sample."""
        from inif.tagging import _tag_by_regexes_doc

        _tag_by_regexes_doc(self, regex_tags)

    def tag_by_text_regex(
        self,
        pattern: str,
        tag: str,
        mode: "TextTagMode | str" = "all",
    ) -> None:
        """Apply text-based regex tagging across all samples."""
        from inif.tagging import TextTagMode, _tag_by_text_regex_doc

        _tag_by_text_regex_doc(self, pattern, tag, mode=TextTagMode(mode))

    def tag_by_predicates(self, predicate_tags: list["PredicateTag"]) -> None:
        """Apply multiple Python predicate taggers across all samples."""
        from inif.tagging import _tag_by_predicates_doc

        _tag_by_predicates_doc(self, predicate_tags)

    def tag_chat_roles(
        self,
        messages_per_sample: list[list[dict[str, str]]],
        tokenizer: Any,
    ) -> None:
        """Tag chat roles for every sample in this document.

        ``messages_per_sample`` must be a list with one message list per
        sample, in the same order as ``self.samples``.
        """
        from inif.tagging import _tag_chat_roles_doc

        _tag_chat_roles_doc(self, messages_per_sample, tokenizer)

    def remove_annotation(self, name: str) -> None:
        """Remove every annotation named ``name`` across all samples."""
        for sample in self.samples:
            sample.remove_annotation(name)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def render_html(
        self,
        compact: bool = False,
        title: str | None = None,
        tokenizer: Any = None,
    ) -> str:
        """Render this document as a self-contained HTML string.

        When ``tokenizer`` is provided, the tokenizer's byte-level
        representation of newlines (e.g. ``Ċ`` for GPT-2 family) is detected
        automatically so that visual line breaks are inserted after newline
        tokens.
        """
        from inif.viewer import _render_html

        return _render_html(self, compact=compact, title=title, tokenizer=tokenizer)

    def show(
        self,
        compact: bool = False,
        title: str | None = None,
        tokenizer: Any = None,
    ) -> Any:
        """Display this document as HTML in a Jupyter notebook."""
        from inif.viewer import _show

        return _show(self, compact=compact, title=title, tokenizer=tokenizer)

    def save_html(
        self,
        path: str | Path,
        compact: bool = False,
        title: str | None = None,
        source: str | Path | None = None,
        tokenizer: Any = None,
    ) -> None:
        """Save this document as a self-contained HTML file.

        When ``title`` is not given, the source filename is used if
        available, otherwise falls back to the model name.
        """
        from inif.viewer import _save_html

        _save_html(
            self,
            path,
            compact=compact,
            title=title,
            source=source,
            tokenizer=tokenizer,
        )


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
