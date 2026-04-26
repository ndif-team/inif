from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator

from pydantic import BaseModel, Field, model_validator

from inif._token_ops import CompiledRegexTag, compile_regex_tags
from inif.models import InifDocument, Metadata, ModelInfo, Sample, TokenOrSeqRef

RegexAnnotation = tuple[str | re.Pattern[str], str]


class FlatTokenStore(BaseModel):
    """Flat token arrays for high-throughput scans over INIF documents.

    This is an internal analysis representation, not a replacement file format.
    Tokens are addressed by global position; ``sample_offsets`` maps sample-local
    ranges back to ``sample_ids``. Annotations are sparse: each annotation name
    stores the global positions carrying it.
    """

    sample_ids: list[str]
    sample_offsets: list[int]
    token_ids: list[int | None]
    token_texts: list[str | None]
    sequence_ids: list[str | None] = Field(default_factory=list)
    annotations: dict[str, list[int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_arrays(self) -> FlatTokenStore:
        n_tokens = len(self.token_ids)
        assert len(self.token_texts) == n_tokens, (
            "token_texts must be parallel to token_ids"
        )
        if not self.sequence_ids:
            self.sequence_ids = [None] * n_tokens
        assert len(self.sequence_ids) == n_tokens, (
            "sequence_ids must be parallel to token_ids"
        )
        assert len(self.sample_offsets) == len(self.sample_ids) + 1, (
            "sample_offsets must have one more entry than sample_ids"
        )
        if self.sample_offsets:
            assert self.sample_offsets[0] == 0, "sample_offsets must start at 0"
            assert self.sample_offsets[-1] == n_tokens, (
                "last sample offset must equal number of tokens"
            )
            for prev, cur in zip(self.sample_offsets, self.sample_offsets[1:]):
                assert prev <= cur, "sample_offsets must be monotonically increasing"
        for name, positions in self.annotations.items():
            for pos in positions:
                assert 0 <= pos < n_tokens, (
                    f"Annotation {name!r} position {pos} out of range "
                    f"(n_tokens={n_tokens})"
                )
        return self

    @classmethod
    def from_document(
        cls,
        doc: InifDocument,
        expand_sequences: bool = True,
    ) -> FlatTokenStore:
        """Build a flat view from ``doc``.

        With ``expand_sequences=True`` (default), sequence refs are expanded
        into their constituent vocab tokens. The sample-flat ``sequence_ids``
        array still records which shared run each materialised position came
        from, even though the expanded :class:`TokenOrSeqRef` objects
        themselves don't carry that provenance.

        With ``expand_sequences=False``, ref positions land as
        ``token_ids[i] is None`` and ``token_texts[i]`` carries the target
        :class:`Sequence` id (mirroring how refs are stored on disk).
        """
        sample_ids: list[str] = []
        sample_offsets: list[int] = [0]
        token_ids: list[int | None] = []
        token_texts: list[str | None] = []
        sequence_ids: list[str | None] = []
        annotations: dict[str, list[int]] = {}

        for sample in doc.samples:
            sample_ids.append(sample.id)
            local_to_global: list[tuple[int, int]] = []
            for token in sample.tokens:
                local_start = len(token_ids)
                if expand_sequences and token.is_sequence_ref:
                    expanded = token.expanded_tokens(doc.sequences)
                else:
                    expanded = [token]
                # For refs, ``token.token`` carries the target Sequence id;
                # remember that as provenance regardless of whether we expand.
                provenance = token.token if token.is_sequence_ref else None
                for expanded_token in expanded:
                    token_ids.append(expanded_token.id)
                    token_texts.append(expanded_token.token)
                    sequence_ids.append(provenance)
                local_to_global.append((local_start, len(token_ids)))

            for annotation in sample.annotations:
                positions = annotations.setdefault(annotation.name, [])
                for start, end in annotation.ranges:
                    positions.extend(
                        range(local_to_global[start][0], local_to_global[end - 1][1])
                    )
            for positions in annotations.values():
                positions.sort()

            sample_offsets.append(len(token_ids))

        return cls(
            sample_ids=sample_ids,
            sample_offsets=sample_offsets,
            token_ids=token_ids,
            token_texts=token_texts,
            sequence_ids=sequence_ids,
            annotations=annotations,
        )

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    def sample_index(self, sample_id: str) -> int:
        sample_id = str(sample_id)
        try:
            return self.sample_ids.index(sample_id)
        except ValueError as exc:
            raise KeyError(f"Sample {sample_id} not found") from exc

    def sample_slice(self, sample_id: str) -> slice:
        idx = self.sample_index(sample_id)
        return slice(self.sample_offsets[idx], self.sample_offsets[idx + 1])

    def sample_position(self, global_position: int) -> tuple[str, int]:
        assert 0 <= global_position < self.total_tokens, (
            f"Global position {global_position} out of range "
            f"(n_tokens={self.total_tokens})"
        )
        idx = bisect_right(self.sample_offsets, global_position) - 1
        return self.sample_ids[idx], global_position - self.sample_offsets[idx]

    def iter_sample_tokens(
        self,
        sample_id: str,
    ) -> Iterator[tuple[int, int, str | None]]:
        token_range = self.sample_slice(sample_id)
        for pos in range(token_range.start, token_range.stop):
            yield pos, self.token_ids[pos], self.token_texts[pos]

    def find_regexes(
        self,
        regex_tags: list[RegexAnnotation],
    ) -> dict[str, list[int]]:
        """Return global token positions matching each regex tag spec."""
        return self.find_compiled_regexes(compile_regex_tags(regex_tags))

    def find_compiled_regexes(
        self,
        regex_tags: list[CompiledRegexTag],
    ) -> dict[str, list[int]]:
        matches: dict[str, list[int]] = {}
        for _, tag in regex_tags:
            matches.setdefault(tag, [])

        last_match_pos: dict[str, int] = {}
        for pos, text in enumerate(self.token_texts):
            if text is None:
                continue
            for pattern, tag in regex_tags:
                if last_match_pos.get(tag) == pos:
                    continue
                if pattern.search(text):
                    matches[tag].append(pos)
                    last_match_pos[tag] = pos
        return matches

    def annotate_regexes(self, regex_tags: list[RegexAnnotation]) -> None:
        """Apply regex annotations over flat arrays without constructing Tokens."""
        self.annotate_compiled_regexes(compile_regex_tags(regex_tags))

    def annotate_compiled_regexes(self, regex_tags: list[CompiledRegexTag]) -> None:
        existing = {tag: set(self.annotations.get(tag, [])) for _, tag in regex_tags}
        for _, tag in regex_tags:
            self.annotations.setdefault(tag, [])

        for pos, text in enumerate(self.token_texts):
            if text is None:
                continue
            for pattern, tag in regex_tags:
                if pos in existing[tag]:
                    continue
                if pattern.search(text):
                    self.annotations[tag].append(pos)
                    existing[tag].add(pos)

        for _, tag in regex_tags:
            self.annotations[tag].sort()

    def positions(self, annotation_name: str) -> list[int]:
        return list(self.annotations.get(annotation_name, []))

    def tokens(
        self, annotation_name: str
    ) -> list[tuple[int, int | None, str | None]]:
        return [
            (pos, self.token_ids[pos], self.token_texts[pos])
            for pos in self.annotations.get(annotation_name, [])
        ]

    def annotation_counts(self) -> dict[str, int]:
        return {name: len(positions) for name, positions in self.annotations.items()}

    def to_document(self, metadata: Metadata | None = None) -> InifDocument:
        """Materialize a flat analysis document.

        The result keeps sample ids, token ids/text, and sparse annotations.
        Sequence-ref provenance lives only on the flat store; the emitted
        :class:`TokenOrSeqRef` objects are plain vocab tokens (or refs when
        ``token_ids[pos] is None``). It intentionally does not recreate
        shared :class:`Sequence` objects.
        """
        samples: list[Sample] = []
        for sample_id, start, end in zip(
            self.sample_ids,
            self.sample_offsets,
            self.sample_offsets[1:],
        ):
            tokens: list[TokenOrSeqRef] = []
            for pos in range(start, end):
                token = TokenOrSeqRef(
                    id=self.token_ids[pos],
                    token=self.token_texts[pos] or "",
                )
                tokens.append(token)
            sample = Sample(id=sample_id, tokens=tokens)
            for name, positions in self.annotations.items():
                local_positions = [
                    pos - start for pos in positions if start <= pos < end
                ]
                sample.annotate_positions(name, local_positions)
            samples.append(sample)

        return InifDocument(
            metadata=metadata.model_copy(deep=True)
            if metadata is not None
            else Metadata(model=ModelInfo(name="flat-token-store")),
            samples=samples,
        )
