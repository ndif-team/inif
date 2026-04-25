from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator

from pydantic import BaseModel, Field, model_validator

from inif._token_ops import CompiledRegexTag, compile_regex_tags
from inif.models import InifDocument, Metadata, ModelInfo, Sample, Token

RegexTag = tuple[str | re.Pattern[str], str]


class FlatTokenStore(BaseModel):
    """Flat token arrays for high-throughput scans over INIF documents.

    This is an internal analysis representation, not a replacement file format.
    Tokens are addressed by global position; ``sample_offsets`` maps sample-local
    ranges back to ``sample_ids``. Tags are sparse: each tag stores the global
    positions carrying it.
    """

    sample_ids: list[str]
    sample_offsets: list[int]
    token_ids: list[int]
    token_texts: list[str | None]
    sequence_ids: list[str | None] = Field(default_factory=list)
    tags: dict[str, list[int]] = Field(default_factory=dict)

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
        for tag, positions in self.tags.items():
            for pos in positions:
                assert 0 <= pos < n_tokens, (
                    f"Tag {tag!r} position {pos} out of range (n_tokens={n_tokens})"
                )
        return self

    @classmethod
    def from_document(
        cls,
        doc: InifDocument,
        expand_sequences: bool = True,
    ) -> FlatTokenStore:
        """Build a flat view from ``doc``.

        With ``expand_sequences=True`` (default), sequence refs are expanded and
        their ``sequence_id`` is retained as provenance on the resulting flat
        tokens.
        """
        sample_ids: list[str] = []
        sample_offsets: list[int] = [0]
        token_ids: list[int] = []
        token_texts: list[str | None] = []
        sequence_ids: list[str | None] = []
        tags: dict[str, list[int]] = {}

        for sample in doc.samples:
            sample_ids.append(sample.id)
            tokens = (
                sample.get_expanded_tokens(doc.sequences)
                if expand_sequences
                else sample.tokens
            )
            for token in tokens:
                pos = len(token_ids)
                token_ids.append(token.id)
                token_texts.append(token.token)
                sequence_ids.append(token.sequence_id)
                for tag in token.tags:
                    tags.setdefault(tag, []).append(pos)
            sample_offsets.append(len(token_ids))

        return cls(
            sample_ids=sample_ids,
            sample_offsets=sample_offsets,
            token_ids=token_ids,
            token_texts=token_texts,
            sequence_ids=sequence_ids,
            tags=tags,
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
        regex_tags: list[RegexTag],
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

    def tag_regexes(self, regex_tags: list[RegexTag]) -> None:
        """Apply regex tags over flat arrays without constructing Token objects."""
        self.tag_compiled_regexes(compile_regex_tags(regex_tags))

    def tag_compiled_regexes(self, regex_tags: list[CompiledRegexTag]) -> None:
        existing = {tag: set(self.tags.get(tag, [])) for _, tag in regex_tags}
        for _, tag in regex_tags:
            self.tags.setdefault(tag, [])

        for pos, text in enumerate(self.token_texts):
            if text is None:
                continue
            for pattern, tag in regex_tags:
                if pos in existing[tag]:
                    continue
                if pattern.search(text):
                    self.tags[tag].append(pos)
                    existing[tag].add(pos)

        for _, tag in regex_tags:
            self.tags[tag].sort()

    def positions_by_tag(self, tag: str) -> list[int]:
        return list(self.tags.get(tag, []))

    def tokens_by_tag(self, tag: str) -> list[tuple[int, int, str | None]]:
        return [
            (pos, self.token_ids[pos], self.token_texts[pos])
            for pos in self.tags.get(tag, [])
        ]

    def tag_counts(self) -> dict[str, int]:
        return {tag: len(positions) for tag, positions in self.tags.items()}

    def to_document(self, metadata: Metadata | None = None) -> InifDocument:
        """Materialize a flat analysis document.

        The result keeps sample ids, token ids/text, sequence provenance, and
        sparse tags. It intentionally emits flat samples and does not recreate
        shared ``Sequence`` objects.
        """
        positions_to_tags: dict[int, list[str]] = {}
        for tag, positions in self.tags.items():
            for pos in positions:
                positions_to_tags.setdefault(pos, []).append(tag)

        samples: list[Sample] = []
        for sample_id, start, end in zip(
            self.sample_ids,
            self.sample_offsets,
            self.sample_offsets[1:],
        ):
            tokens: list[Token] = []
            for pos in range(start, end):
                token = Token(
                    id=self.token_ids[pos],
                    token=self.token_texts[pos],
                    sequence_id=self.sequence_ids[pos],
                )
                for tag in positions_to_tags.get(pos, []):
                    token.add_tag(tag)
                tokens.append(token)
            samples.append(Sample(id=sample_id, tokens=tokens))

        return InifDocument(
            metadata=metadata.model_copy(deep=True)
            if metadata is not None
            else Metadata(model=ModelInfo(name="flat-token-store")),
            samples=samples,
        )
