"""Tests for the raw text converter."""

import tempfile
from pathlib import Path

from inif.converters.text import from_text_files, from_texts


class MockTokenizer:
    """Simple tokenizer that splits on whitespace."""

    name_or_path = "mock-model"
    _commit_hash = "abc123"

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._get_id(w) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "?") for i in ids)


def test_from_texts_basic():
    tokenizer = MockTokenizer()
    doc = from_texts(
        ["hello world", "hello there"],
        tokenizer=tokenizer,
        deduplicate=False,
    )
    assert len(doc.samples) == 2
    assert doc.samples[0].id == "sample_0"
    assert doc.samples[1].id == "sample_1"
    assert [(t.name, t.value) for t in doc.samples[0].texts] == [
        ("text_0", "hello world")
    ]
    assert len(doc.samples[0].tokens) == 2
    assert doc.samples[0].tokens[0].token == "hello"
    assert doc.samples[0].tokens[1].token == "world"


def test_from_texts_byte_level_split_utf8_roundtrips():
    class SplitUtf8Tokenizer:
        name_or_path = "split-utf8"
        _commit_hash = None
        encoder = {"Î": 1, "¸": 2}
        byte_decoder = {"Î": 0xCE, "¸": 0xB8}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert text == "θ"
            return [1, 2]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            if ids == [1, 2]:
                return "θ"
            return "�"

    doc = from_texts(["θ"], tokenizer=SplitUtf8Tokenizer(), deduplicate=False)
    pieces = [t.token for t in doc.samples[0].tokens]

    assert pieces == ["", "θ"]
    assert "".join(pieces) == "θ"


def test_from_texts_custom_ids():
    tokenizer = MockTokenizer()
    doc = from_texts(
        ["hello world"],
        tokenizer=tokenizer,
        sample_ids=["my_sample"],
        deduplicate=False,
    )
    assert doc.samples[0].id == "my_sample"


def test_from_texts_with_dedup():
    tokenizer = MockTokenizer()
    doc = from_texts(
        ["a b c d", "a b c e", "a b c f"],
        tokenizer=tokenizer,
        min_sequence_length=3,
        deduplicate=True,
    )
    assert len(doc.sequences) == 1
    assert [t.token for t in doc.sequences[0].tokens] == ["a", "b", "c"]


def test_from_texts_metadata():
    tokenizer = MockTokenizer()
    doc = from_texts(
        ["hello"],
        tokenizer=tokenizer,
        deduplicate=False,
    )
    assert doc.metadata.model.name == "mock-model"
    assert doc.metadata.model.revision == "abc123"
    assert doc.total_samples == 1


def test_from_text_files():
    tokenizer = MockTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "file1.txt"
        p2 = Path(tmpdir) / "file2.txt"
        p1.write_text("hello world foo")
        p2.write_text("hello world bar")

        doc = from_text_files(
            [p1, p2],
            tokenizer=tokenizer,
            min_sequence_length=2,
            deduplicate=True,
        )

        assert len(doc.samples) == 2
        assert doc.samples[0].id == "file1.txt"
        assert doc.samples[1].id == "file2.txt"
        assert doc.metadata.sources == [str(p1), str(p2)]
        # "hello world" should be deduplicated
        assert len(doc.sequences) >= 1


class ChatMockTokenizer:
    """Mock tokenizer with apply_chat_template for text converter tests.

    Renders each message as ``[role]content[end]`` so the per-message
    terminator (``[end]``) is unambiguous for boundary detection.
    """

    name_or_path = "chat-mock"
    _commit_hash = "chat123"
    all_special_tokens = ["<s>", "[end]"]

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        parts = ["<s>"]
        for msg in messages:
            parts.append(f"[{msg['role']}]")
            parts.append(msg["content"])
            parts.append("[end]")
        formatted = "".join(parts)
        if not tokenize:
            return formatted
        return [ord(c) for c in formatted]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_from_texts_with_messages():
    """from_texts auto-detects chat inputs and annotates chat roles."""
    tokenizer = ChatMockTokenizer()
    messages = [
        [{"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Bye"}],
    ]

    doc = from_texts(
        messages,
        tokenizer=tokenizer,
        tag_chat_roles=True,
        deduplicate=False,
    )

    assert len(doc.samples) == 2
    # texts come from message contents with role-based names
    assert [(t.name, t.value) for t in doc.samples[0].texts] == [("user_0", "Hi")]
    assert [(t.name, t.value) for t in doc.samples[1].texts] == [("user_0", "Bye")]

    # Tokens should include template delimiters (more tokens than just "Hi")
    assert len(doc.samples[0].tokens) > 2

    # Check specific annotations present
    assert doc.samples[0].annotation_positions("user")
    assert doc.samples[0].annotation_positions("template")


def test_from_texts_chat_partitions_tokens_across_messages():
    """Chat inputs populate ``Text.start`` / ``Text.end`` so the per-message
    viewer can scope its token toggle. Adjacent texts chain (``end`` of one
    equals ``start`` of the next) and together cover the full token range."""
    tokenizer = ChatMockTokenizer()
    messages = [
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    ]
    doc = from_texts(messages, tokenizer=tokenizer, deduplicate=False)
    sample = doc.samples[0]
    n = len(sample.tokens)

    starts = [t.start for t in sample.texts]
    ends = [t.end for t in sample.texts]
    assert all(s is not None and e is not None for s, e in zip(starts, ends))
    assert starts[0] == 0
    assert ends[-1] == n
    # Adjacent partition: end[i] == start[i+1].
    for i in range(len(sample.texts) - 1):
        assert ends[i] == starts[i + 1]


def test_from_texts_plain_text_offsets_full_token_range():
    """Plain-string inputs become a single Text whose range covers every
    token in the sample."""
    tokenizer = MockTokenizer()
    doc = from_texts(["hello world foo"], tokenizer=tokenizer, deduplicate=False)
    sample = doc.samples[0]
    assert len(sample.texts) == 1
    assert sample.texts[0].start == 0
    assert sample.texts[0].end == len(sample.tokens)


def test_from_texts_chat_offsets_remap_after_dedup():
    """After dedup, chat-message offsets stay adjacent (no overlaps even when
    a sequence ref now spans a former message boundary)."""
    tokenizer = ChatMockTokenizer()
    messages = [
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "what is 2+2"},
        ],
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "what is the capital"},
        ],
    ]
    doc = from_texts(
        messages,
        tokenizer=tokenizer,
        deduplicate=True,
        min_sequence_length=3,
    )
    # The shared "<s>[system]be nice[/system][user]" prefix becomes a
    # sequence ref; remap must keep system_0.end == user_0.start.
    assert len(doc.sequences) >= 1
    for sample in doc.samples:
        for i in range(len(sample.texts) - 1):
            assert sample.texts[i].end == sample.texts[i + 1].start
            assert sample.texts[i].start <= sample.texts[i].end
        assert sample.texts[-1].end == len(sample.tokens)


def test_from_texts_dedup_does_not_cross_message_boundaries():
    """Dedup runs are bounded by message starts: a sequence ref must live
    entirely within one message, so opening role markers (``<|im_start|>``,
    Kimi's ``<|im_user|>`` etc.) of the NEXT message are never glued onto
    the previous message's tail by a cross-boundary ref."""
    tokenizer = ChatMockTokenizer()
    # Two samples with identical structure → the closing ``[end]`` of one
    # message and the opening ``[user]`` of the next are common across
    # samples. Without boundary-aware dedup the whole ``[end][user]`` run
    # would collapse into one ref straddling the system_0 / user_0 break.
    messages = [
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "what is 2+2"},
        ],
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "what is the capital"},
        ],
    ]
    doc = from_texts(
        messages,
        tokenizer=tokenizer,
        deduplicate=True,
        min_sequence_length=3,
    )
    seq_map = {s.id: s for s in doc.sequences}
    for sample in doc.samples:
        for tok in sample.tokens:
            if not tok.is_sequence_ref:
                continue
            seq = seq_map[tok.token]
            expanded = "".join(t.token or "" for t in seq.tokens)
            # The terminator-then-role-marker run is the canonical
            # cross-boundary run; if it appears inside one ref the
            # per-message partition is broken.
            assert "[end][user]" not in expanded, (
                f"Sequence {tok.token!r} crosses a message boundary: {expanded!r}"
            )
            assert "[end][assistant]" not in expanded, (
                f"Sequence {tok.token!r} crosses a message boundary: {expanded!r}"
            )
