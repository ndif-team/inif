from inif.models import InifDocument, Metadata, ModelInfo, Sample, Sequence, Token
from inif.selectors import select_by_annotation
from inif.tagging import (
    TextTagMode,
    create_span_from_tag,
    remove_tag,
    tag_by_predicate,
    tag_by_predicates,
    tag_by_regex,
    tag_by_regex_all,
    tag_by_regexes,
    tag_by_text_regex,
    tag_chat_roles,
    tag_chat_roles_doc,
    tag_positions,
    tag_special_tokens,
)


def _annotated_tokens(sample: Sample, name: str) -> list[Token]:
    return select_by_annotation(sample, name).tokens


def test_tag_by_regex(sample_flat):
    tag_by_regex(sample_flat, r"^This$", "capitalized")
    tagged = _annotated_tokens(sample_flat, "capitalized")
    assert len(tagged) == 1
    assert tagged[0].token == "This"


def test_tag_by_regex_partial_match(sample_flat):
    tag_by_regex(sample_flat, r"is", "has_is")
    tagged = _annotated_tokens(sample_flat, "has_is")
    # "This" and " is" both contain "is"
    assert len(tagged) == 2


def test_tag_by_regex_no_match(sample_flat):
    tag_by_regex(sample_flat, r"^zzz$", "no_match")
    tagged = _annotated_tokens(sample_flat, "no_match")
    assert len(tagged) == 0


def test_tag_by_regex_all(doc):
    from inif.models import InifDocument, Metadata, ModelInfo

    doc2 = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[
            Sample(
                id="s0",
                tokens=[
                    Token(id=1, token="hello"),
                    Token(id=2, token="world"),
                ],
            ),
            Sample(
                id="s1",
                tokens=[
                    Token(id=1, token="hello"),
                    Token(id=3, token="there"),
                ],
            ),
        ],
    )
    tag_by_regex_all(doc2, r"hello", "greeting")
    for s in doc2.samples:
        assert s.annotation_positions("greeting") == [0]


def test_tag_by_regex_all_flat_sample_does_not_materialize(doc):
    """A flat sample in a doc with sequences should stay on the direct path."""
    doc2 = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        sequences=doc.sequences,
        samples=[
            Sample(
                id="s0",
                tokens=[
                    Token(id=1, token="A"),
                    Token(id=2, token="B"),
                    Token(id=3, token="C"),
                ],
            )
        ],
    )
    before_tokens = list(doc2.samples[0].tokens)

    tag_by_regex_all(doc2, r"^[AB]$", "letter")

    assert doc2.samples[0].tokens == before_tokens
    assert doc2.samples[0].annotation_positions("letter") == [0, 1]


def test_tag_by_regexes_applies_multiple_tags_one_pass(sample_flat):
    tag_by_regexes(
        sample_flat,
        [
            (r"^This$", "capitalized"),
            (r"is", "has_is"),
        ],
    )

    assert sample_flat.annotation_positions("capitalized") == [1]
    assert sample_flat.annotation_positions("has_is") == [1, 2]


def test_tag_by_regex_materializes_inside_ref(sample, sequences):
    """Matches inside a sequence ref should materialize the ref and tag the
    specific matching token, not the ref placeholder."""
    # sample.tokens[0] is a ref to seq_0 = ["<|endoftext|>", "This", " is"]
    n_before = len(sample.tokens)
    tag_by_regex(sample, r"^This$", "capitalized", sequences=sequences)

    # The ref containing "This" was materialized into 3 tokens, so length grew by 2.
    assert len(sample.tokens) == n_before + 2
    # The matching token now lives in sample.tokens with its annotation attached.
    matching = _annotated_tokens(sample, "capitalized")
    assert len(matching) == 1
    assert matching[0].token == "This"
    assert matching[0].id == 1212  # original id is preserved


def test_tag_by_regex_no_materialization_when_no_match(sample, sequences):
    """If the regex doesn't match anything inside a ref, the ref stays compressed."""
    n_before = len(sample.tokens)
    tag_by_regex(sample, r"^nope$", "x", sequences=sequences)
    assert len(sample.tokens) == n_before
    assert sample.tokens[0].is_sequence_ref


def test_tag_by_regexes_materializes_matching_ref_once(sample, sequences):
    n_before = len(sample.tokens)
    tag_by_regexes(
        sample,
        [
            (r"^This$", "capitalized"),
            (r"is", "has_is"),
        ],
        sequences=sequences,
    )

    assert len(sample.tokens) == n_before + 2
    assert sample.tokens[1].token == "This"
    assert sample.annotation_positions("capitalized") == [1]
    assert 1 in sample.annotation_positions("has_is")
    assert sample.tokens[2].token == " is"
    assert 2 in sample.annotation_positions("has_is")


def test_tag_by_predicate(sample_flat):
    tag_by_predicate(sample_flat, lambda t: t.id > 1000, "high_id")
    tagged = _annotated_tokens(sample_flat, "high_id")
    # Tokens with id 50256, 1212, 1332 have id > 1000
    assert len(tagged) == 3


def test_tag_by_predicates_applies_multiple_tags(sample_flat):
    tag_by_predicates(
        sample_flat,
        [
            (lambda t: t.id > 1000, "high_id"),
            (lambda t: t.token is not None and t.token.strip() == "This", "word"),
        ],
    )

    assert sample_flat.annotation_positions("high_id") == [0, 1, 4]
    assert sample_flat.annotation_positions("word") == [1]


def test_tag_by_predicates_can_materialize_sequence_ref(sample, sequences):
    tag_by_predicates(
        sample,
        [(lambda t: t.token == "This", "capitalized")],
        sequences=sequences,
    )

    matching = _annotated_tokens(sample, "capitalized")
    assert len(matching) == 1
    assert matching[0].token == "This"
    assert matching[0].id == 1212


def test_tag_positions(sample_flat):
    tag_positions(sample_flat, [0, 5], "boundary")
    tagged = _annotated_tokens(sample_flat, "boundary")
    assert len(tagged) == 2
    # First and last token
    assert tagged[0].token == "<|endoftext|>"
    assert tagged[1].token == "."


def test_tag_no_duplicates(sample_flat):
    tag_positions(sample_flat, [0], "x")
    tag_positions(sample_flat, [0], "x")
    assert sample_flat.annotation_positions("x") == [0]


def test_remove_tag(sample_flat):
    tag_positions(sample_flat, [0, 1], "temp")
    assert sample_flat.annotation_positions("temp") == [0, 1]
    remove_tag(sample_flat, "temp")
    assert sample_flat.annotation_positions("temp") == []


def test_create_span_from_tag(sample_flat):
    tag_positions(sample_flat, [1, 2, 3], "phrase")
    span = create_span_from_tag(sample_flat, "phrase", "my_phrase")
    assert span.name == "my_phrase"
    assert span.positions == [1, 2, 3]
    assert "phrase" in span.tags
    assert span in sample_flat.spans


def test_tag_by_text_regex_subword():
    """Text regex should match across BPE subword splits."""
    sample = Sample(
        id="bpe",
        tokens=[
            Token(id=1, token=" E"),
            Token(id=2, token="iff"),
            Token(id=3, token="el"),
            Token(id=4, token=" Tower"),
        ],
    )
    tag_by_text_regex(sample, r"(?i)eiffel", "entity")
    tagged = _annotated_tokens(sample, "entity")
    assert len(tagged) == 3
    assert [t.token for t in tagged] == [" E", "iff", "el"]
    # " Tower" should NOT be tagged
    assert 3 not in sample.annotation_positions("entity")


def test_tag_by_text_regex_first():
    """mode='first' should tag only the first token of the match."""
    sample = Sample(
        id="bpe",
        tokens=[
            Token(id=1, token=" E"),
            Token(id=2, token="iff"),
            Token(id=3, token="el"),
            Token(id=4, token=" Tower"),
        ],
    )
    tag_by_text_regex(sample, r"(?i)eiffel", "entity", mode=TextTagMode.FIRST)
    tagged = _annotated_tokens(sample, "entity")
    assert len(tagged) == 1
    assert tagged[0].token == " E"


def test_tag_by_text_regex_last():
    """mode='last' should tag only the last token of the match."""
    sample = Sample(
        id="bpe",
        tokens=[
            Token(id=1, token=" E"),
            Token(id=2, token="iff"),
            Token(id=3, token="el"),
            Token(id=4, token=" Tower"),
        ],
    )
    tag_by_text_regex(sample, r"(?i)eiffel", "entity", mode=TextTagMode.LAST)
    tagged = _annotated_tokens(sample, "entity")
    assert len(tagged) == 1
    assert tagged[0].token == "el"


def test_tag_by_text_regex_no_match():
    sample = Sample(
        id="x",
        tokens=[Token(id=1, token="hello"), Token(id=2, token=" world")],
    )
    tag_by_text_regex(sample, r"xyz", "nope")
    assert sample.annotation_positions("nope") == []


def test_tag_special_tokens(sample_flat):
    class MockTokenizer:
        all_special_ids = [50256]

    tag_special_tokens(sample_flat, MockTokenizer())
    tagged = _annotated_tokens(sample_flat, "special")
    assert len(tagged) == 1
    assert tagged[0].id == 50256


# --- Chat role tagging tests ---


class ChatMockTokenizer:
    """Mock tokenizer with apply_chat_template support.

    Produces: ``<s>[ROLE]content[/ROLE]`` per message.
    Each character of the template becomes one token with id = ord(char).
    """

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
            parts.append(f"[/{msg['role']}]")
        formatted = "".join(parts)
        if not tokenize:
            return formatted
        return [ord(c) for c in formatted]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_tag_chat_roles_basic():
    """Tokens get correct role annotations (system/user/assistant/template)."""
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    tokenizer = ChatMockTokenizer()
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    tokens = [Token(id=tid, token=chr(tid)) for tid in formatted]
    sample = Sample(id="test", tokens=tokens)

    tag_chat_roles(sample, messages, tokenizer)

    annotated = set()
    for annotation in sample.annotations:
        annotated.add(annotation.name)
    assert {"system", "user", "assistant", "template"} <= annotated
    assert sorted(
        pos
        for annotation in sample.annotations
        for start, end in annotation.ranges
        for pos in range(start, end)
    ) == list(range(len(sample.tokens)))


def test_tag_chat_roles_template_tokens():
    """Delimiter tokens get the template annotation."""
    messages = [{"role": "user", "content": "Hi"}]
    tokenizer = ChatMockTokenizer()
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    tokens = [Token(id=tid, token=chr(tid)) for tid in formatted]
    sample = Sample(id="test", tokens=tokens)

    tag_chat_roles(sample, messages, tokenizer)

    # "<s>[user]" = template, "Hi" = user, "[/user]" = template
    # First 3 chars are "<s>" = template
    template_positions = sample.annotation_positions("template")
    user_positions = sample.annotation_positions("user")
    for i in range(3):
        assert i in template_positions
    # "[user]" = 6 chars = template
    for i in range(3, 9):
        assert i in template_positions
    # "Hi" = 2 chars = user
    assert user_positions == [9, 10]
    # "[/user]" = 7 chars = template
    for i in range(11, 18):
        assert i in template_positions


def test_tag_chat_roles_with_sequences():
    """Works with deduplicated docs: ref tokens can be annotated as a unit."""
    messages = [
        {"role": "user", "content": "Hi"},
    ]
    tokenizer = ChatMockTokenizer()
    # Template: "<s>[user]Hi[/user]"
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    all_tokens = [Token(id=tid, token=chr(tid)) for tid in formatted]

    # Simulate dedup: "<s>[user]" (first 9 chars) becomes a sequence
    seq_toks = [Token(id=t.id, token=t.token) for t in all_tokens[:9]]
    seq = Sequence(id="seq_0", n_tokens=len(seq_toks), tokens=seq_toks)
    # Sample has: ref + "H" + "i" + "[" + "/" + "u" + "s" + "e" + "r" + "]"
    sample_tokens = [Token(id=-1, sequence_id="seq_0")] + [
        Token(id=t.id, token=t.token) for t in all_tokens[9:]
    ]
    sample = Sample(id="test", tokens=sample_tokens)

    tag_chat_roles(sample, messages, tokenizer, sequences=[seq])

    # Ref token should be annotated as template.
    assert sample.annotation_positions("template")[0] == 0
    # "H" and "i" should be "user"
    assert sample.annotation_positions("user") == [1, 2]
    # "[/user]" tokens should be "template"
    assert sample.annotation_positions("template") == [0, 3, 4, 5, 6, 7, 8, 9]


def test_tag_chat_roles_doc_basic():
    """tag_chat_roles_doc annotates all samples in a document."""
    tokenizer = ChatMockTokenizer()
    messages_list = [
        [{"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Bye"}],
    ]

    samples = []
    for i, msgs in enumerate(messages_list):
        formatted = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
        tokens = [Token(id=tid, token=chr(tid)) for tid in formatted]
        samples.append(Sample(id=f"s{i}", tokens=tokens))

    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=samples,
    )
    tag_chat_roles_doc(doc, messages_list, tokenizer)

    for sample in doc.samples:
        assert sample.annotation_positions("user")
        assert sample.annotation_positions("template")
