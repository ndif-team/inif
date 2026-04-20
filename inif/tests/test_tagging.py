from inif.models import InifDocument, Metadata, ModelInfo, Sample, Sequence, Token
from inif.tagging import (
    TextTagMode,
    _get_tags,
    create_span_from_tag,
    remove_tag,
    tag_by_predicate,
    tag_by_regex,
    tag_by_regex_all,
    tag_by_text_regex,
    tag_chat_roles,
    tag_chat_roles_doc,
    tag_positions,
    tag_special_tokens,
)


def test_tag_by_regex(sample_flat):
    tag_by_regex(sample_flat, r"^This$", "capitalized")
    tagged = [t for t in sample_flat.tokens if "capitalized" in _get_tags(t)]
    assert len(tagged) == 1
    assert tagged[0].token == "This"


def test_tag_by_regex_partial_match(sample_flat):
    tag_by_regex(sample_flat, r"is", "has_is")
    tagged = [t for t in sample_flat.tokens if "has_is" in _get_tags(t)]
    # "This" and " is" both contain "is"
    assert len(tagged) == 2


def test_tag_by_regex_no_match(sample_flat):
    tag_by_regex(sample_flat, r"^zzz$", "no_match")
    tagged = [t for t in sample_flat.tokens if "no_match" in _get_tags(t)]
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
        assert "greeting" in _get_tags(s.tokens[0])


def test_tag_by_regex_materializes_inside_ref(sample, sequences):
    """Matches inside a sequence ref should materialize the ref and tag the
    specific matching token, not the ref placeholder."""
    # sample.tokens[0] is a ref to seq_0 = ["<|endoftext|>", "This", " is"]
    n_before = len(sample.tokens)
    tag_by_regex(sample, r"^This$", "capitalized", sequences=sequences)

    # The ref containing "This" was materialized into 3 tokens, so length grew by 2.
    assert len(sample.tokens) == n_before + 2
    # The matching token now lives in sample.tokens with its tag attached.
    matching = [t for t in sample.tokens if t.has_tag("capitalized")]
    assert len(matching) == 1
    assert matching[0].token == "This"
    assert matching[0].id == 1212  # original id is preserved


def test_tag_by_regex_no_materialization_when_no_match(sample, sequences):
    """If the regex doesn't match anything inside a ref, the ref stays compressed."""
    n_before = len(sample.tokens)
    tag_by_regex(sample, r"^nope$", "x", sequences=sequences)
    assert len(sample.tokens) == n_before
    assert sample.tokens[0].is_sequence_ref


def test_tag_by_predicate(sample_flat):
    tag_by_predicate(sample_flat, lambda t: t.id > 1000, "high_id")
    tagged = [t for t in sample_flat.tokens if "high_id" in _get_tags(t)]
    # Tokens with id 50256, 1212, 1332 have id > 1000
    assert len(tagged) == 3


def test_tag_positions(sample_flat):
    tag_positions(sample_flat, [0, 5], "boundary")
    tagged = [t for t in sample_flat.tokens if "boundary" in _get_tags(t)]
    assert len(tagged) == 2
    # First and last token
    assert tagged[0].token == "<|endoftext|>"
    assert tagged[1].token == "."


def test_tag_no_duplicates(sample_flat):
    tag_positions(sample_flat, [0], "x")
    tag_positions(sample_flat, [0], "x")
    assert _get_tags(sample_flat.tokens[0]).count("x") == 1


def test_remove_tag(sample_flat):
    tag_positions(sample_flat, [0, 1], "temp")
    assert len([t for t in sample_flat.tokens if "temp" in _get_tags(t)]) == 2
    remove_tag(sample_flat, "temp")
    assert len([t for t in sample_flat.tokens if "temp" in _get_tags(t)]) == 0


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
    tagged = [t for t in sample.tokens if "entity" in _get_tags(t)]
    assert len(tagged) == 3
    assert [t.token for t in tagged] == [" E", "iff", "el"]
    # " Tower" should NOT be tagged
    assert "entity" not in _get_tags(sample.tokens[3])


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
    tagged = [t for t in sample.tokens if "entity" in _get_tags(t)]
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
    tagged = [t for t in sample.tokens if "entity" in _get_tags(t)]
    assert len(tagged) == 1
    assert tagged[0].token == "el"


def test_tag_by_text_regex_no_match():
    sample = Sample(
        id="x",
        tokens=[Token(id=1, token="hello"), Token(id=2, token=" world")],
    )
    tag_by_text_regex(sample, r"xyz", "nope")
    assert all("nope" not in _get_tags(t) for t in sample.tokens)


def test_tag_special_tokens(sample_flat):
    class MockTokenizer:
        all_special_ids = [50256]

    tag_special_tokens(sample_flat, MockTokenizer())
    tagged = [t for t in sample_flat.tokens if "special" in _get_tags(t)]
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
    """Tokens get correct role extras (system/user/assistant/template)."""
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

    roles = [t.model_extra.get("role") for t in sample.tokens]
    # All tokens should have a role
    assert all(r is not None for r in roles)
    # Content tokens should have their message role
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assert "template" in roles


def test_tag_chat_roles_template_tokens():
    """Delimiter tokens get role='template'."""
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
    for i in range(3):
        assert sample.tokens[i].model_extra["role"] == "template"
    # "[user]" = 6 chars = template
    for i in range(3, 9):
        assert sample.tokens[i].model_extra["role"] == "template"
    # "Hi" = 2 chars = user
    assert sample.tokens[9].model_extra["role"] == "user"
    assert sample.tokens[10].model_extra["role"] == "user"
    # "[/user]" = 7 chars = template
    for i in range(11, 18):
        assert sample.tokens[i].model_extra["role"] == "template"


def test_tag_chat_roles_with_sequences():
    """Works with deduplicated doc: sequence refs skipped, non-ref tokens tagged."""
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

    # Ref token should NOT have role
    assert "role" not in (sample.tokens[0].model_extra or {})
    # "H" and "i" should be "user"
    assert sample.tokens[1].model_extra["role"] == "user"
    assert sample.tokens[2].model_extra["role"] == "user"
    # "[/user]" tokens should be "template"
    for tok in sample.tokens[3:]:
        assert tok.model_extra["role"] == "template"


def test_tag_chat_roles_doc_basic():
    """tag_chat_roles_doc tags all samples in a document."""
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
        roles = [t.model_extra.get("role") for t in sample.tokens]
        assert all(r is not None for r in roles)
        assert "user" in roles
        assert "template" in roles
