"""Tests for the Inspect AI converter using mock objects."""

from types import SimpleNamespace

from inif.converters.inspect_ai import (
    _convert_scores,
    _extract_message_dicts,
    _extract_model_info,
    _extract_source_eval,
    _messages_to_tokens,
    from_eval_log,
)


def _make_eval_log(
    model="gpt2",
    task="mmlu",
    samples=None,
    max_tokens=100,
):
    """Create a mock EvalLog-like object."""
    config = SimpleNamespace(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=None,
        top_k=None,
        stop_seqs=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
    )
    plan = SimpleNamespace(config=config)
    eval_spec = SimpleNamespace(
        model=model,
        task=task,
        task_version=None,
        eval_id="eval_abc",
        run_id="run_xyz",
        inspect_version="0.3.0",
    )
    stats = SimpleNamespace(
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:01:00+00:00",
    )
    return SimpleNamespace(
        eval=eval_spec,
        plan=plan,
        stats=stats,
        samples=samples or [],
    )


def _make_message(role, content):
    return SimpleNamespace(role=role, content=content)


def _make_sample(messages, scores=None, target=None, sample_id=None, usage=None):
    return SimpleNamespace(
        id=sample_id,
        messages=messages,
        scores=scores or {},
        target=target,
        usage=usage,
    )


def _make_score(value, answer=None, explanation=None, metadata=None):
    return SimpleNamespace(
        value=value,
        answer=answer,
        explanation=explanation,
        metadata=metadata or {},
    )


def test_extract_model_info():
    log = _make_eval_log(model="meta-llama/Llama-3-8b", max_tokens=200)
    info = _extract_model_info(log)
    assert info.name == "meta-llama/Llama-3-8b"
    assert info.generation_config["max_tokens"] == 200
    assert info.generation_config["temperature"] == 0.7


def test_extract_model_info_with_revision():
    log = _make_eval_log(model="gpt2")

    class MockTokenizer:
        _commit_hash = "abc123"

    info = _extract_model_info(log, tokenizer=MockTokenizer())
    assert info.revision == "abc123"


def test_extract_source_eval():
    log = _make_eval_log(task="hellaswag")
    src = _extract_source_eval(log)
    assert src.framework == "inspect_ai"
    assert src.framework_version == "0.3.0"
    assert src.task == "hellaswag"
    assert src.eval_id == "eval_abc"


def test_convert_scores():
    sample = SimpleNamespace(
        scores={
            "accuracy": _make_score(1.0, answer="A"),
            "f1": _make_score(0.85),
        }
    )
    scores = _convert_scores(sample)
    assert len(scores) == 2
    scorer_names = {s.scorer for s in scores}
    assert "accuracy" in scorer_names
    assert "f1" in scorer_names
    acc = next(s for s in scores if s.scorer == "accuracy")
    assert acc.value == 1.0
    assert acc.answer == "A"


def test_messages_to_tokens_no_tokenizer():
    msg_dicts = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    texts, tokens = _messages_to_tokens(msg_dicts)
    assert len(texts) == 2
    assert texts[0] == "You are helpful."
    assert texts[1] == "Hello!"
    # Without tokenizer, no tokens generated
    assert len(tokens) == 0


def test_messages_to_tokens_fallback_encode():
    """Tokenizer without apply_chat_template falls back to per-message encode."""

    class FallbackTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

        def decode(self, ids):
            return "x" * len(ids)

    msg_dicts = [{"role": "user", "content": "hi"}]
    texts, tokens = _messages_to_tokens(msg_dicts, tokenizer=FallbackTokenizer())
    assert len(tokens) == 2  # "hi" = 2 chars = 2 tokens
    assert texts == ["hi"]
    # No role tagging in _messages_to_tokens anymore
    for t in tokens:
        assert "role" not in (t.model_extra or {})


def test_messages_to_tokens_with_chat_template():
    """Tokenizer with apply_chat_template produces full template tokens."""

    class ChatTokenizer:
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

    msg_dicts = [{"role": "user", "content": "hi"}]
    texts, tokens = _messages_to_tokens(msg_dicts, tokenizer=ChatTokenizer())
    # "<s>[user]hi[/user]" = 18 chars = 18 tokens
    assert len(tokens) == 18
    assert texts == ["hi"]
    # No role tagging in _messages_to_tokens anymore
    for t in tokens:
        assert "role" not in (t.model_extra or {})


def test_from_eval_log_basic():
    messages = [
        _make_message("system", "Answer:"),
        _make_message("user", "What is 2+2?"),
        _make_message("assistant", "4"),
    ]
    scores = {"accuracy": _make_score(1.0, answer="4")}
    sample = _make_sample(messages, scores=scores, target="4", sample_id="q1")
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(log, deduplicate=False)
    assert doc.metadata.model.name == "gpt2"
    assert doc.metadata.source_eval.framework == "inspect_ai"
    assert doc.metadata.total_samples == 1
    assert doc.metadata.total_time == 60.0

    s = doc.samples[0]
    assert s.id == "q1"
    assert s.target == "4"
    assert len(s.scores) == 1
    assert s.scores[0].value == 1.0
    # texts should be plain strings
    assert s.texts == ["Answer:", "What is 2+2?", "4"]


def test_from_eval_log_no_samples():
    log = _make_eval_log(samples=[])
    doc = from_eval_log(log, deduplicate=False)
    assert doc.metadata.total_samples == 0
    assert len(doc.samples) == 0


def test_from_eval_log_with_tokenizer():
    """Tokenizer without apply_chat_template uses per-message encode fallback."""

    class FallbackTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

        def decode(self, ids):
            return "x" * len(ids)

    messages = [_make_message("user", "AB")]
    sample = _make_sample(messages, sample_id=0)
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log, tokenizer=FallbackTokenizer(), deduplicate=False, tag_chat_roles=False
    )
    assert len(doc.samples[0].tokens) == 2
    assert len(doc.samples[0].texts) == 1
    assert doc.samples[0].texts[0] == "AB"


def test_extract_message_dicts():
    """_extract_message_dicts converts Inspect message objects to plain dicts."""
    messages = [
        _make_message("system", "Be helpful."),
        _make_message("user", "Hello!"),
    ]
    dicts = _extract_message_dicts(messages)
    assert len(dicts) == 2
    assert dicts[0] == {"role": "system", "content": "Be helpful."}
    assert dicts[1] == {"role": "user", "content": "Hello!"}


def test_extract_message_dicts_list_content():
    """_extract_message_dicts handles list content (ContentText objects)."""
    part = SimpleNamespace(text="Part 1")
    msg = _make_message("user", [part])
    dicts = _extract_message_dicts([msg])
    assert dicts[0]["content"] == "Part 1"


def test_from_eval_log_chat_roles():
    """Full pipeline with role tagging after dedup."""

    class ChatTokenizer:
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

    messages = [
        _make_message("user", "Hi"),
        _make_message("assistant", "Hello!"),
    ]
    sample = _make_sample(messages, sample_id="q1")
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log, tokenizer=ChatTokenizer(), deduplicate=False, tag_chat_roles=True
    )
    # All non-ref tokens should have role
    for tok in doc.samples[0].tokens:
        if not tok.is_sequence_ref:
            assert "role" in (tok.model_extra or {}), (
                f"Token {tok.token!r} missing role"
            )

    # Check specific roles
    roles = [tok.model_extra.get("role") for tok in doc.samples[0].tokens]
    assert "user" in roles
    assert "assistant" in roles
    assert "template" in roles


def test_from_eval_log_usage():
    usage = SimpleNamespace(input_tokens=50, output_tokens=10)
    sample = _make_sample(
        [_make_message("user", "test")],
        sample_id=0,
        usage=usage,
    )
    log = _make_eval_log(samples=[sample])
    doc = from_eval_log(log, deduplicate=False)
    assert doc.samples[0].input_tokens == 50
    assert doc.samples[0].output_tokens == 10
