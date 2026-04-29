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
    assert texts[0].name == "system_0"
    assert texts[0].value == "You are helpful."
    assert texts[1].name == "user_0"
    assert texts[1].value == "Hello!"
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
    assert [(t.name, t.value) for t in texts] == [("user_0", "hi")]
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
    assert [(t.name, t.value) for t in texts] == [("user_0", "hi")]
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

    doc = from_eval_log(log, tokenizer=_chat_tokenizer(), deduplicate=False)
    assert doc.metadata.model.name == "gpt2"
    assert doc.metadata.source_eval.framework == "inspect_ai"
    assert doc.total_samples == 1
    assert doc.metadata.total_time == 60.0

    s = doc.samples[0]
    assert s.id == "q1"
    assert s.target == "4"
    assert len(s.scores) == 1
    assert s.scores[0].value == 1.0
    # texts are role-named Text objects, system prompt included
    assert [(t.name, t.value) for t in s.texts] == [
        ("system_0", "Answer:"),
        ("user_0", "What is 2+2?"),
        ("assistant_0", "4"),
    ]


def test_from_eval_log_no_samples():
    log = _make_eval_log(samples=[])
    doc = from_eval_log(log, tokenizer=_chat_tokenizer(), deduplicate=False)
    assert doc.total_samples == 0
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
    assert doc.samples[0].texts[0].name == "user_0"
    assert doc.samples[0].texts[0].value == "AB"


def test_from_eval_log_min_sequence_length_controls_dedup():
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    samples = [
        _make_sample([_make_message("user", "ab0")], sample_id=0),
        _make_sample([_make_message("user", "ab1")], sample_id=1),
    ]
    log = _make_eval_log(samples=samples)

    doc_min_2 = from_eval_log(
        log,
        tokenizer=CharTokenizer(),
        deduplicate=True,
        min_sequence_length=2,
        tag_chat_roles=False,
        tag_generated=False,
        extract_logprobs=False,
    )
    doc_min_3 = from_eval_log(
        log,
        tokenizer=CharTokenizer(),
        deduplicate=True,
        min_sequence_length=3,
        tag_chat_roles=False,
        tag_generated=False,
        extract_logprobs=False,
    )

    assert len(doc_min_2.sequences) == 1
    assert [t.token for t in doc_min_2.sequences[0].tokens] == ["a", "b"]
    assert len(doc_min_3.sequences) == 0


def test_extract_message_dicts():
    """_extract_message_dicts converts Inspect message objects to plain dicts."""
    messages = [
        _make_message("system", "Be helpful."),
        _make_message("user", "Hello!"),
    ]
    dicts, reasoning = _extract_message_dicts(messages)
    assert len(dicts) == 2
    assert dicts[0] == {"role": "system", "content": "Be helpful."}
    assert dicts[1] == {"role": "user", "content": "Hello!"}
    assert reasoning == [None, None]


def test_extract_message_dicts_list_content():
    """_extract_message_dicts handles list content (ContentText objects)."""
    part = SimpleNamespace(text="Part 1")
    msg = _make_message("user", [part])
    dicts, reasoning = _extract_message_dicts([msg])
    assert dicts[0]["content"] == "Part 1"
    assert reasoning == [None]


def test_extract_message_dicts_reasoning_part():
    """``ContentReasoning`` parts route through ``reasoning_content`` so the
    chat template (not ``content`` concatenation) decides where to render
    the reasoning."""
    reasoning_part = SimpleNamespace(type="reasoning", reasoning="<think>step</think>")
    text_part = SimpleNamespace(text="answer")
    msg = _make_message("assistant", [reasoning_part, text_part])
    dicts, reasoning = _extract_message_dicts([msg])
    assert dicts[0]["content"] == "answer"
    assert dicts[0]["reasoning"] == "<think>step</think>"
    assert dicts[0]["reasoning_content"] == "<think>step</think>"
    assert reasoning == ["<think>step</think>"]


def test_extract_message_dicts_propagates_tool_calls():
    """Inspect ``ToolCall`` objects surface in OpenAI-shaped ``tool_calls``
    on the message dict — required for chat templates (e.g. Kimi) that route
    assistant messages with tool calls into the suffix where reasoning is
    preserved."""
    tc = SimpleNamespace(
        id="call_0",
        function="search",
        arguments={"q": "x"},
        type="function",
    )
    msg = SimpleNamespace(role="assistant", content="text", tool_calls=[tc])
    dicts, _ = _extract_message_dicts([msg])
    assert dicts[0]["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "search", "arguments": {"q": "x"}},
        }
    ]


def test_extract_message_dicts_propagates_tool_message_id():
    msg = SimpleNamespace(
        role="tool", content="result", tool_call_id="call_0", function="search"
    )
    dicts, _ = _extract_message_dicts([msg])
    assert dicts[0]["tool_call_id"] == "call_0"
    assert dicts[0]["name"] == "search"


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
    sample = doc.samples[0]
    assert sample.annotation_positions("user")
    assert sample.annotation_positions("assistant")
    assert sample.annotation_positions("template")


def _chat_tokenizer():
    """Char-level chat tokenizer used by the generated/logprob tests.

    Mimics Kimi/Qwen-style native reasoning rendering by wrapping any
    ``reasoning_content`` / ``reasoning`` field in ``<think>…</think>``
    before the message's content, so the reasoning span shows up where
    a real reasoning-aware chat template would put it.
    """

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
                rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if rc:
                    parts.append(f"<think>{rc}</think>")
                parts.append(msg.get("content", ""))
                parts.append(f"[/{msg['role']}]")
            formatted = "".join(parts)
            if not tokenize:
                return formatted
            return [ord(c) for c in formatted]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    return ChatTokenizer()


def test_from_eval_log_tags_generated_tokens():
    """Tokens belonging to the LAST assistant message get generated annotation."""
    messages = [
        _make_message("user", "Hi"),
        _make_message("assistant", "Hello!"),
    ]
    sample = _make_sample(messages, sample_id="q1")
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=True,
    )
    tagged = doc.samples[0].select_by_annotation("generated").tokens
    # "Hello!" = 6 chars = 6 tokens
    assert len(tagged) == 6
    assert "".join(t.token for t in tagged) == "Hello!"
    # Tokens from earlier (template, user message) are NOT tagged.
    assert len(doc.samples[0].annotation_positions("generated")) == 6


def test_from_eval_log_tag_generated_disabled():
    messages = [
        _make_message("user", "Hi"),
        _make_message("assistant", "Hello!"),
    ]
    sample = _make_sample(messages, sample_id="q1")
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=False,
    )
    assert doc.samples[0].annotation_positions("generated") == []


def test_from_eval_log_tags_reasoning_blocks():
    """The diff-based tagger detects reasoning-content rendering by rendering
    the chat template with vs without the field — the diff is what the
    template emits FOR that field. For Kimi/Qwen-style ``<think>{rc}</think>``
    wrappers, only the inner content (``rc``) is in the diff window since
    the wrapper is template-emitted regardless of the field's value."""
    reasoning = SimpleNamespace(type="reasoning", reasoning="<think>plan</think>")
    answer = SimpleNamespace(text="Hello!")
    msg = _make_message("assistant", [reasoning, answer])
    sample_in = _make_sample([_make_message("user", "Hi"), msg], sample_id="q1")
    log = _make_eval_log(samples=[sample_in])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=False,
        tag_reasoning=True,
    )
    sample = doc.samples[0]

    reasoning_pos = sample.annotation_positions("reasoning")
    reasoning_text = "".join(sample.tokens[p].token for p in reasoning_pos)
    # The fake tokenizer wraps reasoning conditionally (`if rc:`), so removing
    # the field strips both the source's embedded `<think>plan</think>` AND
    # the wrapper. The diff is the entire double-wrapped block — exactly what
    # the template emits because of this field.
    assert reasoning_text == "<think><think>plan</think></think>"


def test_from_eval_log_tags_tool_call_blocks():
    """Tool-call rendering is detected via the same diff approach, and
    tagged as ``tool_call`` + ``assistant`` (moved out of ``template``)."""

    class ToolCallTokenizer:
        all_special_ids: list[int] = []

        def apply_chat_template(
            self,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            **kwargs,
        ):
            parts = ["<s>"]
            for m in messages:
                parts.append(f"[{m['role']}]")
                rc = m.get("reasoning_content") or m.get("reasoning") or ""
                if rc:
                    parts.append(f"<think>{rc}</think>")
                parts.append(m.get("content", ""))
                tcs = m.get("tool_calls") or []
                if tcs:
                    parts.append("<TC>")
                    for tc in tcs:
                        parts.append(f"[{tc['function']['name']}]")
                    parts.append("</TC>")
                parts.append(f"[/{m['role']}]")
            formatted = "".join(parts)
            if not tokenize:
                return formatted
            return [ord(c) for c in formatted]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    tc = SimpleNamespace(
        id="c0", function="search", arguments={"q": "hi"}, type="function"
    )
    msg = SimpleNamespace(
        role="assistant",
        content="ok",
        tool_calls=[tc],
    )
    sample_in = _make_sample([_make_message("user", "Hi"), msg], sample_id="q1")
    log = _make_eval_log(samples=[sample_in])

    doc = from_eval_log(
        log,
        tokenizer=ToolCallTokenizer(),
        deduplicate=False,
        tag_chat_roles=True,
        tag_generated=False,
        tag_reasoning=True,
    )
    sample = doc.samples[0]

    tool_call_pos = sample.annotation_positions("tool_call")
    rendered_tc = "".join(sample.tokens[p].token for p in tool_call_pos)
    # Diff is the entire <TC>...</TC> block (only emitted when tool_calls
    # is present).
    assert rendered_tc == "<TC>[search]</TC>"

    # Those positions also moved into the `assistant` chat-role annotation.
    assistant_pos = set(sample.annotation_positions("assistant"))
    assert set(tool_call_pos) <= assistant_pos
    # ... and out of the `template` annotation.
    template_pos = set(sample.annotation_positions("template"))
    assert set(tool_call_pos).isdisjoint(template_pos)


def test_from_eval_log_tag_reasoning_disabled():
    reasoning = SimpleNamespace(type="reasoning", reasoning="<think>plan</think>")
    answer = SimpleNamespace(text="Hello!")
    msg = _make_message("assistant", [reasoning, answer])
    sample_in = _make_sample([_make_message("user", "Hi"), msg], sample_id="q1")
    log = _make_eval_log(samples=[sample_in])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=False,
        tag_reasoning=False,
    )
    assert doc.samples[0].annotation_positions("reasoning") == []


def test_from_eval_log_attaches_logprobs():
    """When inspect's output carries per-token logprobs whose count matches
    the assistant token range, attach them to those tokens."""
    messages = [
        _make_message("user", "Hi"),
        _make_message("assistant", "Hello!"),
    ]
    # 6 logprob entries to match "Hello!" → 6 char tokens
    logprob_items = [SimpleNamespace(logprob=-0.1 * (i + 1)) for i in range(6)]
    output = SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(content=logprob_items),
            )
        ]
    )
    sample = _make_sample(messages, sample_id="q1")
    sample.output = output
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=True,
        extract_logprobs=True,
    )
    generated = doc.samples[0].select_by_annotation("generated").tokens
    assert len(generated) == 6
    expected = [-0.1 * (i + 1) for i in range(6)]
    assert [t.get_extra("logprob") for t in generated] == expected
    # Non-response tokens have no logprob attached.
    generated_positions = set(doc.samples[0].annotation_positions("generated"))
    others = [
        t for i, t in enumerate(doc.samples[0].tokens) if i not in generated_positions
    ]
    assert all(not t.has_extra("logprob") for t in others)


def test_from_eval_log_logprobs_skipped_on_count_mismatch():
    """If logprob count doesn't match the response token count, attachment is
    silently skipped — but the 'generated' tag still applies."""
    messages = [
        _make_message("user", "Hi"),
        _make_message("assistant", "Hello!"),
    ]
    # Wrong count (3 vs 6 chars)
    logprob_items = [SimpleNamespace(logprob=-0.5) for _ in range(3)]
    output = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=logprob_items))]
    )
    sample = _make_sample(messages, sample_id="q1")
    sample.output = output
    log = _make_eval_log(samples=[sample])

    doc = from_eval_log(
        log,
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=False,
        tag_generated=True,
        extract_logprobs=True,
    )
    # Generated annotation still applied
    assert doc.samples[0].annotation_positions("generated")
    # No logprobs attached
    assert all(not t.has_extra("logprob") for t in doc.samples[0].tokens)


def test_from_eval_log_usage():
    usage = SimpleNamespace(input_tokens=50, output_tokens=10)
    sample = _make_sample(
        [_make_message("user", "test")],
        sample_id=0,
        usage=usage,
    )
    log = _make_eval_log(samples=[sample])
    doc = from_eval_log(log, tokenizer=_chat_tokenizer(), deduplicate=False)
    assert doc.samples[0].input_tokens == 50
    assert doc.samples[0].output_tokens == 10
