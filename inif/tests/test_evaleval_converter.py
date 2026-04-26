"""Tests for the every_eval_ever (EEE) converter.

All tests build in-memory records that conform to ``instance_level_eval_0.2.2``.
No HuggingFace downloads happen in CI (see the ``pytest.mark.skipif`` guard on
the one integration test at the bottom).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inif.converters.evaleval import (
    _build_doc_extra,
    _build_sample_metadata,
    _build_score,
    _messages_from_record,
    from_eval_json,
    from_instance_records,
)
from inif.io import to_dict
from inif.schema import validate
from inif.selectors import select_by_annotation

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_record(
    sample_id="q1",
    evaluation_id="eval_001",
    evaluation_result_id="eval_001_acc",
    evaluation_name="hellaswag",
    model_id="meta-llama/Llama-3.2-1B",
    interaction_type="single_turn",
    input_raw="What is 2+2?",
    references=("4",),
    choices=None,
    output_raw=("The answer is 4.",),
    reasoning_trace=None,
    messages=None,
    score=1.0,
    is_correct=True,
    extracted_value="4",
    extraction_method="regex",
    token_usage=None,
    performance=None,
    error=None,
    eval_metadata=None,
    sample_hash=None,
    schema_version="instance_level_eval_0.2.2",
):
    record: dict = {
        "schema_version": schema_version,
        "evaluation_id": evaluation_id,
        "evaluation_result_id": evaluation_result_id,
        "model_id": model_id,
        "evaluation_name": evaluation_name,
        "sample_id": sample_id,
        "interaction_type": interaction_type,
        "input": {
            "raw": input_raw,
            "reference": list(references),
        },
        "answer_attribution": [
            {
                "turn_idx": 0,
                "source": "output.raw",
                "extracted_value": extracted_value,
                "extraction_method": extraction_method,
                "is_terminal": True,
            }
        ],
        "evaluation": {"score": score, "is_correct": is_correct},
    }
    if choices is not None:
        record["input"]["choices"] = list(choices)
    if interaction_type == "single_turn":
        record["output"] = {"raw": list(output_raw)}
        if reasoning_trace is not None:
            record["output"]["reasoning_trace"] = list(reasoning_trace)
    else:
        record["messages"] = list(messages or [])
    if token_usage is not None:
        record["token_usage"] = dict(token_usage)
    if performance is not None:
        record["performance"] = dict(performance)
    if error is not None:
        record["error"] = error
    if eval_metadata is not None:
        record["metadata"] = dict(eval_metadata)
    if sample_hash is not None:
        record["sample_hash"] = sample_hash
    return record


def _chat_tokenizer():
    """Char-level fake tokenizer exposing ``apply_chat_template``.

    Mirrors the one used by ``test_inspect_converter.py`` so the two converter
    tests exercise the same code paths through ``messages_to_tokens``.
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
                parts.append(msg["content"])
                parts.append(f"[/{msg['role']}]")
            formatted = "".join(parts)
            if not tokenize:
                return formatted
            return [ord(c) for c in formatted]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    return ChatTokenizer()


def _make_aggregate(
    evaluation_id="eval_001",
    evaluation_result_id="eval_001_acc",
    model_id="meta-llama/Llama-3.2-1B",
    developer="Meta",
    eval_library_name="inspect_ai",
    eval_library_version="0.3.0",
    metric_id="accuracy",
    aggregate_score=0.75,
):
    return {
        "schema_version": "0.2.2",
        "evaluation_id": evaluation_id,
        "retrieved_timestamp": "1700000000",
        "source_metadata": {
            "source_type": "evaluation_run",
            "source_organization_name": "test-org",
            "evaluator_relationship": "first_party",
        },
        "model_info": {
            "name": model_id,
            "id": model_id,
            "developer": developer,
            "inference_platform": "HuggingFace",
        },
        "eval_library": {"name": eval_library_name, "version": eval_library_version},
        "evaluation_results": [
            {
                "evaluation_name": "hellaswag",
                "evaluation_result_id": evaluation_result_id,
                "source_data": {"hf_dataset": "Rowan/hellaswag"},
                "metric_config": {
                    "metric_id": metric_id,
                    "metric_name": "Accuracy",
                    "metric_kind": "accuracy",
                    "lower_is_better": False,
                    "score_type": "binary",
                },
                "score_details": {"score": aggregate_score},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Message reconstruction
# ---------------------------------------------------------------------------


def test_messages_from_record_single_turn_with_reasoning():
    record = _make_record(
        reasoning_trace=["Let me think... ", "2+2=4. "],
        output_raw=("The answer is 4.",),
    )
    msgs = _messages_from_record(record)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "What is 2+2?"
    assert msgs[1]["content"] == "Let me think... 2+2=4. The answer is 4."


def test_messages_from_record_multi_turn_orders_by_turn_idx():
    record = _make_record(
        interaction_type="multi_turn",
        output_raw=(),
        messages=[
            {"turn_idx": 1, "role": "assistant", "content": "Hi there."},
            {"turn_idx": 0, "role": "user", "content": "Hello."},
            {"turn_idx": 2, "role": "user", "content": "Bye."},
        ],
    )
    msgs = _messages_from_record(record)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [m["content"] for m in msgs] == ["Hello.", "Hi there.", "Bye."]


def test_messages_from_record_agentic_serializes_tool_calls():
    record = _make_record(
        interaction_type="agentic",
        output_raw=(),
        messages=[
            {"turn_idx": 0, "role": "user", "content": "Search weather."},
            {
                "turn_idx": 1,
                "role": "assistant",
                "content": "Calling tool.",
                "tool_calls": [
                    {"id": "c1", "name": "search", "arguments": {"q": "weather"}}
                ],
            },
            {"turn_idx": 2, "role": "tool", "content": "sunny"},
        ],
    )
    msgs = _messages_from_record(record)
    assert msgs[1]["role"] == "assistant"
    assert "<tool_call name=search " in msgs[1]["content"]
    assert '"q": "weather"' in msgs[1]["content"]
    assert msgs[2]["role"] == "tool"


# ---------------------------------------------------------------------------
# from_instance_records — interaction types
# ---------------------------------------------------------------------------


def test_from_instance_records_single_turn():
    doc = from_instance_records(
        [_make_record()], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    assert doc.total_samples == 1
    s = doc.samples[0]
    assert s.id == "q1"
    assert s.target == "4"
    assert s.references == ["4"]
    assert s.interaction_type == "single_turn"
    assert [(t.name, t.value) for t in s.texts] == [
        ("user_0", "What is 2+2?"),
        ("assistant_0", "The answer is 4."),
    ]
    # Tokens are a load-bearing INIF invariant — always populated.
    assert len(s.tokens) > 0
    assert len(s.scores) == 1
    assert s.scores[0].value == 1.0
    assert s.scores[0].answer == "4"
    assert s.scores[0].metadata["is_correct"] is True


def test_include_messages_false_still_tokenizes():
    doc = from_instance_records(
        [_make_record()],
        tokenizer=_chat_tokenizer(),
        include_messages=False,
        deduplicate=False,
        tag_chat_roles=False,
    )
    s = doc.samples[0]
    assert s.texts == []
    assert len(s.tokens) > 0
    joined = "".join(t.token or "" for t in s.tokens)
    assert "What is 2+2?" in joined
    assert "The answer is 4." in joined


def test_from_instance_records_multi_turn_with_tokenizer():
    record = _make_record(
        interaction_type="multi_turn",
        output_raw=(),
        messages=[
            {"turn_idx": 0, "role": "user", "content": "Hi"},
            {"turn_idx": 1, "role": "assistant", "content": "Yo"},
        ],
    )
    doc = from_instance_records(
        [record],
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_chat_roles=True,
    )
    assert doc.samples[0].interaction_type == "multi_turn"
    sample = doc.samples[0]
    assert sample.annotation_positions("user")
    assert sample.annotation_positions("assistant")
    assert sample.annotation_positions("template")


def test_from_instance_records_agentic_tool_call_tokens_preserved():
    record = _make_record(
        interaction_type="agentic",
        output_raw=(),
        messages=[
            {"turn_idx": 0, "role": "user", "content": "go"},
            {
                "turn_idx": 1,
                "role": "assistant",
                "content": "ok",
                "tool_calls": [{"id": "c1", "name": "f", "arguments": {}}],
            },
        ],
    )
    doc = from_instance_records(
        [record], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    assert doc.samples[0].interaction_type == "agentic"
    joined = "".join(t.token or "" for t in doc.samples[0].tokens)
    assert "tool_call name=f" in joined


# ---------------------------------------------------------------------------
# Tier 1 first-class field promotion
# ---------------------------------------------------------------------------


def test_tier1_fields_populate_from_record():
    record = _make_record(
        choices=("A", "B", "C", "D"),
        sample_hash="deadbeef" * 8,
        error=None,
    )
    doc = from_instance_records(
        [record], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    s = doc.samples[0]
    assert s.choices == ["A", "B", "C", "D"]
    assert s.sample_hash == "deadbeef" * 8
    assert s.error is None
    assert s.interaction_type == "single_turn"
    assert s.references == ["4"]
    # First-class fields no longer leak into metadata
    promoted_fields = (
        "interaction_type",
        "error",
        "sample_hash",
        "choices",
        "references",
    )
    for promoted in promoted_fields:
        assert promoted not in s.metadata, (
            f"{promoted!r} should be a first-class field, not a metadata key"
        )


def test_target_singleton_vs_multi_reference():
    # len(reference) == 1 → Sample.target is a str AND references is the full list.
    doc1 = from_instance_records(
        [_make_record(references=("4",))],
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
    )
    assert doc1.samples[0].target == "4"
    assert doc1.samples[0].references == ["4"]

    # Multiple references → target stays None, references holds them all.
    doc2 = from_instance_records(
        [_make_record(references=("4", "four"))],
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
    )
    assert doc2.samples[0].target is None
    assert doc2.samples[0].references == ["4", "four"]


def test_error_promoted_to_first_class_field():
    record = _make_record(error="API timeout after 60s")
    doc = from_instance_records(
        [record], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    assert doc.samples[0].error == "API timeout after 60s"
    assert "error" not in doc.samples[0].metadata


def test_error_absent_when_null():
    doc = from_instance_records(
        [_make_record()], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    assert doc.samples[0].error is None


# ---------------------------------------------------------------------------
# Scoring + answer attribution
# ---------------------------------------------------------------------------


def test_answer_attribution_terminal_drives_score_answer():
    record = _make_record()
    record["answer_attribution"] = [
        {
            "turn_idx": 0,
            "source": "messages[1].content",
            "extracted_value": "first-guess",
            "extraction_method": "regex",
            "is_terminal": False,
        },
        {
            "turn_idx": 0,
            "source": "output.raw",
            "extracted_value": "final",
            "extraction_method": "exact_match",
            "is_terminal": True,
        },
    ]
    doc = from_instance_records(
        [record], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    s = doc.samples[0]
    assert s.scores[0].answer == "final"
    assert s.scores[0].metadata["extraction_method"] == "exact_match"
    # Non-terminal attribution survives in sample metadata for later analysis.
    assert len(s.metadata["intermediate_answers"]) == 1
    assert s.metadata["intermediate_answers"][0]["extracted_value"] == "first-guess"


# ---------------------------------------------------------------------------
# Reasoning-trace tagging
# ---------------------------------------------------------------------------


def test_reasoning_tag_applied_via_char_span():
    record = _make_record(
        reasoning_trace=("I think", " step by step."),
        output_raw=("Answer: 4",),
    )
    doc = from_instance_records(
        [record],
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_reasoning=True,
    )
    sample = doc.samples[0]
    reasoning_tokens = select_by_annotation(sample, "reasoning").tokens
    tagged_text = "".join(t.token or "" for t in reasoning_tokens)
    # The reasoning trace should be covered entirely by reasoning-annotated tokens.
    assert tagged_text == "I think step by step."
    # Tokens outside the reasoning span stay unannotated.
    reasoning_positions = set(sample.annotation_positions("reasoning"))
    post_reasoning = "".join(
        t.token or ""
        for i, t in enumerate(sample.tokens)
        if i not in reasoning_positions
    )
    assert "Answer: 4" in post_reasoning


def test_reasoning_tag_applied_per_multi_turn_message():
    record = _make_record(
        interaction_type="multi_turn",
        output_raw=(),
        messages=[
            {"turn_idx": 0, "role": "user", "content": "Question 1"},
            {
                "turn_idx": 1,
                "role": "assistant",
                "reasoning_trace": "think one. ",
                "content": "Answer 1",
            },
            {"turn_idx": 2, "role": "user", "content": "Question 2"},
            {
                "turn_idx": 3,
                "role": "assistant",
                "reasoning_trace": "think two. ",
                "content": "Answer 2",
            },
        ],
    )
    doc = from_instance_records(
        [record],
        tokenizer=_chat_tokenizer(),
        deduplicate=False,
        tag_reasoning=True,
        tag_generated=False,
        tag_chat_roles=False,
    )
    sample = doc.samples[0]
    reasoning_tokens = select_by_annotation(sample, "reasoning").tokens
    tagged_text = "".join(t.token or "" for t in reasoning_tokens)
    assert tagged_text == "think one. think two. "
    reasoning_positions = set(sample.annotation_positions("reasoning"))
    untagged_text = "".join(
        t.token or ""
        for i, t in enumerate(sample.tokens)
        if i not in reasoning_positions
    )
    assert "Answer 1" in untagged_text
    assert "Answer 2" in untagged_text


# ---------------------------------------------------------------------------
# Aggregate metadata + token usage + performance
# ---------------------------------------------------------------------------


def test_aggregate_metadata_merged():
    record = _make_record(evaluation_result_id="eval_001_acc")
    agg = _make_aggregate(evaluation_result_id="eval_001_acc", aggregate_score=0.825)
    doc = from_instance_records(
        [record], aggregate=agg, tokenizer=_chat_tokenizer(), deduplicate=False
    )

    md = doc.metadata
    assert md.model.name == "meta-llama/Llama-3.2-1B"
    assert md.extra["inference"]["developer"] == "Meta"
    assert md.extra["eval_library"]["name"] == "inspect_ai"
    assert md.extra["metric_config"]["metric_id"] == "accuracy"
    assert md.extra["aggregate_score"]["score"] == 0.825
    assert md.source_eval.framework == "evaleval"
    assert md.source_eval.framework_version == "instance_level_eval_0.2.2"
    assert md.source_eval.eval_id == "eval_001"
    assert md.source_eval.run_id == "eval_001_acc"


def test_token_usage_and_performance_surface_on_sample():
    record = _make_record(
        token_usage={
            "input_tokens": 12,
            "output_tokens": 34,
            "total_tokens": 46,
            "reasoning_tokens": 20,
        },
        performance={"latency_ms": 500.0, "time_to_first_token_ms": 50.0},
    )
    doc = from_instance_records(
        [record], tokenizer=_chat_tokenizer(), deduplicate=False
    )
    s = doc.samples[0]
    assert s.input_tokens == 12
    assert s.output_tokens == 34
    assert s.metadata["reasoning_tokens"] == 20
    assert s.metadata["performance"]["latency_ms"] == 500.0


# ---------------------------------------------------------------------------
# Deduplication + IO round-trip
# ---------------------------------------------------------------------------


def test_deduplication_collapses_shared_prefix():
    # Three records share the same user prompt → a common token run across all
    # samples should collapse into one sequence after dedup.
    records = [
        _make_record(sample_id=f"q{i}", output_raw=(f"Answer {i}",)) for i in range(3)
    ]
    doc = from_instance_records(
        records,
        tokenizer=_chat_tokenizer(),
        deduplicate=True,
        min_sequence_length=3,
        tag_chat_roles=False,
        tag_generated=False,
        tag_reasoning=False,
    )
    assert len(doc.sequences) >= 1
    assert all(any(t.is_sequence_ref for t in s.tokens) for s in doc.samples), (
        "every sample should reference the shared sequence after dedup"
    )


def test_from_eval_json_roundtrip(tmp_path: Path):
    agg_path = tmp_path / "agg.json"
    inst_path = tmp_path / "instances.jsonl"
    agg_path.write_text(json.dumps(_make_aggregate()))
    with inst_path.open("w") as f:
        for rec in [
            _make_record(sample_id="q1"),
            _make_record(sample_id="q2", score=0.0, is_correct=False),
        ]:
            f.write(json.dumps(rec) + "\n")

    doc = from_eval_json(
        agg_path, inst_path, tokenizer=_chat_tokenizer(), deduplicate=False
    )
    assert doc.total_samples == 2
    assert doc.metadata.extra["metric_config"]["metric_id"] == "accuracy"
    assert str(inst_path) in doc.metadata.sources
    # Round-trip through the top-level INIF schema validator to confirm the
    # produced document (including the new Tier 1 fields) is well-formed.
    validate(to_dict(doc))


def test_build_helpers_are_pure_dict_operations():
    # These helpers should produce the same output whether the caller is the
    # public API or a downstream tool that wants the intermediate dicts —
    # guards against them gaining hidden side effects on the input record.
    record = _make_record(eval_metadata={"subject": "arithmetic"})
    record_copy = json.loads(json.dumps(record))
    _build_sample_metadata(record)
    _build_score(record)
    _build_doc_extra([record], aggregate=None)
    assert record == record_copy


# ---------------------------------------------------------------------------
# Optional HF integration smoke test (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("CI", "").lower() in ("1", "true"),
    reason="HF dataset download — opt-in integration test",
)
def test_from_hf_dataset_smoke():
    """End-to-end smoke test against a small real config. Skipped in CI."""
    pytest.importorskip("datasets")
    from inif.converters.evaleval import from_hf_dataset

    doc = from_hf_dataset("theory_of_mind_samples", limit=3, deduplicate=False)
    assert doc.total_samples <= 3
    assert doc.metadata.source_eval is not None
    assert doc.metadata.source_eval.framework == "evaleval"


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------


class _FakeAutoTokenizer:
    """Stand-in for ``transformers.AutoTokenizer`` used by resolver tests.

    Records every ``from_pretrained`` call. By default returns a fake instance
    whose ``name_or_path`` reflects the requested id; raise behavior is
    controlled by ``raise_for``.
    """

    calls: list[str] = []
    raise_for: set[str] = set()

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.raise_for = set()

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append(model_id)
        if model_id in cls.raise_for:
            raise OSError(f"fake-not-found: {model_id}")

        class _Tok:
            name_or_path = model_id

            def apply_chat_template(
                self,
                messages,
                tokenize=True,
                add_generation_prompt=False,  # noqa: ARG002
                return_dict=False,  # noqa: ARG002
                **kwargs,  # noqa: ARG002
            ):
                formatted = "".join(msg.get("content", "") for msg in messages)
                if not tokenize:
                    return formatted
                return [ord(c) for c in formatted]

            def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
                return "".join(chr(i) for i in ids)

        return _Tok()


def _patch_autotokenizer(monkeypatch):
    import transformers

    _FakeAutoTokenizer.reset()
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)


_RESOLVER_KWARGS = dict(
    include_messages=False,
    deduplicate=False,
    tag_chat_roles=False,
    tag_generated=False,
    tag_reasoning=False,
)


def test_resolver_auto_loads_from_record_model_id(monkeypatch):
    _patch_autotokenizer(monkeypatch)
    record = _make_record(model_id="meta-llama/Llama-3.2-1B")
    from_instance_records([record], **_RESOLVER_KWARGS)
    assert _FakeAutoTokenizer.calls == ["meta-llama/Llama-3.2-1B"]


def test_resolver_strips_provider_prefix(monkeypatch):
    _patch_autotokenizer(monkeypatch)
    record = _make_record(model_id="together/moonshotai/Kimi-K2.5")
    from_instance_records([record], **_RESOLVER_KWARGS)
    assert _FakeAutoTokenizer.calls == ["moonshotai/Kimi-K2.5"]


def test_resolver_raises_when_model_id_missing():
    record = _make_record(model_id="")
    with pytest.raises(ValueError, match="no model id is available"):
        from_instance_records([record], deduplicate=False)


def test_resolver_raises_when_tokenizer_unavailable(monkeypatch):
    _patch_autotokenizer(monkeypatch)
    _FakeAutoTokenizer.raise_for = {"openai/gpt-4"}
    record = _make_record(model_id="openai/gpt-4")
    with pytest.raises(ValueError, match="Could not auto-load a tokenizer"):
        from_instance_records([record], **_RESOLVER_KWARGS)


def test_resolver_explicit_id_warns_on_mismatch(monkeypatch):
    _patch_autotokenizer(monkeypatch)
    record = _make_record(model_id="meta-llama/Llama-3.2-1B")
    with pytest.warns(UserWarning, match="Tokenizer mismatch"):
        from_instance_records(
            [record], tokenizer="mistralai/Mistral-7B", **_RESOLVER_KWARGS
        )


def test_resolver_explicit_id_no_warning_when_matched(monkeypatch, recwarn):
    _patch_autotokenizer(monkeypatch)
    record = _make_record(model_id="meta-llama/Llama-3.2-1B")
    from_instance_records(
        [record], tokenizer="meta-llama/Llama-3.2-1B", **_RESOLVER_KWARGS
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_resolver_instance_warns_on_mismatch():
    record = _make_record(model_id="meta-llama/Llama-3.2-1B")

    class _Tok:
        name_or_path = "Qwen/Qwen2.5-3B-Instruct"

        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
            return "".join(chr(i) for i in ids)

    with pytest.warns(UserWarning, match="Tokenizer mismatch"):
        from_instance_records(
            [record],
            tokenizer=_Tok(),
            include_messages=False,
            deduplicate=False,
            tag_chat_roles=False,
            tag_generated=False,
            tag_reasoning=False,
        )


def test_resolver_treats_none_as_auto(monkeypatch):
    """``tokenizer=None`` is an alias for ``"auto"``: tokens are still produced,
    never skipped (tokens are a load-bearing INIF invariant)."""
    _patch_autotokenizer(monkeypatch)
    record = _make_record(model_id="meta-llama/Llama-3.2-1B")
    from_instance_records([record], tokenizer=None, **_RESOLVER_KWARGS)
    assert _FakeAutoTokenizer.calls == ["meta-llama/Llama-3.2-1B"]
