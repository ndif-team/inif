from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from typing import Any

from inif.converters._decode import (
    ByteSliceDecoder,
    has_byte_level_decoder,
)
from inif.converters._tokenize import (
    find_message_token_range,
    messages_to_tokens,
    offset_mapping_decode,
    resolve_tokenizer,
    tag_template_field_renderings,
)
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    SourceEval,
    Text,
    TokenOrSeqRef,
)


def _get_package_versions() -> dict[str, str]:
    packages = {}
    for pkg in ["inif", "transformers", "inspect_ai"]:
        try:
            packages[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


_GENERATION_CONFIG_FIELDS: tuple[str, ...] = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_seqs",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "best_of",
    "logit_bias",
    "num_choices",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "parallel_tool_calls",
    "max_tool_output",
    "cache_prompt",
    "verbosity",
    "effort",
    "reasoning_effort",
    "reasoning_tokens",
    "reasoning_summary",
    "reasoning_history",
    "response_schema",
    "extra_headers",
    "extra_body",
    "modalities",
    "max_connections",
    "max_retries",
    "timeout",
)


def _extract_model_info(eval_log: Any, tokenizer: Any = None) -> ModelInfo:
    model_name = eval_log.eval.model
    gen_config: dict[str, Any] = {}

    # Generation params live on plan.config and on eval.model_generate_config —
    # different inspect releases populate one vs. the other depending on
    # whether the values came in via ``GenerateConfig`` or via the model
    # provider's defaults. Read from both, with eval.model_generate_config
    # winning on conflicts since it reflects the resolved config the provider
    # actually saw.
    config_sources = [
        getattr(eval_log.plan, "config", None) if eval_log.plan else None,
        getattr(eval_log.eval, "model_generate_config", None),
    ]
    for config in config_sources:
        if config is None:
            continue
        for attr in _GENERATION_CONFIG_FIELDS:
            val = getattr(config, attr, None)
            if val is not None:
                gen_config[attr] = val

    # Record that the converter rewrote Qwen3-style ``last_query_index``
    # gating so historical reasoning blocks are preserved in the rendered
    # tokens. Downstream consumers (e.g. interpretability conversions that
    # need to replay what the model actually saw at turn N) must reapply
    # the original gating themselves.
    gen_config["preserve_reasoning"] = True

    revision = None
    if tokenizer is not None:
        revision = getattr(tokenizer, "_commit_hash", None)

    return ModelInfo(
        name=model_name,
        revision=revision,
        generation_config=gen_config,
    )


_EVAL_CONFIG_FIELDS: tuple[str, ...] = (
    "limit",
    "sample_id",
    "epochs",
    "epochs_reducer",
    "fail_on_error",
    "continue_on_fail",
    "retry_on_error",
    "message_limit",
    "token_limit",
    "time_limit",
    "working_limit",
    "cost_limit",
    "max_samples",
    "max_tasks",
    "max_subprocesses",
    "max_sandboxes",
)


def _extract_source_eval(eval_log: Any) -> SourceEval:
    eval_spec = eval_log.eval
    task_version = getattr(eval_spec, "task_version", None)
    if task_version is not None:
        task_version = str(task_version)

    # Eval-time limits and run options that aren't generation params but ARE
    # needed to reproduce / interpret the run (sample limits, per-sample
    # message caps, time budgets, error-handling policy). Stash on
    # ``SourceEval.extra`` so the canonical source-eval shape stays
    # framework-agnostic while keeping the data accessible.
    extra: dict[str, Any] = {}
    eval_config = getattr(eval_spec, "config", None)
    if eval_config is not None:
        eval_extras: dict[str, Any] = {}
        for attr in _EVAL_CONFIG_FIELDS:
            val = getattr(eval_config, attr, None)
            if val is not None:
                eval_extras[attr] = val
        if eval_extras:
            extra["eval_config"] = eval_extras

    task_args = getattr(eval_spec, "task_args", None) or {}
    if task_args:
        extra["task_args"] = dict(task_args)
    solver = getattr(eval_spec, "solver", None)
    if solver:
        extra["solver"] = solver
    solver_args = getattr(eval_spec, "solver_args", None)
    if solver_args:
        extra["solver_args"] = dict(solver_args)
    model_args = getattr(eval_spec, "model_args", None) or {}
    if model_args:
        extra["model_args"] = dict(model_args)

    return SourceEval(
        framework="inspect_ai",
        framework_version=eval_spec.inspect_version
        if hasattr(eval_spec, "inspect_version")
        else None,
        task=eval_spec.task,
        task_version=task_version,
        eval_id=eval_spec.eval_id if hasattr(eval_spec, "eval_id") else None,
        run_id=eval_spec.run_id if hasattr(eval_spec, "run_id") else None,
        extra=extra,
    )


def _extract_tool_calls(msg: Any) -> list[dict[str, Any]] | None:
    """Convert Inspect ``ToolCall`` objects to the OpenAI/Kimi-compatible
    dict shape that HF chat templates expect."""
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        return None
    out: list[dict[str, Any]] = []
    for tc in raw:
        fn_name = getattr(tc, "function", None) or ""
        fn_args = getattr(tc, "arguments", None)
        if fn_args is None:
            fn_args = {}
        out.append(
            {
                "id": getattr(tc, "id", "") or "",
                "type": getattr(tc, "type", "function") or "function",
                "function": {"name": fn_name, "arguments": fn_args},
            }
        )
    return out


def _extract_message_dicts(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Convert Inspect AI message objects to ``(msg_dicts, reasoning_per_msg)``.

    Reasoning is routed through the chat template's native ``reasoning_content``
    slot (not concatenated into ``content``) so the template wraps it inside
    its own reasoning delimiters (``<think>…</think>`` for Kimi/Qwen, etc.)
    instead of dumping the prose as plain content text after an empty
    template-emitted wrapper. ``tool_calls`` and ``tool_call_id`` are
    propagated when present — Kimi's template, in particular, only routes
    assistant messages with ``tool_calls`` into the suffix (where
    ``reasoning_content`` is rendered); without this propagation, every
    intermediate reasoning gets stripped to ``<think></think>``.

    ``reasoning_per_msg[i]`` is the raw reasoning string for message ``i`` so
    :func:`tag_reasoning_blocks` can locate the rendered reasoning span in
    the formatted chat-template output.
    """
    msg_dicts: list[dict[str, Any]] = []
    reasoning_per_msg: list[str | None] = []
    for msg in messages:
        if isinstance(msg.content, str):
            text = msg.content
            reasoning: str | None = None
        elif isinstance(msg.content, list):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            for part in msg.content:
                if getattr(part, "type", None) == "reasoning":
                    rt = getattr(part, "reasoning", None) or ""
                    if rt:
                        reasoning_parts.append(rt)
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
            text = "".join(text_parts)
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
        else:
            text = str(msg.content)
            reasoning = None

        d: dict[str, Any] = {"role": msg.role, "content": text}
        if reasoning is not None:
            # Both spellings — different chat templates pick different ones
            # (Kimi reads ``reasoning`` first then falls back to ``reasoning_content``;
            # other templates use only ``reasoning_content``).
            d["reasoning"] = reasoning
            d["reasoning_content"] = reasoning
        tcs = _extract_tool_calls(msg)
        if tcs is not None:
            d["tool_calls"] = tcs
        tc_id = getattr(msg, "tool_call_id", None)
        if tc_id:
            d["tool_call_id"] = tc_id
        tc_fn = getattr(msg, "function", None)
        if tc_fn:
            d["name"] = tc_fn

        msg_dicts.append(d)
        reasoning_per_msg.append(reasoning)
    return msg_dicts, reasoning_per_msg


def _convert_scores(sample: Any) -> list[SampleScore]:
    scores = []
    if not hasattr(sample, "scores") or not sample.scores:
        return scores
    for scorer_name, score_obj in sample.scores.items():
        value = score_obj.value
        # Convert Score enum/string to primitive
        if hasattr(value, "value"):
            value = value.value
        scores.append(
            SampleScore(
                scorer=scorer_name,
                value=value,
                answer=getattr(score_obj, "answer", None),
                explanation=getattr(score_obj, "explanation", None),
                metadata=dict(getattr(score_obj, "metadata", {}) or {}),
            )
        )
    return scores


def _extract_logprobs(inspect_sample: Any) -> list[float | None] | None:
    """Pull per-token logprobs for the model's response, or None if absent."""
    output = getattr(inspect_sample, "output", None)
    if not output:
        return None
    choices = getattr(output, "choices", None)
    if not choices:
        return None
    logprobs = getattr(choices[0], "logprobs", None)
    if not logprobs:
        return None
    content = getattr(logprobs, "content", None)
    if not content:
        return None
    return [getattr(item, "logprob", None) for item in content]


def _annotate_response_tokens(
    sample: Sample,
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
    inspect_sample: Any,
    tag_generated: bool,
    extract_logprobs: bool,
) -> None:
    """Annotate the model's response tokens and attach per-token logprobs.

    Both behaviors require a HF tokenizer with ``apply_chat_template`` and a
    successful round-trip from tokens back to the formatted string. When that
    fails, the call is a no-op (these are convenience annotations, not invariants).
    """
    if not (tag_generated or extract_logprobs):
        return
    if tokenizer is None or not msg_dicts:
        return

    rng = find_message_token_range(
        sample.tokens, msg_dicts, tokenizer, role="assistant", which="last"
    )
    if rng is None:
        return
    start, end = rng

    if tag_generated:
        sample.annotate("generated", [(start, end)], metadata={"source": "converter"})

    if extract_logprobs:
        logprobs = _extract_logprobs(inspect_sample)
        if logprobs is not None and len(logprobs) == end - start:
            for offset, lp in enumerate(logprobs):
                if lp is not None:
                    sample.tokens[start + offset].set_extra("logprob", lp)


def from_eval_log(
    eval_log: Any,
    tokenizer: Any = "auto",
    include_messages: bool = True,
    deduplicate: bool = True,
    min_sequence_length: int = 5,
    tag_chat_roles: bool = True,
    tag_generated: bool = True,
    tag_reasoning: bool = True,
    extract_logprobs: bool = True,
) -> InifDocument:
    """Convert an Inspect AI EvalLog to an InifDocument.

    Args:
        eval_log: An inspect_ai.log.EvalLog object.
        tokenizer: One of:

            - ``"auto"`` (default) or ``None`` — auto-load
              ``AutoTokenizer.from_pretrained`` based on
              ``eval_log.eval.model``, after stripping any routing prefix
              (``together/``, ``hf/``, …). Raises ``ValueError`` if the
              model id is missing or the tokenizer cannot be loaded
              (closed-source ids like ``openai/gpt-4`` will hit this).
            - a model id string — loaded via ``AutoTokenizer.from_pretrained``;
              warns if the resolved id disagrees with the source eval's
              model id.
            - a tokenizer instance — used as-is; warns if its
              ``name_or_path`` disagrees with the source eval's model id.

            Tokens are a load-bearing invariant of every INIF sample, so
            there is no way to opt out of tokenization — every option here
            yields a usable tokenizer or raises.

        include_messages: Whether to include message-level text segments.
        deduplicate: Whether to run sequence deduplication.
        min_sequence_length: Minimum length for common sequence detection.
        tag_chat_roles: Whether to record per-message chat-template roles
            (``system``/``user``/``assistant``/``template``) on
            ``Sample.annotations`` with ``metadata={"source": "message_role"}``.
        tag_generated: Whether to tag the model's response tokens with
            ``"generated"``. The response is the last assistant message.
        tag_reasoning: Whether to annotate ``ContentReasoning`` blocks pulled
            from each assistant message. Tokens overlapping any reasoning span
            get ``reasoning`` + ``assistant``; tokens within the span whose
            id appears in ``tokenizer.all_special_ids`` additionally get
            ``template``. Best-effort: silently skipped on lossy tokenizers
            or when the reasoning text can't be located in the rendered
            chat template.
        extract_logprobs: Whether to attach per-token logprobs from the eval
            output to the response tokens (best-effort: requires the tokenizer
            and the eval-source tokenization to agree on token count).
    """
    source_model_id = (
        getattr(eval_log.eval, "model", None) if eval_log.eval is not None else None
    )
    tokenizer = resolve_tokenizer(tokenizer, source_model_id)

    model_info = _extract_model_info(eval_log, tokenizer)
    source_eval = _extract_source_eval(eval_log)

    # Extract timing
    total_time = None
    if eval_log.stats:
        started = getattr(eval_log.stats, "started_at", None)
        completed = getattr(eval_log.stats, "completed_at", None)
        if started and completed:
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed)
            total_time = (completed - started).total_seconds()

    samples: list[Sample] = []
    inspect_samples: list[Any] = []
    all_msg_dicts: list[list[dict[str, str]]] = []
    # Decide per-archive how to decode tokens into per-token strings so the
    # concat always matches the tokenizer's full-sequence decode. The order
    # of preference is (1) offset_mapping, (2) byte-slice, (3) per-token
    # decode with cache — see ``_messages_to_tokens`` docstring. We only
    # probe each candidate once and reuse the decision across samples.
    byte_decoder: ByteSliceDecoder | None = None
    decode_cache: dict[int, str] | None = None
    use_offset_mapping = False
    if eval_log.samples:
        first_msgs: list[dict[str, str]] = []
        probe_sample = next(
            (s for s in eval_log.samples if getattr(s, "messages", None)),
            None,
        )
        if probe_sample is not None:
            first_msgs, _ = _extract_message_dicts(probe_sample.messages)
        if first_msgs and offset_mapping_decode(first_msgs, tokenizer) is not None:
            use_offset_mapping = True
        elif has_byte_level_decoder(tokenizer):
            byte_decoder = ByteSliceDecoder(tokenizer)
        else:
            decode_cache = {}
    if eval_log.samples:
        for i, inspect_sample in enumerate(eval_log.samples):
            sample_id_raw = getattr(inspect_sample, "id", i)
            if sample_id_raw is None:
                sample_id_raw = i
            sample_id = str(sample_id_raw)

            texts: list[Text] = []
            sample_tokens: list[TokenOrSeqRef] = []
            target = None

            if hasattr(inspect_sample, "target") and inspect_sample.target:
                target = (
                    inspect_sample.target
                    if isinstance(inspect_sample.target, str)
                    else str(inspect_sample.target)
                )

            msg_dicts: list[dict[str, str]] = []
            if include_messages and hasattr(inspect_sample, "messages"):
                msg_dicts, _reasoning = _extract_message_dicts(inspect_sample.messages)
                texts, sample_tokens = messages_to_tokens(
                    msg_dicts,
                    tokenizer,
                    decode_cache=decode_cache,
                    byte_decoder=byte_decoder,
                    use_offset_mapping=use_offset_mapping,
                )
            all_msg_dicts.append(msg_dicts)
            inspect_samples.append(inspect_sample)

            scores = _convert_scores(inspect_sample)

            # Extract usage stats
            input_toks = None
            output_toks = None
            if hasattr(inspect_sample, "usage") and inspect_sample.usage:
                usage = inspect_sample.usage
                input_toks = getattr(usage, "input_tokens", None)
                output_toks = getattr(usage, "output_tokens", None)

            sample = Sample(
                id=sample_id,
                texts=texts,
                tokens=sample_tokens,
                scores=scores,
                target=target,
                input_tokens=input_toks,
                output_tokens=output_toks,
            )

            # Annotate response tokens BEFORE dedup so generated ranges and
            # logprobs prevent response-specific runs from collapsing.
            _annotate_response_tokens(
                sample,
                msg_dicts,
                tokenizer,
                inspect_sample,
                tag_generated=tag_generated,
                extract_logprobs=extract_logprobs,
            )

            samples.append(sample)

    metadata = Metadata(
        model=model_info,
        packages=_get_package_versions(),
        source_eval=source_eval,
        created_at=datetime.now(timezone.utc),
        total_time=total_time,
    )

    doc = InifDocument(
        metadata=metadata,
        samples=samples,
    )

    if deduplicate:
        doc = doc.deduplicate_sequences(min_length=min_sequence_length)

    if tag_chat_roles:
        doc.tag_chat_roles(all_msg_dicts, tokenizer)

    if tag_reasoning:
        sequences = doc.sequences or None
        for sample, msg_dicts in zip(doc.samples, all_msg_dicts):
            tag_template_field_renderings(
                sample, msg_dicts, tokenizer, sequences=sequences
            )

    return doc


def from_eval_file(path: str, **kwargs: Any) -> InifDocument:
    """Load an Inspect AI eval log file and convert to InifDocument."""
    from inspect_ai.log import read_eval_log

    eval_log = read_eval_log(path)
    doc = from_eval_log(eval_log, **kwargs)
    doc.metadata.sources.append(str(path))
    return doc


# Re-export Sequence for downstream typing convenience (used internally above
# only via type hints). Kept to avoid silent import errors after refactors.
__all__ = ["from_eval_log", "from_eval_file", "Sequence"]
