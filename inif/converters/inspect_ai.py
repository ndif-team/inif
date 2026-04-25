from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from typing import Any

from inif.converters._decode import (
    ByteSliceDecoder,
    decode_token_ids,
    has_byte_level_decoder,
    offset_mapping_decode_text,
)
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    SourceEval,
    Token,
)
from inif.sequences import deduplicate_sequences


def _get_package_versions() -> dict[str, str]:
    packages = {}
    for pkg in ["inif", "transformers", "inspect_ai"]:
        try:
            packages[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


def _extract_model_info(eval_log: Any, tokenizer: Any = None) -> ModelInfo:
    model_name = eval_log.eval.model
    gen_config = {}
    if eval_log.plan and eval_log.plan.config:
        config = eval_log.plan.config
        for attr in [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_seqs",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ]:
            val = getattr(config, attr, None)
            if val is not None:
                gen_config[attr] = val

    revision = None
    if tokenizer is not None:
        revision = getattr(tokenizer, "_commit_hash", None)

    return ModelInfo(
        name=model_name,
        revision=revision,
        generation_config=gen_config,
    )


def _extract_source_eval(eval_log: Any) -> SourceEval:
    eval_spec = eval_log.eval
    task_version = getattr(eval_spec, "task_version", None)
    if task_version is not None:
        task_version = str(task_version)
    return SourceEval(
        framework="inspect_ai",
        framework_version=eval_spec.inspect_version
        if hasattr(eval_spec, "inspect_version")
        else None,
        task=eval_spec.task,
        task_version=task_version,
        eval_id=eval_spec.eval_id if hasattr(eval_spec, "eval_id") else None,
        run_id=eval_spec.run_id if hasattr(eval_spec, "run_id") else None,
    )


def _extract_message_dicts(messages: list[Any]) -> list[dict[str, str]]:
    """Convert Inspect AI message objects to plain dicts."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            text = msg.content
        elif isinstance(msg.content, list):
            text_parts = []
            for part in msg.content:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
            text = "".join(text_parts)
        else:
            text = str(msg.content)
        result.append({"role": msg.role, "content": text})
    return result


def _offset_mapping_decode(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> tuple[list[int], list[str]] | None:
    """Decode via ``return_offsets_mapping`` if the tokenizer supports it.

    Fast tokenizers (most HF-fast SentencePiece + BPE variants) expose byte
    offsets from a single ``tokenizer(text, return_offsets_mapping=True)``
    call, so we can slice per-token strings directly from the formatted chat
    template — bulletproof even when the tokenizer's decode path is lossy.

    Returns ``(ids, pieces)`` on success (``"".join(pieces) == formatted``)
    or ``None`` if offsets aren't supported / the re-tokenized ids don't
    match ``apply_chat_template(tokenize=True)`` (template-internal
    normalization). The caller should try the next tier on ``None``.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        ids_from_template = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except (TypeError, KeyError, ValueError):
        return None

    pieces = offset_mapping_decode_text(formatted, list(ids_from_template), tokenizer)
    if pieces is None:
        return None
    return list(ids_from_template), pieces


def _messages_to_tokens(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
    decode_cache: dict[int, str] | None = None,
    byte_decoder: "ByteSliceDecoder | None" = None,
    use_offset_mapping: bool = True,
) -> tuple[list[str], list[Token]]:
    """Convert message dicts to plain text strings and Tokens.

    When the tokenizer supports ``apply_chat_template``, uses it to produce the
    full token sequence including template delimiters. Otherwise falls back to
    per-message ``tokenizer.encode()``.

    Role tagging is NOT done here — it is a separate post-dedup step via
    ``tag_chat_roles_doc``.

    Returns (texts, tokens).

    Three-tier decode strategy (in order of preference):

    1. **offset_mapping** (HF fast tokenizers that support it — most fast
       SentencePiece + BPE): per-token strings are exact slices of the
       formatted chat template, so ``"".join(token.token) == formatted``
       for every sample, no matter which chars appear. Tried first; on
       mismatch with template-internal ids we move on.

    2. **byte-slice** (byte-level BPE without offset support, e.g. Kimi's
       tiktoken variant): reconstruct per-token bytes via
       ``tokenizer.byte_decoder`` + ``encoder`` (no decode calls), decode
       the whole byte stream once, attribute each char to the token whose
       byte range contains the char's last byte. Also bulletproof.

    3. **per-token decode with cache** (fallback for tokenizers with
       neither of the above — mostly old SentencePiece slow tokenizers):
       ``tokenizer.decode([tid])`` returns U+FFFD when a multi-byte char
       is split across BPE tokens. ``tag_chat_roles`` silently skips
       those samples.

    Callers supply ``byte_decoder`` (shared across samples of one archive)
    and / or ``decode_cache`` (same) to amortize per-id work.
    """
    texts = [msg["content"] for msg in messages]
    tokens: list[Token] = []

    if tokenizer is None:
        return texts, tokens

    if hasattr(tokenizer, "apply_chat_template"):
        if use_offset_mapping:
            result = _offset_mapping_decode(messages, tokenizer)
            if result is not None:
                ids, pieces = result
                for tid, piece in zip(ids, pieces):
                    tokens.append(Token(id=tid, token=piece))
                return texts, tokens

        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        try:
            pieces = decode_token_ids(
                list(token_ids),
                tokenizer,
                decode_cache=decode_cache,
                byte_decoder=byte_decoder,
                strict_roundtrip=True,
            )
        except ValueError as e:
            raise ValueError(
                "Per-token decode does not round-trip to the full decode: "
                f"{type(tokenizer).__name__} does not support "
                "``return_offsets_mapping`` and does not expose byte-level "
                "BPE internals (``encoder`` + ``byte_decoder``), so INIF "
                "cannot produce reliable per-token strings for content "
                "containing multi-byte UTF-8 characters split across tokens. "
                "Use a tokenizer that supports either mechanism, or strip the "
                "problematic characters before tokenizing."
            ) from e
        for tid, piece in zip(token_ids, pieces):
            tokens.append(Token(id=tid, token=piece))
    else:
        for msg in messages:
            text = msg["content"]
            if text:
                encoded = tokenizer.encode(text, add_special_tokens=False)
                for tid in encoded:
                    token_str = tokenizer.decode([tid])
                    tokens.append(Token(id=tid, token=token_str))

    return texts, tokens


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


def _last_assistant_content(msg_dicts: list[dict[str, str]]) -> str | None:
    for msg in reversed(msg_dicts):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            return content if content else None
    return None


def _find_assistant_token_range(
    tokens: list[Token],
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
) -> tuple[int, int] | None:
    """Locate the [start, end) token indices that cover the LAST assistant
    message's content within the chat-template-formatted token stream.

    Returns ``None`` when the formatted string can't be reconstructed from the
    token sequence (e.g. a tokenizer that drops bytes), or when the content
    can't be located unambiguously.
    """
    content = _last_assistant_content(msg_dicts)
    if content is None:
        return None
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    formatted = tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=False
    )
    decoded = [t.token or "" for t in tokens]
    if "".join(decoded) != formatted:
        return None

    char_start = formatted.rfind(content)
    if char_start < 0:
        return None
    char_end = char_start + len(content)

    pos = 0
    tok_start: int | None = None
    tok_end: int | None = None
    for i, s in enumerate(decoded):
        s_start, s_end = pos, pos + len(s)
        if tok_start is None and s_end > char_start:
            tok_start = i
        if s_start < char_end:
            tok_end = i + 1
        pos = s_end
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


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
    """Tag the model's response tokens and attach per-token logprobs.

    Both behaviors require a HF tokenizer with ``apply_chat_template`` and a
    successful round-trip from tokens back to the formatted string. When that
    fails, the call is a no-op (these are convenience annotations, not invariants).
    """
    if not (tag_generated or extract_logprobs):
        return
    if tokenizer is None or not msg_dicts:
        return

    rng = _find_assistant_token_range(sample.tokens, msg_dicts, tokenizer)
    if rng is None:
        return
    start, end = rng

    if tag_generated:
        for i in range(start, end):
            sample.tokens[i].add_tag("generated")

    if extract_logprobs:
        logprobs = _extract_logprobs(inspect_sample)
        if logprobs is not None and len(logprobs) == end - start:
            for offset, lp in enumerate(logprobs):
                if lp is not None:
                    sample.tokens[start + offset].set_extra("logprob", lp)


def from_eval_log(
    eval_log: Any,
    tokenizer: Any = None,
    include_messages: bool = True,
    deduplicate: bool = True,
    min_sequence_length: int = 3,
    tag_chat_roles: bool = True,
    tag_generated: bool = True,
    extract_logprobs: bool = True,
) -> InifDocument:
    """Convert an Inspect AI EvalLog to an InifDocument.

    Args:
        eval_log: An inspect_ai.log.EvalLog object.
        tokenizer: A HuggingFace tokenizer, or a model identifier string
            (e.g. ``"openai/gpt-oss-20b"``) that will be loaded via
            ``AutoTokenizer.from_pretrained``. If ``None``, tokenization is
            skipped (messages still produce ``texts`` but no ``tokens``).
        include_messages: Whether to include message-level text segments.
        deduplicate: Whether to run sequence deduplication.
        min_sequence_length: Minimum length for common sequence detection.
        tag_chat_roles: Whether to add 'role' extra field to tokens.
        tag_generated: Whether to tag the model's response tokens with
            ``"generated"``. The response is the last assistant message.
        extract_logprobs: Whether to attach per-token logprobs from the eval
            output to the response tokens (best-effort: requires the tokenizer
            and the eval-source tokenization to agree on token count).
    """
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

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
    if tokenizer is not None and eval_log.samples:
        first_msgs: list[dict[str, str]] = []
        probe_sample = next(
            (s for s in eval_log.samples if getattr(s, "messages", None)),
            None,
        )
        if probe_sample is not None:
            first_msgs = _extract_message_dicts(probe_sample.messages)
        if first_msgs and _offset_mapping_decode(first_msgs, tokenizer) is not None:
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

            texts: list[str] = []
            sample_tokens: list[Token] = []
            target = None

            if hasattr(inspect_sample, "target") and inspect_sample.target:
                target = (
                    inspect_sample.target
                    if isinstance(inspect_sample.target, str)
                    else str(inspect_sample.target)
                )

            msg_dicts: list[dict[str, str]] = []
            if include_messages and hasattr(inspect_sample, "messages"):
                msg_dicts = _extract_message_dicts(inspect_sample.messages)
                texts, sample_tokens = _messages_to_tokens(
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

            # Annotate response tokens BEFORE dedup so the per-token tags and
            # logprobs become part of the extras that prevent collapsing.
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

    if deduplicate and tokenizer is not None:
        doc = deduplicate_sequences(doc, min_length=min_sequence_length)

    if tag_chat_roles and tokenizer is not None:
        from inif.tagging import tag_chat_roles_doc

        tag_chat_roles_doc(doc, all_msg_dicts, tokenizer)

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
