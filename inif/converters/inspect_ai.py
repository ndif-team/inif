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


# Back-compat aliases — existing test modules import these private names.
_offset_mapping_decode = offset_mapping_decode
_messages_to_tokens = messages_to_tokens


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
    min_sequence_length: int = 3,
    tag_chat_roles: bool = True,
    tag_generated: bool = True,
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
            first_msgs = _extract_message_dicts(probe_sample.messages)
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
                msg_dicts = _extract_message_dicts(inspect_sample.messages)
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
        doc = deduplicate_sequences(doc, min_length=min_sequence_length)

    if tag_chat_roles:
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
