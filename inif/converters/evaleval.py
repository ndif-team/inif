"""Convert `every_eval_ever <https://github.com/evaleval/every_eval_ever>`_
instance-level records into :class:`~inif.models.InifDocument`.

The EEE schema (``instance_level_eval_0.2.2``) is the interchange format used
by the `EEE_datastore <https://huggingface.co/datasets/evaleval/EEE_datastore>`_
HuggingFace dataset, which bundles traces from Inspect AI, HELM, and
lm-eval-harness under a shared schema. This module accepts that schema as
the source of truth and produces INIF documents whose tokenization, role
tags, and deduplication match the Inspect converter's conventions.

Three entry points:

- :func:`from_instance_records` — raw list of EEE-schema dicts (the core path).
- :func:`from_eval_json` — two local files on disk (aggregate + instances).
- :func:`from_hf_dataset` — pulls instance records straight from
  ``evaleval/EEE_datastore`` via HuggingFace ``datasets``.
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from inif.converters._decode import ByteSliceDecoder, has_byte_level_decoder
from inif.converters._tokenize import (
    find_message_token_range,
    messages_to_tokens,
    offset_mapping_decode,
    resolve_tokenizer,
    tag_char_span,
)
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    SourceEval,
    Token,
)
from inif.sequences import deduplicate_sequences

FRAMEWORK_NAME = "evaleval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_package_versions(extra_pkgs: Iterable[str] = ()) -> dict[str, str]:
    packages: dict[str, str] = {}
    for pkg in ["inif", "transformers", "datasets", *extra_pkgs]:
        try:
            packages[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


def _messages_from_record(record: dict) -> list[dict[str, str]]:
    """Build a chat-template-compatible message list for a single record.

    Handles all three interaction types:

    - ``single_turn``: one ``user`` + one ``assistant`` turn; the user content
      is ``input.raw`` and the assistant content is
      ``(reasoning_trace or "") + output.raw[0]``.
    - ``multi_turn`` / ``agentic``: uses ``messages[]`` directly, ordered by
      ``turn_idx``. Per-turn ``reasoning_trace`` is prepended to the turn's
      content. ``tool_calls`` are rendered as a compact ``<tool_call .../>``
      suffix so tool invocations survive into the tokenized stream.

    No ``system`` message is synthesised here — if one exists in
    ``messages[]`` it is preserved, but single-turn records have no slot for a
    system prompt in the EEE schema.
    """
    interaction = record.get("interaction_type", "single_turn")
    if interaction == "single_turn":
        input_obj = record.get("input") or {}
        output_obj = record.get("output") or {}
        user_content = input_obj.get("raw", "") or ""
        assistant_parts: list[str] = []
        reasoning = output_obj.get("reasoning_trace") or []
        if reasoning:
            assistant_parts.append("".join(reasoning))
        raw_outputs = output_obj.get("raw") or []
        if raw_outputs:
            assistant_parts.append(raw_outputs[0])
        assistant_content = "".join(assistant_parts)
        messages = [{"role": "user", "content": user_content}]
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    # multi_turn / agentic
    raw_msgs = sorted(
        (record.get("messages") or []),
        key=lambda m: m.get("turn_idx", 0),
    )
    out: list[dict[str, str]] = []
    for msg in raw_msgs:
        parts: list[str] = []
        reasoning = msg.get("reasoning_trace")
        if reasoning:
            parts.append(reasoning)
        content = msg.get("content") or ""
        parts.append(content)
        for call in msg.get("tool_calls") or []:
            name = call.get("name", "")
            args = call.get("arguments") or {}
            parts.append(f"<tool_call name={name} args={json.dumps(args)}/>")
        out.append({"role": msg.get("role", "user"), "content": "".join(parts)})
    return out


def _terminal_attribution(record: dict) -> dict | None:
    """Return the ``answer_attribution`` entry with ``is_terminal=True``, if any."""
    for att in record.get("answer_attribution") or []:
        if att.get("is_terminal"):
            return att
    return None


def _build_score(record: dict) -> SampleScore | None:
    evaluation = record.get("evaluation")
    if not evaluation:
        return None
    terminal = _terminal_attribution(record)
    metadata: dict[str, Any] = {}
    if "is_correct" in evaluation:
        metadata["is_correct"] = evaluation["is_correct"]
    if terminal is not None:
        metadata["extraction_method"] = terminal.get("extraction_method")
        metadata["source"] = terminal.get("source")
    return SampleScore(
        scorer=record.get("evaluation_name", FRAMEWORK_NAME),
        value=evaluation.get("score", 0.0),
        answer=(terminal or {}).get("extracted_value"),
        metadata=metadata,
    )


def _build_sample_metadata(record: dict) -> dict[str, Any]:
    """Build the ``Sample.metadata`` dict for everything that isn't promoted to
    a first-class field (see :class:`inif.models.Sample` for the promoted set:
    ``interaction_type``, ``error``, ``sample_hash``, ``choices``,
    ``references``).
    """
    md: dict[str, Any] = {}

    input_obj = record.get("input") or {}
    if input_obj.get("formatted") is not None:
        md["formatted_input"] = input_obj["formatted"]

    evaluation = record.get("evaluation") or {}
    for k in ("num_turns", "tool_calls_count"):
        if evaluation.get(k) is not None:
            md[k] = evaluation[k]

    token_usage = record.get("token_usage") or {}
    reasoning_tokens = token_usage.get("reasoning_tokens")
    if reasoning_tokens is not None:
        md["reasoning_tokens"] = reasoning_tokens

    performance = record.get("performance")
    if performance:
        md["performance"] = dict(performance)

    eval_md = record.get("metadata")
    if eval_md:
        md["eval_metadata"] = dict(eval_md)

    non_terminal = [
        a for a in (record.get("answer_attribution") or []) if not a.get("is_terminal")
    ]
    if non_terminal:
        md["intermediate_answers"] = non_terminal

    return md


def _reasoning_text_for_record(record: dict) -> str | None:
    """Concatenated reasoning text, if any, for char-span tagging.

    Mirrors the concatenation done in :func:`_messages_from_record` so the
    substring lookup inside ``apply_chat_template`` actually finds it.
    """
    interaction = record.get("interaction_type", "single_turn")
    if interaction == "single_turn":
        trace = (record.get("output") or {}).get("reasoning_trace") or []
        return "".join(trace) or None
    pieces: list[str] = []
    for msg in sorted(
        (record.get("messages") or []), key=lambda m: m.get("turn_idx", 0)
    ):
        r = msg.get("reasoning_trace")
        if r:
            pieces.append(r)
    return "\n".join(pieces) if pieces else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_instance_records(
    records: list[dict],
    aggregate: dict | None = None,
    *,
    tokenizer: Any = "auto",
    include_messages: bool = True,
    deduplicate: bool = True,
    min_sequence_length: int = 3,
    tag_chat_roles: bool = True,
    tag_generated: bool = True,
    tag_reasoning: bool = True,
) -> InifDocument:
    """Convert a list of EEE instance records into an :class:`InifDocument`.

    Args:
        records: Instance-level records conforming to ``instance_level_eval_0.2.2``.
        aggregate: Optional aggregate record (``eval.schema.json``). Contributes
            model developer info, eval library version, and metric config to
            document-level metadata.
        tokenizer: One of:

            - ``"auto"`` (default) or ``None`` — auto-load
              ``AutoTokenizer.from_pretrained`` based on each record's
              ``model_id`` (with aggregate fallback). Raises if the id is
              missing or the tokenizer cannot be loaded (closed-source ids
              like ``openai/gpt-4`` will hit this).
            - a model id string — loaded via ``AutoTokenizer.from_pretrained``;
              warns if the resolved id disagrees with the source records'
              ``model_id``.
            - a tokenizer instance — used as-is; warns if its
              ``name_or_path`` disagrees with the source records' ``model_id``.

            Tokens are a load-bearing invariant of every INIF sample, so
            there is no way to opt out of tokenization — every option here
            yields a usable tokenizer or raises.

        include_messages: Include message texts on each ``Sample``.
        deduplicate: Run sequence deduplication over the produced samples.
        min_sequence_length: Minimum length for common-sequence detection.
        tag_chat_roles: Add ``role`` extras to tokens via character-span matching.
        tag_generated: Tag the last assistant message's tokens with ``"generated"``.
        tag_reasoning: Tag tokens overlapping the reasoning-trace span with
            ``"reasoning"``. Best-effort: silently skipped when the token stream
            doesn't round-trip cleanly to the formatted chat template.

    Returns:
        An :class:`InifDocument` with one :class:`Sample` per record.
    """
    model_name = _resolve_model_name(records, aggregate)
    tokenizer = resolve_tokenizer(
        tokenizer, model_name if model_name != "unknown" else None
    )
    model_info = ModelInfo(
        name=model_name,
        huggingface_id=model_name if "/" in model_name else None,
        revision=getattr(tokenizer, "_commit_hash", None) if tokenizer else None,
    )
    source_eval = _build_source_eval(records, aggregate)

    # Pick a decode strategy once per archive, reused across every sample —
    # same three-tier approach as the Inspect converter.
    byte_decoder: ByteSliceDecoder | None = None
    decode_cache: dict[int, str] | None = None
    use_offset_mapping = False
    if records:
        probe_msgs = _messages_from_record(records[0])
        if probe_msgs and offset_mapping_decode(probe_msgs, tokenizer) is not None:
            use_offset_mapping = True
        elif has_byte_level_decoder(tokenizer):
            byte_decoder = ByteSliceDecoder(tokenizer)
        else:
            decode_cache = {}

    samples: list[Sample] = []
    all_msg_dicts: list[list[dict[str, str]]] = []
    reasoning_texts: list[str | None] = []

    for record in records:
        msg_dicts = _messages_from_record(record) if include_messages else []
        texts: list[str] = []
        sample_tokens: list[Token] = []
        if include_messages:
            texts, sample_tokens = messages_to_tokens(
                msg_dicts,
                tokenizer,
                decode_cache=decode_cache,
                byte_decoder=byte_decoder,
                use_offset_mapping=use_offset_mapping,
            )

        score = _build_score(record)
        scores = [score] if score is not None else []

        input_obj = record.get("input") or {}
        refs = list(input_obj.get("reference") or [])
        target = refs[0] if len(refs) == 1 else None
        choices = list(input_obj["choices"]) if input_obj.get("choices") else None

        token_usage = record.get("token_usage") or {}
        sample = Sample(
            id=str(record.get("sample_id", "")),
            texts=texts,
            tokens=sample_tokens,
            scores=scores,
            target=target,
            references=refs,
            choices=choices,
            interaction_type=record.get("interaction_type"),
            error=record.get("error"),
            sample_hash=record.get("sample_hash"),
            input_tokens=token_usage.get("input_tokens"),
            output_tokens=token_usage.get("output_tokens"),
            metadata=_build_sample_metadata(record),
        )

        if tag_generated:
            rng = find_message_token_range(
                sample.tokens, msg_dicts, tokenizer, role="assistant", which="last"
            )
            if rng is not None:
                for i in range(rng[0], rng[1]):
                    sample.tokens[i].add_tag("generated")

        reasoning = _reasoning_text_for_record(record) if tag_reasoning else None
        if reasoning:
            tag_char_span(sample, msg_dicts, tokenizer, reasoning, "reasoning")
        reasoning_texts.append(reasoning)

        all_msg_dicts.append(msg_dicts)
        samples.append(sample)

    metadata = Metadata(
        model=model_info,
        packages=_get_package_versions(),
        source_eval=source_eval,
        created_at=datetime.now(timezone.utc),
        extra=_build_doc_extra(records, aggregate),
    )
    doc = InifDocument(metadata=metadata, samples=samples)

    if deduplicate:
        doc = deduplicate_sequences(doc, min_length=min_sequence_length)

    if tag_chat_roles:
        from inif.tagging import tag_chat_roles_doc

        tag_chat_roles_doc(doc, all_msg_dicts, tokenizer)

    return doc


def from_eval_json(
    aggregate_path: str | Path | None,
    instances_path: str | Path,
    **kwargs: Any,
) -> InifDocument:
    """Load EEE JSON files from disk and convert.

    ``aggregate_path`` is optional; ``instances_path`` can be either a single
    JSON file containing a list of records, or a ``.jsonl`` file with one
    record per line.
    """
    aggregate: dict | None = None
    if aggregate_path is not None:
        aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))

    records = _load_instances(Path(instances_path))
    doc = from_instance_records(records, aggregate=aggregate, **kwargs)
    doc.metadata.sources.append(str(instances_path))
    if aggregate_path is not None:
        doc.metadata.sources.append(str(aggregate_path))
    return doc


def from_hf_dataset(
    config: str,
    split: str = "samples",
    *,
    aggregate_config: str | None = None,
    aggregate_split: str = "train",
    repo: str = "evaleval/EEE_datastore",
    limit: int | None = None,
    revision: str | None = None,
    **kwargs: Any,
) -> InifDocument:
    """Convert an EEE config from the HuggingFace datastore.

    Args:
        config: A config name from ``evaleval/EEE_datastore``, typically one
            ending in ``_samples`` (e.g. ``"theory_of_mind_samples"``).
        split: Dataset split for instance records (usually ``"samples"``).
        aggregate_config: Optional paired aggregate config (without ``_samples``
            suffix). When provided, the first row is used as the aggregate.
        aggregate_split: Split name for the aggregate config (defaults to
            ``"train"``).
        repo: HuggingFace repo id (override for testing).
        limit: Maximum number of records to convert (``None`` = all).
        revision: HuggingFace dataset revision / commit.
        **kwargs: Forwarded to :func:`from_instance_records`.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split, revision=revision, streaming=True)
    records: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        records.append(dict(row))

    aggregate: dict | None = None
    if aggregate_config is not None:
        agg_ds = load_dataset(
            repo, aggregate_config, split=aggregate_split, revision=revision
        )
        if len(agg_ds) > 0:
            aggregate = dict(agg_ds[0])

    doc = from_instance_records(records, aggregate=aggregate, **kwargs)
    doc.metadata.sources.append(f"hf://{repo}/{config}[{split}]")
    return doc


# ---------------------------------------------------------------------------
# Internal: record/aggregate → metadata
# ---------------------------------------------------------------------------


def _resolve_model_name(records: list[dict], aggregate: dict | None) -> str:
    if aggregate:
        info = aggregate.get("model_info") or {}
        hf_id = info.get("id") or info.get("name")
        if hf_id:
            return hf_id
    for rec in records:
        if rec.get("model_id"):
            return rec["model_id"]
    return "unknown"


def _build_source_eval(records: list[dict], aggregate: dict | None) -> SourceEval:
    first = records[0] if records else {}
    schema_version = first.get("schema_version")
    task = first.get("evaluation_name")
    eval_id = first.get("evaluation_id")
    run_id = first.get("evaluation_result_id")
    extra: dict[str, Any] = {}
    if aggregate:
        extra["retrieved_timestamp"] = aggregate.get("retrieved_timestamp")
        src_md = aggregate.get("source_metadata")
        if src_md:
            extra["source_metadata"] = src_md
    return SourceEval(
        framework=FRAMEWORK_NAME,
        framework_version=schema_version,
        task=task,
        eval_id=eval_id,
        run_id=run_id,
        extra={k: v for k, v in extra.items() if v is not None},
    )


def _build_doc_extra(records: list[dict], aggregate: dict | None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if aggregate is None:
        return extra

    info = aggregate.get("model_info") or {}
    inference = {
        k: info[k]
        for k in ("developer", "inference_platform", "inference_engine")
        if info.get(k)
    }
    if inference:
        extra["inference"] = inference

    eval_library = aggregate.get("eval_library") or {}
    if eval_library.get("name"):
        extra["eval_library"] = dict(eval_library)

    results = aggregate.get("evaluation_results") or []
    if results:
        # The aggregate may cover multiple metrics; pick the one whose
        # ``evaluation_result_id`` matches this batch when possible, else
        # fall back to the first entry.
        first_record_run_id = (
            records[0].get("evaluation_result_id") if records else None
        )
        match = next(
            (
                r
                for r in results
                if r.get("evaluation_result_id") == first_record_run_id
            ),
            results[0],
        )
        if match.get("metric_config"):
            extra["metric_config"] = dict(match["metric_config"])
        if match.get("score_details"):
            extra["aggregate_score"] = dict(match["score_details"])
    return extra


def _load_instances(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Some EEE exports wrap records as {"instances": [...]}; accept both.
        if "instances" in data and isinstance(data["instances"], list):
            return list(data["instances"])
        return [data]
    assert isinstance(data, list), (
        f"Expected list or wrapped dict of EEE records, got {type(data).__name__}"
    )
    return list(data)


__all__ = [
    "from_instance_records",
    "from_eval_json",
    "from_hf_dataset",
]
