from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    Token,
)
from inif.sequences import deduplicate_sequences


def _get_package_versions() -> dict[str, str]:
    packages = {}
    for pkg in ["inif", "transformers"]:
        try:
            packages[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


def _load_tokenizer(tokenizer: Any) -> Any:
    assert tokenizer is not None, "tokenizer is required"
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def _tokenize_text(text: str, tokenizer: Any) -> tuple[list[int], list[str]]:
    """Tokenize text and return (token_ids, token_strings)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    strings = [tokenizer.decode([tid]) for tid in ids]
    return ids, strings


def _model_info_from_tokenizer(tokenizer: Any) -> ModelInfo:
    name = getattr(tokenizer, "name_or_path", "unknown")
    revision = getattr(tokenizer, "_commit_hash", None)
    return ModelInfo(name=name, revision=revision)


def _is_chat_input(item: Any) -> bool:
    """A chat input is a list of ``{"role", "content"}`` dicts."""
    return isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict)


def from_texts(
    texts: list[str] | list[list[dict[str, str]]],
    tokenizer: Any,
    sample_ids: list[str] | None = None,
    min_sequence_length: int = 3,
    deduplicate: bool = True,
    tag_chat_roles: bool = False,
) -> InifDocument:
    """Converts one or more inputs into ``Sample`` objects in an ``InifDocument``.

    Args:
        texts: Either a list of plain strings, or a list of chat message lists
            (``list[dict[str, str]]`` with ``"role"``/``"content"`` keys) —
            detected per-element. Chat inputs are tokenized via
            ``tokenizer.apply_chat_template``; plain strings via
            ``tokenizer.encode``.
        tokenizer: A HuggingFace tokenizer, or a model identifier string
            (e.g. ``"openai/gpt-oss-20b"``) that will be loaded via
            ``AutoTokenizer.from_pretrained``.
        sample_ids: Optional list of sample IDs (defaults to "sample_0", ...).
        min_sequence_length: Minimum length for common sequence detection.
        deduplicate: Whether to run sequence deduplication. Default: True.
        tag_chat_roles: Whether to tag chat-input tokens with roles after dedup.
            No-op for plain-string inputs.
    """
    tok = _load_tokenizer(tokenizer)
    model_info = _model_info_from_tokenizer(tok)

    samples = []
    chat_messages: list[list[dict[str, str]] | None] = []
    for i, item in enumerate(texts):
        sid = sample_ids[i] if sample_ids else f"sample_{i}"
        if _is_chat_input(item):
            assert hasattr(tok, "apply_chat_template"), (
                "tokenizer must support apply_chat_template for chat inputs"
            )
            token_ids = tok.apply_chat_template(
                item,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            tokens = [
                Token(id=tid, token=tok.decode([tid], skip_special_tokens=False))
                for tid in token_ids
            ]
            sample_texts = [msg["content"] for msg in item]
            chat_messages.append(item)
        else:
            ids, strings = _tokenize_text(item, tok)
            tokens = [Token(id=tid, token=tstr) for tid, tstr in zip(ids, strings)]
            sample_texts = [item]
            chat_messages.append(None)
        samples.append(Sample(id=sid, tokens=tokens, texts=sample_texts))

    metadata = Metadata(
        model=model_info,
        packages=_get_package_versions(),
        created_at=datetime.now(timezone.utc),
    )

    doc = InifDocument(metadata=metadata, samples=samples)

    if deduplicate:
        doc = deduplicate_sequences(doc, min_length=min_sequence_length)

    if tag_chat_roles and any(m is not None for m in chat_messages):
        from inif.tagging import tag_chat_roles as _tag_chat_roles

        for sample, msgs in zip(doc.samples, chat_messages):
            if msgs is not None:
                _tag_chat_roles(sample, msgs, tok, doc.sequences or None)

    return doc


def from_text_files(
    paths: list[str | Path],
    tokenizer: Any,
    min_sequence_length: int = 3,
    deduplicate: bool = True,
) -> InifDocument:
    """Each file becomes one Sample. Finds common sequences across samples.

    Args:
        paths: List of file paths to process.
        tokenizer: A HuggingFace tokenizer, or a model identifier string
            (e.g. ``"openai/gpt-oss-20b"``) that will be loaded via
            ``AutoTokenizer.from_pretrained``.
        min_sequence_length: Minimum length for common sequence detection.
        deduplicate: Whether to run sequence deduplication.
    """
    texts = []
    sample_ids = []
    for p in paths:
        p = Path(p)
        texts.append(p.read_text(encoding="utf-8"))
        sample_ids.append(p.name)

    doc = from_texts(
        texts,
        tokenizer=tokenizer,
        sample_ids=sample_ids,
        min_sequence_length=min_sequence_length,
        deduplicate=deduplicate,
    )
    doc.metadata.sources = [str(p) for p in paths]
    return doc
