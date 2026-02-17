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


def _load_tokenizer(tokenizer: Any = None, tokenizer_name: str | None = None) -> Any:
    if tokenizer is not None:
        return tokenizer
    assert tokenizer_name is not None, "Either tokenizer or tokenizer_name required"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _tokenize_text(text: str, tokenizer: Any) -> tuple[list[int], list[str]]:
    """Tokenize text and return (token_ids, token_strings)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    strings = [tokenizer.decode([tid]) for tid in ids]
    return ids, strings


def _model_info_from_tokenizer(
    tokenizer: Any, tokenizer_name: str | None = None
) -> ModelInfo:
    name = tokenizer_name or getattr(tokenizer, "name_or_path", "unknown")
    revision = getattr(tokenizer, "_commit_hash", None)
    return ModelInfo(name=name, revision=revision)


def from_texts(
    texts: list[str],
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
    sample_ids: list[str] | None = None,
    min_sequence_length: int = 3,
    deduplicate: bool = True,
    messages: list[list[dict[str, str]]] | None = None,
    tag_chat_roles: bool = False,
) -> InifDocument:
    """Each text string becomes one Sample.

    Args:
        texts: List of text strings to process.
        tokenizer: A HuggingFace tokenizer.
        tokenizer_name: HuggingFace model name to load tokenizer from.
        sample_ids: Optional list of sample IDs (defaults to "sample_0", ...).
        min_sequence_length: Minimum length for common sequence detection.
        deduplicate: Whether to run sequence deduplication.
        messages: Optional list of message lists per sample. When provided and
            the tokenizer supports ``apply_chat_template``, uses it instead of
            plain ``tokenizer.encode``. ``texts[i]`` is still used for
            ``Sample.texts``.
        tag_chat_roles: Whether to tag tokens with chat roles after dedup.
            Requires ``messages`` to be provided; no-op otherwise.
    """
    tok = _load_tokenizer(tokenizer, tokenizer_name)
    model_info = _model_info_from_tokenizer(tok, tokenizer_name)

    samples = []
    for i, text in enumerate(texts):
        sid = sample_ids[i] if sample_ids else f"sample_{i}"
        if messages is not None and hasattr(tok, "apply_chat_template"):
            token_ids = tok.apply_chat_template(
                messages[i],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            tokens = [
                Token(
                    id=tid,
                    token=tok.decode([tid], skip_special_tokens=False),
                )
                for tid in token_ids
            ]
        else:
            ids, strings = _tokenize_text(text, tok)
            tokens = [Token(id=tid, token=tstr) for tid, tstr in zip(ids, strings)]
        samples.append(Sample(id=sid, tokens=tokens, texts=[text]))

    metadata = Metadata(
        model=model_info,
        packages=_get_package_versions(),
        created_at=datetime.now(timezone.utc).isoformat(),
        total_samples=len(samples),
    )

    doc = InifDocument(metadata=metadata, samples=samples)

    if deduplicate:
        doc = deduplicate_sequences(doc, min_length=min_sequence_length)

    if tag_chat_roles and messages is not None:
        from inif.tagging import tag_chat_roles_doc

        tag_chat_roles_doc(doc, messages, tok)

    return doc


def from_text_files(
    paths: list[str | Path],
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
    min_sequence_length: int = 3,
    deduplicate: bool = True,
) -> InifDocument:
    """Each file becomes one Sample. Finds common sequences across samples.

    Args:
        paths: List of file paths to process.
        tokenizer: A HuggingFace tokenizer.
        tokenizer_name: HuggingFace model name to load tokenizer from.
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
        tokenizer_name=tokenizer_name,
        sample_ids=sample_ids,
        min_sequence_length=min_sequence_length,
        deduplicate=deduplicate,
    )
    doc.metadata.sources = [str(p) for p in paths]
    return doc
