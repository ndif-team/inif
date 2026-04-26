"""Shared helpers for converting chat-style message dicts into INIF Tokens.

Used by multiple eval-format converters (Inspect AI, evaleval). The logic here
is framework-agnostic: it consumes a list of ``{"role": ..., "content": ...}``
dicts and a HuggingFace tokenizer.
"""

from __future__ import annotations

import warnings
from typing import Any

from inif.converters._decode import (
    ByteSliceDecoder,
    decode_token_ids,
    offset_mapping_decode_text,
)
from inif.models import Sample, Token


def offset_mapping_decode(
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


def messages_to_tokens(
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
            result = offset_mapping_decode(messages, tokenizer)
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


def find_message_token_range(
    tokens: list[Token],
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
    role: str = "assistant",
    which: str = "last",
) -> tuple[int, int] | None:
    """Locate the [start, end) token indices covering a message's content.

    ``which`` selects between the first / last occurrence of the role in
    ``msg_dicts``. Returns ``None`` when the formatted string can't be
    reconstructed from the token sequence (e.g. a tokenizer that drops bytes),
    or when the content can't be located unambiguously.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    assert which in ("first", "last"), f"which must be 'first' or 'last', got {which!r}"
    iterator = reversed(msg_dicts) if which == "last" else iter(msg_dicts)
    content: str | None = None
    for msg in iterator:
        if msg.get("role") == role:
            c = msg.get("content")
            if c:
                content = c
                break
    if content is None:
        return None

    formatted = tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=False
    )
    decoded = [t.token or "" for t in tokens]
    if "".join(decoded) != formatted:
        return None

    char_start = (
        formatted.rfind(content) if which == "last" else formatted.find(content)
    )
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


def tag_char_span(
    sample: Sample,
    msg_dicts: list[dict[str, str]],
    tokenizer: Any,
    text: str,
    tag: str,
) -> bool:
    """Tag every token whose decoded piece overlaps ``text`` inside the
    tokenizer's ``apply_chat_template`` rendering of ``msg_dicts``.

    Returns True on success, False when the token stream doesn't round-trip to
    the formatted string or ``text`` isn't found.
    """
    if not text or not hasattr(tokenizer, "apply_chat_template"):
        return False
    formatted = tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=False
    )
    decoded = [t.token or "" for t in sample.tokens]
    if "".join(decoded) != formatted:
        return False
    char_start = formatted.find(text)
    if char_start < 0:
        return False
    char_end = char_start + len(text)
    pos = 0
    for i, s in enumerate(decoded):
        s_start, s_end = pos, pos + len(s)
        if s_end > char_start and s_start < char_end:
            sample.tokens[i].add_tag(tag)
        pos = s_end
    return True


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------

# Routing prefixes some eval frameworks prepend to model ids (e.g. Inspect AI's
# ``together/moonshotai/Kimi-K2.5``). Stripped before handing the id to
# ``AutoTokenizer.from_pretrained`` so the underlying HF id resolves cleanly.
# Keyed on a small allowlist to avoid chopping a legitimate org from a HF id
# (``meta-llama/Llama-3.1-8B`` must NOT lose its first component).
_KNOWN_PROVIDERS = frozenset(
    {
        "azureml",
        "anthropic",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks",
        "google",
        "groq",
        "hf",
        "huggingface",
        "mistral",
        "mockllm",
        "ollama",
        "openai",
        "perplexity",
        "replicate",
        "together",
        "vertex",
        "vllm",
    }
)


def _strip_provider_prefix(model_id: str) -> str:
    """Strip a routing prefix like ``together/`` from a model id.

    Only strips when the head matches the provider allowlist AND the tail
    still contains a ``/``, so HF-format ids like ``meta-llama/Llama-3.1-8B``
    are left untouched.
    """
    if "/" not in model_id:
        return model_id
    head, _, rest = model_id.partition("/")
    if head.lower() in _KNOWN_PROVIDERS and "/" in rest:
        return rest
    return model_id


def _ids_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return _strip_provider_prefix(a) == _strip_provider_prefix(b)


def _warn_tokenizer_mismatch(tokenizer: Any, source_model_id: str | None) -> None:
    if not source_model_id:
        return
    tokenizer_id = getattr(tokenizer, "name_or_path", None)
    if not tokenizer_id or _ids_match(tokenizer_id, source_model_id):
        return
    warnings.warn(
        f"Tokenizer mismatch: tokenizer ({tokenizer_id!r}) differs from the "
        f"source eval's model id ({source_model_id!r}). Token-level data "
        "will reflect the supplied tokenizer, not the original model. Pass "
        '``tokenizer="auto"`` to auto-load the matching tokenizer, or '
        "ignore this warning if the mismatch is intentional.",
        UserWarning,
        stacklevel=3,
    )


def resolve_tokenizer(tokenizer: Any, source_model_id: str | None) -> Any:
    """Resolve the tokenizer used for INIF conversion.

    Tokens are a load-bearing invariant of every INIF sample, so this never
    returns ``None``: callers always get back a usable tokenizer or a
    ``ValueError`` explaining why one couldn't be obtained.

    - ``tokenizer="auto"`` (the default for converter entry points) and
      ``tokenizer=None`` (treated as ``"auto"`` for ergonomics):
      ``AutoTokenizer.from_pretrained(source_model_id)`` with the routing
      prefix stripped (``together/`` etc.). Raises ``ValueError`` if
      ``source_model_id`` is missing or the load fails (closed-source ids
      like ``openai/gpt-4`` will hit this path).
    - ``tokenizer`` is a string other than ``"auto"``: load it via
      ``AutoTokenizer.from_pretrained``, then warn (but don't fail) if its
      id disagrees with ``source_model_id``.
    - ``tokenizer`` is a tokenizer instance: use as-is, warn (but don't fail)
      if its ``name_or_path`` disagrees with ``source_model_id``.
    """
    if tokenizer is None or (isinstance(tokenizer, str) and tokenizer == "auto"):
        if not source_model_id:
            raise ValueError(
                "Cannot auto-load a tokenizer: no model id is available in "
                "the source. Pass ``tokenizer=<id-or-instance>`` explicitly."
            )
        from transformers import AutoTokenizer

        canonical_id = _strip_provider_prefix(source_model_id)
        try:
            return AutoTokenizer.from_pretrained(canonical_id, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f"Could not auto-load a tokenizer for model id "
                f"{source_model_id!r} (canonical: {canonical_id!r}). This is "
                "common for closed-source models (OpenAI, Anthropic, …). "
                "Pass ``tokenizer=<hf-id-or-instance>`` with an HF tokenizer "
                "that approximates the source model — token-level data will "
                "then be approximate but tokens will still be produced. "
                f"Underlying error: {e}"
            ) from e

    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        _warn_tokenizer_mismatch(loaded, source_model_id)
        return loaded

    _warn_tokenizer_mismatch(tokenizer, source_model_id)
    return tokenizer
