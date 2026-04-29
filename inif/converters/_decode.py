from __future__ import annotations

from typing import Any


def decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])


def decode_many(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def has_byte_level_decoder(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` exposes byte-level BPE internals."""
    return hasattr(tokenizer, "encoder") and hasattr(tokenizer, "byte_decoder")


class ByteSliceDecoder:
    """Per-token string decoder for byte-level BPE tokenizers.

    Multi-byte UTF-8 characters can be split across tokens, making isolated
    ``tokenizer.decode([tid])`` lossy. This decoder reconstructs raw bytes,
    decodes the whole byte stream once, then attributes each character to the
    token whose byte range contains that character's final byte.
    """

    __slots__ = ("_tokenizer", "_id_to_piece", "_byte_decoder", "_id_to_bytes")

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._id_to_piece: dict[int, str] = {v: k for k, v in tokenizer.encoder.items()}
        self._byte_decoder: dict[str, int] = tokenizer.byte_decoder
        self._id_to_bytes: dict[int, bytes] = {}

    def _bytes_for(self, tid: int) -> bytes:
        b = self._id_to_bytes.get(tid)
        if b is not None:
            return b

        piece = self._id_to_piece.get(tid)
        if piece is None:
            # Added special tokens often live outside the base byte-level
            # encoder. Their isolated decode is the only reliable literal.
            b = decode_one(self._tokenizer, tid).encode("utf-8")
        else:
            try:
                b = bytes(self._byte_decoder[c] for c in piece)
            except KeyError:
                # Special token whose piece chars are not byte-decoder keys.
                b = piece.encode("utf-8")
        self._id_to_bytes[tid] = b
        return b

    def decode(self, token_ids: list[int]) -> list[str]:
        per_token_bytes = [self._bytes_for(tid) for tid in token_ids]
        full_str = b"".join(per_token_bytes).decode("utf-8", errors="replace")

        char_end_byte: list[int] = []
        byte_pos = 0
        for ch in full_str:
            byte_pos += len(ch.encode("utf-8"))
            char_end_byte.append(byte_pos)

        pieces: list[str] = []
        char_idx = 0
        token_end = 0
        n_chars = len(full_str)
        for token_bytes in per_token_bytes:
            token_end += len(token_bytes)
            buf: list[str] = []
            while char_idx < n_chars and char_end_byte[char_idx] <= token_end:
                buf.append(full_str[char_idx])
                char_idx += 1
            pieces.append("".join(buf))
        return pieces


def offset_mapping_decode_text(
    text: str,
    expected_ids: list[int],
    tokenizer: Any,
) -> list[str] | None:
    """Decode token strings by slicing ``text`` with tokenizer offsets.

    Multi-byte UTF-8 characters that get split across byte-level BPE tokens
    (e.g. ``♠`` U+2660 → 3 bytes → 2 tokens on Qwen3.5) report the SAME
    ``(start, end)`` offset on every fragment, since char-level slicing can't
    represent partial chars. The naive ``text[start:end]`` would emit the
    char once per fragment, so the joined per-token strings would be longer
    than ``text``. We track ``last_consumed`` and only emit the chars that
    haven't already been claimed by an earlier token; subsequent fragments of
    the same char yield ``""``. The final ``"".join(pieces) == text`` check
    is the load-bearing invariant — it catches any remaining mismatch (gaps,
    tokenizer normalization that removes/adds chars) and signals the caller
    to fall back to byte-slice decode.
    """
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, KeyError, ValueError):
        return None

    if "offset_mapping" not in enc:
        return None
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if list(ids) != list(expected_ids):
        return None

    pieces: list[str] = []
    last_consumed = 0
    for start, end in offsets:
        actual_start = max(start, last_consumed)
        if actual_start < end:
            pieces.append(text[actual_start:end])
            last_consumed = end
        else:
            pieces.append("")
    if "".join(pieces) != text:
        return None
    return pieces


def decode_token_ids(
    token_ids: list[int],
    tokenizer: Any,
    decode_cache: dict[int, str] | None = None,
    byte_decoder: ByteSliceDecoder | None = None,
    *,
    strict_roundtrip: bool = False,
) -> list[str]:
    """Decode token ids into per-token strings.

    ``strict_roundtrip`` raises when the per-token strings do not concatenate
    to the full tokenizer decode. In non-strict mode we only raise for the
    common dangerous corruption case: isolated decodes introduce U+FFFD while
    the full decode does not.
    """
    if byte_decoder is not None:
        return byte_decoder.decode(token_ids)

    cache = decode_cache if decode_cache is not None else {}
    pieces: list[str] = []
    for tid in token_ids:
        piece = cache.get(tid)
        if piece is None:
            piece = decode_one(tokenizer, tid)
            cache[tid] = piece
        pieces.append(piece)

    full_decode = decode_many(tokenizer, token_ids)
    joined = "".join(pieces)
    replacement_corruption = "\ufffd" in joined and "\ufffd" not in full_decode
    if joined != full_decode and (strict_roundtrip or replacement_corruption):
        raise ValueError(
            "Per-token decode does not round-trip to the full decode. "
            "Use a tokenizer with return_offsets_mapping or byte-level BPE "
            "internals (encoder + byte_decoder)."
        )
    return pieces
