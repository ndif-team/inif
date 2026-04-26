from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from inif.models import InifDocument, Metadata, Sample, SampleScore, Sequence

_FORMAT = "inif-indexed"
_VERSION = 1
_MANIFEST = "manifest.json"
_METADATA = "metadata.json"
_SEQUENCES = "sequences.json"
_SAMPLES_DIR = "samples"
_JOURNAL_START = "_journal/start.json"
_JOURNAL_SUMMARIES_DIR = "_journal/summaries"
_DEFAULT_TEXT_PREVIEW_CHARS = 1000


def _dump_model(model: Any, compact: bool) -> dict:
    kwargs: dict[str, Any] = {"mode": "json", "by_alias": True}
    if compact:
        kwargs["exclude_none"] = True
        kwargs["exclude_defaults"] = True
    return model.model_dump(**kwargs)


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _read_json(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name, "r") as f:
        return json.loads(f.read().decode("utf-8"))


def _score_summary(score: SampleScore) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scorer": score.scorer,
        "value": score.value,
    }
    if score.answer is not None:
        summary["answer"] = score.answer
    return summary


def _preview_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sample_summary(sample: Sample, max_preview_chars: int) -> dict[str, Any]:
    text_preview = "\n".join(sample.texts)
    summary: dict[str, Any] = {
        "id": sample.id,
        "n_tokens": len(sample.tokens),
        "n_texts": len(sample.texts),
        "n_spans": len(sample.spans),
        "scores": [_score_summary(score) for score in sample.scores],
        "text_preview": _preview_text(text_preview, max_preview_chars),
        "texts_preview": [
            _preview_text(text, max_preview_chars) for text in sample.texts
        ],
    }
    if sample.target is not None:
        summary["target"] = sample.target
    if sample.error is not None:
        summary["error"] = sample.error
    if sample.input_tokens is not None:
        summary["input_tokens"] = sample.input_tokens
    if sample.output_tokens is not None:
        summary["output_tokens"] = sample.output_tokens
    return summary


def _load_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    names = set(zf.namelist())
    if _MANIFEST in names:
        manifest = _read_json(zf, _MANIFEST)
    else:
        manifest = _read_json(zf, _JOURNAL_START)
        summary_paths = sorted(
            name
            for name in names
            if name.startswith(f"{_JOURNAL_SUMMARIES_DIR}/") and name.endswith(".json")
        )
        manifest["samples"] = [_read_json(zf, path) for path in summary_paths]
        manifest["total_samples"] = len(manifest["samples"])
    assert manifest["format"] == _FORMAT, "Not an indexed INIF archive"
    assert manifest["version"] == _VERSION, (
        f"Unsupported indexed INIF version {manifest['version']}"
    )
    return manifest


def _sample_entries_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["samples"]:
        sample_id = str(entry["id"])
        assert sample_id not in entries, f"Sample id {sample_id!r} is not unique"
        entries[sample_id] = entry
    return entries


def _read_sample_entry(zf: zipfile.ZipFile, entry: dict[str, Any]) -> Sample:
    return Sample.model_validate(_read_json(zf, entry["path"]))


def _referenced_sequence_ids(samples: Iterable[Sample]) -> set[str]:
    referenced: set[str] = set()
    for sample in samples:
        for token in sample.tokens:
            if token.is_sequence_ref and token.sequence_id is not None:
                referenced.add(token.sequence_id)
    return referenced


def _prune_sequences(
    sequences: list[Sequence], samples: list[Sample]
) -> list[Sequence]:
    referenced = _referenced_sequence_ids(samples)
    return [seq for seq in sequences if seq.id in referenced]


class IndexedInifWriter:
    """Incremental writer for the single-file indexed INIF archive format."""

    def __init__(
        self,
        path: str | Path,
        metadata: Metadata,
        sequences: list[Sequence] | None = None,
        *,
        compact: bool = True,
        compression: int = zipfile.ZIP_DEFLATED,
        max_preview_chars: int = _DEFAULT_TEXT_PREVIEW_CHARS,
    ) -> None:
        self.path = Path(path)
        self.compact = compact
        self.compression = compression
        self.max_preview_chars = max_preview_chars
        self._zip = zipfile.ZipFile(self.path, "w")
        self._closed = False
        self._samples: list[dict[str, Any]] = []

        self._zip.writestr(
            _JOURNAL_START,
            _json_bytes(
                {
                    "format": _FORMAT,
                    "version": _VERSION,
                    "metadata_path": _METADATA,
                    "sequences_path": _SEQUENCES,
                }
            ),
            compress_type=zipfile.ZIP_STORED,
        )
        self._zip.writestr(
            _METADATA,
            _json_bytes(_dump_model(metadata, compact)),
            compress_type=zipfile.ZIP_STORED,
        )
        self._zip.writestr(
            _SEQUENCES,
            _json_bytes([_dump_model(seq, compact) for seq in (sequences or [])]),
            compress_type=compression,
        )

    def write_sample(self, sample: Sample) -> None:
        assert not self._closed, "Cannot write to a closed indexed INIF archive"
        idx = len(self._samples)
        sample_path = f"{_SAMPLES_DIR}/{idx:08d}.json"
        self._zip.writestr(
            sample_path,
            _json_bytes(_dump_model(sample, self.compact)),
            compress_type=self.compression,
        )

        entry = _sample_summary(sample, self.max_preview_chars)
        entry["path"] = sample_path
        entry["index"] = idx
        self._samples.append(entry)
        self._zip.writestr(
            f"{_JOURNAL_SUMMARIES_DIR}/{idx:08d}.json",
            _json_bytes(entry),
            compress_type=zipfile.ZIP_STORED,
        )

    def flush(self) -> None:
        """Flush the archive so already-written samples are readable."""
        assert not self._closed, "Cannot flush a closed indexed INIF archive"
        self._zip.close()
        self._zip = zipfile.ZipFile(self.path, "a")

    def close(self) -> None:
        if self._closed:
            return
        manifest = {
            "format": _FORMAT,
            "version": _VERSION,
            "metadata_path": _METADATA,
            "sequences_path": _SEQUENCES,
            "total_samples": len(self._samples),
            "samples": self._samples,
        }
        self._zip.writestr(
            _MANIFEST,
            _json_bytes(manifest),
            compress_type=zipfile.ZIP_STORED,
        )
        self._zip.close()
        self._closed = True

    def __enter__(self) -> IndexedInifWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


def save_indexed(
    doc: InifDocument,
    path: str | Path,
    *,
    compact: bool = True,
    compression: int = zipfile.ZIP_DEFLATED,
    max_preview_chars: int = _DEFAULT_TEXT_PREVIEW_CHARS,
) -> None:
    """Save ``doc`` as a single indexed archive.

    The archive stores metadata and sequences once, then writes each sample as a
    separate compressed member. Readers can load the header, summaries, one
    sample, or all samples without inflating the entire document up front.
    """
    with IndexedInifWriter(
        path,
        doc.metadata,
        doc.sequences,
        compact=compact,
        compression=compression,
        max_preview_chars=max_preview_chars,
    ) as writer:
        for sample in doc.samples:
            writer.write_sample(sample)


def read_indexed_header(path: str | Path) -> InifDocument:
    """Read only metadata and sequences from an indexed INIF archive."""
    with zipfile.ZipFile(path, "r") as zf:
        _load_manifest(zf)
        metadata = Metadata.model_validate(_read_json(zf, _METADATA))
        sequences = [Sequence.model_validate(seq) for seq in _read_json(zf, _SEQUENCES)]
    return InifDocument(metadata=metadata, sequences=sequences)


def read_indexed_sample_summaries(path: str | Path) -> list[dict[str, Any]]:
    """Return compact sample summaries from the archive manifest."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest = _load_manifest(zf)
        return [
            {k: v for k, v in entry.items() if k != "path"}
            for entry in manifest["samples"]
        ]


def iter_indexed_samples(path: str | Path) -> Iterator[Sample]:
    """Yield samples from an indexed INIF archive in manifest order."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest = _load_manifest(zf)
        for entry in manifest["samples"]:
            yield _read_sample_entry(zf, entry)


def read_indexed_samples(
    path: str | Path,
    sample_ids: Iterable[str | int],
) -> list[Sample]:
    """Read selected samples by id from an indexed INIF archive."""
    requested = [str(sample_id) for sample_id in sample_ids]
    assert len(requested) == len(set(requested)), "Requested sample ids must be unique"
    with zipfile.ZipFile(path, "r") as zf:
        manifest = _load_manifest(zf)
        entries = _sample_entries_by_id(manifest)
        missing = [sample_id for sample_id in requested if sample_id not in entries]
        if missing:
            raise KeyError(f"Samples not found: {missing!r}")
        return [_read_sample_entry(zf, entries[sample_id]) for sample_id in requested]


def read_indexed_sample(path: str | Path, sample_id: str | int) -> Sample:
    """Read one sample by id from an indexed INIF archive."""
    samples = read_indexed_samples(path, [sample_id])
    return samples[0]


def load_indexed(
    path: str | Path,
    sample_ids: Iterable[str | int] | None = None,
) -> InifDocument:
    """Load an indexed archive, optionally with only selected samples."""
    header = read_indexed_header(path)
    samples = (
        list(iter_indexed_samples(path))
        if sample_ids is None
        else read_indexed_samples(path, sample_ids)
    )
    return InifDocument(
        metadata=header.metadata,
        sequences=header.sequences
        if sample_ids is None
        else _prune_sequences(header.sequences, samples),
        samples=samples,
    )
