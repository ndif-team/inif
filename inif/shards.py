from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from inif.io import load, save, to_dict
from inif.models import InifDocument, Metadata, Sample, Sequence

_MANIFEST = "manifest.json"
_FORMAT = "inif-sharded"
_VERSION = 1


def _referenced_sequence_ids(samples: list[Sample]) -> set[str]:
    referenced: set[str] = set()
    for sample in samples:
        for token in sample.tokens:
            if token.is_sequence_ref and token.sequence_id is not None:
                referenced.add(token.sequence_id)
    return referenced


def _shard_suffix(compress: bool) -> str:
    return ".inif" if compress else ".inif.json"


def save_shards(
    doc: InifDocument,
    path: str | Path,
    samples_per_shard: int = 100,
    compress: bool = True,
    compact: bool = True,
    indent: int | None = 4,
) -> None:
    """Save ``doc`` as a manifest plus canonical INIF JSON shard files."""
    assert samples_per_shard > 0, "samples_per_shard must be positive"

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": _FORMAT,
        "version": _VERSION,
        "metadata": to_dict(
            InifDocument(metadata=doc.metadata.model_copy(deep=True)),
            compact=compact,
        )["metadata"],
        "total_samples": len(doc.samples),
        "samples_per_shard": samples_per_shard,
        "shards": [],
    }

    seq_map = doc.sequence_map
    suffix = _shard_suffix(compress)
    for shard_index, start in enumerate(range(0, len(doc.samples), samples_per_shard)):
        chunk = doc.samples[start : start + samples_per_shard]
        referenced = _referenced_sequence_ids(chunk)
        missing = referenced - set(seq_map)
        assert not missing, f"Missing sequence definitions: {sorted(missing)}"
        sequences = [
            seq.model_copy(deep=True) for seq in doc.sequences if seq.id in referenced
        ]
        shard_doc = InifDocument(
            metadata=doc.metadata.model_copy(deep=True),
            sequences=sequences,
            samples=[sample.model_copy(deep=True) for sample in chunk],
        )
        shard_name = f"shard_{shard_index:05d}{suffix}"
        save(
            shard_doc,
            path / shard_name,
            compress=compress,
            compact=compact,
            indent=indent,
        )
        manifest["shards"].append(
            {
                "path": shard_name,
                "sample_ids": [sample.id for sample in chunk],
                "n_samples": len(chunk),
            }
        )

    with open(path / _MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path / _MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["format"] == _FORMAT, "Not an INIF shard manifest"
    assert manifest["version"] == _VERSION, (
        f"Unsupported INIF shard version {manifest['version']}"
    )
    return manifest


def iter_shards(path: str | Path) -> Iterator[InifDocument]:
    """Yield each canonical INIF shard as a self-contained document."""
    path = Path(path)
    manifest = _load_manifest(path)
    for shard in manifest["shards"]:
        yield load(path / shard["path"])


def iter_samples(path: str | Path) -> Iterator[Sample]:
    """Yield samples from a sharded document.

    Samples are yielded in manifest order. Use :func:`iter_shards` when callers
    need the shard-local sequence table alongside compressed samples.
    """
    for shard in iter_shards(path):
        for sample in shard.samples:
            yield sample


def _add_sequence(sequence_map: dict[str, Sequence], sequence: Sequence) -> None:
    if sequence.id in sequence_map:
        assert sequence_map[sequence.id].model_dump(
            mode="json",
            by_alias=True,
        ) == sequence.model_dump(mode="json", by_alias=True), (
            f"Conflicting sequence definition for {sequence.id!r}"
        )
        return
    sequence_map[sequence.id] = sequence.model_copy(deep=True)


def load_shards(path: str | Path) -> InifDocument:
    """Load all shards and merge them into a single ``InifDocument``."""
    path = Path(path)
    manifest = _load_manifest(path)
    metadata = Metadata.model_validate(manifest["metadata"])
    sequence_map: dict[str, Sequence] = {}
    samples: list[Sample] = []

    for shard in iter_shards(path):
        for sequence in shard.sequences:
            _add_sequence(sequence_map, sequence)
        samples.extend(sample.model_copy(deep=True) for sample in shard.samples)

    assert len(samples) == manifest["total_samples"], (
        f"Manifest expected {manifest['total_samples']} samples, got {len(samples)}"
    )
    return InifDocument(
        metadata=metadata,
        sequences=list(sequence_map.values()),
        samples=samples,
    )
