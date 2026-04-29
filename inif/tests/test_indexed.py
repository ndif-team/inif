import json
import zipfile

import pytest

from inif.indexed import IndexedInifWriter, load_indexed, save_indexed
from inif.io import iter_samples, read_info, read_samples
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    Sequence,
    Text,
    TokenOrSeqRef,
)


def _indexed_doc() -> InifDocument:
    seq = Sequence(
        id="shared",
        n_tokens=2,
        tokens=[TokenOrSeqRef(id=10, token="A"), TokenOrSeqRef(id=11, token="B")],
    )
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="indexed-test")),
        sequences=[seq],
        samples=[
            Sample(
                id="s0",
                tokens=[
                    TokenOrSeqRef(id=None, token="shared"),
                    TokenOrSeqRef(id=20, token=" C"),
                ],
                texts=[Text(name="text_0", value="A B C")],
                target="C",
                input_tokens=2,
                output_tokens=1,
            ),
            Sample(
                id="s1",
                tokens=[TokenOrSeqRef(id=30, token="D")],
                texts=[Text(name="text_0", value="D")],
                error="mock error",
            ),
            Sample(
                id="s2",
                tokens=[TokenOrSeqRef(id=None, token="shared")],
                texts=[Text(name="text_0", value="A B")],
            ),
        ],
    )


def test_save_indexed_writes_single_archive_with_manifest(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"

    save_indexed(doc, path)

    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "metadata.json" in names
        assert "sequences.json" in names
        assert "_journal/start.json" in names
        assert "_journal/summaries/00000000.json" in names
        assert "samples/00000000.json" in names
        assert zf.getinfo("manifest.json").compress_type == zipfile.ZIP_STORED
        assert zf.getinfo("metadata.json").compress_type == zipfile.ZIP_STORED
        assert (
            zf.getinfo("_journal/summaries/00000000.json").compress_type
            == zipfile.ZIP_STORED
        )
        assert zf.getinfo("samples/00000000.json").compress_type == zipfile.ZIP_DEFLATED
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        summary = json.loads(
            zf.read("_journal/summaries/00000000.json").decode("utf-8")
        )

    assert manifest["format"] == "inif-indexed"
    assert manifest["version"] == 1
    assert manifest["total_samples"] == 3
    assert [entry["id"] for entry in manifest["samples"]] == ["s0", "s1", "s2"]
    assert manifest["samples"][0]["text_preview"] == "A B C"
    assert summary["id"] == "s0"
    assert summary["text_preview"] == "A B C"


def test_load_indexed_roundtrip(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    loaded = load_indexed(path)

    assert loaded.metadata.model.name == "indexed-test"
    assert [sample.id for sample in loaded.samples] == ["s0", "s1", "s2"]
    assert loaded.sequences[0].tokens[1].token == "B"
    # Sequence refs round-trip with id=None and the target id in ``token``.
    assert loaded.samples[0].tokens[0].is_sequence_ref
    assert loaded.samples[0].tokens[0].token == "shared"


def test_save_load_dispatches_inif_suffix_to_indexed_archive(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"

    doc.save(path)
    loaded = InifDocument.load(path)

    assert loaded.metadata.model.name == "indexed-test"
    assert [sample.id for sample in loaded.samples] == ["s0", "s1", "s2"]


def test_load_indexed_can_load_selected_samples(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    loaded = load_indexed(path, sample_ids=["s2"])

    assert [sample.id for sample in loaded.samples] == ["s2"]
    assert [sequence.id for sequence in loaded.sequences] == ["shared"]

    loaded_without_refs = load_indexed(path, sample_ids=["s1"])
    assert [sample.id for sample in loaded_without_refs.samples] == ["s1"]
    assert loaded_without_refs.sequences == []


def test_incremental_writer_appends_samples(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"

    with IndexedInifWriter(path, doc.metadata, doc.sequences) as writer:
        writer.write_sample(doc.samples[0])
        writer.write_sample(doc.samples[1])

    loaded = load_indexed(path)
    assert [sample.id for sample in loaded.samples] == ["s0", "s1"]


def test_incremental_writer_flush_makes_partial_archive_readable(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"

    with IndexedInifWriter(path, doc.metadata, doc.sequences) as writer:
        writer.write_sample(doc.samples[0])
        writer.flush()

        # Use the unified read API while the writer is still open.
        info = read_info(path)
        sample = read_samples(path, "s0")[0]

        assert [s["id"] for s in info.samples] == ["s0"]
        assert info.samples[0]["text_preview"] == "A B C"
        assert sample.id == "s0"

        writer.write_sample(doc.samples[1])

    loaded = load_indexed(path)
    assert [sample.id for sample in loaded.samples] == ["s0", "s1"]


# ---------------------------------------------------------------------------
# Unified read API — exercised on both .inif and .inif.json
# ---------------------------------------------------------------------------


@pytest.fixture(params=["doc.inif", "doc.inif.json"])
def saved_doc_path(request, tmp_path):
    """Save the canonical fixture document under both supported suffixes."""
    doc = _indexed_doc()
    path = tmp_path / request.param
    doc.save(path)
    return path


def test_iter_samples_yields_in_document_order(saved_doc_path):
    assert [s.id for s in iter_samples(saved_doc_path)] == ["s0", "s1", "s2"]


def test_read_samples_subset_in_requested_order(saved_doc_path):
    samples = read_samples(saved_doc_path, ["s2", "s0"])
    assert [s.id for s in samples] == ["s2", "s0"]
    assert samples[1].tokens[1].token == " C"


def test_read_samples_accepts_single_id_string(saved_doc_path):
    samples = read_samples(saved_doc_path, "s1")
    assert [s.id for s in samples] == ["s1"]
    assert samples[0].error == "mock error"
    assert samples[0].tokens[0].token == "D"


def test_read_samples_missing_id_raises_keyerror(saved_doc_path):
    with pytest.raises(KeyError):
        read_samples(saved_doc_path, "missing")


def test_read_samples_rejects_duplicate_ids(saved_doc_path):
    with pytest.raises(AssertionError, match="unique"):
        read_samples(saved_doc_path, ["s0", "s0"])


def test_read_info_returns_metadata_and_summaries(saved_doc_path):
    info = read_info(saved_doc_path)

    assert info.metadata.model.name == "indexed-test"
    assert [s["id"] for s in info.samples] == ["s0", "s1", "s2"]
    assert info.samples[0]["n_tokens"] == 2
    assert info.samples[0]["target"] == "C"
    assert info.samples[0]["text_preview"] == "A B C"
    assert info.samples[1]["error"] == "mock error"
    # No sequences and no full token streams in the header view.
    assert "tokens" not in info.samples[0]
    # Indexed-archive bookkeeping fields should be stripped from both formats.
    assert "path" not in info.samples[0]
    assert "index" not in info.samples[0]


def test_read_info_does_not_inflate_sequences(saved_doc_path):
    info = read_info(saved_doc_path)
    # ``read_info`` returns a DocumentInfo (not an InifDocument) — so there is
    # no ``sequences`` attribute at all, by construction.
    assert not hasattr(info, "sequences")
