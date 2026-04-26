import json
import zipfile

import pytest

from inif.indexed import (
    IndexedInifWriter,
    iter_indexed_samples,
    load_indexed,
    read_indexed_header,
    read_indexed_sample,
    read_indexed_sample_summaries,
    read_indexed_samples,
    save_indexed,
)
from inif.io import load, save
from inif.models import InifDocument, Metadata, ModelInfo, Sample, Sequence, Token


def _indexed_doc() -> InifDocument:
    seq = Sequence(
        id="shared",
        n_tokens=2,
        tokens=[Token(id=10, token="A"), Token(id=11, token="B")],
    )
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="indexed-test")),
        sequences=[seq],
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=-1, sequence_id="shared"), Token(id=20, token=" C")],
                texts=["A B C"],
                target="C",
                input_tokens=2,
                output_tokens=1,
            ),
            Sample(
                id="s1",
                tokens=[Token(id=30, token="D")],
                texts=["D"],
                error="mock error",
            ),
            Sample(
                id="s2",
                tokens=[Token(id=-1, sequence_id="shared")],
                texts=["A B"],
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
    assert manifest["samples"][0]["texts_preview"] == ["A B C"]
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
    assert loaded.samples[0].tokens[0].sequence_id == "shared"


def test_save_load_dispatches_inif_suffix_to_indexed_archive(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"

    save(doc, path)
    loaded = load(path)

    assert loaded.metadata.model.name == "indexed-test"
    assert [sample.id for sample in loaded.samples] == ["s0", "s1", "s2"]


def test_read_indexed_header_avoids_samples(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    header = read_indexed_header(path)

    assert header.metadata.model.name == "indexed-test"
    assert [seq.id for seq in header.sequences] == ["shared"]
    assert header.samples == []


def test_iter_indexed_samples_streams_in_manifest_order(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    assert [sample.id for sample in iter_indexed_samples(path)] == ["s0", "s1", "s2"]


def test_read_indexed_sample_by_id(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    sample = read_indexed_sample(path, "s1")

    assert sample.id == "s1"
    assert sample.error == "mock error"
    assert sample.tokens[0].token == "D"


def test_read_indexed_samples_by_id_loads_subset_in_requested_order(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    samples = read_indexed_samples(path, ["s2", "s0"])

    assert [sample.id for sample in samples] == ["s2", "s0"]
    assert samples[1].tokens[1].token == " C"


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


def test_read_indexed_sample_missing_raises_keyerror(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    with pytest.raises(KeyError):
        read_indexed_sample(path, "missing")


def test_read_indexed_sample_summaries_do_not_load_full_tokens(tmp_path):
    doc = _indexed_doc()
    path = tmp_path / "doc.inif"
    save_indexed(doc, path)

    summaries = read_indexed_sample_summaries(path)

    assert summaries[0]["id"] == "s0"
    assert summaries[0]["n_tokens"] == 2
    assert summaries[0]["target"] == "C"
    assert summaries[0]["text_preview"] == "A B C"
    assert "tokens" not in summaries[0]
    assert summaries[1]["error"] == "mock error"


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

        summaries = read_indexed_sample_summaries(path)
        sample = read_indexed_sample(path, "s0")

        assert [summary["id"] for summary in summaries] == ["s0"]
        assert summaries[0]["text_preview"] == "A B C"
        assert sample.id == "s0"

        writer.write_sample(doc.samples[1])

    loaded = load_indexed(path)
    assert [sample.id for sample in loaded.samples] == ["s0", "s1"]
