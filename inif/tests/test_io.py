import tempfile
from pathlib import Path

import pytest

from inif.io import from_dict, load, save, to_dict
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Token,
)


def test_to_dict_compact(doc):
    d = to_dict(doc, compact=True)
    # None values should be stripped
    assert "revision" not in d["metadata"]["model"]
    # Empty dicts should be stripped
    assert "loading_config" not in d["metadata"]["model"]
    # Non-empty values should remain
    assert "generation_config" in d["metadata"]["model"]


def test_to_dict_not_compact(doc):
    d = to_dict(doc, compact=False)
    assert "revision" in d["metadata"]["model"]
    assert d["metadata"]["model"]["revision"] is None


def test_roundtrip_dict(doc):
    d = to_dict(doc, compact=False)
    doc2 = from_dict(d)

    assert doc2.metadata.model.name == doc.metadata.model.name
    assert doc2.metadata.inif_version == doc.metadata.inif_version
    assert len(doc2.sequences) == len(doc.sequences)
    assert len(doc2.samples) == len(doc.samples)

    s1 = doc.samples[0]
    s2 = doc2.samples[0]
    assert s1.id == s2.id
    assert len(s1.tokens) == len(s2.tokens)
    assert s1.tokens[0].id == s2.tokens[0].id
    assert s1.scores[0].value == s2.scores[0].value


def test_roundtrip_dict_compact(doc):
    d = to_dict(doc, compact=True)
    doc2 = from_dict(d)
    # Compact strips None/empty, but from_dict fills defaults
    assert doc2.metadata.model.name == "gpt2"
    assert doc2.metadata.model.loading_config == {}
    assert len(doc2.samples[0].tokens) == len(doc.samples[0].tokens)


def test_save_load_json(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        save(doc, path)
        doc2 = load(path)

        assert doc2.metadata.model.name == "gpt2"
        assert len(doc2.samples) == 1
        assert doc2.samples[0].id == "sample_0"


def test_save_load_inif_indexed_archive(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif"
        save(doc, path)
        doc2 = load(path)

        assert doc2.metadata.model.name == "gpt2"
        assert len(doc2.samples) == 1


def test_save_rejects_compress_override(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        with pytest.raises(AssertionError, match="compress"):
            save(doc, path, compress=True)


def test_save_load_preserves_extra_fields():
    """Verify extra fields on Token survive round-trip serialization."""
    tok = Token(id=1, token="hello", source="user", logprob=-0.5)
    tok.set_extra("data", {"logit_lens": {"layer_0": {}}})

    from inif.models import Sample

    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="test")),
        samples=[Sample(id="s0", tokens=[tok])],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        save(doc, path, compact=False)
        doc2 = load(path)

        t = doc2.samples[0].tokens[0]
        assert t.get_extra("source") == "user"
        assert t.get_extra("logprob") == -0.5
        assert t.get_extra("data")["logit_lens"]["layer_0"] == {}


def test_sequence_ids_survive_save_load(doc):
    """Sequence token ids must round-trip through save/load — losing them
    would corrupt every expanded token's vocabulary id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        save(doc, path)
        doc2 = load(path)

    for s_orig, s_reload in zip(doc.sequences, doc2.sequences):
        assert [t.id for t in s_reload.tokens] == [t.id for t in s_orig.tokens]
        assert [t.token for t in s_reload.tokens] == [t.token for t in s_orig.tokens]


def test_load_rejects_compress_override(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        save(doc, path)
        with pytest.raises(AssertionError, match="compress"):
            load(path, compress=False)


def test_gzip_suffix_is_rejected(doc):
    """INIF now has two storage modes: plain JSON or indexed .inif archives."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json.gz"
        with pytest.raises(AssertionError, match="gzip"):
            save(doc, path)


def test_inifx_suffix_is_rejected(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inifx"
        with pytest.raises(AssertionError, match="inifx"):
            save(doc, path)
        with pytest.raises(AssertionError, match="inifx"):
            load(path)


def test_minimal_document():
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="test")))
    d = to_dict(doc, compact=True)
    doc2 = from_dict(d)
    assert doc2.metadata.model.name == "test"
    assert doc2.samples == []
    assert doc2.sequences == []


def test_schema_documents_token_extras():
    """The generated JSON schema embeds TokenExtras under $defs so external
    validators / UIs can discover the conventional Token extra fields."""
    from inif.schema import get_schema

    schema = get_schema()
    assert "TokenExtras" in schema["$defs"]
    extras_props = schema["$defs"]["TokenExtras"]["properties"]
    for key in ("logprob", "logit_lens"):
        assert key in extras_props, f"Missing conventional extra: {key}"
    assert "tags" not in extras_props
    assert "role" not in extras_props
