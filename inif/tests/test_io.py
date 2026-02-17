import json
import tempfile
from pathlib import Path

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


def test_save_load_gzip(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json.gz"
        save(doc, path)
        doc2 = load(path)

        assert doc2.metadata.model.name == "gpt2"
        assert len(doc2.samples) == 1


def test_save_force_compress(doc):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.inif.json"
        save(doc, path, compress=True)
        # File is gzip-compressed even without .gz extension
        import gzip

        with gzip.open(path, "rt") as f:
            data = json.load(f)
        assert data["metadata"]["model"]["name"] == "gpt2"


def test_save_load_preserves_extra_fields():
    """Verify extra fields on Token survive round-trip serialization."""
    tok = Token(id=1, token="hello", role="user", logprob=-0.5)
    tok.__dict__["data"] = {"logit_lens": {"layer_0": {}}}
    if tok.model_extra is not None:
        tok.model_extra["data"] = {"logit_lens": {"layer_0": {}}}

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
        assert t.model_extra["role"] == "user"
        assert t.model_extra["logprob"] == -0.5
        assert t.model_extra["data"]["logit_lens"]["layer_0"] == {}


def test_minimal_document():
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="test")))
    d = to_dict(doc, compact=True)
    doc2 = from_dict(d)
    assert doc2.metadata.model.name == "test"
    assert doc2.samples == []
    assert doc2.sequences == []
