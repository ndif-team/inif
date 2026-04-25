import json

from inif.models import InifDocument, Metadata, ModelInfo, Sample, Sequence, Token
from inif.shards import iter_samples, iter_shards, load_shards, save_shards


def _sharded_doc():
    seq = Sequence(
        id="shared",
        n_tokens=2,
        tokens=[Token(id=10, token="A"), Token(id=11, token="B")],
    )
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="test-model")),
        sequences=[seq],
        samples=[
            Sample(id="s0", tokens=[Token(id=-1, sequence_id="shared")]),
            Sample(id="s1", tokens=[Token(id=20, token="C")]),
            Sample(id="s2", tokens=[Token(id=-1, sequence_id="shared")]),
        ],
    )


def test_save_shards_writes_manifest_and_canonical_shards(tmp_path):
    doc = _sharded_doc()
    save_shards(doc, tmp_path, samples_per_shard=1, compress=False)

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "inif-sharded"
    assert manifest["version"] == 1
    assert manifest["metadata"]["model"]["name"] == "test-model"
    assert manifest["total_samples"] == 3
    assert [shard["sample_ids"] for shard in manifest["shards"]] == [
        ["s0"],
        ["s1"],
        ["s2"],
    ]
    assert [shard["path"] for shard in manifest["shards"]] == [
        "shard_00000.inif.json",
        "shard_00001.inif.json",
        "shard_00002.inif.json",
    ]
    assert (tmp_path / "shard_00000.inif.json").exists()


def test_iter_shards_prunes_sequences_per_shard(tmp_path):
    doc = _sharded_doc()
    save_shards(doc, tmp_path, samples_per_shard=1, compress=False)

    shards = list(iter_shards(tmp_path))

    assert [shard.samples[0].id for shard in shards] == ["s0", "s1", "s2"]
    assert [len(shard.sequences) for shard in shards] == [1, 0, 1]
    assert shards[0].sequences[0].id == "shared"
    assert shards[2].sequences[0].tokens[1].token == "B"


def test_iter_samples_streams_manifest_order(tmp_path):
    doc = _sharded_doc()
    save_shards(doc, tmp_path, samples_per_shard=2, compress=False)

    assert [sample.id for sample in iter_samples(tmp_path)] == ["s0", "s1", "s2"]


def test_load_shards_merges_samples_and_deduplicates_sequences(tmp_path):
    doc = _sharded_doc()
    save_shards(doc, tmp_path, samples_per_shard=1, compress=False)

    merged = load_shards(tmp_path)

    assert merged.metadata.model.name == "test-model"
    assert [sample.id for sample in merged.samples] == ["s0", "s1", "s2"]
    assert [seq.id for seq in merged.sequences] == ["shared"]
    assert merged.samples[0].tokens[0].sequence_id == "shared"


def test_load_empty_sharded_document(tmp_path):
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="empty")))
    save_shards(doc, tmp_path, samples_per_shard=2, compress=False)

    merged = load_shards(tmp_path)

    assert merged.metadata.model.name == "empty"
    assert merged.samples == []
    assert merged.sequences == []
