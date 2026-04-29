"""Tests for the CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from inif.cli import main
from inif.models import InifDocument, Metadata, ModelInfo


class MockTokenizer:
    name_or_path = "mock-model"
    _commit_hash = None

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._get_id(w) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "?") for i in ids)


def test_convert_txt_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input file
        inp = Path(tmpdir) / "input.txt"
        inp.write_text("hello world test")
        out = Path(tmpdir) / "output.inif.json"

        with patch(
            "inif.converters.text._load_tokenizer",
            return_value=MockTokenizer(),
        ):
            main(["convert", "txt", str(inp), "-o", str(out), "-m", "mock"])

        assert out.exists()

        doc = InifDocument.load(out)
        assert len(doc.samples) == 1
        assert [(t.name, t.value) for t in doc.samples[0].texts] == [
            ("text_0", "hello world test")
        ]


def test_convert_txt_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input files
        (Path(tmpdir) / "a.txt").write_text("foo bar baz")
        (Path(tmpdir) / "b.txt").write_text("foo bar qux")
        out = Path(tmpdir) / "output.inif.json"

        with patch(
            "inif.converters.text._load_tokenizer",
            return_value=MockTokenizer(),
        ):
            main(
                [
                    "convert",
                    "txt",
                    tmpdir,
                    "-o",
                    str(out),
                    "-m",
                    "mock",
                    "--glob",
                    "*.txt",
                ]
            )

        assert out.exists()


def test_convert_eval_passes_min_sequence_length():
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="mock")))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "output.inif.json"
        with (
            patch(
                "inif.converters.inspect_ai.from_eval_file", return_value=doc
            ) as conv,
            patch.object(InifDocument, "save"),
        ):
            main(
                [
                    "convert",
                    "eval",
                    "input.eval",
                    "-o",
                    str(out),
                    "-m",
                    "mock",
                    "--min-seq-length",
                    "7",
                ]
            )

    assert conv.call_args.kwargs["min_sequence_length"] == 7


def test_cli_help(capsys):
    """CLI should print help without error."""
    main([])
    captured = capsys.readouterr()
    assert "inif" in captured.out.lower() or "usage" in captured.out.lower()
