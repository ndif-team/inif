from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _convert_txt(args: argparse.Namespace) -> None:
    from inif.converters.text import from_text_files
    from inif.io import save

    # Resolve input paths
    input_paths: list[str | Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            pattern = args.glob or "*.txt"
            input_paths.extend(sorted(p.glob(pattern)))
        else:
            input_paths.append(p)

    if not input_paths:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    doc = from_text_files(
        paths=input_paths,
        tokenizer=args.model,
        min_sequence_length=args.min_seq_length,
        deduplicate=not args.no_dedup,
    )

    if args.revision:
        doc.metadata.model.revision = args.revision

    if args.output:
        output = Path(args.output)
    else:
        output = Path(input_paths[0]).with_suffix(".inif.json")
    save(doc, output)
    print(f"Saved to {output}")


def _view(args: argparse.Namespace) -> None:
    import tempfile
    import webbrowser

    from inif.io import load
    from inif.viewer import save_html

    doc = load(args.input)
    if args.output:
        save_html(
            doc, args.output, compact=args.compact, title=args.title, source=args.input
        )
        print(f"Saved to {args.output}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            tmp_path = f.name
        save_html(
            doc, tmp_path, compact=args.compact, title=args.title, source=args.input
        )
        webbrowser.open(f"file://{tmp_path}")
        print(f"Opened {tmp_path} in browser")


def _convert_eval(args: argparse.Namespace) -> None:
    from inif.converters.inspect_ai import from_eval_file
    from inif.io import save

    for inp in args.inputs:
        kwargs = {
            "deduplicate": not args.no_dedup,
            "min_sequence_length": args.min_seq_length,
            "tag_chat_roles": not args.no_tag_chat_roles,
        }
        if args.model:
            kwargs["tokenizer"] = args.model

        doc = from_eval_file(inp, **kwargs)

        if args.revision:
            doc.metadata.model.revision = args.revision

        if args.output:
            output = Path(args.output)
        else:
            p = Path(inp)
            stem = p.stem
            if stem.endswith(".eval"):
                stem = stem[: -len(".eval")]
            output = p.parent / f"{stem}.inif.json"

        save(doc, output)
        print(f"Saved to {output}")


def _convert_evaleval(args: argparse.Namespace) -> None:
    from inif.converters.evaleval import from_eval_json, from_hf_dataset
    from inif.io import save

    kwargs: dict = {
        "deduplicate": not args.no_dedup,
        "min_sequence_length": args.min_seq_length,
        "tag_chat_roles": not args.no_tag_chat_roles,
    }
    if args.model:
        kwargs["tokenizer"] = args.model
    if args.limit is not None:
        kwargs.setdefault("limit", args.limit)

    if args.hf_config:
        doc = from_hf_dataset(
            args.hf_config,
            split=args.hf_split,
            aggregate_config=args.hf_aggregate_config,
            repo=args.hf_repo,
            limit=args.limit,
            **{k: v for k, v in kwargs.items() if k != "limit"},
        )
        default_stem = args.hf_config
    else:
        kwargs.pop("limit", None)
        doc = from_eval_json(
            args.aggregate,
            args.input,
            **kwargs,
        )
        stem = Path(args.input).stem
        default_stem = stem.removesuffix(".jsonl").removesuffix(".json")

    if args.revision:
        doc.metadata.model.revision = args.revision

    output = Path(args.output) if args.output else Path(f"{default_stem}.inif.json")
    save(doc, output)
    print(f"Saved to {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="inif",
        description="INterpretability Interchange Format CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # view subcommand
    view_parser = subparsers.add_parser("view", help="View inif file as HTML")
    view_parser.add_argument(
        "input", help="Input inif file (.inif.json, .inif.json.gz, or .inif)"
    )
    view_parser.add_argument(
        "-o", "--output", help="Output HTML path (opens browser if omitted)"
    )
    view_parser.add_argument("--compact", action="store_true", help="Hide empty panels")
    view_parser.add_argument("--title", help="Custom page title")

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert", help="Convert files to inif format"
    )
    convert_sub = convert_parser.add_subparsers(dest="format")

    # convert txt
    txt_parser = convert_sub.add_parser("txt", help="Convert raw text files")
    txt_parser.add_argument("inputs", nargs="+", help="Input files or dirs")
    txt_parser.add_argument("-o", "--output", help="Output path")
    txt_parser.add_argument("-m", "--model", required=True, help="Tokenizer name")
    txt_parser.add_argument("--revision", help="Model revision")
    txt_parser.add_argument("--min-seq-length", type=int, default=3, help="Min seq len")
    txt_parser.add_argument("--glob", help="Glob pattern for dir filtering")
    txt_parser.add_argument("--no-dedup", action="store_true", help="Skip dedup")

    # convert eval
    eval_parser = convert_sub.add_parser("eval", help="Convert Inspect AI eval logs")
    eval_parser.add_argument("inputs", nargs="+", help="Input eval log files")
    eval_parser.add_argument("-o", "--output", help="Output path")
    eval_parser.add_argument("-m", "--model", help="Tokenizer model name")
    eval_parser.add_argument("--revision", help="Model revision")
    eval_parser.add_argument(
        "--min-seq-length", type=int, default=3, help="Min seq len"
    )
    eval_parser.add_argument(
        "--no-tag-chat-roles",
        action="store_true",
        help="Don't add role tags to tokens",
    )
    eval_parser.add_argument("--no-dedup", action="store_true", help="Skip dedup")
    eval_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")

    # convert evaleval
    eee_parser = convert_sub.add_parser(
        "evaleval",
        help="Convert every_eval_ever instance-level JSON or EEE_datastore config",
    )
    eee_parser.add_argument(
        "input",
        nargs="?",
        help="Instances file (.json or .jsonl). Omit when using --hf-config.",
    )
    eee_parser.add_argument(
        "--aggregate", help="Optional aggregate eval.json file", default=None
    )
    eee_parser.add_argument(
        "--hf-config",
        help="Config name from evaleval/EEE_datastore, e.g. "
        "'theory_of_mind_samples'. When given, pulls rows via HuggingFace.",
    )
    eee_parser.add_argument(
        "--hf-split", default="samples", help="HF split (default: samples)"
    )
    eee_parser.add_argument(
        "--hf-repo", default="evaleval/EEE_datastore", help="HF repo id"
    )
    eee_parser.add_argument(
        "--hf-aggregate-config",
        help="Paired aggregate config (non-samples counterpart)",
    )
    eee_parser.add_argument("--limit", type=int, help="Max instance records to convert")
    eee_parser.add_argument("-o", "--output", help="Output path")
    eee_parser.add_argument("-m", "--model", help="Tokenizer model name")
    eee_parser.add_argument("--revision", help="Model revision")
    eee_parser.add_argument("--min-seq-length", type=int, default=3, help="Min seq len")
    eee_parser.add_argument(
        "--no-tag-chat-roles",
        action="store_true",
        help="Don't add role tags to tokens",
    )
    eee_parser.add_argument("--no-dedup", action="store_true", help="Skip dedup")

    args = parser.parse_args(argv)

    if args.command == "view":
        _view(args)
    elif args.command == "convert":
        if args.format == "txt":
            _convert_txt(args)
        elif args.format == "eval":
            _convert_eval(args)
        elif args.format == "evaleval":
            _convert_evaleval(args)
        else:
            convert_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
