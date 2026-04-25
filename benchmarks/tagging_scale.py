from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from inif.models import Sample, Token
from inif.tagging import tag_by_regexes

TOKEN_CYCLE = (
    "The",
    " ",
    "answer",
    " ",
    "is",
    " ",
    "A",
    ".",
    "\n",
    "carbon",
    " ",
    "12",
    " ",
    "protein",
    " ",
    "B",
    ",",
    " electron",
    " ",
    "42",
    " ",
    "molecule",
    " ",
    "C",
    ";",
    " oxygen",
    " ",
    "391",
    " ",
    "D",
    "\n\n",
    "plain",
)

DEFAULT_REGEX_TAGS = (
    (r"\d+", "number"),
    (r"^\s*[ABCD]\s*$", "choice_letter"),
    (r"(?i)carbon|oxygen|protein|electron|molecule", "domain"),
    (r"(?i)^answer$", "answer_word"),
    (r"^\s+$", "whitespace"),
    (r"^[,.;]+$", "punct"),
)


@dataclass
class BenchResult:
    mode: str
    tokens: int
    strategies: int
    seconds: float
    matches: dict[str, int]
    max_rss_mb: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else float("inf")

    @property
    def checks_per_second(self) -> float:
        n_checks = self.tokens * self.strategies
        return n_checks / self.seconds if self.seconds else float("inf")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tokens": self.tokens,
            "strategies": self.strategies,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "checks_per_second": round(self.checks_per_second, 1),
            "matches": self.matches,
            "max_rss_mb": round(self.max_rss_mb, 1),
        }


def max_rss_mb() -> float:
    try:
        import resource
    except ImportError:
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024 / 1024
    return rss / 1024


def token_chunks(total: int, chunk_size: int) -> Iterator[tuple[int, list[str]]]:
    cycle_len = len(TOKEN_CYCLE)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = [TOKEN_CYCLE[i % cycle_len] for i in range(start, end)]
        yield start, chunk


def run_streaming_regex(total: int, chunk_size: int) -> BenchResult:
    """Run many regex tag strategies over flat token text chunks.

    This is the closest benchmark to a future Rust accelerator boundary:
    flat token strings in, per-strategy match decisions out, with no Pydantic
    object mutation in the hot loop.
    """
    compiled = [(re.compile(pattern), tag) for pattern, tag in DEFAULT_REGEX_TAGS]
    matches = {tag: 0 for _, tag in compiled}
    t0 = perf_counter()
    for _, chunk in token_chunks(total, chunk_size):
        for text in chunk:
            for pattern, tag in compiled:
                if pattern.search(text):
                    matches[tag] += 1
    seconds = perf_counter() - t0
    return BenchResult(
        mode="streaming_regex",
        tokens=total,
        strategies=len(compiled),
        seconds=seconds,
        matches=matches,
        max_rss_mb=max_rss_mb(),
    )


def run_object_regex(total: int) -> BenchResult:
    """Run the public Pydantic-backed tagger on a materialized Sample.

    Do not run this at 100M tokens. It estimates the overhead of the current
    Python object graph at smaller scales and highlights where a compact token
    store would help.
    """
    cycle_len = len(TOKEN_CYCLE)
    t_build = perf_counter()
    sample = Sample(
        id="bench",
        tokens=[
            Token(id=i % cycle_len, token=TOKEN_CYCLE[i % cycle_len])
            for i in range(total)
        ],
    )
    build_seconds = perf_counter() - t_build

    t0 = perf_counter()
    tag_by_regexes(sample, list(DEFAULT_REGEX_TAGS))
    seconds = perf_counter() - t0
    matches = {
        tag: sum(1 for token in sample.tokens if token.has_tag(tag))
        for _, tag in DEFAULT_REGEX_TAGS
    }
    matches["_build_ms"] = int(round(build_seconds * 1000))
    return BenchResult(
        mode="object_regex",
        tokens=total,
        strategies=len(DEFAULT_REGEX_TAGS),
        seconds=seconds,
        matches=matches,
        max_rss_mb=max_rss_mb(),
    )


def print_result(result: BenchResult) -> None:
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic INIF tagging scale benchmark."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100_000_000,
        help="Number of synthetic tokens for the streaming benchmark.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Streaming chunk size.",
    )
    parser.add_argument(
        "--object-tokens",
        type=int,
        default=0,
        help="Optional materialized Pydantic Token benchmark size.",
    )
    args = parser.parse_args()

    print_result(run_streaming_regex(args.tokens, args.chunk_size))
    if args.object_tokens:
        gc.collect()
        print_result(run_object_regex(args.object_tokens))


if __name__ == "__main__":
    main()
