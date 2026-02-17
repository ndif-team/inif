# CLAUDE.md

This file provides guidance to Claude Code when working with the `inif` package.

## What is inif

INterpretability Interchange Format — a JSON-based format for tokenized LLM generation traces with support for tagging, position selection, and storing interpretability outputs. Designed as the interchange layer between eval frameworks (Inspect AI) and interpretability tools (nnterp/nnsight).

## Development Commands

- **Install**: `make dev` (or `uv sync --all-extras`)
- **Test**: `uv run pytest inif/tests/`
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`
- **Type check**: `uv run ty check inif/`
- **Generate schema**: `make schema`

## Code Philosophy

Research-oriented library. Follow nnterp conventions:

- **Fail-fast**: Use assertions for invariants, no try-catch fallbacks
- **No bloat**: Only implement what was asked. No defensive coding for non-common inputs
- **Pydantic models**: All data models use `pydantic.BaseModel` with `Field(default_factory=...)` for mutables
- **Compact serialization**: Strip None values and empty collections by default
- **Lazy imports**: Optional dependencies (inspect-ai, jsonschema) imported at call site

## Architecture

- `models.py` — Core Pydantic models (InifDocument, Sample, Token, Sequence, etc.)
- `schema.py` — JSON schema dict + validation
- `io.py` — save/load (.inif.json, .inif.json.gz)
- `selectors.py` — Position selection by index/tag/span/sequence_id/score
- `tagging.py` — Regex-based auto-tagging, span creation
- `sequences.py` — Sequence deduplication (lmout-style set-intersection) and expansion
- `converters/inspect_ai.py` — Inspect AI EvalLog converter
- `converters/text.py` — Raw text file / string processing
- `cli.py` — CLI entry point (`inif convert txt`, `inif convert eval`)

## Key Conventions

- **Token.id**: `>= 0` = vocabulary token ID, `-1` = reference to a sequence
- **Token.sequence_id**: string ref to `Sequence.id` (only set on sequence ref tokens)
- **Token extra fields**: Tags, logprob, role, logit_lens data etc. stored as Pydantic extra fields (`model_config = {"extra": "allow"}`). Access via `getattr(token, "field", default)`, write via `token.__dict__["field"]` + `token.model_extra["field"]`.
- **Sequence.id**: string identifier (e.g. `"sequence_0"`)
- **Sample.texts**: plain strings (message contents or raw text)
- **Deduplication**: finds token sequences common to ALL samples via set-intersection of contiguous n-grams
