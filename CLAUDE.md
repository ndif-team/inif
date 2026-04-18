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

- **Token.id**: `>= 0` = vocabulary token ID, `-1` = reference to a sequence. A `@model_validator` enforces the sentinel: vocab tokens require `token: str`, sequence-ref tokens require `sequence_id: str`.
- **Token.sequence_id**: string ref to `Sequence.id` (only set on sequence ref tokens)
- **Token extras API**: Tags, logprob, role, logit_lens data, etc. live in `model_extra` (pydantic `extra="allow"`). Use the dedicated helpers — they keep `__dict__` and `model_extra` in sync so the field both shows up under attribute access and serializes:
  - tags: `token.tags`, `token.has_tag(t)`, `token.add_tag(t)`, `token.remove_tag(t)`
  - generic: `token.get_extra(key, default)`, `token.set_extra(key, value)`, `token.has_extra(key)`, `token.pop_extra(key)`, `token.extras` (snapshot dict)
- **`TokenExtras`**: documentation-only Pydantic model declaring the conventional extras (`tags`, `role`, `logprob`, `logit_lens`); embedded under `$defs.TokenExtras` in the JSON schema for external validators/UIs.
- **Sequence**: stores both `tokens: list[str]` and `ids: list[int]` (parallel arrays). Required so dedup → expand round-trips preserve real vocabulary IDs even after a sample has been compressed and later materialized. Optional `name: str` is a human-readable label distinct from the immutable `id`; use `seq.display_name` (falls back to id).
- **Sample.id**: always `str`. A `@field_validator(mode="before")` coerces ints (Inspect AI uses int sample ids by default).
- **Sample.spans**: validated against `len(tokens)` at construction; out-of-range positions raise `ValidationError`.
- **Sample.materialize_position(expanded_pos, sequences)**: expands the containing sequence ref in-place when `expanded_pos` falls inside one, returning `(actual_index, real_token)`. Other refs and other samples are left untouched.
- **InifDocument.total_samples**: computed property (`len(samples)`) — there is no stored field.
- **InifDocument.subset(predicate)**: returns a new doc with only matching samples; sequences not referenced by the kept samples are pruned (self-contained sub-document).
- **Sequence.id**: string identifier (e.g. `"sequence_0"`)
- **Sample.texts**: plain strings (message contents or raw text)
- **Metadata.created_at**: `datetime` (pydantic auto-parses ISO strings on load; `to_dict` uses `mode="json"` to emit ISO strings).
- **Deduplication**: finds token sequences common to ALL samples via set-intersection of contiguous n-grams; replacement requires both the token strings AND ids to match.

## Inspect AI converter conventions

- **`tag_generated=True`** (default): tokens belonging to the LAST assistant message are tagged `"generated"`. Identification uses character-span matching against `apply_chat_template` output, so it works for any HuggingFace chat template.
- **`extract_logprobs=True`** (default): per-token logprobs from `inspect_sample.output.choices[0].logprobs.content` are attached to the response tokens via `set_extra("logprob", ...)`. Best-effort: skipped silently when the eval-source tokenization disagrees with our tokenizer on token count.
- **`filter_samples_by_score(doc, scorer, predicate)`** returns `list[Sample]` (compose with the position selectors). The old `select_by_score` is gone.

## IO conventions

- **`save(doc, path, compress=None)`** and **`load(path, compress=None)`** are symmetric. Suffix-based detection: `.gz` and `.inif` ⇒ gzipped; everything else ⇒ plain JSON. `compress=True/False` overrides the detection.
