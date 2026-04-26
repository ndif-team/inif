# CLAUDE.md

This file provides guidance to Claude Code when working with the `inif` package.

## What is inif

INterpretability Interchange Format — a JSON-based format for tokenized LLM generation traces with support for token annotations, position selection, and storing interpretability outputs. Designed as the interchange layer between eval frameworks (Inspect AI) and interpretability tools (nnterp/nnsight).

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

- `models.py` — Core Pydantic models (InifDocument, Sample, TokenOrSeqRef, Sequence, Text, etc.)
- `schema.py` — JSON schema dict + validation
- `io.py` — save/load (.inif.json, .inif indexed archive) plus the unified `iter_samples` / `read_samples` / `read_info` readers
- `indexed.py` — single-file indexed `.inif` archive writer + internal readers (used by `io.py`)
- `selectors.py` — Position selection by index/annotation/span/sequence_id/score
- `tagging.py` — Regex-based auto-tagging, span creation
- `sequences.py` — Sequence deduplication (lmout-style set-intersection) and expansion
- `converters/inspect_ai.py` — Inspect AI EvalLog converter
- `converters/evaleval.py` — every_eval_ever (EEE) instance-level → `InifDocument`
- `converters/_tokenize.py` — shared `apply_chat_template` → `TokenOrSeqRef` helpers (used by both eval converters); also `name_messages` for role-named `Text` segments
- `converters/text.py` — Raw text file / string processing
- `cli.py` — CLI entry point (`inif convert txt`, `inif convert eval`, `inif convert evaleval`)

## Key Conventions

- **`TokenOrSeqRef`**: a single Pydantic class for both vocab tokens and sequence references. Vocab tokens have `id: int >= 0` and `token: str` (the decoded piece). Sequence refs have `id is None` and use `token: str` to carry the target `Sequence.id` — there is no separate `sequence_id` field. `is_sequence_ref` is the property to check; the `sequence_id` *property* (read-only) returns `token` for refs and `None` for vocab tokens, kept as a convenience accessor for the common "is this a ref to seq X?" check.
- **Token extras API**: Sparse per-token values such as logprob, logit_lens data, probes, etc. live in `model_extra` (pydantic `extra="allow"`). Use the dedicated helpers — they keep `__dict__` and `model_extra` in sync so the field both shows up under attribute access and serializes: `token.get_extra(key, default)`, `token.set_extra(key, value)`, `token.has_extra(key)`, `token.pop_extra(key)`, `token.extras` (snapshot dict).
- **`TokenAnnotation`**: repeated labels live at `Sample.annotations` as `{name, ranges, metadata}`. Ranges are half-open token offsets (`[start, end)`) and are merged when metadata matches. Chat roles, generated output, reasoning traces, and regex labels are all annotations; roles are just auto-parsed annotation names from messages.
- **`TokenExtras`**: documentation-only Pydantic model declaring the conventional token extras (`logprob`, `logit_lens`); embedded under `$defs.TokenExtras` in the JSON schema for external validators/UIs.
- **`Text`**: each entry on `Sample.texts` is `{name, value, metadata}`. Default naming is role-based for chat (`"system_0"`, `"user_0"`, `"assistant_0"`, `"user_1"`, …) and index-based for plain text (`"text_0"`, …). System prompts are included.
- **Sequence**: stores `tokens: list[TokenOrSeqRef]` (always real vocab tokens). Required so dedup → expand round-trips preserve real vocabulary IDs even after a sample has been compressed and later materialized.
- **Sample.id**: always `str`. A `@field_validator(mode="before")` coerces ints (Inspect AI uses int sample ids by default).
- **Sample.spans / Sample.annotations**: validated against `len(tokens)` at construction; out-of-range positions/ranges raise `ValidationError`.
- **Sample first-class fields aligned with EEE**: `target` (singleton convenience), `references: list[str]` (full ground-truth list), `choices: list[str] | None` (MCQ options), `interaction_type: str | None` (`"single_turn"` / `"multi_turn"` / `"agentic"`), `error: str | None` (API timeouts, refusals, etc.), `sample_hash: str | None` (cross-model comparison key). These used to live under `Sample.metadata` for the evaleval converter; they are now top-level so filters / viewers / downstream tools can rely on them without key archaeology.
- **Sample.materialize_position(expanded_pos, sequences)**: expands the containing sequence ref in-place when `expanded_pos` falls inside one, returning `(actual_index, real_token)`. Other refs and other samples are left untouched.
- **InifDocument.total_samples**: computed property (`len(samples)`) — there is no stored field.
- **InifDocument.subset(predicate)**: returns a new doc with only matching samples; sequences not referenced by the kept samples are pruned (self-contained sub-document).
- **Sequence.id**: string identifier (e.g. `"sequence_0"`)
- **Sample.texts**: list of `Text` (`{name, value, metadata}`); see `Text` above.
- **Metadata.created_at**: `datetime` (pydantic auto-parses ISO strings on load; `to_dict` uses `mode="json"` to emit ISO strings).
- **Deduplication**: finds token sequences common to ALL samples via set-intersection of contiguous n-grams; replacement requires both the token strings AND ids to match.

## Tokenizer resolution (Inspect + evaleval)

- **Always present**: tokens are a load-bearing INIF invariant — there is no way to opt out of tokenization at the converter level.
- **Default `tokenizer="auto"`** (and the alias `tokenizer=None`): both converters read the source's model id (`eval_log.eval.model` for Inspect, `record["model_id"]` / aggregate fallback for evaleval), strip routing prefixes (`together/`, `hf/`, `bedrock/`, …) via the `_KNOWN_PROVIDERS` allowlist, and call `AutoTokenizer.from_pretrained(canonical_id, trust_remote_code=True)`. Raises `ValueError` if the id is missing or unloadable (closed-source ids like `openai/gpt-4` will hit this — the error message tells the user to pass an HF stand-in).
- **Explicit string / instance**: still loaded / used, but compared against the source's model id via the same prefix-stripping comparison; on mismatch a `UserWarning` is emitted (`"Tokenizer mismatch: …"`). The mismatch may be intentional (forcing one model's tokens through another's tokenizer for cross-model studies); the warning just keeps users informed.
- The shared resolver lives in `inif/converters/_tokenize.py::resolve_tokenizer` and is imported by both `from_eval_log` (Inspect) and `from_instance_records` (evaleval).

## Inspect AI converter conventions

- **`tag_generated=True`** (default): tokens belonging to the LAST assistant message are annotated `"generated"`. Identification uses character-span matching against `apply_chat_template` output, so it works for any HuggingFace chat template.
- **`extract_logprobs=True`** (default): per-token logprobs from `inspect_sample.output.choices[0].logprobs.content` are attached to the response tokens via `set_extra("logprob", ...)`. Best-effort: skipped silently when the eval-source tokenization disagrees with our tokenizer on token count.
- **`filter_samples_by_score(doc, scorer, predicate)`** returns `list[Sample]` (compose with the position selectors). The old `select_by_score` is gone.

## Evaleval converter conventions

- **Schema**: targets `instance_level_eval_0.2.2` from `evaleval/every_eval_ever`. Trusted as-is — users who want jsonschema validation should run it themselves against the upstream schema before calling the converter.
- **Three entry points** in `inif/converters/evaleval.py`: `from_instance_records(records, aggregate=None, ...)` (core), `from_eval_json(aggregate_path, instances_path, ...)` (local JSON / JSONL), `from_hf_dataset(config, split="samples", aggregate_config=None, limit=None, ...)` (streams rows from `evaleval/EEE_datastore`).
- **Interaction types**: `single_turn` synthesises `user` + `assistant` messages; `multi_turn` / `agentic` use the `messages[]` array directly, ordered by `turn_idx`. `tool_calls` are rendered into assistant content as `<tool_call name=… args={…}/>` so the invocations survive tokenization.
- **Reasoning traces**: when `tag_reasoning=True` (default), reasoning-trace tokens are char-span annotated with `"reasoning"`. Best-effort: silently skipped when the tokenizer's per-token decode doesn't round-trip to `apply_chat_template`.
- **Score**: `SampleScore` is built from `evaluation.score` (`scorer = evaluation_name`). The terminal `answer_attribution` entry feeds `SampleScore.answer`; non-terminal entries land in `Sample.metadata["intermediate_answers"]`.
- **Metadata mapping**: `token_usage.input_tokens`/`output_tokens` → `Sample.input_tokens`/`output_tokens`; `input.reference` → `Sample.references` (+ `Sample.target` when length 1); `input.choices` → `Sample.choices`; `interaction_type`, `error`, `sample_hash` → same-named first-class `Sample` fields. `performance`, `num_turns`, `tool_calls_count`, `reasoning_tokens`, per-sample EEE `metadata`, non-terminal `answer_attribution` entries → `Sample.metadata`. Aggregate record (when supplied) contributes `Metadata.extra["inference"]`, `["eval_library"]`, `["metric_config"]`, `["aggregate_score"]`.
- **Optional extra**: `pip install inif[evaleval]` pulls `datasets` (imported lazily inside `from_hf_dataset`).

## IO conventions

- **`save(doc, path)`** and **`load(path)`** are symmetric. Suffix-based detection: `.inif` ⇒ indexed archive; `.inif.json` / `.json` ⇒ plain JSON. `.gz`, `.inifx`, and `compress=True/False` overrides are no longer supported.
- **Unified read API in `inif.io`**: `iter_samples(path)` (streams `Sample`s), `read_samples(path, sample_ids)` (single id or iterable; returns `list[Sample]` in requested order), `read_info(path)` (returns a `DocumentInfo` with `metadata` plus per-sample summary dicts; never inflates sequences). All three accept both `.inif` and `.inif.json` paths, dispatching to the indexed-archive readers or to a full `load` as appropriate. The format-specific helpers (`iter_indexed_samples`, `read_indexed_*`) are now internal — public callers should use the unified API.
