# AGENTS.md

This file provides guidance to Codex when working with the `inif` package.

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

- `models.py` — Core Pydantic models (InifDocument, Sample, TokenOrSeqRef, Sequence, Text, etc.). All public verbs (save / load / to_dict / from_dict, sequence (de)duplication, selection, tagging, viewer, etc.) live here as methods on `InifDocument` and `Sample`. The other modules below host the underscore-prefixed implementations the methods delegate to.
- `schema.py` — JSON schema dict + validation
- `io.py` — backs `InifDocument.save` / `.load` / `.to_dict` / `.from_dict` (suffix-based dispatch between `.inif.json` and `.inif`); also exposes the path-based unified readers `iter_samples` / `read_samples` / `read_info`
- `indexed.py` — single-file indexed `.inif` archive writer + internal readers (used by `io.py`)
- `selectors.py` — `_select_by_*` helpers (called by `Sample` selection methods) and `_filter_samples_by_score` (called by `InifDocument.filter_samples_by_score`); also exports `TokenSelection`
- `tagging.py` — `_tag_by_*` / `_tag_chat_*` / `_create_span_from_tag` helpers powering the `Sample.tag_*` and `InifDocument.tag_*` methods; also exports `TextTagMode`, `RegexTag`, `PredicateTag`, `TokenPredicate`
- `sequences.py` — `_deduplicate_sequences` / `_expand_sequences` helpers backing the `InifDocument` methods of the same name
- `viewer.py` — `_render_html` / `_show` / `_save_html` helpers backing the `InifDocument` methods
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
- **Sample first-class fields aligned with EEE**: `target` (singleton convenience), `references: list[str]` (full ground-truth list), `choices: list[str] | None` (MCQ options), `interaction_type: str | None` (`"single_turn"` / `"multi_turn"` / `"agentic"`), `error: str | None` (API timeouts, refusals, etc.), `sample_hash: str | None` (cross-model comparison key). These live at the top level of `Sample` so filters / viewers / downstream tools can read them without reaching into `metadata`.
- **Sample.materialize_position(expanded_pos, sequences)**: expands the containing sequence ref in-place when `expanded_pos` falls inside one, returning `(actual_index, real_token)`. Other refs and other samples are left untouched.
- **InifDocument.total_samples**: computed property (`len(samples)`) — there is no stored field.
- **InifDocument.subset(predicate)**: returns a new doc with only matching samples; sequences not referenced by the kept samples are pruned (self-contained sub-document).
- **Sequence.id**: string identifier (e.g. `"sequence_0"`)
- **Sample.texts**: list of `Text` (`{name, value, metadata}`); see `Text` above.
- **Metadata.created_at**: `datetime` (pydantic auto-parses ISO strings on load; `to_dict` uses `mode="json"` to emit ISO strings).
- **Deduplication**: finds token sequences common to ALL samples via set-intersection of contiguous n-grams; replacement requires both the token strings AND ids to match. Default `min_length=5` — short shared runs aren't worth the indirection in practice; pass an explicit smaller value when you really need to capture them.
- **Method-only public API**: anything that operates on a class instance is a method on that class. Save / load, `to_dict` / `from_dict`, sequence (de)duplication, selection, tagging, score-based filtering, viewer rendering — all live on `InifDocument` and `Sample`. The corresponding underscore-prefixed functions in `io.py` / `sequences.py` / `selectors.py` / `tagging.py` / `viewer.py` are internal implementations and are not re-exported from `inif.__init__`.

## Tokenizer resolution (Inspect + evaleval)

- **Always present**: tokens are a load-bearing INIF invariant — there is no way to opt out of tokenization at the converter level.
- **Default `tokenizer="auto"`** (and the alias `tokenizer=None`): both converters read the source's model id (`eval_log.eval.model` for Inspect, `record["model_id"]` / aggregate fallback for evaleval), strip routing prefixes (`together/`, `hf/`, `bedrock/`, …) via the `_KNOWN_PROVIDERS` allowlist, and call `AutoTokenizer.from_pretrained(canonical_id, trust_remote_code=True)`. Raises `ValueError` if the id is missing or unloadable (closed-source ids like `openai/gpt-4` will hit this — the error message tells the user to pass an HF stand-in).
- **Explicit string / instance**: still loaded / used, but compared against the source's model id via the same prefix-stripping comparison; on mismatch a `UserWarning` is emitted (`"Tokenizer mismatch: …"`). The mismatch may be intentional (forcing one model's tokens through another's tokenizer for cross-model studies); the warning just keeps users informed.
- The shared resolver lives in `inif/converters/_tokenize.py::resolve_tokenizer` and is imported by both `from_eval_log` (Inspect) and `from_instance_records` (evaleval).

## Inspect AI converter conventions

- **`tag_generated=True`** (default): tokens belonging to the LAST assistant message are annotated `"generated"`. Identification uses character-span matching against `apply_chat_template` output, so it works for any HuggingFace chat template.
- **`tag_reasoning=True`** (default): runs the shared `tag_template_field_renderings` helper, which detects each per-message structured field (`reasoning`, `tool_calls`) by rendering the chat template twice — once with the field, once without — and using the diff to locate the chars the template emitted FOR that field. Tokens in the diff window get the field's annotation (`reasoning` / `tool_call`) AND the message's role (`assistant`), moved out of `template` if `tag_chat_roles` had labelled them there. Fully model-agnostic: the chat template itself decides where each field renders. `_extract_message_dicts` separates reasoning from `content` (passed via `reasoning_content` + `reasoning`), propagates `tool_calls` (OpenAI/Kimi shape) and `tool_call_id`/`name`, and `render_chat_template` always passes `preserve_thinking=True` so chat templates that strip reasoning from history (Kimi) preserve it for analysis. The chat-template kwarg is silently ignored by templates that don't recognize it.
- **`extract_logprobs=True`** (default): per-token logprobs from `inspect_sample.output.choices[0].logprobs.content` are attached to the response tokens via `set_extra("logprob", ...)`. Best-effort: skipped silently when the eval-source tokenization disagrees with our tokenizer on token count.
- **`InifDocument.filter_samples_by_score(scorer, predicate)`** returns `list[Sample]` (compose with the per-sample selection methods like `Sample.select_by_annotation`).

## Evaleval converter conventions

- **Schema**: targets `instance_level_eval_0.2.2` from `evaleval/every_eval_ever`. Trusted as-is — users who want jsonschema validation should run it themselves against the upstream schema before calling the converter.
- **Three entry points** in `inif/converters/evaleval.py`: `from_instance_records(records, aggregate=None, ...)` (core), `from_eval_json(aggregate_path, instances_path, ...)` (local JSON / JSONL), `from_hf_dataset(config, split="samples", aggregate_config=None, limit=None, ...)` (streams rows from `evaleval/EEE_datastore`).
- **Interaction types**: `single_turn` synthesises `user` + `assistant` messages; `multi_turn` / `agentic` use the `messages[]` array directly, ordered by `turn_idx`. `tool_calls` are rendered into assistant content as `<tool_call name=… args={…}/>` so the invocations survive tokenization.
- **Reasoning traces**: when `tag_reasoning=True` (default), each turn's `reasoning_trace` is routed through the chat template's native `reasoning_content` slot (NOT prepended to `content`), and the shared `tag_template_field_renderings` helper detects the rendered chars via render-with-vs-without diff. Tokens get `reasoning` + the message's role (moved out of `template`). Same helper also tags `tool_calls` rendering as `tool_call` + role. Best-effort: silently skipped when the tokenizer's per-token decode doesn't round-trip to `apply_chat_template`.
- **Score**: `SampleScore` is built from `evaluation.score` (`scorer = evaluation_name`). The terminal `answer_attribution` entry feeds `SampleScore.answer`; non-terminal entries land in `Sample.metadata["intermediate_answers"]`.
- **Metadata mapping**: `token_usage.input_tokens`/`output_tokens` → `Sample.input_tokens`/`output_tokens`; `input.reference` → `Sample.references` (+ `Sample.target` when length 1); `input.choices` → `Sample.choices`; `interaction_type`, `error`, `sample_hash` → same-named first-class `Sample` fields. `performance`, `num_turns`, `tool_calls_count`, `reasoning_tokens`, per-sample EEE `metadata`, non-terminal `answer_attribution` entries → `Sample.metadata`. Aggregate record (when supplied) contributes `Metadata.extra["inference"]`, `["eval_library"]`, `["metric_config"]`, `["aggregate_score"]`.
- **Optional extra**: `pip install inif[evaleval]` pulls `datasets` (imported lazily inside `from_hf_dataset`).

## IO conventions

- **`InifDocument.save(path)`** (instance method) and **`InifDocument.load(path)`** (classmethod) are symmetric. Suffix-based detection: `.inif` ⇒ indexed archive; `.inif.json` / `.json` ⇒ plain JSON.
- **Unified read API in `inif.io`** (path-based, no instance): `iter_samples(path)` (streams `Sample`s), `read_samples(path, sample_ids)` (single id or iterable; returns `list[Sample]` in requested order), `read_info(path)` (returns a `DocumentInfo` with `metadata` plus per-sample summary dicts; never inflates sequences). All three accept both `.inif` and `.inif.json` paths, dispatching to the indexed-archive readers or to the full document loader as appropriate. Format-specific helpers (`iter_indexed_samples`, `read_indexed_*`) are internal — public callers should use the unified API or `InifDocument.load`.
