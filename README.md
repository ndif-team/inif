# INterpretability Interchange Format (INIF)

*This package is in early development and the API may change without deprecation. Feedback and contributions are very welcome!*

The **INterpretability Interchange Format** (INIF) is a JSON-based format for tokenized LLM generation traces with support for contiguous token annotations, position selection, and efficient storage of interpretability outputs.

Designed as the interchange layer between generation and evaluation frameworks (e.g. [Inspect AI](https://inspect.ai-safety-institute.org.uk/)) and interpretability tools in the NDIF ecosystem ([nnsight](https://github.com/ndif-team/nnsight), [nnterp](https://github.com/ndif-team/nnterp) and [workbench](https://github.com/ndif-team/workbench)).

## Installation

```bash
pip install inif
```

With [Inspect AI](https://inspect.aisi.org.uk/) converter support:

```bash
pip install "inif[inspect]"
```

## Quick start

### From text

```python
from inif import load, save
from inif.converters.text import from_texts

doc = from_texts(
    ["The capital of France is Paris.", "Hello world!"],
    tokenizer="gpt2",
)
save(doc, "traces.inif.json")
```

### From Inspect AI eval logs

```python
from inif.converters.inspect_ai import from_eval_file

doc = from_eval_file("logs/my_eval.eval")
```

### Viewing

```python
from inif import load, show, save_html

doc = load("traces.inif.json")
show(doc)  # in Jupyter
save_html(doc, "traces.html")  # self-contained HTML
```

## CLI

```bash
# Convert text files to inif
inif convert txt input.txt -m gpt2

# Convert Inspect AI eval logs
inif convert eval logs/my_eval.eval

# View as interactive HTML in the browser
inif view traces.inif.json
```

## Format overview

An `.inif.json` file contains:

```text
InifDocument
├── metadata          — model info, source eval, packages, timestamps
├── sequences[]       — deduplicated token patterns shared across samples
└── samples[]         — tokenized generation traces
    ├── tokens[]      — token id + string, plus sparse extras (logprob, logit lens, probes, ...)
    ├── annotations[] — named token ranges with optional metadata
    ├── texts[]       — original message strings
    ├── spans[]       — named position ranges
    └── scores[]      — evaluation scores (scorer, value, answer)
```

**Token ID convention**: `id >= 0` is a vocabulary token, `id == -N` references a shared `Sequence` via `sequence_id`.

**Annotations**: repeated labels such as chat roles, generated output, reasoning traces, and regex matches live in `Sample.annotations` as named half-open ranges. This avoids repeating `"role": "assistant"` or `"tags": [...]` on every token in a long contiguous region.

**Extensible tokens**: sparse per-token values such as logprobs and interpretability outputs (logit lens, probes, etc.) are stored as token extras.

Use `.inif.json` for plain JSON and `.inif` for the indexed archive format. The
archive keeps per-sample text previews in the manifest and stores each full
sample payload as a separate compressed member, so callers can browse summaries
without inflating token dictionaries:

```python
from inif import (
    IndexedInifWriter,
    iter_indexed_samples,
    read_indexed_header,
    read_indexed_sample,
    read_indexed_samples,
    read_indexed_sample_summaries,
    save,
)

save(doc, "traces.inif")                     # indexed archive
header = read_indexed_header("traces.inif")  # metadata + sequences only
summaries = read_indexed_sample_summaries("traces.inif")
sample = read_indexed_sample("traces.inif", "sample_42")
subset = read_indexed_samples("traces.inif", ["sample_1", "sample_7"])

for sample in iter_indexed_samples("traces.inif"):
    ...

with IndexedInifWriter("streaming.inif", doc.metadata, doc.sequences) as writer:
    for sample in doc.samples:
        writer.write_sample(sample)
        writer.flush()  # make the partial archive readable
```

The archive stores metadata and sequences once, then stores each sample as a
separate compressed member with an uncompressed preview summary. This supports
incremental writes, header-only reads, per-sample random access, and streaming
iteration while preserving the same `InifDocument` model.

## Key features

### Annotation

```python
from inif import tag_by_regex_all, tag_by_text_regex, create_span_from_tag

# Annotate tokens matching a regex pattern
tag_by_regex_all(doc, r"^\d+$", "number")

# Annotate by concatenated text (multi-token matches)
tag_by_text_regex(sample, r"Paris", "city")

# Convert annotations to named spans
create_span_from_tag(sample, "city", "answer_span")
```

### Selection

```python
from inif import select_by_annotation, select_by_span, select_by_position

selection = select_by_annotation(sample, "number")
selection = select_by_span(sample, "answer_span")
selection = select_by_position(sample, slice(5, 10))
```

### Sequence deduplication

Common token sequences across samples (e.g. shared system prompts) are automatically deduplicated via set-intersection and stored as `Sequence` objects referenced by tokens.

```python
from inif import deduplicate_sequences, expand_sequences

deduplicate_sequences(doc, min_length=3)  # compress
expand_sequences(doc)                     # flatten back
```

### Interactive HTML viewer

`save_html` / `show` produce a self-contained HTML page with:

- Collapsible sidebar with sample list and pass/fail indicators
- Token-level display with hover tooltips showing all extra fields
- Toggleable annotation highlighting with color legends
- Span border annotations and extra-field underline indicators
- Newline-aware token wrapping

## Development

```bash
make dev              # install dev environment
make test             # run tests
make format           # ruff format
make lint             # ruff check
make typecheck        # ty check
make schema           # regenerate JSON schema
```

## License

MIT
