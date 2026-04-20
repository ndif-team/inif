# INterpretability Interchange Format (INIF)

*This package is in early development and the API may change without deprecation. Feedback and contributions are very welcome!*

The **INterpretability Interchange Format** (INIF) is a JSON-based format for tokenized LLM generation traces with support for tagging, position selection, and support for efficient storing interpretability outputs.

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
    ├── tokens[]      — token id + string, plus extra fields (tags, role, logprob, data, ...)
    ├── texts[]       — original message strings
    ├── spans[]       — named position ranges
    └── scores[]      — evaluation scores (scorer, value, answer)
```

**Token ID convention**: `id >= 0` is a vocabulary token, `id == -N` references a shared `Sequence` via `sequence_id`.

**Extensible tokens**: tags, logprobs, chat roles, and interpretability outputs (logit lens, probes, etc.) are stored as extra fields on each token.

## Key features

### Tagging

```python
from inif import tag_by_regex_all, tag_by_text_regex, create_span_from_tag

# Tag tokens matching a regex pattern
tag_by_regex_all(doc, r"^\d+$", "number")

# Tag by concatenated text (multi-token matches)
tag_by_text_regex(sample, r"Paris", "city")

# Convert tags to named spans
create_span_from_tag(sample, "city", "answer_span")
```

### Selection

```python
from inif import select_by_tag, select_by_span, select_by_position

selection = select_by_tag(sample, "number")
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
- Toggleable role and tag highlighting with color legends
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
