from inif.flat import FlatTokenStore
from inif.indexed import IndexedInifWriter
from inif.io import (
    DocumentInfo,
    iter_samples,
    read_info,
    read_samples,
)
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    SourceEval,
    Span,
    Text,
    TokenAnnotation,
    TokenExtras,
    TokenOrSeqRef,
)
from inif.schema import get_schema, validate, write_schema
from inif.selectors import TokenSelection
from inif.tagging import PredicateTag, RegexTag, TextTagMode, TokenPredicate

__all__ = [
    # Models
    "InifDocument",
    "Metadata",
    "ModelInfo",
    "Sample",
    "SampleScore",
    "Sequence",
    "SourceEval",
    "Span",
    "Text",
    "TokenAnnotation",
    "TokenExtras",
    "TokenOrSeqRef",
    "FlatTokenStore",
    # IO (path-based readers; load/save/from_dict/to_dict live on InifDocument)
    "DocumentInfo",
    "IndexedInifWriter",
    "iter_samples",
    "read_info",
    "read_samples",
    # Schema
    "get_schema",
    "validate",
    "write_schema",
    # Selection / tagging types (operations are methods on Sample / InifDocument)
    "PredicateTag",
    "RegexTag",
    "TextTagMode",
    "TokenPredicate",
    "TokenSelection",
]
