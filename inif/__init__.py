from inif.flat import FlatTokenStore
from inif.indexed import IndexedInifWriter
from inif.io import (
    DocumentInfo,
    from_dict,
    iter_samples,
    load,
    read_info,
    read_samples,
    save,
    to_dict,
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
from inif.selectors import (
    TokenSelection,
    filter_samples_by_score,
    select_by_annotation,
    select_by_position,
    select_by_sequence_id,
    select_by_span,
)
from inif.sequences import deduplicate_sequences, expand_sequences
from inif.tagging import (
    TextTagMode,
    create_span_from_tag,
    remove_tag,
    remove_tag_all,
    tag_by_predicate,
    tag_by_predicates,
    tag_by_predicates_all,
    tag_by_regex,
    tag_by_regex_all,
    tag_by_regexes,
    tag_by_regexes_all,
    tag_by_text_regex,
    tag_by_text_regex_all,
    tag_chat_roles,
    tag_chat_roles_doc,
    tag_positions,
    tag_special_tokens,
)
from inif.viewer import render_html, save_html, show

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
    # IO
    "DocumentInfo",
    "from_dict",
    "IndexedInifWriter",
    "iter_samples",
    "load",
    "read_info",
    "read_samples",
    "save",
    "to_dict",
    # Schema
    "get_schema",
    "validate",
    "write_schema",
    # Selectors
    "TokenSelection",
    "filter_samples_by_score",
    "select_by_annotation",
    "select_by_position",
    "select_by_sequence_id",
    "select_by_span",
    # Tagging
    "TextTagMode",
    "create_span_from_tag",
    "remove_tag",
    "remove_tag_all",
    "tag_by_predicate",
    "tag_by_predicates",
    "tag_by_predicates_all",
    "tag_by_regex",
    "tag_by_regex_all",
    "tag_by_regexes",
    "tag_by_regexes_all",
    "tag_by_text_regex",
    "tag_by_text_regex_all",
    "tag_chat_roles",
    "tag_chat_roles_doc",
    "tag_positions",
    "tag_special_tokens",
    # Sequences
    "deduplicate_sequences",
    "expand_sequences",
    # Viewer
    "render_html",
    "save_html",
    "show",
]
