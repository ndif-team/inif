from __future__ import annotations

from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    Text,
    TokenOrSeqRef,
)


def test_render_html_basic(doc):
    html = doc.render_html()
    assert "<!DOCTYPE html>" in html
    assert "gpt2" in html
    assert "sample_0" in html


def test_render_html_tokens(doc):
    html = doc.render_html()
    assert " a" in html
    assert " test" in html
    assert "." in html


def test_render_html_metadata(doc):
    html = doc.render_html()
    assert "inspect_ai" in html
    assert "mmlu" in html


def test_render_html_scores(doc):
    html = doc.render_html()
    assert "accuracy" in html
    assert "1.0" in html
    assert "f1" in html
    assert "0.8" in html


def test_render_html_spans(doc):
    html = doc.render_html()
    assert "answer" in html


def test_render_html_sequence_refs(doc):
    html = doc.render_html()
    assert "seq-ref" in html


def test_render_html_compact(doc):
    html = doc.render_html(compact=True)
    assert "<!DOCTYPE html>" in html
    assert "gpt2" in html


def test_render_html_title(doc):
    html = doc.render_html(title="My Custom Title")
    assert "My Custom Title" in html


def test_render_html_extra_fields(doc):
    html = doc.render_html()
    # logprob is an extra field on one of the tokens
    assert "logprob" in html


def test_save_html(doc, tmp_path):
    out = tmp_path / "test.html"
    doc.save_html(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content


def test_save_html_source_as_title(doc, tmp_path):
    """When no title is given, the source filename is used as the title."""
    out = tmp_path / "test.html"
    doc.save_html(out, source="/data/my_analysis.inif.json")
    content = out.read_text(encoding="utf-8")
    assert "my_analysis.inif.json" in content
    # model name should not be the title
    assert "<h1>inif: gpt2</h1>" not in content


def test_save_html_title_overrides_source(doc, tmp_path):
    """Explicit title takes precedence over source filename."""
    out = tmp_path / "test.html"
    doc.save_html(out, title="Custom", source="/data/file.inif.json")
    content = out.read_text(encoding="utf-8")
    assert "Custom" in content
    assert "file.inif.json" not in content


def test_render_html_minimal():
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="minimal")))
    html = doc.render_html()
    assert "<!DOCTYPE html>" in html
    assert "No samples" in html


def test_render_html_multiple_samples():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="multi")),
        samples=[
            Sample(id="s0", tokens=[TokenOrSeqRef(id=1, token="hello")]),
            Sample(id="s1", tokens=[TokenOrSeqRef(id=2, token="world")]),
        ],
    )
    html = doc.render_html()
    # Sidebar with sample items instead of tabs
    assert "inif-sample-item" in html
    assert "inif-sidebar" in html
    assert "s0" in html
    assert "s1" in html


def test_render_html_xss_safety():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="<script>alert(1)</script>")),
        samples=[
            Sample(
                id="xss",
                tokens=[TokenOrSeqRef(id=1, token="<script>alert('xss')</script>")],
            ),
        ],
    )
    html = doc.render_html()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_save_html_from_indexed_inif(doc, tmp_path):
    """View command round-trip: save as indexed .inif, load, render HTML."""
    inif_path = tmp_path / "test.inif"
    doc.save(inif_path)

    reloaded = InifDocument.load(inif_path)
    out = tmp_path / "from_inif.html"
    reloaded.save_html(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content
    assert "sample_0" in content


def test_save_html_from_plain_json(doc, tmp_path):
    """View command round-trip: save as .inif.json, load, render HTML."""
    json_path = tmp_path / "test.inif.json"
    doc.save(json_path)

    reloaded = InifDocument.load(json_path)
    out = tmp_path / "from_json.html"
    reloaded.save_html(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content


# --- Sidebar tests ---


def test_render_html_sidebar_layout(doc):
    """Sidebar layout structure present for docs with samples."""
    html = doc.render_html()
    assert "inif-layout" in html
    assert "inif-sidebar" in html
    assert "inif-main" in html
    assert "inif-sidebar-header" in html
    assert "inif-sidebar-content" in html
    assert "inif-sample-list" in html


def test_render_html_sidebar_collapse_toggle(doc):
    """Sidebar has collapse toggle button."""
    html = doc.render_html()
    assert "inif-sidebar-toggle" in html
    # Contains the left-pointing arrow ‹
    assert "‹" in html


def test_render_html_sidebar_metadata_inside(doc):
    """Metadata is rendered inside the sidebar."""
    html = doc.render_html()
    # Metadata should be inside sidebar-content div (not the CSS)
    div_marker = '<div class="inif-sidebar-content">'
    sidebar_start = html.index(div_marker)
    sidebar_end = html.index("inif-sample-list", sidebar_start)
    sidebar_section = html[sidebar_start:sidebar_end]
    assert "Metadata" in sidebar_section
    assert "gpt2" in sidebar_section


def test_render_html_no_old_tabs(doc):
    """Old tab-based elements should not be present."""
    html = doc.render_html()
    assert "inif-sample-tab" not in html
    assert "inif-sample-tabs" not in html


# --- Exact match indicator tests ---


def test_render_html_exact_match_pass():
    """Sample with exact_match=1.0 shows check mark."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=1.0)],
            ),
        ],
    )
    html = doc.render_html()
    assert "inif-em-pass" in html
    assert "✓" in html


def test_render_html_exact_match_fail():
    """Sample with exact_match=0.0 shows X mark."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=0.0)],
            ),
        ],
    )
    html = doc.render_html()
    assert "inif-em-fail" in html
    assert "✗" in html


def test_render_html_exact_match_other_value():
    """Sample with exact_match != 0 or 1 shows no indicator."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=0.5)],
            ),
        ],
    )
    html = doc.render_html()
    # Check marks/X marks should not appear in the sample list
    assert "✓" not in html
    assert "✗" not in html


def test_render_html_other_numeric_scorer_also_shows_indicator():
    """Any scorer whose value is 0.0 or 1.0 drives the sidebar indicator — not
    just exact_match. Covers Inspect scorers like ``accuracy``/``choice`` and
    any custom scorer that emits binary results."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[SampleScore(scorer="accuracy", value=1.0)],
            ),
        ],
    )
    html = doc.render_html()
    assert "inif-em-pass" in html


def test_render_html_is_correct_metadata_drives_indicator():
    """Scorers that carry explicit ``metadata.is_correct`` (as produced by the
    evaleval converter) take precedence, so EEE samples get check/cross marks
    even though their scorer name matches the benchmark, not ``exact_match``."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[
                    SampleScore(
                        scorer="theory_of_mind",
                        value=0.5,  # non-binary
                        metadata={"is_correct": False},
                    )
                ],
            ),
        ],
    )
    html = doc.render_html()
    assert "inif-em-fail" in html


def test_render_html_non_binary_value_shows_no_indicator():
    """A purely continuous score with no is_correct metadata stays blank."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="hi")],
                scores=[SampleScore(scorer="bleu", value=0.72)],
            ),
        ],
    )
    html = doc.render_html()
    assert "✓" not in html
    assert "✗" not in html


# --- Annotation highlight tests ---


def _set_extra(tok, key, value):
    tok.set_extra(key, value)


def _make_annotation_doc():
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="annotation_test")),
        samples=[
            Sample(
                id="a0",
                tokens=[
                    TokenOrSeqRef(id=1, token="System:"),
                    TokenOrSeqRef(id=2, token="Hello"),
                    TokenOrSeqRef(id=3, token="Hi"),
                ],
                annotations=[
                    {"name": "system", "ranges": [(0, 1)]},
                    {"name": "user", "ranges": [(1, 2)]},
                    {"name": "assistant", "ranges": [(2, 3)]},
                ],
            ),
        ],
    )


def test_render_html_annotation_highlight_data_attrs():
    """Annotated tokens get annotation data attributes."""
    doc = _make_annotation_doc()
    html = doc.render_html()
    assert 'data-annotations="system"' in html
    assert 'data-annotations="user"' in html
    assert 'data-annotations="assistant"' in html
    assert "data-annotation-bg=" in html


def test_render_html_role_annotation_highlight_colors():
    """Common chat-role annotations get stable background colors."""
    doc = _make_annotation_doc()
    html = doc.render_html()
    assert "#d4e6f1" in html  # system
    assert "#d5f5e3" in html  # user
    assert "#fdebd0" in html  # assistant


def test_render_html_has_annotations_attr():
    """Token strip gets data-has-annotations attribute."""
    doc = _make_annotation_doc()
    html = doc.render_html()
    assert 'data-has-annotations="true"' in html


def test_render_html_no_annotations_attr():
    """Token strip without annotations gets data-has-annotations=false."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_annotation")),
        samples=[
            Sample(id="nr0", tokens=[TokenOrSeqRef(id=1, token="plain")]),
        ],
    )
    html = doc.render_html()
    assert 'data-has-annotations="false"' in html


def test_render_html_unknown_annotation():
    """Unknown annotations still get highlight data."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="fallback")),
        samples=[
            Sample(
                id="fb0",
                tokens=[TokenOrSeqRef(id=1, token="x")],
                annotations=[{"name": "custom_annotation", "ranges": [(0, 1)]}],
            )
        ],
    )
    html = doc.render_html()
    assert 'data-annotations="custom_annotation"' in html
    assert "data-annotation-bg=" in html


# --- Control panel tests ---


def test_render_html_control_panel_with_annotations():
    """Control panel checkbox is checked when sample has annotations."""
    doc = _make_annotation_doc()
    html = doc.render_html()
    assert "inif-control-panel" in html
    assert "inif-annotation-toggle" in html
    assert "checked" in html
    # Should NOT be disabled
    assert 'class="inif-annotation-toggle" checked' in html


def test_render_html_control_panel_no_annotations():
    """Control panel checkbox is disabled when sample has no annotations."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_annotation")),
        samples=[
            Sample(id="nr0", tokens=[TokenOrSeqRef(id=1, token="plain")]),
        ],
    )
    html = doc.render_html()
    assert "inif-control-panel" in html
    assert "disabled" in html
    assert 'label class="disabled"' in html


def test_render_html_annotation_legend_in_control_panel():
    """Annotation legend with swatches appears inside the control panel."""
    doc = _make_annotation_doc()
    html = doc.render_html()
    assert "inif-annotation-legend" in html
    assert "inif-annotation-swatch" in html
    # Annotation names appear in the legend
    cp_start = html.index('<div class="inif-control-panel">')
    cp_end = html.index("</div>", html.index("inif-annotation-legend", cp_start))
    cp_section = html[cp_start:cp_end]
    assert "system" in cp_section
    assert "user" in cp_section
    assert "assistant" in cp_section


def test_render_html_no_annotation_legend_without_annotations():
    """No annotation legend when sample has no annotations."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_annotation")),
        samples=[
            Sample(id="nr0", tokens=[TokenOrSeqRef(id=1, token="plain")]),
        ],
    )
    html = doc.render_html()
    assert '<div class="inif-annotation-legend">' not in html


# --- Sample header tests ---


def test_render_html_sample_header_stats(doc):
    """Sample header contains stats (target, tokens)."""
    html = doc.render_html()
    assert "inif-sample-header" in html
    assert "inif-sample-stats" in html
    assert "Target" in html
    assert "Input tokens" in html
    assert "Output tokens" in html


def test_render_html_annotation_bg_data_attr():
    """Token with annotation gets data-annotation-bg attribute."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="annotation_test")),
        samples=[
            Sample(
                id="t0",
                tokens=[TokenOrSeqRef(id=1, token="tagged")],
                annotations=[{"name": "my_annotation", "ranges": [(0, 1)]}],
            )
        ],
    )
    html = doc.render_html()
    assert "data-annotation-bg=" in html


def test_render_html_multiple_annotations_listed():
    """When a token has multiple annotations, all names are exposed and the
    higher-priority one (user-defined > auto sub-text > chat role) drives
    the background color."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="multi_annotation")),
        samples=[
            Sample(
                id="p0",
                tokens=[TokenOrSeqRef(id=1, token="both")],
                annotations=[
                    {"name": "user", "ranges": [(0, 1)]},
                    {"name": "some_annotation", "ranges": [(0, 1)]},
                ],
            )
        ],
    )
    html = doc.render_html()
    assert 'data-annotations="some_annotation,user"' in html


def test_render_html_annotation_priority_ordering():
    """Chat-role tags (assistant/template/...) yield to auto sub-text tags
    (reasoning/tool_call), which in turn yield to user-defined tags."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="priority")),
        samples=[
            Sample(
                id="p0",
                tokens=[
                    TokenOrSeqRef(id=1, token="a"),
                    TokenOrSeqRef(id=2, token="b"),
                    TokenOrSeqRef(id=3, token="c"),
                ],
                annotations=[
                    # Token 0: chat-role only
                    {"name": "assistant", "ranges": [(0, 3)]},
                    # Token 1: chat-role + auto sub-text → reasoning wins
                    {"name": "reasoning", "ranges": [(1, 3)]},
                    # Token 2: chat-role + auto sub-text + user-defined →
                    # user-defined wins
                    {"name": "my_probe", "ranges": [(2, 3)]},
                ],
            )
        ],
    )
    html = doc.render_html()
    assert 'data-annotations="assistant"' in html
    assert 'data-annotations="reasoning,assistant"' in html
    assert 'data-annotations="my_probe,reasoning,assistant"' in html


# --- Extra field underline tests ---


def test_render_html_extra_field_underline_single():
    """Token with one extra field gets a box-shadow underline."""
    tok = TokenOrSeqRef(id=1, token="x")
    _set_extra(tok, "logit_lens", {"layer_5": "."})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = doc.render_html()
    assert "box-shadow:" in html
    assert "inset 0 -2px 0 0" in html


def test_render_html_extra_field_underline_two_fields():
    """Token with two extra fields gets stacked underlines."""
    tok = TokenOrSeqRef(id=1, token="x")
    _set_extra(tok, "logit_lens", {"layer_5": "."})
    _set_extra(tok, "probe", {"acc": 0.9})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = doc.render_html()
    assert "box-shadow:" in html
    # Two inset shadows stacked
    assert "inset 0 -2px 0 0" in html
    assert "inset 0 -4px 0 0" in html


def test_render_html_extra_field_different_tokens():
    """Two tokens with different extra fields get different colors."""
    tok_a = TokenOrSeqRef(id=1, token="a")
    _set_extra(tok_a, "logit_lens", {})
    tok_b = TokenOrSeqRef(id=2, token="b")
    _set_extra(tok_b, "probe", {})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok_a, tok_b])],
    )
    html = doc.render_html()
    # Both tokens get underlines
    assert html.count("box-shadow:") == 2


def test_render_html_extras_legend_in_control_panel():
    """Extra fields legend appears inside the control panel."""
    tok = TokenOrSeqRef(id=1, token="x")
    _set_extra(tok, "logit_lens", {})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = doc.render_html()
    # Legend elements present
    assert "inif-extras-legend" in html
    assert "inif-extra-swatch" in html
    # Legend is inside the control panel
    cp_start = html.index('<div class="inif-control-panel">')
    cp_end = html.index("</div>\n</div>\n</div>", cp_start)
    cp_section = html[cp_start:cp_end]
    assert "logit_lens" in cp_section
    assert "inif-extras-legend" in cp_section


def test_render_html_no_extras_legend_plain_token():
    """No extras legend when tokens have no extra fields."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="plain")),
        samples=[
            Sample(id="p0", tokens=[TokenOrSeqRef(id=1, token="hi")]),
        ],
    )
    html = doc.render_html()
    assert '<div class="inif-extras-legend">' not in html


def test_render_html_annotations_not_underlined():
    """Sample annotations do not produce extra-field underlines."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="skip")),
        samples=[
            Sample(
                id="s0",
                tokens=[TokenOrSeqRef(id=1, token="x")],
                annotations=[{"name": "user", "ranges": [(0, 1)]}],
            )
        ],
    )
    html = doc.render_html()
    assert "box-shadow:" not in html
    assert '<div class="inif-extras-legend">' not in html


def test_render_html_extra_padding_multiple():
    """Multiple extra fields add padding-bottom for stacked lines."""
    tok = TokenOrSeqRef(id=1, token="x")
    _set_extra(tok, "field_a", 1)
    _set_extra(tok, "field_b", 2)
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="pad")),
        samples=[Sample(id="p0", tokens=[tok])],
    )
    html = doc.render_html()
    assert "padding-bottom:4px" in html


# --- Sidebar collapse CSS tests ---


def test_render_html_sidebar_collapsed_hides_text(doc):
    """CSS hides the header text span when sidebar is collapsed."""
    html = doc.render_html()
    assert ".inif-sidebar.collapsed .inif-sidebar-header > span" in html
    assert "display: none" in html


def test_render_html_sidebar_collapsed_overflow(doc):
    """CSS sets overflow hidden on collapsed sidebar."""
    html = doc.render_html()
    # Find the collapsed rule in CSS
    css_start = html.index("<style>")
    css_end = html.index("</style>")
    css = html[css_start:css_end]
    assert ".inif-sidebar.collapsed" in css
    assert "overflow: hidden" in css


# --- Sequence ref expansion tests ---


def test_render_html_seq_ref_expanded_into_individual_tokens():
    """Sequence refs are expanded into individual token spans."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="seq_exp")),
        sequences=[
            Sequence(
                id="s0",
                n_tokens=3,
                tokens=[
                    TokenOrSeqRef(id=1, token="Hello"),
                    TokenOrSeqRef(id=2, token=" world"),
                    TokenOrSeqRef(id=3, token="!"),
                ],
            ),
        ],
        samples=[
            Sample(
                id="x0",
                tokens=[
                    TokenOrSeqRef(id=None, token="s0"),
                    TokenOrSeqRef(id=1, token=" end"),
                ],
            ),
        ],
    )
    html = doc.render_html()
    # Each sub-token from sequence should be a separate seq-ref span
    assert html.count("seq-ref") >= 3
    # Individual tokens appear separately (not concatenated)
    assert "Hello" in html
    assert "world" in html  # " world" after escaping leading space
    assert "!" in html
    assert "·end" in html


def test_render_html_seq_ref_expanded_uses_real_token_ids():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="seq_ids")),
        sequences=[
            Sequence(
                id="s0",
                n_tokens=2,
                tokens=[
                    TokenOrSeqRef(id=11, token="a"),
                    TokenOrSeqRef(id=12, token="b"),
                ],
            ),
        ],
        samples=[
            Sample(
                id="x0",
                tokens=[TokenOrSeqRef(id=None, token="s0")],
            ),
        ],
    )

    html = doc.render_html()

    assert html.count('class="inif-token seq-ref"') == 2
    assert 'data-token-id="11"' in html
    assert 'data-token-id="12"' in html
    assert 'data-token-id="-1"' not in html


def test_render_html_seq_ref_wraps_like_normal_tokens():
    """Expanded seq-ref tokens are inline-block spans that can wrap."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="wrap")),
        sequences=[
            Sequence(
                id="s0",
                n_tokens=2,
                tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            ),
        ],
        samples=[
            Sample(
                id="w0",
                tokens=[TokenOrSeqRef(id=None, token="s0")],
            ),
        ],
    )
    html = doc.render_html()
    # Two separate seq-ref spans, not one big one
    count = html.count('class="inif-token seq-ref"')
    assert count == 2


def test_render_html_seq_ref_fallback_missing_seq():
    """Seq ref with unknown sequence_id shows placeholder."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="miss")),
        samples=[
            Sample(
                id="m0",
                tokens=[TokenOrSeqRef(id=None, token="nonexistent")],
            ),
        ],
    )
    html = doc.render_html()
    assert "seq-ref" in html
    assert "[nonexistent]" in html


def test_render_html_seq_ref_shares_position():
    """Expanded seq-ref sub-tokens share the original position."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="pos")),
        sequences=[
            Sequence(
                id="s0",
                n_tokens=2,
                tokens=[TokenOrSeqRef(id=1, token="a"), TokenOrSeqRef(id=2, token="b")],
            ),
        ],
        samples=[
            Sample(
                id="p0",
                tokens=[
                    TokenOrSeqRef(id=1, token="before"),
                    TokenOrSeqRef(id=None, token="s0"),
                ],
            ),
        ],
    )
    html = doc.render_html()
    # "before" is at position 0, both seq-ref tokens at position 1
    assert 'data-position="0"' in html
    assert 'data-position="1"' in html


# --- Newline handling tests ---


def test_render_html_newline_shown_as_symbol():
    """Newline chars in tokens are displayed as the ↵ symbol."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="nl")),
        samples=[
            Sample(
                id="n0",
                tokens=[
                    TokenOrSeqRef(id=1, token="hello"),
                    TokenOrSeqRef(id=2, token="\n"),
                    TokenOrSeqRef(id=3, token="world"),
                ],
            ),
        ],
    )
    html = doc.render_html()
    assert "↵" in html  # ↵ symbol


def test_render_html_newline_inserts_line_break():
    """A flex line-break div is inserted after tokens containing newlines."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="nl_br")),
        samples=[
            Sample(
                id="n1",
                tokens=[
                    TokenOrSeqRef(id=1, token="line1"),
                    TokenOrSeqRef(id=2, token="\n"),
                    TokenOrSeqRef(id=3, token="line2"),
                ],
            ),
        ],
    )
    html = doc.render_html()
    # The newline token span should be followed by a line-break div
    idx_newline = html.index("↵")
    after_newline = html[idx_newline:]
    span_end = after_newline.index("</span>")
    after_span = after_newline[span_end + len("</span>") :].lstrip()
    assert after_span.startswith('<div class="inif-line-break">')


def test_render_html_newline_in_seq_ref_inserts_line_break():
    """Newlines inside expanded seq-ref tokens also insert line break."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="nl_seq")),
        sequences=[
            Sequence(
                id="s0",
                n_tokens=2,
                tokens=[
                    TokenOrSeqRef(id=1, token="first\n"),
                    TokenOrSeqRef(id=2, token="second"),
                ],
            ),
        ],
        samples=[
            Sample(
                id="n2",
                tokens=[TokenOrSeqRef(id=None, token="s0")],
            ),
        ],
    )
    html = doc.render_html()
    assert "↵" in html
    idx_sym = html.index("↵")
    after = html[idx_sym:]
    span_end = after.index("</span>")
    rest = after[span_end + len("</span>") :].lstrip()
    assert rest.startswith('<div class="inif-line-break">')


def test_render_html_no_line_break_without_newline():
    """Tokens without newlines do not get a line-break div."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_nl")),
        samples=[
            Sample(
                id="n3",
                tokens=[
                    TokenOrSeqRef(id=1, token="hello"),
                    TokenOrSeqRef(id=2, token=" world"),
                ],
            ),
        ],
    )
    html = doc.render_html()
    assert "inif-line-break" not in html.split("</style>")[-1]


# --- Per-message panel tests (markdown body + token toggle) ---


def _make_messages_doc():
    """A sample whose texts carry start/end so the new viewer renders
    per-message panels with markdown bodies and per-message token strips."""
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="msg_panels")),
        samples=[
            Sample(
                id="m0",
                tokens=[
                    TokenOrSeqRef(id=1, token="<sys>"),
                    TokenOrSeqRef(id=2, token="hi"),
                    TokenOrSeqRef(id=3, token="!"),
                    TokenOrSeqRef(id=4, token="<usr>"),
                    TokenOrSeqRef(id=5, token="ok"),
                ],
                texts=[
                    Text(name="system_0", value="Hello, **world**!", start=0, end=3),
                    Text(name="user_0", value="ok", start=3, end=5),
                ],
            ),
        ],
    )


def test_render_html_message_panels_present():
    """Texts with offsets render as one ``inif-message`` per Text."""
    html = _make_messages_doc().render_html()
    assert '<div class="inif-messages">' in html
    # Two messages.
    assert html.count('class="inif-message"') == 2
    # Role-derived data attribute is set so CSS can color-code message headers.
    assert 'data-role="system"' in html
    assert 'data-role="user"' in html


def test_render_html_message_panels_have_toggle_and_md_source():
    """Each message has an eye toggle button and a hidden raw-markdown source
    block that the JS markdown renderer reads via ``textContent``."""
    html = _make_messages_doc().render_html()
    assert html.count("inif-message-toggle") >= 2
    assert 'class="inif-message-md-source"' in html
    # Markdown is left raw (HTML-escaped) inside the source block; the JS
    # renderer wraps it for display.
    assert "Hello, **world**!" in html
    # Rendered container is present (initially empty, populated client-side).
    assert 'class="inif-message-md-rendered"' in html


def test_render_html_message_panels_token_strip_scoped():
    """The hidden ``inif-message-tokens`` block contains only that message's
    tokens — `<sys>` belongs to the first panel; `<usr>` to the second."""
    html = _make_messages_doc().render_html()
    # Skip the CSS so role-selector strings (``[data-role="system"]``) don't
    # confuse the text search; only the per-sample panel HTML matters here.
    body = html.split("</style>", 1)[-1]
    first_msg_start = body.index('<div class="inif-message" data-role="system"')
    second_msg_start = body.index('<div class="inif-message" data-role="user"')
    first_msg = body[first_msg_start:second_msg_start]
    second_msg = body[second_msg_start:]
    # `<sys>` (escaped) appears in message 0 only.
    assert "&lt;sys&gt;" in first_msg
    assert "&lt;sys&gt;" not in second_msg
    # `<usr>` appears in message 1 only.
    assert "&lt;usr&gt;" in second_msg
    assert "&lt;usr&gt;" not in first_msg


def test_render_html_messages_panel_skipped_when_no_offsets():
    """A document whose texts lack start/end falls back to the legacy
    full-document token strip + ``Texts`` list."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="legacy")),
        samples=[
            Sample(
                id="l0",
                tokens=[TokenOrSeqRef(id=1, token="x")],
                texts=[Text(name="text_0", value="x")],  # no start/end
            ),
        ],
    )
    html = doc.render_html()
    assert '<div class="inif-messages">' not in html
    # Legacy "Texts" panel is shown as a fallback for old archives.
    assert '<div class="inif-texts-panel">' in html


def test_render_html_no_legacy_texts_panel_when_messages_render():
    """When per-message panels are rendered, the legacy ``Texts`` block is
    suppressed to avoid showing the same content twice."""
    html = _make_messages_doc().render_html()
    assert '<div class="inif-texts-panel">' not in html


def test_render_html_message_panels_show_offsets_in_meta():
    """The message header shows the ``[start, end)`` offsets so users can
    cross-reference token positions in the toggled token view."""
    html = _make_messages_doc().render_html()
    assert "[0, 3)" in html
    assert "[3, 5)" in html


def test_render_html_message_md_collapse_assets_present():
    """The CSS clip rule and the JS that mounts a "Show full text" overlay
    button on overflowing messages both ship with every render — actual
    overflow detection happens client-side, since text height depends on
    the rendered viewport."""
    html = _make_messages_doc().render_html()
    # Clip rule for the 8-line cap.
    assert ".inif-message-md.collapsible .inif-message-md-rendered" in html
    # The expand button is created by JS, not server-side, so its label
    # only appears in the inlined script.
    assert "Show full text" in html
    # The expanded class disables the clip when the user reveals the rest.
    assert ".inif-message-md.collapsible.expanded" in html
    # The token-view body has no clipping (it must always show all tokens).
    assert "inif-message-tokens" in html
    assert ".inif-message-tokens.collapsible" not in html
