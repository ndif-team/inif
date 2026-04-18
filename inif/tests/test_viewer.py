from __future__ import annotations

from inif.io import load, save
from inif.models import (
    InifDocument,
    Metadata,
    ModelInfo,
    Sample,
    SampleScore,
    Sequence,
    Token,
)
from inif.viewer import render_html, save_html


def test_render_html_basic(doc):
    html = render_html(doc)
    assert "<!DOCTYPE html>" in html
    assert "gpt2" in html
    assert "sample_0" in html


def test_render_html_tokens(doc):
    html = render_html(doc)
    assert " a" in html
    assert " test" in html
    assert "." in html


def test_render_html_metadata(doc):
    html = render_html(doc)
    assert "inspect_ai" in html
    assert "mmlu" in html


def test_render_html_scores(doc):
    html = render_html(doc)
    assert "accuracy" in html
    assert "1.0" in html
    assert "f1" in html
    assert "0.8" in html


def test_render_html_spans(doc):
    html = render_html(doc)
    assert "answer" in html


def test_render_html_sequence_refs(doc):
    html = render_html(doc)
    assert "seq-ref" in html


def test_render_html_compact(doc):
    html = render_html(doc, compact=True)
    assert "<!DOCTYPE html>" in html
    assert "gpt2" in html


def test_render_html_title(doc):
    html = render_html(doc, title="My Custom Title")
    assert "My Custom Title" in html


def test_render_html_extra_fields(doc):
    html = render_html(doc)
    # logprob is an extra field on one of the tokens
    assert "logprob" in html


def test_save_html(doc, tmp_path):
    out = tmp_path / "test.html"
    save_html(doc, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content


def test_save_html_source_as_title(doc, tmp_path):
    """When no title is given, the source filename is used as the title."""
    out = tmp_path / "test.html"
    save_html(doc, out, source="/data/my_analysis.inif.json")
    content = out.read_text(encoding="utf-8")
    assert "my_analysis.inif.json" in content
    # model name should not be the title
    assert "<h1>inif: gpt2</h1>" not in content


def test_save_html_title_overrides_source(doc, tmp_path):
    """Explicit title takes precedence over source filename."""
    out = tmp_path / "test.html"
    save_html(doc, out, title="Custom", source="/data/file.inif.json")
    content = out.read_text(encoding="utf-8")
    assert "Custom" in content
    assert "file.inif.json" not in content


def test_render_html_minimal():
    doc = InifDocument(metadata=Metadata(model=ModelInfo(name="minimal")))
    html = render_html(doc)
    assert "<!DOCTYPE html>" in html
    assert "No samples" in html


def test_render_html_multiple_samples():
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="multi")),
        samples=[
            Sample(id="s0", tokens=[Token(id=1, token="hello")]),
            Sample(id="s1", tokens=[Token(id=2, token="world")]),
        ],
    )
    html = render_html(doc)
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
                tokens=[Token(id=1, token="<script>alert('xss')</script>")],
            ),
        ],
    )
    html = render_html(doc)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_save_html_from_gzipped_inif(doc, tmp_path):
    """View command round-trip: save as gzipped .inif, load, render HTML."""
    gz_path = tmp_path / "test.inif"
    save(doc, gz_path)  # .inif is gzip-compressed

    reloaded = load(gz_path)
    out = tmp_path / "from_gz.html"
    save_html(reloaded, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content
    assert "sample_0" in content


def test_save_html_from_gzipped_json(doc, tmp_path):
    """View command round-trip: save as .inif.json.gz, load, render HTML."""
    gz_path = tmp_path / "test.inif.json.gz"
    save(doc, gz_path)

    reloaded = load(gz_path)
    out = tmp_path / "from_json_gz.html"
    save_html(reloaded, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "gpt2" in content


# --- Sidebar tests ---


def test_render_html_sidebar_layout(doc):
    """Sidebar layout structure present for docs with samples."""
    html = render_html(doc)
    assert "inif-layout" in html
    assert "inif-sidebar" in html
    assert "inif-main" in html
    assert "inif-sidebar-header" in html
    assert "inif-sidebar-content" in html
    assert "inif-sample-list" in html


def test_render_html_sidebar_collapse_toggle(doc):
    """Sidebar has collapse toggle button."""
    html = render_html(doc)
    assert "inif-sidebar-toggle" in html
    # Contains the left-pointing arrow ‹
    assert "\u2039" in html


def test_render_html_sidebar_metadata_inside(doc):
    """Metadata is rendered inside the sidebar."""
    html = render_html(doc)
    # Metadata should be inside sidebar-content div (not the CSS)
    div_marker = '<div class="inif-sidebar-content">'
    sidebar_start = html.index(div_marker)
    sidebar_end = html.index("inif-sample-list", sidebar_start)
    sidebar_section = html[sidebar_start:sidebar_end]
    assert "Metadata" in sidebar_section
    assert "gpt2" in sidebar_section


def test_render_html_no_old_tabs(doc):
    """Old tab-based elements should not be present."""
    html = render_html(doc)
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
                tokens=[Token(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=1.0)],
            ),
        ],
    )
    html = render_html(doc)
    assert "inif-em-pass" in html
    assert "\u2713" in html


def test_render_html_exact_match_fail():
    """Sample with exact_match=0.0 shows X mark."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=0.0)],
            ),
        ],
    )
    html = render_html(doc)
    assert "inif-em-fail" in html
    assert "\u2717" in html


def test_render_html_exact_match_other_value():
    """Sample with exact_match != 0 or 1 shows no indicator."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=1, token="hi")],
                scores=[SampleScore(scorer="exact_match", value=0.5)],
            ),
        ],
    )
    html = render_html(doc)
    # Check marks/X marks should not appear in the sample list
    assert "\u2713" not in html
    assert "\u2717" not in html


def test_render_html_no_exact_match_scorer():
    """Sample without exact_match scorer shows no indicator."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="em_test")),
        samples=[
            Sample(
                id="s0",
                tokens=[Token(id=1, token="hi")],
                scores=[SampleScore(scorer="accuracy", value=1.0)],
            ),
        ],
    )
    html = render_html(doc)
    # Check marks/X marks should not appear in the sample list
    assert "\u2713" not in html
    assert "\u2717" not in html


# --- Role highlight tests ---


def _set_extra(tok, key, value):
    tok.set_extra(key, value)


def _make_role_doc():
    tok_sys = Token(id=1, token="System:")
    _set_extra(tok_sys, "role", "system")

    tok_user = Token(id=2, token="Hello")
    _set_extra(tok_user, "role", "user")

    tok_asst = Token(id=3, token="Hi")
    _set_extra(tok_asst, "role", "assistant")

    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="role_test")),
        samples=[
            Sample(id="r0", tokens=[tok_sys, tok_user, tok_asst]),
        ],
    )


def test_render_html_role_highlight_data_attrs():
    """Tokens with roles get data-role and data-role-bg attributes."""
    doc = _make_role_doc()
    html = render_html(doc)
    assert 'data-role="system"' in html
    assert 'data-role="user"' in html
    assert 'data-role="assistant"' in html
    assert "data-role-bg=" in html


def test_render_html_role_highlight_colors():
    """Role tokens get correct background colors from palette."""
    doc = _make_role_doc()
    html = render_html(doc)
    assert "#d4e6f1" in html  # system
    assert "#d5f5e3" in html  # user
    assert "#fdebd0" in html  # assistant


def test_render_html_role_has_roles_attr():
    """Token strip gets data-has-roles attribute."""
    doc = _make_role_doc()
    html = render_html(doc)
    assert 'data-has-roles="true"' in html


def test_render_html_no_roles_attr():
    """Token strip without roles gets data-has-roles=false."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_role")),
        samples=[
            Sample(id="nr0", tokens=[Token(id=1, token="plain")]),
        ],
    )
    html = render_html(doc)
    assert 'data-has-roles="false"' in html


def test_render_html_role_unknown_fallback():
    """Unknown role gets fallback color #f0f0f0."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "role", "custom_role")
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="fallback")),
        samples=[Sample(id="fb0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "#f0f0f0" in html
    assert 'data-role="custom_role"' in html


# --- Control panel tests ---


def test_render_html_control_panel_with_roles():
    """Control panel checkbox is checked when sample has roles."""
    doc = _make_role_doc()
    html = render_html(doc)
    assert "inif-control-panel" in html
    assert "inif-role-toggle" in html
    assert "checked" in html
    # Should NOT be disabled
    assert 'class="inif-role-toggle" checked' in html


def test_render_html_control_panel_no_roles():
    """Control panel checkbox is disabled when sample has no roles."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_role")),
        samples=[
            Sample(id="nr0", tokens=[Token(id=1, token="plain")]),
        ],
    )
    html = render_html(doc)
    assert "inif-control-panel" in html
    assert "disabled" in html
    assert 'label class="disabled"' in html


def test_render_html_role_legend_in_control_panel():
    """Role legend with swatches appears inside the control panel."""
    doc = _make_role_doc()
    html = render_html(doc)
    assert "inif-role-legend" in html
    assert "inif-role-swatch" in html
    # Role names appear in the legend
    cp_start = html.index('<div class="inif-control-panel">')
    cp_end = html.index("</div>", html.index("inif-role-legend", cp_start))
    cp_section = html[cp_start:cp_end]
    assert "system" in cp_section
    assert "user" in cp_section
    assert "assistant" in cp_section


def test_render_html_no_role_legend_without_roles():
    """No role legend when sample has no roles."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_role")),
        samples=[
            Sample(id="nr0", tokens=[Token(id=1, token="plain")]),
        ],
    )
    html = render_html(doc)
    # inif-role-legend appears in CSS, but no actual legend div rendered
    assert '<div class="inif-role-legend">' not in html


# --- Tag highlight toggle tests ---


def _make_tag_doc():
    t0 = Token(id=1, token="hello")
    _set_extra(t0, "tags", ["greeting"])
    t1 = Token(id=2, token=" world")
    _set_extra(t1, "tags", ["noun"])
    return InifDocument(
        metadata=Metadata(model=ModelInfo(name="tag_hl")),
        samples=[
            Sample(id="t0", tokens=[t0, t1]),
        ],
    )


def test_render_html_tag_toggle_present_with_tags():
    """Tag highlight checkbox is enabled when sample has tags."""
    doc = _make_tag_doc()
    html = render_html(doc)
    assert 'class="inif-tag-toggle" checked' in html


def test_render_html_tag_toggle_disabled_without_tags():
    """Tag highlight checkbox is disabled when sample has no tags."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_tag")),
        samples=[
            Sample(id="nt0", tokens=[Token(id=1, token="plain")]),
        ],
    )
    html = render_html(doc)
    assert 'class="inif-tag-toggle" disabled' in html


def test_render_html_tag_legend_in_control_panel():
    """Tag legend with swatches appears inside the control panel."""
    doc = _make_tag_doc()
    html = render_html(doc)
    assert "inif-tag-legend" in html
    assert "inif-tag-swatch" in html
    cp_start = html.index('<div class="inif-control-panel">')
    cp_end = html.index("</div>", html.index("inif-tag-legend", cp_start))
    cp_section = html[cp_start:cp_end]
    assert "greeting" in cp_section
    assert "noun" in cp_section


def test_render_html_no_tag_legend_without_tags():
    """No tag legend when sample has no tags."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="no_tag")),
        samples=[
            Sample(id="nt0", tokens=[Token(id=1, token="plain")]),
        ],
    )
    html = render_html(doc)
    assert '<div class="inif-tag-legend">' not in html


# --- Sample header tests ---


def test_render_html_sample_header_stats(doc):
    """Sample header contains stats (target, tokens)."""
    html = render_html(doc)
    assert "inif-sample-header" in html
    assert "inif-sample-stats" in html
    assert "Target" in html
    assert "Input tokens" in html
    assert "Output tokens" in html


def test_render_html_tag_bg_data_attr():
    """Token with tag gets data-tag-bg attribute."""
    tok = Token(id=1, token="tagged")
    _set_extra(tok, "tags", ["my_tag"])
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="tag_test")),
        samples=[Sample(id="t0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "data-tag-bg=" in html


def test_render_html_role_overrides_tag():
    """When token has both role and tag, role bg takes priority."""
    tok = Token(id=1, token="both")
    _set_extra(tok, "role", "user")
    _set_extra(tok, "tags", ["some_tag"])
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="priority")),
        samples=[Sample(id="p0", tokens=[tok])],
    )
    html = render_html(doc)
    # Both data attrs present
    assert "data-role-bg=" in html
    assert "data-tag-bg=" in html
    # Role color in inline style (takes priority)
    assert "#d5f5e3" in html


# --- Extra field underline tests ---


def test_render_html_extra_field_underline_single():
    """Token with one extra field gets a box-shadow underline."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "logit_lens", {"layer_5": "."})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "box-shadow:" in html
    assert "inset 0 -2px 0 0" in html


def test_render_html_extra_field_underline_two_fields():
    """Token with two extra fields gets stacked underlines."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "logit_lens", {"layer_5": "."})
    _set_extra(tok, "probe", {"acc": 0.9})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "box-shadow:" in html
    # Two inset shadows stacked
    assert "inset 0 -2px 0 0" in html
    assert "inset 0 -4px 0 0" in html


def test_render_html_extra_field_different_tokens():
    """Two tokens with different extra fields get different colors."""
    tok_a = Token(id=1, token="a")
    _set_extra(tok_a, "logit_lens", {})
    tok_b = Token(id=2, token="b")
    _set_extra(tok_b, "probe", {})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok_a, tok_b])],
    )
    html = render_html(doc)
    # Both tokens get underlines
    assert html.count("box-shadow:") == 2


def test_render_html_extras_legend_in_control_panel():
    """Extra fields legend appears inside the control panel."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "logit_lens", {})
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="ext")),
        samples=[Sample(id="e0", tokens=[tok])],
    )
    html = render_html(doc)
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
            Sample(id="p0", tokens=[Token(id=1, token="hi")]),
        ],
    )
    html = render_html(doc)
    assert '<div class="inif-extras-legend">' not in html


def test_render_html_role_tags_not_underlined():
    """role and tags fields do NOT produce underlines."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "role", "user")
    _set_extra(tok, "tags", ["t"])
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="skip")),
        samples=[Sample(id="s0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "box-shadow:" not in html
    assert '<div class="inif-extras-legend">' not in html


def test_render_html_extra_padding_multiple():
    """Multiple extra fields add padding-bottom for stacked lines."""
    tok = Token(id=1, token="x")
    _set_extra(tok, "field_a", 1)
    _set_extra(tok, "field_b", 2)
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="pad")),
        samples=[Sample(id="p0", tokens=[tok])],
    )
    html = render_html(doc)
    assert "padding-bottom:4px" in html


# --- Sidebar collapse CSS tests ---


def test_render_html_sidebar_collapsed_hides_text(doc):
    """CSS hides the header text span when sidebar is collapsed."""
    html = render_html(doc)
    assert ".inif-sidebar.collapsed .inif-sidebar-header > span" in html
    assert "display: none" in html


def test_render_html_sidebar_collapsed_overflow(doc):
    """CSS sets overflow hidden on collapsed sidebar."""
    html = render_html(doc)
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
                id="s0", tokens=["Hello", " world", "!"], ids=[1, 2, 3], n_tokens=3
            ),
        ],
        samples=[
            Sample(
                id="x0",
                tokens=[
                    Token(id=-1, sequence_id="s0"),
                    Token(id=1, token=" end"),
                ],
            ),
        ],
    )
    html = render_html(doc)
    # Each sub-token from sequence should be a separate seq-ref span
    assert html.count("seq-ref") >= 3
    # Individual tokens appear separately (not concatenated)
    assert "Hello" in html
    assert "world" in html  # " world" after escaping leading space
    assert "!" in html
    assert "\u00b7end" in html


def test_render_html_seq_ref_wraps_like_normal_tokens():
    """Expanded seq-ref tokens are inline-block spans that can wrap."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="wrap")),
        sequences=[
            Sequence(id="s0", tokens=["a", "b"], ids=[1, 2], n_tokens=2),
        ],
        samples=[
            Sample(
                id="w0",
                tokens=[Token(id=-1, sequence_id="s0")],
            ),
        ],
    )
    html = render_html(doc)
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
                tokens=[Token(id=-1, sequence_id="nonexistent")],
            ),
        ],
    )
    html = render_html(doc)
    assert "seq-ref" in html
    assert "[nonexistent]" in html


def test_render_html_seq_ref_shares_position():
    """Expanded seq-ref sub-tokens share the original position."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="pos")),
        sequences=[
            Sequence(id="s0", tokens=["a", "b"], ids=[1, 2], n_tokens=2),
        ],
        samples=[
            Sample(
                id="p0",
                tokens=[
                    Token(id=1, token="before"),
                    Token(id=-1, sequence_id="s0"),
                ],
            ),
        ],
    )
    html = render_html(doc)
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
                    Token(id=1, token="hello"),
                    Token(id=2, token="\n"),
                    Token(id=3, token="world"),
                ],
            ),
        ],
    )
    html = render_html(doc)
    assert "\u21b5" in html  # ↵ symbol


def test_render_html_newline_inserts_line_break():
    """A flex line-break div is inserted after tokens containing newlines."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="nl_br")),
        samples=[
            Sample(
                id="n1",
                tokens=[
                    Token(id=1, token="line1"),
                    Token(id=2, token="\n"),
                    Token(id=3, token="line2"),
                ],
            ),
        ],
    )
    html = render_html(doc)
    # The newline token span should be followed by a line-break div
    idx_newline = html.index("\u21b5")
    after_newline = html[idx_newline:]
    span_end = after_newline.index("</span>")
    after_span = after_newline[span_end + len("</span>") :].lstrip()
    assert after_span.startswith('<div class="inif-line-break">')


def test_render_html_newline_in_seq_ref_inserts_line_break():
    """Newlines inside expanded seq-ref tokens also insert line break."""
    doc = InifDocument(
        metadata=Metadata(model=ModelInfo(name="nl_seq")),
        sequences=[
            Sequence(id="s0", tokens=["first\n", "second"], ids=[1, 2], n_tokens=2),
        ],
        samples=[
            Sample(
                id="n2",
                tokens=[Token(id=-1, sequence_id="s0")],
            ),
        ],
    )
    html = render_html(doc)
    assert "\u21b5" in html
    idx_sym = html.index("\u21b5")
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
                    Token(id=1, token="hello"),
                    Token(id=2, token=" world"),
                ],
            ),
        ],
    )
    html = render_html(doc)
    assert "inif-line-break" not in html.split("</style>")[-1]
