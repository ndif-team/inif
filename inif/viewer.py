from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from inif.io import to_dict
from inif.models import InifDocument

# 12 muted pastel colors for tags
_TAG_PALETTE = [
    "#b3d9ff",  # light blue
    "#ffd9b3",  # light orange
    "#d9b3ff",  # light purple
    "#b3ffb3",  # light green
    "#ffb3d9",  # light pink
    "#ffffb3",  # light yellow
    "#b3ffff",  # light cyan
    "#ffcccc",  # light red
    "#d9ffb3",  # light lime
    "#e6ccff",  # light lavender
    "#ccf2e6",  # light mint
    "#ffe6cc",  # light peach
]

# Separate palette for span border colors (more saturated)
_SPAN_PALETTE = [
    "#3366cc",
    "#cc6633",
    "#6633cc",
    "#33cc33",
    "#cc3366",
    "#cccc33",
    "#33cccc",
    "#cc3333",
    "#66cc33",
    "#9933cc",
    "#33cc99",
    "#cc9933",
]

# Role-based background colors for token highlighting
_ROLE_PALETTE = {
    "system": "#d4e6f1",
    "user": "#d5f5e3",
    "assistant": "#fdebd0",
    "template": "#eaecee",
}

# Saturated colors for extra-field underlines
_EXTRA_PALETTE = [
    "#e67e22",  # orange
    "#8e44ad",  # purple
    "#2980b9",  # blue
    "#27ae60",  # green
    "#c0392b",  # red
    "#16a085",  # teal
    "#d35400",  # dark orange
    "#7f8c8d",  # grey
]

# Token dict keys that do NOT produce an underline
_EXTRA_SKIP = {"id", "token", "seq_id", "sequence_id", "role", "tags", "_seq_ref"}


def _detect_newline_chars(tokenizer: Any) -> frozenset[str]:
    """Return the set of characters that represent newlines for *tokenizer*.

    Always includes the literal ``\\n``.  When a tokenizer is given its
    byte-level representation of newline (e.g. ``Ċ`` for GPT-2 family) is
    added automatically.
    """
    chars: set[str] = {"\n"}
    if tokenizer is not None:
        ids = tokenizer.encode("\n", add_special_tokens=False)
        if ids:
            tok = tokenizer.convert_ids_to_tokens(ids[0])
            if tok and len(tok) == 1:
                chars.add(tok)
    return frozenset(chars)


_DEFAULT_NL = frozenset({"\n"})


def _escape(s: str, newline_chars: frozenset[str] = _DEFAULT_NL) -> str:
    escaped = html.escape(str(s))
    if escaped and escaped[0] == " ":
        escaped = "\u00b7" + escaped[1:]
    if len(escaped) > 1 and escaped[-1] == " ":
        escaped = escaped[:-1] + "\u00b7"
    for ch in newline_chars:
        escaped = escaped.replace(ch, "\u21b5")
    return escaped


def _has_newline(tok_data: dict, newline_chars: frozenset[str] = _DEFAULT_NL) -> bool:
    """Check whether a token's text contains a newline character."""
    text = tok_data.get("token") or ""
    return any(ch in text for ch in newline_chars)


def _render_css() -> str:
    return """<style>
.inif-viewer {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 100%;
    margin: 0 auto;
    padding: 16px;
    color: #333;
    background: #fff;
    line-height: 1.5;
}
.inif-viewer h1 { font-size: 1.4em; margin: 0 0 12px 0; }
.inif-viewer h2 { font-size: 1.1em; margin: 16px 0 8px 0; color: #555; }
.inif-viewer h3 { font-size: 1em; margin: 12px 0 6px 0; color: #666; }
.inif-viewer table {
    border-collapse: collapse;
    width: 100%;
    margin: 4px 0 12px 0;
    font-size: 0.9em;
}
.inif-viewer th, .inif-viewer td {
    text-align: left;
    padding: 4px 10px;
    border-bottom: 1px solid #e0e0e0;
}
.inif-viewer th { background: #f5f5f5; font-weight: 600; }
.inif-layout {
    display: flex;
    gap: 0;
}
.inif-sidebar {
    width: 220px;
    flex-shrink: 0;
    border-right: 1px solid #e0e0e0;
    overflow-y: auto;
    transition: width 0.2s;
}
.inif-sidebar.collapsed {
    width: 36px;
    overflow: hidden;
}
.inif-sidebar.collapsed .inif-sidebar-content {
    display: none;
}
.inif-sidebar.collapsed .inif-sidebar-header {
    justify-content: center;
    padding: 6px 4px;
}
.inif-sidebar.collapsed .inif-sidebar-header > span {
    display: none;
}
.inif-sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 8px;
    border-bottom: 1px solid #e0e0e0;
    font-weight: 600;
    font-size: 0.9em;
}
.inif-sidebar-toggle {
    background: none;
    border: 1px solid #ccc;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.85em;
    padding: 1px 6px;
    color: #666;
}
.inif-sidebar-toggle:hover { background: #eee; }
.inif-sidebar-content {
    padding: 0;
}
.inif-sidebar-content table {
    font-size: 0.8em;
    margin: 0;
}
.inif-sidebar-content th, .inif-sidebar-content td {
    padding: 2px 6px;
}
.inif-sample-list {
    list-style: none;
    margin: 0;
    padding: 0;
}
.inif-sample-item {
    padding: 6px 10px;
    cursor: pointer;
    font-size: 0.85em;
    border-bottom: 1px solid #f0f0f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.inif-sample-item:hover { background: #f0f0f0; }
.inif-sample-item.active { background: #e3f2fd; font-weight: 600; }
.inif-em-pass { color: #27ae60; font-weight: bold; }
.inif-em-fail { color: #e74c3c; font-weight: bold; }
.inif-main {
    flex: 1;
    min-width: 0;
    padding-left: 16px;
}
.inif-sample-header {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 8px;
}
.inif-sample-stats { flex: 1; }
.inif-sample-stats table { margin: 0; }
.inif-control-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    padding-top: 4px;
}
.inif-control-panel label {
    font-size: 0.85em;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
}
.inif-control-panel label.disabled {
    color: #aaa;
    cursor: default;
}
.inif-role-legend, .inif-tag-legend, .inif-extras-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 0.8em;
}
.inif-role-swatch {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 3px;
}
.inif-tag-swatch {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 3px;
}
.inif-extra-swatch {
    display: inline-block;
    width: 12px;
    height: 3px;
    vertical-align: middle;
    margin-right: 3px;
}
.inif-token-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 1px;
    margin: 6px 0 12px 0;
    padding: 4px;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}
.inif-line-break { flex-basis: 100%; height: 0; }
.inif-token {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.85em;
    display: inline-block;
    padding: 2px 3px;
    border-radius: 2px;
    cursor: default;
    white-space: pre;
    border: 1px solid transparent;
}
.inif-token:hover { outline: 2px solid #666; z-index: 1; }
.inif-token.seq-ref {
    border: 1px dashed #999;
    background: #eee;
    font-style: italic;
}
.inif-token.in-span { border-bottom: 2px solid; }
.inif-spans-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 4px 0 8px 0;
    font-size: 0.85em;
}
.inif-span-swatch {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 4px;
}
.inif-texts-panel .inif-text-item {
    margin: 4px 0;
    padding: 6px 10px;
    background: #f9f9f9;
    border-left: 3px solid #ccc;
    font-size: 0.9em;
    white-space: pre-wrap;
}
.inif-scores-table td:first-child { font-weight: 600; }
.inif-hidden { display: none; }
.inif-tooltip {
    position: fixed;
    background: #333;
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.8em;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    max-width: 400px;
    white-space: pre-wrap;
    word-break: break-word;
    z-index: 10000;
    pointer-events: none;
    display: none;
}
.inif-sample-panel { padding: 8px 0; }
</style>"""


def _render_js() -> str:
    return """<script>
(function() {
    var tip = document.createElement('div');
    tip.className = 'inif-tooltip';
    document.body.appendChild(tip);

    document.addEventListener('mouseenter', function(e) {
        var t = e.target;
        if (!t.classList || !t.classList.contains('inif-token')) return;
        var data = t.getAttribute('data-tooltip');
        if (!data || data === '{}') return;
        try { data = JSON.stringify(JSON.parse(data), null, 2); } catch(_) {}
        tip.textContent = data;
        tip.style.display = 'block';
    }, true);

    document.addEventListener('mousemove', function(e) {
        if (tip.style.display === 'block') {
            tip.style.left = (e.clientX + 12) + 'px';
            tip.style.top = (e.clientY + 12) + 'px';
        }
    }, true);

    document.addEventListener('mouseleave', function(e) {
        var t = e.target;
        if (t.classList && t.classList.contains('inif-token')) {
            tip.style.display = 'none';
        }
    }, true);

    /* Sidebar item click: show/hide sample panels */
    document.addEventListener('click', function(e) {
        var item = e.target.closest('.inif-sample-item');
        if (!item) return;
        var viewer = item.closest('.inif-viewer');
        if (!viewer) return;
        var idx = item.getAttribute('data-sample-idx');
        var items = viewer.querySelectorAll('.inif-sample-item');
        var panels = viewer.querySelectorAll('.inif-sample-panel');
        for (var i = 0; i < items.length; i++) {
            var a = items[i].getAttribute('data-sample-idx');
            items[i].classList.toggle('active', a === idx);
        }
        for (var j = 0; j < panels.length; j++) {
            var b = panels[j].getAttribute('data-sample-idx');
            panels[j].style.display = b === idx ? '' : 'none';
        }
    });

    /* Sidebar collapse toggle */
    document.addEventListener('click', function(e) {
        var btn = e.target.closest('.inif-sidebar-toggle');
        if (!btn) return;
        var sidebar = btn.closest('.inif-sidebar');
        if (!sidebar) return;
        sidebar.classList.toggle('collapsed');
        var c = sidebar.classList.contains('collapsed');
        btn.textContent = c ? '\\u203a' : '\\u2039';
    });

    /* Shared: recompute token backgrounds from toggle states */
    function updateTokenBgs(panel) {
        var rc = panel.querySelector('.inif-role-toggle');
        var tc = panel.querySelector('.inif-tag-toggle');
        var roleOn = rc && rc.checked;
        var tagOn = tc && tc.checked;
        var tokens = panel.querySelectorAll('.inif-token');
        for (var i = 0; i < tokens.length; i++) {
            var tok = tokens[i];
            var bg = '';
            if (roleOn) {
                var rb = tok.getAttribute('data-role-bg');
                if (rb) bg = rb;
            }
            if (!bg && tagOn) {
                var tb = tok.getAttribute('data-tag-bg');
                if (tb) bg = tb;
            }
            tok.style.background = bg;
        }
    }

    /* Role highlight checkbox */
    document.addEventListener('change', function(e) {
        var cb = e.target;
        if (!cb.classList) return;
        if (!cb.classList.contains('inif-role-toggle')
            && !cb.classList.contains('inif-tag-toggle')) return;
        var panel = cb.closest('.inif-sample-panel');
        if (panel) updateTokenBgs(panel);
    });
})();
</script>"""


def _render_metadata_panel(doc_data: dict) -> str:
    meta = doc_data.get("metadata", {})
    model = meta.get("model", {})
    source_eval = meta.get("source_eval")

    rows: list[tuple[str, str]] = []
    if model.get("name"):
        rows.append(("Model", _escape(model["name"])))
    if model.get("revision"):
        rows.append(("Revision", _escape(model["revision"])))
    if model.get("huggingface_id"):
        rows.append(("HuggingFace ID", _escape(model["huggingface_id"])))
    if source_eval:
        if source_eval.get("framework"):
            rows.append(("Framework", _escape(source_eval["framework"])))
        if source_eval.get("task"):
            rows.append(("Task", _escape(source_eval["task"])))
    if meta.get("created_at"):
        rows.append(("Created", _escape(meta["created_at"])))
    n_samples = len(doc_data.get("samples", []))
    if n_samples:
        rows.append(("Total samples", str(n_samples)))
    if meta.get("total_time") is not None:
        rows.append(("Total time", f"{meta['total_time']:.2f}s"))
    pkgs = meta.get("packages", {})
    if pkgs:
        pkg_str = ", ".join(f"{k} {v}" for k, v in pkgs.items())
        rows.append(("Packages", _escape(pkg_str)))

    if not rows:
        return ""

    lines = ["<h3>Metadata</h3>", "<table>"]
    for label, value in rows:
        lines.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _score_is_correct(score: dict) -> bool | None:
    """Infer pass/fail for a single score, framework-agnostic.

    Priority: explicit ``metadata.is_correct`` (how the evaleval converter
    records correctness), then an Inspect-style ``"C"``/``"I"`` value, then a
    numeric 0/1. Returns ``None`` when the score doesn't clearly represent a
    binary outcome \u2014 callers should keep looking.
    """
    meta = score.get("metadata") or {}
    if isinstance(meta.get("is_correct"), bool):
        return meta["is_correct"]
    val = score.get("value")
    if val == "C":
        return True
    if val == "I":
        return False
    if val is True or val == 1 or val == 1.0:
        return True
    if val is False or val == 0 or val == 0.0:
        return False
    return None


def _get_exact_match_indicator(sample_data: dict) -> str:
    for sc in sample_data.get("scores", []):
        result = _score_is_correct(sc)
        if result is True:
            return '<span class="inif-em-pass">\u2713</span>'
        if result is False:
            return '<span class="inif-em-fail">\u2717</span>'
    return ""


def _render_sidebar(doc_data: dict, samples: list[dict]) -> str:
    parts = ['<div class="inif-sidebar">']
    parts.append('<div class="inif-sidebar-header">')
    parts.append("<span>Samples</span>")
    parts.append('<button class="inif-sidebar-toggle">\u2039</button>')
    parts.append("</div>")
    parts.append('<div class="inif-sidebar-content">')

    metadata_html = _render_metadata_panel(doc_data)
    if metadata_html:
        parts.append(metadata_html)

    parts.append('<ul class="inif-sample-list">')
    for i, s in enumerate(samples):
        active = " active" if i == 0 else ""
        sid = _escape(str(s.get("id", i)))
        indicator = _get_exact_match_indicator(s)
        indicator_html = f" {indicator}" if indicator else ""
        parts.append(
            f'<li class="inif-sample-item{active}" data-sample-idx="{i}">'
            f"<span>{sid}</span>{indicator_html}</li>"
        )
    parts.append("</ul>")
    parts.append("</div>")  # sidebar-content
    parts.append("</div>")  # sidebar
    return "\n".join(parts)


def _sample_has_roles(sample_data: dict) -> bool:
    return any("role" in tok for tok in sample_data.get("tokens", []))


def _sample_has_tags(sample_data: dict) -> bool:
    return any("tags" in tok for tok in sample_data.get("tokens", []))


def _collect_active_roles(tokens: list[dict]) -> dict[str, str]:
    """Map each role present in the sample to its color."""
    roles: dict[str, str] = {}
    for tok in tokens:
        role = tok.get("role")
        if role and isinstance(role, str) and role not in roles:
            roles[role] = _ROLE_PALETTE.get(role, "#f0f0f0")
    return roles


def _collect_active_tags(tokens: list[dict]) -> dict[str, str]:
    """Map each tag present in the sample to its color."""
    tags: dict[str, str] = {}
    for tok in tokens:
        tok_tags = tok.get("tags")
        if tok_tags and isinstance(tok_tags, list):
            for t in tok_tags:
                if t not in tags:
                    tags[t] = _TAG_PALETTE[hash(t) % len(_TAG_PALETTE)]
    return tags


def _collect_extra_field_colors(
    tokens: list[dict],
) -> dict[str, str]:
    """Map each extra field name to a color, sorted alphabetically."""
    fields: set[str] = set()
    for tok in tokens:
        for k in tok:
            if k not in _EXTRA_SKIP:
                fields.add(k)
    return {
        f: _EXTRA_PALETTE[i % len(_EXTRA_PALETTE)] for i, f in enumerate(sorted(fields))
    }


def _render_role_legend(active_roles: dict[str, str]) -> str:
    if not active_roles:
        return ""
    parts = ['<div class="inif-role-legend">']
    for name, color in active_roles.items():
        esc = _escape(name)
        parts.append(
            f'<span><span class="inif-role-swatch" '
            f'style="background:{color}"></span>'
            f"{esc}</span>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_tag_legend(active_tags: dict[str, str]) -> str:
    if not active_tags:
        return ""
    parts = ['<div class="inif-tag-legend">']
    for name, color in active_tags.items():
        esc = _escape(name)
        parts.append(
            f'<span><span class="inif-tag-swatch" '
            f'style="background:{color}"></span>'
            f"{esc}</span>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_extras_legend(
    extra_colors: dict[str, str],
) -> str:
    if not extra_colors:
        return ""
    parts = ['<div class="inif-extras-legend">']
    for name, color in extra_colors.items():
        esc = _escape(name)
        parts.append(
            f'<span><span class="inif-extra-swatch" '
            f'style="background:{color}"></span>'
            f"{esc}</span>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_sample_header(
    sample_data: dict,
    has_roles: bool,
    has_tags: bool,
    active_roles: dict[str, str],
    active_tags: dict[str, str],
    extra_colors: dict[str, str],
) -> str:
    parts = ['<div class="inif-sample-header">']

    # Left: stats
    stats_rows: list[tuple[str, str]] = []
    if sample_data.get("target") is not None:
        stats_rows.append(("Target", _escape(str(sample_data["target"]))))
    if sample_data.get("input_tokens") is not None:
        stats_rows.append(("Input tokens", str(sample_data["input_tokens"])))
    if sample_data.get("output_tokens") is not None:
        stats_rows.append(("Output tokens", str(sample_data["output_tokens"])))
    if sample_data.get("total_time") is not None:
        stats_rows.append(("Time", f"{sample_data['total_time']:.2f}s"))

    parts.append('<div class="inif-sample-stats">')
    if stats_rows:
        parts.append("<table>")
        for label, value in stats_rows:
            parts.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
        parts.append("</table>")
    parts.append("</div>")

    # Right: control panel (column layout)
    parts.append('<div class="inif-control-panel">')
    if has_roles:
        parts.append(
            '<label><input type="checkbox" class="inif-role-toggle" checked>'
            " Highlight roles</label>"
        )
    else:
        parts.append(
            '<label class="disabled">'
            '<input type="checkbox" class="inif-role-toggle" disabled>'
            " Highlight roles</label>"
        )

    if has_tags:
        parts.append(
            '<label><input type="checkbox" class="inif-tag-toggle" checked>'
            " Highlight tags</label>"
        )
    else:
        parts.append(
            '<label class="disabled">'
            '<input type="checkbox" class="inif-tag-toggle" disabled>'
            " Highlight tags</label>"
        )

    role_legend = _render_role_legend(active_roles)
    if role_legend:
        parts.append(role_legend)

    tag_legend = _render_tag_legend(active_tags)
    if tag_legend:
        parts.append(tag_legend)

    extras_legend = _render_extras_legend(extra_colors)
    if extras_legend:
        parts.append(extras_legend)

    parts.append("</div>")  # control-panel

    parts.append("</div>")  # sample-header
    return "\n".join(parts)


def _render_token(
    tok_data: dict,
    position: int,
    span_positions: dict[int, str],
    seq_map: dict[str, dict],
    extra_colors: dict[str, str] | None = None,
    newline_chars: frozenset[str] = _DEFAULT_NL,
) -> str:
    tok_id = tok_data.get("id", 0)
    tok_str = tok_data.get("token") or ""
    seq_id = tok_data.get("seq_id") or tok_data.get("sequence_id")
    is_ref = tok_id < 0 or bool(tok_data.get("_seq_ref"))

    # Extra fields for tooltip (exclude id, token, seq_id/sequence_id)
    skip = ("id", "token", "seq_id", "sequence_id", "_seq_ref")
    extra = {k: v for k, v in tok_data.items() if k not in skip}
    tooltip_json = html.escape(json.dumps(extra, default=str), quote=True)

    classes = ["inif-token"]
    style_parts = []
    data_attrs: list[str] = []

    # Tag coloring
    tag_bg = ""
    tags = tok_data.get("tags")
    if tags and isinstance(tags, list) and len(tags) > 0:
        tag_bg = _TAG_PALETTE[hash(tags[0]) % len(_TAG_PALETTE)]

    # Role coloring
    role = tok_data.get("role")
    role_bg = ""
    if role and isinstance(role, str):
        role_bg = _ROLE_PALETTE.get(role, "#f0f0f0")

    # Store data attributes for JS toggling
    if tag_bg:
        data_attrs.append(f'data-tag-bg="{tag_bg}"')
    if role_bg:
        data_attrs.append(f'data-role="{html.escape(str(role), quote=True)}"')
        data_attrs.append(f'data-role-bg="{role_bg}"')

    # Default: role bg takes priority, then tag bg
    if role_bg:
        style_parts.append(f"background:{role_bg}")
    elif tag_bg:
        style_parts.append(f"background:{tag_bg}")

    # Sequence ref — only apply fallback text if not already expanded
    if is_ref:
        classes.append("seq-ref")
        if not tok_str:
            # Not expanded by caller — show concatenated fallback
            if seq_id and seq_id in seq_map:
                seq_toks = seq_map[seq_id].get("tokens", [])
                tok_str = "".join(t.get("token") or "" for t in seq_toks) or (
                    f"[{seq_id}]"
                )
            else:
                tok_str = f"[{seq_id or '?'}]"

    # Span membership
    if position in span_positions:
        classes.append("in-span")
        style_parts.append(f"border-bottom-color:{span_positions[position]}")

    # Extra-field underlines via box-shadow
    if extra_colors:
        tok_extras = sorted(
            k for k in tok_data if k not in _EXTRA_SKIP and k in extra_colors
        )
        if tok_extras:
            shadows = []
            for j, field in enumerate(tok_extras):
                offset = -2 * (j + 1)
                color = extra_colors[field]
                shadows.append(f"inset 0 {offset}px 0 0 {color}")
            style_parts.append(f"box-shadow:{','.join(shadows)}")
            if len(tok_extras) > 1:
                pb = 2 * len(tok_extras)
                style_parts.append(f"padding-bottom:{pb}px")

    display_text = _escape(tok_str, newline_chars)
    cls_attr = " ".join(classes)
    style_attr = f' style="{";".join(style_parts)}"' if style_parts else ""
    data_str = (" " + " ".join(data_attrs)) if data_attrs else ""

    return (
        f'<span class="{cls_attr}"{style_attr} '
        f'data-tooltip="{tooltip_json}" '
        f'data-position="{position}" '
        f'data-token-id="{tok_id}"{data_str}>'
        f"{display_text}</span>"
    )


def _render_token_strip(
    sample_data: dict,
    sequences: list[dict],
    extra_colors: dict[str, str],
    newline_chars: frozenset[str] = _DEFAULT_NL,
) -> str:
    seq_map = {s["id"]: s for s in sequences}

    # Build span position -> color map
    span_positions: dict[int, str] = {}
    for i, span in enumerate(sample_data.get("spans", [])):
        color = _SPAN_PALETTE[i % len(_SPAN_PALETTE)]
        for pos in span.get("positions", []):
            if pos not in span_positions:
                span_positions[pos] = color

    has_roles = _sample_has_roles(sample_data)
    tokens = sample_data.get("tokens", [])
    data_has_roles = "true" if has_roles else "false"
    hr = data_has_roles
    parts = [f'<div class="inif-token-strip" data-has-roles="{hr}">']
    for idx, tok in enumerate(tokens):
        tok_id = tok.get("id", 0)
        seq_id = tok.get("seq_id") or tok.get("sequence_id")
        # Expand sequence refs into individual wrappable tokens
        if tok_id < 0 and seq_id and seq_id in seq_map:
            seq_toks = seq_map[seq_id].get("tokens", [])
            if seq_toks:
                for sub_tok in seq_toks:
                    sub_str = sub_tok.get("token") or ""
                    sub = {
                        "id": sub_tok.get("id", tok_id),
                        "token": sub_str,
                        "seq_id": seq_id,
                        "_seq_ref": True,
                    }
                    parts.append(
                        _render_token(
                            sub,
                            idx,
                            span_positions,
                            seq_map,
                            extra_colors,
                            newline_chars,
                        )
                    )
                    if any(ch in sub_str for ch in newline_chars):
                        parts.append('<div class="inif-line-break"></div>')
                continue
        parts.append(
            _render_token(
                tok, idx, span_positions, seq_map, extra_colors, newline_chars
            )
        )
        if _has_newline(tok, newline_chars):
            parts.append('<div class="inif-line-break"></div>')
    parts.append("</div>")
    return "\n".join(parts)


def _render_spans_legend(sample_data: dict) -> str:
    spans = sample_data.get("spans", [])
    if not spans:
        return ""
    parts = ["<h3>Spans</h3>", '<div class="inif-spans-legend">']
    for i, span in enumerate(spans):
        color = _SPAN_PALETTE[i % len(_SPAN_PALETTE)]
        name = _escape(span.get("name", f"span_{i}"))
        parts.append(
            f'<span><span class="inif-span-swatch" style="background:{color}"></span>'
            f"{name}</span>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_texts_panel(sample_data: dict) -> str:
    texts = sample_data.get("texts", [])
    if not texts:
        return ""
    parts = ["<h3>Texts</h3>", '<div class="inif-texts-panel">']
    for i, text in enumerate(texts):
        safe_text = html.escape(str(text))
        parts.append(
            f'<div class="inif-text-item"><strong>{i}:</strong> {safe_text}</div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_scores_panel(sample_data: dict) -> str:
    scores = sample_data.get("scores", [])
    if not scores:
        return ""
    parts = ["<h3>Scores</h3>", '<table class="inif-scores-table">']
    parts.append(
        "<tr><th>Scorer</th><th>Value</th><th>Answer</th><th>Explanation</th></tr>"
    )
    for sc in scores:
        scorer = _escape(str(sc.get("scorer", "")))
        value = _escape(str(sc.get("value", "")))
        answer = (
            _escape(str(sc.get("answer", ""))) if sc.get("answer") is not None else ""
        )
        explanation = (
            _escape(str(sc.get("explanation", "")))
            if sc.get("explanation") is not None
            else ""
        )
        parts.append(
            f"<tr><td>{scorer}</td><td>{value}</td><td>{answer}</td><td>{explanation}</td></tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def _render_sample_panel(
    idx: int,
    sample_data: dict,
    sequences: list[dict],
    compact: bool,
    newline_chars: frozenset[str] = _DEFAULT_NL,
) -> str:
    tokens = sample_data.get("tokens", [])
    has_roles = _sample_has_roles(sample_data)
    has_tags = _sample_has_tags(sample_data)
    active_roles = _collect_active_roles(tokens)
    active_tags = _collect_active_tags(tokens)
    extra_colors = _collect_extra_field_colors(tokens)
    parts = []

    header_html = _render_sample_header(
        sample_data, has_roles, has_tags, active_roles, active_tags, extra_colors
    )
    if header_html:
        parts.append(header_html)
    elif compact:
        pass  # skip empty

    parts.append(
        _render_token_strip(sample_data, sequences, extra_colors, newline_chars)
    )

    spans_html = _render_spans_legend(sample_data)
    if spans_html:
        parts.append(spans_html)

    texts_html = _render_texts_panel(sample_data)
    if texts_html:
        parts.append(texts_html)

    scores_html = _render_scores_panel(sample_data)
    if scores_html:
        parts.append(scores_html)

    return "\n".join(parts)


def render_html(
    doc: InifDocument,
    compact: bool = False,
    title: str | None = None,
    tokenizer: Any = None,
) -> str:
    """Render an InifDocument as a self-contained HTML string.

    When *tokenizer* is provided, the tokenizer's byte-level representation
    of newlines (e.g. ``Ċ`` for GPT-2 family) is detected automatically so
    that visual line breaks are inserted after newline tokens.
    """
    newline_chars = _detect_newline_chars(tokenizer)
    doc_data = to_dict(doc, compact=False)
    display_title = title or f"inif: {doc.metadata.model.name}"
    sequences = doc_data.get("sequences", [])
    samples = doc_data.get("samples", [])

    parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{_escape(display_title)}</title>",
        '<meta charset="utf-8">',
        _render_css(),
        "</head><body>",
        '<div class="inif-viewer">',
        f"<h1>{_escape(display_title)}</h1>",
    ]

    if samples:
        parts.append('<div class="inif-layout">')

        # Sidebar
        parts.append(_render_sidebar(doc_data, samples))

        # Main content
        parts.append('<div class="inif-main">')
        for i, s in enumerate(samples):
            display = "" if i == 0 else ' style="display:none"'
            parts.append(
                f'<div class="inif-sample-panel" data-sample-idx="{i}"{display}>'
            )
            parts.append(_render_sample_panel(i, s, sequences, compact, newline_chars))
            parts.append("</div>")
        parts.append("</div>")  # inif-main

        parts.append("</div>")  # inif-layout
    else:
        parts.append("<p><em>No samples in document.</em></p>")

    parts.append("</div>")
    parts.append(_render_js())
    parts.append("</body></html>")
    return "\n".join(parts)


def show(
    doc: InifDocument,
    compact: bool = False,
    title: str | None = None,
    tokenizer: Any = None,
) -> Any:
    """Display an InifDocument as HTML in a Jupyter notebook.

    Pass *tokenizer* to enable line breaks after BPE newline tokens.
    """
    from IPython.display import HTML

    return HTML(render_html(doc, compact=compact, title=title, tokenizer=tokenizer))


def save_html(
    doc: InifDocument,
    path: str | Path,
    compact: bool = False,
    title: str | None = None,
    source: str | Path | None = None,
    tokenizer: Any = None,
) -> None:
    """Save an InifDocument as a self-contained HTML file.

    When *title* is not given, the source filename is used if available,
    otherwise falls back to the model name.  Pass *tokenizer* to enable
    line breaks after BPE newline tokens.
    """
    path = Path(path)
    if title is None and source is not None:
        title = Path(source).name
    html_str = render_html(doc, compact=compact, title=title, tokenizer=tokenizer)
    path.write_text(html_str, encoding="utf-8")
