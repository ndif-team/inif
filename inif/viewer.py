from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from inif.io import _to_dict
from inif.models import InifDocument

# 12 muted pastel colors for annotations
_ANNOTATION_PALETTE = [
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

# Stable colors for common chat-role annotations. Kept muted so they read
# as background / chrome — auto sub-text and user-defined tags overlay
# higher-priority colors on top.
_ROLE_ANNOTATION_PALETTE = {
    "system": "#d4e6f1",
    "user": "#d5f5e3",
    "assistant": "#fdebd0",
    "tool": "#f4ecf7",
    "template": "#eaecee",
}

# Stable colors for auto sub-text annotations emitted by the converter
# (reasoning / tool_call). Picked for contrast against the chat-role
# palette so a token tagged both "assistant" and "reasoning" reads as
# clearly reasoning, not assistant.
_AUTO_SUBTEXT_PALETTE = {
    "reasoning": "#e6ccff",  # light lavender — pops against orange assistant
    "tool_call": "#b3ffd9",  # light mint — pops against purple tool / orange assistant
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
_EXTRA_SKIP = {"id", "token", "_seq_ref"}


def _annotation_color(name: str) -> str:
    if name in _AUTO_SUBTEXT_PALETTE:
        return _AUTO_SUBTEXT_PALETTE[name]
    if name in _ROLE_ANNOTATION_PALETTE:
        return _ROLE_ANNOTATION_PALETTE[name]
    return _ANNOTATION_PALETTE[hash(name) % len(_ANNOTATION_PALETTE)]


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
        escaped = "·" + escaped[1:]
    if len(escaped) > 1 and escaped[-1] == " ":
        escaped = escaped[:-1] + "·"
    for ch in newline_chars:
        escaped = escaped.replace(ch, "↵")
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
.inif-annotation-legend, .inif-extras-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 0.8em;
}
.inif-annotation-swatch {
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
.inif-messages {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 6px 0 12px 0;
}
.inif-message {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    background: #fff;
}
.inif-message-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    border-bottom: 1px solid #e0e0e0;
    background: #fafafa;
    font-size: 0.85em;
}
.inif-message-header .inif-message-name {
    font-weight: 600;
    color: #555;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
}
.inif-message-header .inif-message-meta {
    color: #999;
    font-size: 0.85em;
}
.inif-message[data-role="system"] .inif-message-header { background: #d4e6f1; }
.inif-message[data-role="user"] .inif-message-header { background: #d5f5e3; }
.inif-message[data-role="assistant"] .inif-message-header { background: #fdebd0; }
.inif-message[data-role="tool"] .inif-message-header { background: #f4ecf7; }
.inif-message-toggle {
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.8em;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 3px 10px;
    color: #555;
    line-height: 1.2;
    min-width: 64px;
    text-align: center;
}
.inif-message-toggle:hover { background: #eee; }
.inif-message-toggle.active { background: #555; color: #fff; border-color: #555; }
.inif-message-body {
    padding: 10px 14px;
    font-size: 0.95em;
}
.inif-message-md-source { display: none; }
.inif-message-md-rendered {
    word-wrap: break-word;
}
/* Long messages clip to ~8 lines; the wrapper carries a relative anchor for
   the absolute-positioned "Show full text" overlay button. */
.inif-message-md.collapsible {
    position: relative;
}
.inif-message-md.collapsible .inif-message-md-rendered {
    max-height: 13em;
    overflow: hidden;
    -webkit-mask-image: linear-gradient(to bottom, #000 70%, transparent 100%);
            mask-image: linear-gradient(to bottom, #000 70%, transparent 100%);
}
.inif-message-md.collapsible.expanded .inif-message-md-rendered {
    max-height: none;
    -webkit-mask-image: none;
            mask-image: none;
}
.inif-message-md-expand {
    position: absolute;
    right: 8px;
    bottom: 8px;
    background: #fff;
    border: 1px solid #999;
    border-radius: 3px;
    padding: 2px 10px;
    font-size: 0.8em;
    color: #555;
    cursor: pointer;
    transition: opacity 0.15s;
}
/* Collapsed: button is always visible so the user can see there's more
   content hidden under the fade. Expanded: button only shows on hover so
   it doesn't compete with the now-fully-visible text. */
.inif-message-md.collapsible.expanded .inif-message-md-expand {
    opacity: 0;
    pointer-events: none;
}
.inif-message-md.collapsible.expanded:hover .inif-message-md-expand {
    opacity: 1;
    pointer-events: auto;
}
.inif-message-md-expand:hover { background: #eee; }
.inif-message-md-rendered p {
    margin: 0 0 8px 0;
}
.inif-message-md-rendered p:last-child {
    margin-bottom: 0;
}
.inif-message-md-rendered h1,
.inif-message-md-rendered h2,
.inif-message-md-rendered h3,
.inif-message-md-rendered h4,
.inif-message-md-rendered h5,
.inif-message-md-rendered h6 {
    margin: 12px 0 6px 0;
    font-weight: 600;
    color: #444;
}
.inif-message-md-rendered h1 { font-size: 1.25em; }
.inif-message-md-rendered h2 { font-size: 1.15em; }
.inif-message-md-rendered h3 { font-size: 1.05em; }
.inif-message-md-rendered h4,
.inif-message-md-rendered h5,
.inif-message-md-rendered h6 { font-size: 1em; }
.inif-message-md-rendered a {
    color: #2980b9;
    text-decoration: none;
}
.inif-message-md-rendered a:hover { text-decoration: underline; }
.inif-message-md-rendered code.inif-inline-code {
    background: #f4f4f4;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.9em;
    color: #c0392b;
}
.inif-message-md-rendered pre.inif-code-block {
    background: #2b2b2b;
    color: #f8f8f2;
    padding: 10px 12px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 6px 0;
    font-size: 0.85em;
    line-height: 1.4;
}
.inif-message-md-rendered pre.inif-code-block code {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    background: transparent;
    color: inherit;
    padding: 0;
}
.inif-message-md-rendered blockquote {
    border-left: 3px solid #d0d0d0;
    padding: 2px 12px;
    color: #666;
    margin: 8px 0;
    background: #f9f9f9;
}
.inif-message-md-rendered ul,
.inif-message-md-rendered ol {
    margin: 4px 0 8px 24px;
    padding: 0;
}
.inif-message-md-rendered li { margin: 2px 0; }
.inif-message-md-rendered table {
    margin: 6px 0;
    border-collapse: collapse;
}
.inif-message-md-rendered th,
.inif-message-md-rendered td {
    border: 1px solid #ddd;
    padding: 4px 8px;
}
.inif-message-md-rendered .inif-math-inline {
    background: #fffaf0;
    border: 1px solid #f0e2c0;
    padding: 0 4px;
    border-radius: 3px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.9em;
    color: #8e6e2a;
}
.inif-message-md-rendered .inif-math-block {
    background: #fffaf0;
    border: 1px solid #f0e2c0;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 4px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.95em;
    color: #8e6e2a;
    text-align: center;
    overflow-x: auto;
}
.inif-message-empty {
    color: #999;
    font-style: italic;
    padding: 4px 0;
}
.inif-message-tokens {
    padding: 8px 10px;
}
/* Per-section panels (reasoning / content / tool calls) inside a message
   body. The reasoning + tool-calls panels carry a label and a tinted left
   border so they read as distinct boxes; the plain-content section is
   unstyled so it looks like the regular message body. */
.inif-section { margin: 6px 0; }
.inif-section:first-child { margin-top: 0; }
.inif-section:last-child { margin-bottom: 0; }
.inif-section-label {
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.inif-section-label-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 4px;
}
.inif-section-label-row .inif-section-label { margin-bottom: 0; }
.inif-section-meta {
    font-size: 0.75em;
    color: #999;
    margin-bottom: 4px;
}
.inif-section-range {
    font-size: 0.7em;
    color: #aaa;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
}
.inif-section-reasoning {
    background: #fffaf0;
    border-left: 3px solid #d4a574;
    padding: 8px 12px;
    border-radius: 0 4px 4px 0;
    font-size: 0.92em;
    color: #5a4628;
}
.inif-section-reasoning .inif-section-label { color: #b07a3a; }
.inif-section-tool-calls {
    background: #f6f1fb;
    border-left: 3px solid #9b6bc7;
    padding: 8px 12px;
    border-radius: 0 4px 4px 0;
}
.inif-section-tool-calls .inif-section-label { color: #6e3fa3; }
.inif-tool-call {
    background: #fff;
    border: 1px solid #e0d6e8;
    border-radius: 4px;
    overflow: hidden;
}
.inif-tool-call + .inif-tool-call { margin-top: 6px; }
.inif-tool-call-header {
    background: #efe4f7;
    padding: 4px 10px;
    font-size: 0.85em;
    color: #4a2a6e;
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.inif-tool-call-name {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-weight: 600;
}
.inif-tool-call-id {
    font-size: 0.85em;
    color: #8a6db0;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
}
.inif-tool-call-args {
    margin: 0;
    padding: 8px 10px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.8em;
    background: #fafafa;
    color: #333;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
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
    return r"""<script>
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
        btn.textContent = c ? '›' : '‹';
    });

    /* Per-message eye toggle: switch between markdown and tokens. The text
       wrapper holds every section (reasoning / content / tool calls) so a
       single toggle hides them all together. */
    document.addEventListener('click', function(e) {
        var btn = e.target.closest('.inif-message-toggle');
        if (!btn) return;
        var msg = btn.closest('.inif-message');
        if (!msg) return;
        var text = msg.querySelector('.inif-message-text');
        var toks = msg.querySelector('.inif-message-tokens');
        var showingTokens = btn.classList.toggle('active');
        if (text) text.hidden = showingTokens;
        if (toks) toks.hidden = !showingTokens;
        btn.textContent = showingTokens ? 'text' : 'tokens';
        btn.setAttribute('title', showingTokens ? 'Show text' : 'Show tokens');
    });

    /* Shared: recompute token backgrounds from toggle states */
    function updateTokenBgs(panel) {
        var ac = panel.querySelector('.inif-annotation-toggle');
        var annotationOn = ac && ac.checked;
        var tokens = panel.querySelectorAll('.inif-token');
        for (var i = 0; i < tokens.length; i++) {
            var tok = tokens[i];
            var bg = '';
            if (annotationOn) {
                var tb = tok.getAttribute('data-annotation-bg');
                if (tb) bg = tb;
            }
            tok.style.background = bg;
        }
    }

    /* Annotation highlight checkbox */
    document.addEventListener('change', function(e) {
        var cb = e.target;
        if (!cb.classList) return;
        if (!cb.classList.contains('inif-annotation-toggle')) return;
        var panel = cb.closest('.inif-sample-panel');
        if (panel) updateTokenBgs(panel);
    });

    /* --- Minimal markdown renderer ------------------------------------- */

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderMarkdown(src) {
        if (!src) return '';
        // Normalize line endings.
        src = src.replace(/\r\n?/g, '\n');

        // 1. Pull out code fences first so their contents stay literal.
        var codeBlocks = [];
        src = src.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, function(_, lang, code) {
            var i = codeBlocks.length;
            codeBlocks.push({lang: (lang || '').trim(), code: code.replace(/\n$/, '')});
            return '\x01CB' + i + '\x01';
        });

        // 2. Block math $$...$$
        var blockMath = [];
        src = src.replace(/\$\$([\s\S]+?)\$\$/g, function(_, expr) {
            var i = blockMath.length;
            blockMath.push(expr);
            return '\x01BM' + i + '\x01';
        });

        // 3. Inline code `...`
        var inlineCode = [];
        src = src.replace(/`([^`\n]+)`/g, function(_, c) {
            var i = inlineCode.length;
            inlineCode.push(c);
            return '\x01IC' + i + '\x01';
        });

        // 4. Inline math $...$ (single-line, no consecutive $$).
        var inlineMath = [];
        src = src.replace(/(^|[^\$])\$([^\$\n]+?)\$(?!\$)/g, function(m, pre, expr) {
            var i = inlineMath.length;
            inlineMath.push(expr);
            return pre + '\x01IM' + i + '\x01';
        });

        // Escape the remaining text.
        src = escapeHtml(src);

        // Headers (### ... etc.).
        src = src.replace(/^######\s+(.*)$/gm, '<h6>$1</h6>');
        src = src.replace(/^#####\s+(.*)$/gm, '<h5>$1</h5>');
        src = src.replace(/^####\s+(.*)$/gm, '<h4>$1</h4>');
        src = src.replace(/^###\s+(.*)$/gm, '<h3>$1</h3>');
        src = src.replace(/^##\s+(.*)$/gm, '<h2>$1</h2>');
        src = src.replace(/^#\s+(.*)$/gm, '<h1>$1</h1>');

        // Blockquote (one level).
        src = src.replace(/(^|\n)((?:&gt;\s.*(?:\n|$))+)/g, function(_, lead, block) {
            var inner = block.replace(/^&gt;\s?/gm, '').replace(/\n$/, '');
            return lead + '<blockquote>' + inner.replace(/\n/g, '<br>') +
                '</blockquote>\n';
        });

        // Bullet lists.
        src = src.replace(/(^|\n)((?:[-*]\s.+(?:\n|$))+)/g, function(_, lead, block) {
            var lines = block.trim().split('\n');
            var items = lines.map(function(l) {
                return '<li>' + l.replace(/^[-*]\s+/, '') + '</li>';
            });
            return lead + '<ul>' + items.join('') + '</ul>\n';
        });

        // Numbered lists.
        src = src.replace(/(^|\n)((?:\d+\.\s.+(?:\n|$))+)/g, function(_, lead, block) {
            var lines = block.trim().split('\n');
            var items = lines.map(function(l) {
                return '<li>' + l.replace(/^\d+\.\s+/, '') + '</li>';
            });
            return lead + '<ol>' + items.join('') + '</ol>\n';
        });

        // Bold and italic.
        src = src.replace(/\*\*([^\*\n]+)\*\*/g, '<strong>$1</strong>');
        src = src.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
        src = src.replace(/(^|[^\*])\*([^\*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
        src = src.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, '$1<em>$2</em>');

        // Markdown links [text](url).
        src = src.replace(/\[([^\]]+)\]\(([^\)\s]+)\)/g, function(_, text, url) {
            return '<a href="' + url + '" target="_blank" rel="noopener">' +
                text + '</a>';
        });

        // Auto-linkify bare URLs.
        src = src.replace(/(^|[\s\(])((?:https?|ftp):\/\/[^\s<>")]+)/g,
            function(_, lead, url) {
                return lead + '<a href="' + url + '" target="_blank" rel="noopener">' +
                    url + '</a>';
            });

        // Wrap remaining paragraphs.
        var blocks = src.split(/\n{2,}/);
        src = blocks.map(function(b) {
            b = b.replace(/^\n+|\n+$/g, '');
            if (!b) return '';
            if (/^<(h[1-6]|ul|ol|blockquote|pre|table|div|figure)/i.test(b)) return b;
            if (/^\x01CB\d+\x01$/.test(b) || /^\x01BM\d+\x01$/.test(b)) return b;
            return '<p>' + b.replace(/\n/g, '<br>') + '</p>';
        }).join('\n');

        // Restore code fences, math, inline code.
        src = src.replace(/\x01CB(\d+)\x01/g, function(_, i) {
            var b = codeBlocks[+i];
            var lang = b.lang ? ' data-lang="' + escapeHtml(b.lang) + '"' : '';
            return '<pre class="inif-code-block"' + lang + '><code>' +
                escapeHtml(b.code) + '</code></pre>';
        });
        src = src.replace(/\x01BM(\d+)\x01/g, function(_, i) {
            return '<div class="inif-math-block">' +
                escapeHtml(blockMath[+i]) + '</div>';
        });
        src = src.replace(/\x01IC(\d+)\x01/g, function(_, i) {
            return '<code class="inif-inline-code">' +
                escapeHtml(inlineCode[+i]) + '</code>';
        });
        src = src.replace(/\x01IM(\d+)\x01/g, function(_, i) {
            return '<span class="inif-math-inline">' +
                escapeHtml(inlineMath[+i]) + '</span>';
        });

        return src;
    }

    /* Render markdown for every embedded message text on load, then mark
       any message whose rendered text overflows the 8-line clip box as
       collapsible (so the "Show full text" overlay shows on hover). */
    function renderAllMarkdown(root) {
        var srcs = (root || document).querySelectorAll('.inif-message-md-source');
        for (var i = 0; i < srcs.length; i++) {
            var src = srcs[i];
            var md = src.parentNode;
            var rendered = md.querySelector('.inif-message-md-rendered');
            if (!rendered) continue;
            var raw = src.textContent;
            rendered.innerHTML = raw.length
                ? renderMarkdown(raw)
                : '<span class="inif-message-empty">(empty)</span>';
        }
        // After layout, decide which messages need the collapse treatment.
        // ``scrollHeight > clientHeight`` means the rendered content is
        // taller than the 8-line clip; for those, mount the overlay button.
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(markCollapsibleMessages);
        } else {
            markCollapsibleMessages();
        }
    }

    function markCollapsibleMessages() {
        var mds = document.querySelectorAll('.inif-message-md');
        for (var i = 0; i < mds.length; i++) {
            var md = mds[i];
            // Reset any prior decision before measuring.
            md.classList.remove('collapsible');
            md.classList.remove('expanded');
            var rendered = md.querySelector('.inif-message-md-rendered');
            if (!rendered) continue;
            // Temporarily clip to detect overflow against the 8-line max.
            md.classList.add('collapsible');
            var overflows = rendered.scrollHeight - rendered.clientHeight > 1;
            if (!overflows) {
                md.classList.remove('collapsible');
                continue;
            }
            if (md.querySelector('.inif-message-md-expand')) continue;
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'inif-message-md-expand';
            btn.textContent = 'Show full text';
            md.appendChild(btn);
        }
    }

    /* Toggle the per-message expand button. */
    document.addEventListener('click', function(e) {
        var btn = e.target.closest('.inif-message-md-expand');
        if (!btn) return;
        var md = btn.closest('.inif-message-md');
        if (!md) return;
        var expanded = md.classList.toggle('expanded');
        btn.textContent = expanded ? 'Show less' : 'Show full text';
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            renderAllMarkdown();
        });
    } else {
        renderAllMarkdown();
    }
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
    binary outcome — callers should keep looking.
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
            return '<span class="inif-em-pass">✓</span>'
        if result is False:
            return '<span class="inif-em-fail">✗</span>'
    return ""


def _render_sidebar(doc_data: dict, samples: list[dict]) -> str:
    parts = ['<div class="inif-sidebar">']
    parts.append('<div class="inif-sidebar-header">')
    parts.append("<span>Samples</span>")
    parts.append('<button class="inif-sidebar-toggle">‹</button>')
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


def _sample_has_annotations(sample_data: dict) -> bool:
    return bool(sample_data.get("annotations"))


def _collect_active_annotations(sample_data: dict) -> dict[str, str]:
    """Map each annotation present in the sample to its color."""
    annotations: dict[str, str] = {}
    for annotation in sample_data.get("annotations", []):
        name = annotation.get("name")
        if name and isinstance(name, str) and name not in annotations:
            annotations[name] = _annotation_color(name)
    return annotations


# Annotation precedence for token highlighting. Lower-priority groups sit
# behind higher-priority ones, so a token tagged both ``assistant`` and
# ``reasoning`` paints with the reasoning color (and any user-defined tag
# wins over both). Within a group, the first-encountered annotation wins.
_CHAT_ROLE_ANNOTATIONS = frozenset({"system", "user", "assistant", "tool", "template"})
_AUTO_SUBTEXT_ANNOTATIONS = frozenset({"reasoning", "tool_call"})


def _annotation_priority(name: str) -> int:
    """Return 0 (chat-template role) / 1 (auto sub-text) / 2 (user-defined).

    Higher values render in front of lower values, so the highest-priority
    annotation drives a token's background color.
    """
    if name in _CHAT_ROLE_ANNOTATIONS:
        return 0
    if name in _AUTO_SUBTEXT_ANNOTATIONS:
        return 1
    return 2


def _token_annotation_names(sample_data: dict) -> dict[int, list[str]]:
    """Return ``pos → annotation_names`` ordered by display priority.

    Highest-priority annotations come FIRST so callers that pick
    ``annotations[0]`` (token background, tooltip lead) automatically use
    the most informative tag for the position.
    """
    by_pos: dict[int, list[str]] = {}
    for annotation in sample_data.get("annotations", []):
        name = annotation.get("name")
        if not name:
            continue
        for start, end in annotation.get("ranges", []):
            for pos in range(start, end):
                by_pos.setdefault(pos, []).append(name)
    for pos, names in by_pos.items():
        # Stable sort by descending priority — preserves source order for
        # ties (two user-defined tags on the same token keep the order they
        # were declared in).
        by_pos[pos] = sorted(names, key=lambda n: -_annotation_priority(n))
    return by_pos


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


def _render_annotation_legend(active_annotations: dict[str, str]) -> str:
    if not active_annotations:
        return ""
    parts = ['<div class="inif-annotation-legend">']
    for name, color in active_annotations.items():
        esc = _escape(name)
        parts.append(
            f'<span><span class="inif-annotation-swatch" '
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
    has_annotations: bool,
    active_annotations: dict[str, str],
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
    if has_annotations:
        parts.append(
            '<label><input type="checkbox" class="inif-annotation-toggle" checked>'
            " Highlight annotations</label>"
        )
    else:
        parts.append(
            '<label class="disabled">'
            '<input type="checkbox" class="inif-annotation-toggle" disabled>'
            " Highlight annotations</label>"
        )

    annotation_legend = _render_annotation_legend(active_annotations)
    if annotation_legend:
        parts.append(annotation_legend)

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
    annotations: list[str],
    annotation_colors: dict[str, str],
    extra_colors: dict[str, str] | None = None,
    newline_chars: frozenset[str] = _DEFAULT_NL,
) -> str:
    # Sequence refs serialize without ``id`` (None is stripped); the ``token``
    # field then carries the target Sequence id. ``_seq_ref`` is set by the
    # token-strip pass when it pre-expands a ref into per-piece sub-tokens.
    raw_id = tok_data.get("id")
    tok_id = 0 if raw_id is None else raw_id
    tok_str = tok_data.get("token") or ""
    is_ref = raw_id is None or bool(tok_data.get("_seq_ref"))
    seq_id = tok_str if raw_id is None else tok_data.get("_ref_target")

    # Extra fields for tooltip (exclude id, token, _seq_ref bookkeeping)
    skip = ("id", "token", "_seq_ref", "_ref_target")
    extra = {k: v for k, v in tok_data.items() if k not in skip}
    if annotations:
        extra["annotations"] = annotations
    tooltip_json = html.escape(json.dumps(extra, default=str), quote=True)

    classes = ["inif-token"]
    style_parts = []
    data_attrs: list[str] = []

    annotation_bg = ""
    if annotations:
        annotation_bg = annotation_colors.get(
            annotations[0], _annotation_color(annotations[0])
        )
        data_attrs.append(f'data-annotation-bg="{annotation_bg}"')
        data_attrs.append(
            f'data-annotations="{html.escape(",".join(annotations), quote=True)}"'
        )
        style_parts.append(f"background:{annotation_bg}")

    # Sequence ref — only apply fallback text if not already expanded
    if is_ref:
        classes.append("seq-ref")
        # When the token-strip pass already expanded the ref into a piece,
        # ``raw_id`` is the piece's vocab id and ``tok_str`` is the piece;
        # leave both alone. When the ref was passed through unexpanded,
        # ``tok_str`` currently holds the target Sequence id — replace it
        # with a concatenated fallback or a placeholder.
        if raw_id is None:
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
    token_range: tuple[int, int] | None = None,
) -> str:
    """Render a strip of token spans for the sample.

    When ``token_range`` is given, only tokens whose native (sample.tokens)
    index falls in ``[start, end)`` are rendered. Position numbers are
    preserved (so annotation / span overlays still line up across
    per-message and full-document views).
    """
    seq_map = {s["id"]: s for s in sequences}

    # Build span position -> color map
    span_positions: dict[int, str] = {}
    for i, span in enumerate(sample_data.get("spans", [])):
        color = _SPAN_PALETTE[i % len(_SPAN_PALETTE)]
        for pos in span.get("positions", []):
            if pos not in span_positions:
                span_positions[pos] = color

    annotation_names = _token_annotation_names(sample_data)
    tokens = sample_data.get("tokens", [])
    if token_range is not None:
        start, end = token_range
        start = max(0, start)
        end = min(len(tokens), end)
        token_iter = list(enumerate(tokens))[start:end]
    else:
        token_iter = list(enumerate(tokens))
    has_annotations = _sample_has_annotations(sample_data)
    ha = "true" if has_annotations else "false"
    annotation_colors = _collect_active_annotations(sample_data)
    parts = [f'<div class="inif-token-strip" data-has-annotations="{ha}">']
    for idx, tok in token_iter:
        raw_id = tok.get("id")
        # Sequence refs serialize as ``{"token": "<seq_id>"}`` (id is None and
        # stripped by compact mode). Expand each ref into wrappable per-piece
        # sub-tokens so newlines inside the run still break visually.
        if raw_id is None:
            seq_id = tok.get("token")
            if seq_id and seq_id in seq_map:
                seq_toks = seq_map[seq_id].get("tokens", [])
            else:
                seq_toks = []
            if seq_toks:
                for sub_tok in seq_toks:
                    sub_str = sub_tok.get("token") or ""
                    sub = {
                        "id": sub_tok.get("id"),
                        "token": sub_str,
                        "_seq_ref": True,
                        "_ref_target": seq_id,
                    }
                    parts.append(
                        _render_token(
                            sub,
                            idx,
                            span_positions,
                            seq_map,
                            annotation_names.get(idx, []),
                            annotation_colors,
                            extra_colors,
                            newline_chars,
                        )
                    )
                    if any(ch in sub_str for ch in newline_chars):
                        parts.append('<div class="inif-line-break"></div>')
                continue
        parts.append(
            _render_token(
                tok,
                idx,
                span_positions,
                seq_map,
                annotation_names.get(idx, []),
                annotation_colors,
                extra_colors,
                newline_chars,
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


def _role_from_text_name(name: str) -> str | None:
    """Pull the role prefix off a role-named text (``user_0`` → ``user``)."""
    if "_" not in name:
        return None
    head, _, tail = name.rpartition("_")
    if head and tail.isdigit():
        return head
    return None


def _texts_with_offsets(sample_data: dict) -> list[dict]:
    """Filter ``sample.texts`` down to the entries that carry both offsets."""
    texts = sample_data.get("texts", [])
    out: list[dict] = []
    for t in texts:
        if not isinstance(t, dict):
            continue
        if t.get("start") is not None and t.get("end") is not None:
            out.append(t)
    return out


def _render_markdown_section(raw_value: str) -> str:
    """Wrap raw markdown source in the `.inif-message-md` shell the JS
    renderer scans on load. Each shell is independently collapsible."""
    return (
        '<div class="inif-message-md">'
        f'<pre class="inif-message-md-source">{html.escape(raw_value)}</pre>'
        '<div class="inif-message-md-rendered"></div>'
        "</div>"
    )


def _format_tool_call_args(value: str) -> str:
    """Pretty-print a tool-call arguments JSON string.

    The converter stores ``arguments`` as a JSON-encoded string on each
    tool-call child; re-parsing for indentation gives a readable block.
    Falls back to the raw string when parsing fails (e.g. a non-JSON
    payload).
    """
    if not value:
        return ""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _render_tool_call_child(child: dict) -> str:
    """Render one tool-call child as a name header + args code block."""
    name = child.get("name") or "(unnamed)"
    md = child.get("metadata") or {}
    call_id = md.get("id") or ""
    args = _format_tool_call_args(child.get("value") or "")
    parts = ['<div class="inif-tool-call">']
    parts.append('<div class="inif-tool-call-header">')
    parts.append(f'<span class="inif-tool-call-name">{html.escape(str(name))}</span>')
    if call_id:
        parts.append(
            f'<span class="inif-tool-call-id">{html.escape(str(call_id))}</span>'
        )
    parts.append("</div>")
    parts.append(f'<pre class="inif-tool-call-args">{html.escape(args)}</pre>')
    parts.append("</div>")
    return "".join(parts)


_SECTION_LABELS = {
    "reasoning": "Reasoning",
    "tool_calls": "Tool calls",
}


def _section_range_badge(child: dict) -> str:
    start = child.get("start")
    end = child.get("end")
    if start is None or end is None:
        return ""
    return (
        f'<span class="inif-section-range">[{int(start)}, {int(end)}) · '
        f"{int(end) - int(start)} tokens</span>"
    )


def _render_child_section(child: dict) -> str:
    """Render one direct child of a message Text as a labeled section.

    The child's ``name`` selects the visual treatment: ``reasoning`` and
    ``tool_calls`` get their own tinted box with a header label and a
    per-section token range badge; ``content`` is rendered as a bare
    markdown body so it reads as the "main" message text. Anything else
    falls back to a generic markdown block labeled with the child's name.
    """
    name = child.get("name", "")
    range_badge = _section_range_badge(child)
    if name == "content":
        body = _render_markdown_section(child.get("value") or "")
        if range_badge:
            return (
                '<div class="inif-section inif-section-content">'
                f'<div class="inif-section-meta">{range_badge}</div>'
                f"{body}</div>"
            )
        return f'<div class="inif-section inif-section-content">{body}</div>'
    if name == "tool_calls":
        parts = ['<div class="inif-section inif-section-tool-calls">']
        parts.append('<div class="inif-section-label-row">')
        parts.append('<div class="inif-section-label">Tool calls</div>')
        if range_badge:
            parts.append(range_badge)
        parts.append("</div>")
        for call in child.get("children") or []:
            parts.append(_render_tool_call_child(call))
        parts.append("</div>")
        return "".join(parts)
    label = _SECTION_LABELS.get(name, name)
    css_class = (
        "inif-section-reasoning" if name == "reasoning" else "inif-section-generic"
    )
    parts = [f'<div class="inif-section {css_class}">']
    parts.append('<div class="inif-section-label-row">')
    parts.append(f'<div class="inif-section-label">{html.escape(str(label))}</div>')
    if range_badge:
        parts.append(range_badge)
    parts.append("</div>")
    parts.append(_render_markdown_section(child.get("value") or ""))
    parts.append("</div>")
    return "".join(parts)


def _render_message_text_sections(text: dict) -> str:
    """Render the text-mode body of one message.

    With nested children: emit one labeled section per child (reasoning,
    content, tool calls). Without children: render the message's own
    ``value`` as a single markdown block. When everything is empty,
    render a ``(empty)`` placeholder.
    """
    children = text.get("children") or []
    if children:
        return "".join(_render_child_section(c) for c in children)
    raw_value = text.get("value") or ""
    if not raw_value:
        return '<span class="inif-message-empty">(empty)</span>'
    return _render_markdown_section(raw_value)


def _render_messages_panel(
    sample_data: dict,
    sequences: list[dict],
    extra_colors: dict[str, str],
    newline_chars: frozenset[str] = _DEFAULT_NL,
) -> str:
    """Render the per-message panels with markdown body + token toggle.

    Each Text with ``start`` / ``end`` becomes one collapsible message
    panel: by default it shows reasoning / content / tool calls as
    distinct sections, and clicking the eye toggle swaps the whole text
    wrapper for the token strip covering ``[start, end)`` of
    ``sample.tokens``. Returns ``""`` when there are no offsetted texts
    (caller falls back to a full token strip).
    """
    texts = _texts_with_offsets(sample_data)
    if not texts:
        return ""
    parts = ['<div class="inif-messages">']
    for text in texts:
        name = text.get("name", "")
        role = _role_from_text_name(name)
        start = int(text["start"])
        end = int(text["end"])
        n = end - start
        role_attr = f' data-role="{html.escape(role, quote=True)}"' if role else ""
        meta_html = (
            f'<span class="inif-message-meta">[{start}, {end}) · {n} tokens</span>'
        )
        parts.append(
            f'<div class="inif-message"{role_attr} '
            f'data-text-name="{html.escape(name, quote=True)}" '
            f'data-start="{start}" data-end="{end}">'
        )
        parts.append('<div class="inif-message-header">')
        parts.append(f'<span class="inif-message-name">{html.escape(name)}</span>')
        parts.append(meta_html)
        parts.append(
            '<button class="inif-message-toggle" type="button" '
            'title="Show tokens">tokens</button>'
        )
        parts.append("</div>")
        # Text body (default): reasoning / content / tool-calls sections.
        parts.append('<div class="inif-message-body inif-message-text">')
        parts.append(_render_message_text_sections(text))
        parts.append("</div>")
        # Tokens body (hidden until toggle).
        parts.append('<div class="inif-message-body inif-message-tokens" hidden>')
        if n > 0:
            parts.append(
                _render_token_strip(
                    sample_data,
                    sequences,
                    extra_colors,
                    newline_chars,
                    token_range=(start, end),
                )
            )
        else:
            parts.append('<div class="inif-message-empty">(no tokens)</div>')
        parts.append("</div>")
        parts.append("</div>")  # message
    parts.append("</div>")  # messages
    return "\n".join(parts)


def _render_texts_panel(sample_data: dict) -> str:
    """Legacy fallback: a flat list of ``{name, value}`` entries.

    Only used when the document predates the message-panel layout (no
    ``start`` / ``end`` on any text); current converters always populate
    those, so this is purely for backwards compatibility with old archives.
    """
    texts = sample_data.get("texts", [])
    if not texts:
        return ""
    if any(_texts_with_offsets({"texts": [t]}) for t in texts):
        return ""
    parts = ["<h3>Texts</h3>", '<div class="inif-texts-panel">']
    for i, text in enumerate(texts):
        if isinstance(text, dict):
            name = html.escape(str(text.get("name", str(i))))
            value = html.escape(str(text.get("value", "")))
        else:
            name = str(i)
            value = html.escape(str(text))
        parts.append(
            f'<div class="inif-text-item"><strong>{name}:</strong> {value}</div>'
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
    has_annotations = _sample_has_annotations(sample_data)
    active_annotations = _collect_active_annotations(sample_data)
    extra_colors = _collect_extra_field_colors(tokens)
    parts = []

    header_html = _render_sample_header(
        sample_data, has_annotations, active_annotations, extra_colors
    )
    if header_html:
        parts.append(header_html)
    elif compact:
        pass  # skip empty

    messages_html = _render_messages_panel(
        sample_data, sequences, extra_colors, newline_chars
    )
    if messages_html:
        parts.append(messages_html)
    else:
        # Old archives without per-text offsets: fall back to a single
        # full-document token strip plus the legacy "Texts" panel.
        parts.append(
            _render_token_strip(sample_data, sequences, extra_colors, newline_chars)
        )
        texts_html = _render_texts_panel(sample_data)
        if texts_html:
            parts.append(texts_html)

    spans_html = _render_spans_legend(sample_data)
    if spans_html:
        parts.append(spans_html)

    scores_html = _render_scores_panel(sample_data)
    if scores_html:
        parts.append(scores_html)

    return "\n".join(parts)


def _render_html(
    doc: InifDocument,
    compact: bool = False,
    title: str | None = None,
    tokenizer: Any = None,
) -> str:
    """Implementation backing :meth:`InifDocument.render_html`.

    When *tokenizer* is provided, the tokenizer's byte-level representation
    of newlines (e.g. ``Ċ`` for GPT-2 family) is detected automatically so
    that visual line breaks are inserted after newline tokens.
    """
    newline_chars = _detect_newline_chars(tokenizer)
    doc_data = _to_dict(doc, compact=False)
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


def _show(
    doc: InifDocument,
    compact: bool = False,
    title: str | None = None,
    tokenizer: Any = None,
) -> Any:
    """Implementation backing :meth:`InifDocument.show`.

    Pass *tokenizer* to enable line breaks after BPE newline tokens.
    """
    from IPython.display import HTML

    return HTML(_render_html(doc, compact=compact, title=title, tokenizer=tokenizer))


def _save_html(
    doc: InifDocument,
    path: str | Path,
    compact: bool = False,
    title: str | None = None,
    source: str | Path | None = None,
    tokenizer: Any = None,
) -> None:
    """Implementation backing :meth:`InifDocument.save_html`.

    When *title* is not given, the source filename is used if available,
    otherwise falls back to the model name.  Pass *tokenizer* to enable
    line breaks after BPE newline tokens.
    """
    path = Path(path)
    if title is None and source is not None:
        title = Path(source).name
    html_str = _render_html(doc, compact=compact, title=title, tokenizer=tokenizer)
    path.write_text(html_str, encoding="utf-8")
