"""
ui/static_analysis_page.py — Static Security Analysis UI (Phase 3).

Allows the user to paste or upload Python source code, run Bandit-based
static analysis, and inspect structured findings with severity badges,
code snippets, and CWE references.
"""

from __future__ import annotations

import io
import json

import streamlit as st

from agents.static_analysis_agent import (
    FindingConfidence,
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisInput,
    StaticAnalysisResult,
    analyse,
)
from tools.language_utils import detect_language


# ─────────────────────────────────────────────────────────────────────────────
# Severity & confidence styling helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_COLOR = {
    FindingSeverity.CRITICAL: ("#ff4b4b", "#2d0a0a"),
    FindingSeverity.HIGH:     ("#f0a500", "#2d1a00"),
    FindingSeverity.MEDIUM:   ("#3b82f6", "#0a1628"),
    FindingSeverity.LOW:      ("#57606a", "#1c2230"),
    FindingSeverity.INFO:     ("#8b949e", "#1c2230"),
}

_SEVERITY_ICON = {
    FindingSeverity.CRITICAL: "🔴",
    FindingSeverity.HIGH:     "🟠",
    FindingSeverity.MEDIUM:   "🔵",
    FindingSeverity.LOW:      "⚪",
    FindingSeverity.INFO:     "ℹ️",
}

_CONFIDENCE_COLOR = {
    FindingConfidence.HIGH:   "#3fb950",
    FindingConfidence.MEDIUM: "#d29922",
    FindingConfidence.LOW:    "#8b949e",
}

# ─────────────────────────────────────────────────────────────────────────────
# Example vulnerable snippets for quick testing
# ─────────────────────────────────────────────────────────────────────────────

_EXAMPLE_SNIPPETS = {
    "OS Command Injection": """\
import os

user_input = input("Enter command: ")
os.system(user_input)
""",
    "Subprocess Shell Injection": """\
import subprocess

cmd = "ls -la"
subprocess.call(cmd, shell=True)
subprocess.Popen("echo hello", shell=True)
""",
    "eval() on User Input": """\
data = input("Enter expression: ")
result = eval(data)
print(result)
""",
    "Hardcoded Password + Weak Hash": """\
import hashlib

password = "SuperSecret123!"
hashed = hashlib.md5(password.encode()).hexdigest()
""",
    "SQL Injection": """\
import sqlite3

def get_user(username):
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '%s'" % username
    cursor.execute(query)
    return cursor.fetchall()
""",
    "Multiple Vulnerabilities": """\
import os
import subprocess
import hashlib

# OS command injection
os.system(input("cmd: "))

# Shell injection via subprocess
subprocess.Popen("echo hello", shell=True)

# Weak hash
hashlib.md5(b"data").hexdigest()

# Eval on user input
result = eval(input("expr: "))
""",
    "Clean Code (No Issues)": """\
from __future__ import annotations

import os
import hashlib


def get_secret_from_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"{name} is not set")
    return value


def hash_data(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add(a: int, b: int) -> int:
    return a + b
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_badge(severity: FindingSeverity) -> str:
    color, _ = _SEVERITY_COLOR[severity]
    icon     = _SEVERITY_ICON[severity]
    label    = severity.value.upper()
    return (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};border-radius:4px;padding:2px 8px;'
        f'font-size:.75rem;font-weight:600;">{icon} {label}</span>'
    )


def _confidence_badge(confidence: FindingConfidence) -> str:
    color = _CONFIDENCE_COLOR[confidence]
    label = confidence.value.upper()
    return (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};border-radius:4px;padding:2px 8px;'
        f'font-size:.72rem;font-weight:500;">⬤ {label} confidence</span>'
    )


def _render_summary_banner(result: StaticAnalysisResult) -> None:
    if result.is_clean and not result.errors:
        st.success(
            "**No security issues detected.** Bandit found no vulnerabilities in this code.",
            icon="✅",
        )
        return

    if result.is_clean and result.errors:
        st.warning(
            f"**No findings** — but {len(result.errors)} error(s) occurred during analysis.",
            icon="⚠️",
        )
        return

    sev    = result.highest_severity
    color  = _SEVERITY_COLOR[sev][0]
    icon   = _SEVERITY_ICON[sev]

    # Count by severity
    counts: dict[FindingSeverity, int] = {}
    for f in result.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    breakdown = "  ·  ".join(
        f"{_SEVERITY_ICON[s]} {n} {s.value}"
        for s, n in counts.items()
    )

    if sev in (FindingSeverity.CRITICAL, FindingSeverity.HIGH):
        st.error(
            f"**{result.finding_count} security issue(s) detected** — {breakdown}.",
            icon="🚨",
        )
    elif sev == FindingSeverity.MEDIUM:
        st.warning(
            f"**{result.finding_count} security issue(s) detected** — {breakdown}.",
            icon="⚠️",
        )
    else:
        st.info(
            f"**{result.finding_count} low-severity finding(s)** — {breakdown}.",
            icon="ℹ️",
        )


def _render_finding_card(f: StaticAnalysisFinding) -> None:
    """Render a single finding as a styled card."""
    color, bg = _SEVERITY_COLOR[f.severity]

    cwe_line = ""
    if f.cwe_id:
        cwe_line = (
            f'<span style="font-size:.78rem;color:#3b82f6;'
            f'margin-right:.5rem;">🛡️ {f.cwe_id}</span>'
        )

    more_info_line = ""
    if f.more_info:
        more_info_line = (
            f'<a href="{f.more_info}" target="_blank" '
            f'style="font-size:.75rem;color:#3b82f6;text-decoration:none;">'
            f'📖 Bandit docs →</a>'
        )

    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {color}40;
                    border-left:4px solid {color};border-radius:8px;
                    padding:.85rem 1.1rem;margin-bottom:.6rem;">
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;margin-bottom:.4rem;">
                <span style="font-weight:600;font-size:.9rem;
                             color:#e6edf3;flex:1;padding-right:.5rem;">{f.title}</span>
                <div style="display:flex;gap:.4rem;flex-shrink:0;">
                    {_severity_badge(f.severity)}
                    {_confidence_badge(f.confidence)}
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:.5rem;font-size:.8rem;color:#8b949e;">
                <span>📍 Line <strong style="color:#e6edf3;">{f.line}</strong></span>
                <span>🔬 Test: <strong style="color:#e6edf3;">{f.test_id}</strong>
                    <span style="color:#57606a;">({f.test_name})</span></span>
                <span>{cwe_line}{more_info_line}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if f.code_snippet:
        with st.expander(f"📝 Code snippet — line {f.line}", expanded=False):
            st.code(f.code_snippet, language="python", line_numbers=False)


def _render_findings_list(result: StaticAnalysisResult) -> None:
    st.markdown("#### Security Findings")

    if result.is_clean:
        st.caption("No findings to display.")
        return

    for f in result.findings:
        _render_finding_card(f)


def _render_metrics(result: StaticAnalysisResult) -> None:
    """Show Bandit metrics overview."""
    if not result.bandit_metrics:
        return

    with st.expander("📊 Bandit Metrics", expanded=False):
        # The metrics dict is keyed by filename, plus _totals
        totals = result.bandit_metrics.get("_totals", {})
        if totals:
            severity_counts = {
                "HIGH": totals.get("SEVERITY.HIGH", 0),
                "MEDIUM": totals.get("SEVERITY.MEDIUM", 0),
                "LOW": totals.get("SEVERITY.LOW", 0),
            }
            confidence_counts = {
                "HIGH": totals.get("CONFIDENCE.HIGH", 0),
                "MEDIUM": totals.get("CONFIDENCE.MEDIUM", 0),
                "LOW": totals.get("CONFIDENCE.LOW", 0),
            }

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Severity**")
                for label, count in severity_counts.items():
                    st.markdown(f"- {label}: **{count}**")
            with col2:
                st.markdown("**By Confidence**")
                for label, count in confidence_counts.items():
                    st.markdown(f"- {label}: **{count}**")
        else:
            st.json(result.bandit_metrics)


def _render_errors(result: StaticAnalysisResult) -> None:
    """Show any errors that occurred during analysis."""
    if not result.errors:
        return

    with st.expander(f"⚠️ {len(result.errors)} Error(s)", expanded=True):
        for err in result.errors:
            st.warning(err, icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────────
# Public render function
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Main entry point — called by the page router in main.py."""

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="font-size:2rem;font-weight:700;letter-spacing:-.02em;
                   margin-bottom:.25rem;color:#e6edf3;">
            🛡️ Static Security Analysis
        </h1>
        <p style="color:#8b949e;font-size:1rem;margin-bottom:1.5rem;">
            Run Bandit static analysis to detect security vulnerabilities in Python code.
            For other languages, use the RAG Security Scanner or Logic Reviewer which use LLM-based analysis.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Input mode selector ───────────────────────────────────────────────
    tab_paste, tab_upload, tab_example = st.tabs([
        "📝 Paste Code", "📁 Upload File", "🧪 Examples",
    ])

    source_code: str | None = None
    filename = "input.py"

    with tab_paste:
        pasted = st.text_area(
            "Paste Python code here",
            height=280,
            placeholder="import os\nos.system(input('cmd: '))\n...",
            key="sa_paste_area",
        )
        if pasted.strip():
            source_code = pasted
            filename = "pasted_code.py"

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a Python (.py) file",
            type=["py"],
            key="sa_upload",
            help="Bandit works on Python files. For other languages use the RAG Security Scanner.",
        )
        if uploaded is not None:
            source_code = uploaded.read().decode("utf-8", errors="replace")
            filename = uploaded.name

    with tab_example:
        example_name = st.selectbox(
            "Select a sample snippet",
            options=list(_EXAMPLE_SNIPPETS.keys()),
            key="sa_example_select",
        )
        if st.button("Load Example", key="sa_load_example"):
            source_code = _EXAMPLE_SNIPPETS[example_name]
            filename = f"example_{example_name.lower().replace(' ', '_')}.py"
            st.session_state["sa_loaded_example"] = source_code
            st.session_state["sa_loaded_filename"] = filename

    # Persist example across reruns
    if source_code is None and "sa_loaded_example" in st.session_state:
        source_code = st.session_state["sa_loaded_example"]
        filename = st.session_state.get("sa_loaded_filename", "example.py")

    st.divider()

    # ── Analyse button ────────────────────────────────────────────────────
    if source_code:
        col_preview, col_action = st.columns([3, 1])
        with col_preview:
            with st.expander("📄 Code Preview", expanded=False):
                _, lang_id = detect_language(filename)
                st.code(source_code, language=lang_id, line_numbers=True)
        with col_action:
            run_analysis = st.button(
                "🔍 Run Bandit Analysis",
                type="primary",
                use_container_width=True,
                key="sa_run_btn",
            )
    else:
        run_analysis = False
        st.info(
            "Paste code, upload a file, or load an example to begin analysis.",
            icon="👆",
        )

    # ── Run and display results ───────────────────────────────────────────
    if run_analysis and source_code:
        with st.spinner("Running Bandit static analysis..."):
            result = analyse(
                StaticAnalysisInput(source_code=source_code, filename=filename)
            )

        # Store in session state for persistence
        st.session_state["sa_result"] = result
        st.session_state["sa_source"] = source_code

    # Display results if available
    if "sa_result" in st.session_state:
        result: StaticAnalysisResult = st.session_state["sa_result"]
        analyzed_source: str = st.session_state.get("sa_source", "")

        st.divider()

        # ── Summary banner ────────────────────────────────────────────────
        _render_summary_banner(result)

        # ── Metrics row ───────────────────────────────────────────────────
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Findings", result.finding_count)
        with metric_cols[1]:
            sev_label = result.highest_severity.value.upper() if result.highest_severity else "—"
            st.metric("Highest Severity", sev_label)
        with metric_cols[2]:
            high_count = sum(
                1 for f in result.findings
                if f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH)
            )
            st.metric("High / Critical", high_count)
        with metric_cols[3]:
            st.metric("Errors", len(result.errors))

        # ── Tabs for different views ──────────────────────────────────────
        tab_findings, tab_code, tab_json = st.tabs([
            "🔍 Findings", "📄 Analysed Code", "📋 JSON Export",
        ])

        with tab_findings:
            _render_findings_list(result)
            _render_metrics(result)
            _render_errors(result)

        with tab_code:
            st.code(analyzed_source, language="python", line_numbers=True)

        with tab_json:
            # Build exportable JSON
            export_data = {
                "filename": result.filename,
                "finding_count": result.finding_count,
                "highest_severity": (
                    result.highest_severity.value if result.highest_severity else None
                ),
                "summary": result.summary(),
                "findings": [
                    {
                        "title": f.title,
                        "description": f.description,
                        "severity": f.severity.value,
                        "confidence": f.confidence.value,
                        "file": f.file,
                        "line": f.line,
                        "tool": f.tool,
                        "test_id": f.test_id,
                        "test_name": f.test_name,
                        "cwe_id": f.cwe_id,
                        "code_snippet": f.code_snippet,
                        "more_info": f.more_info,
                    }
                    for f in result.findings
                ],
                "errors": result.errors,
            }
            json_str = json.dumps(export_data, indent=2)
            st.code(json_str, language="json")
            st.download_button(
                "⬇️ Download JSON Report",
                data=json_str,
                file_name=f"bandit_report_{filename}.json",
                mime="application/json",
                key="sa_download_json",
            )
