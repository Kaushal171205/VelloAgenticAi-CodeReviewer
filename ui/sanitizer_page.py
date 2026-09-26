"""
ui/sanitizer_page.py — Secret Sanitization UI (Phase 2).

Allows the user to paste or upload Python source code, run the sanitizer
agent, and inspect original code, sanitized code, and structured findings
side-by-side.
"""

from __future__ import annotations

import io

import streamlit as st

from agents.sanitizer import Severity, SanitizerInput, SanitizerResult, sanitize

# ─────────────────────────────────────────────────────────────────────────────
# Severity styling helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_COLOR = {
    Severity.CRITICAL: ("#ff4b4b", "#2d0a0a"),
    Severity.HIGH:     ("#f0a500", "#2d1a00"),
    Severity.MEDIUM:   ("#3b82f6", "#0a1628"),
    Severity.LOW:      ("#57606a", "#1c2230"),
}

_SEVERITY_ICON = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH:     "🟠",
    Severity.MEDIUM:   "🔵",
    Severity.LOW:      "⚪",
}

_EXAMPLE_SNIPPETS = {
    "Google API Key": """\
import os

# Hardcoded credentials — BAD practice
GOOGLE_API_KEY = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"

def get_maps_data(location: str) -> dict:
    import requests
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": location, "key": GOOGLE_API_KEY},
    )
    return resp.json()
""",
    "Database URL + Password": """\
# Configuration with embedded credentials
DATABASE_URL = "postgresql://admin:S3cr3tPa55@prod.db.example.com:5432/mydb"
DB_PASSWORD  = "correct_horse_battery_staple"

import psycopg2

def get_connection():
    return psycopg2.connect(DATABASE_URL)
""",
    "Multiple Secrets": """\
import os

# Multiple hardcoded secrets
GOOGLE_KEY   = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"
AWS_KEY      = "AKIAIOSFODNN7EXAMPLE"
DB_PASSWORD  = "super_secret_pass_2024"
DATABASE_URL = "postgresql://admin:S3cr3tPa55@db.example.com/prod"
API_TOKEN    = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890ab"

def connect():
    pass
""",
    "Clean Code (No Secrets)": """\
from __future__ import annotations
import os

def get_api_key() -> str:
    \"\"\"Always load secrets from environment variables.\"\"\"
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")
    return key

def fetch_data(endpoint: str) -> dict:
    import requests
    headers = {"Authorization": f"Bearer {get_api_key()}"}
    return requests.get(endpoint, headers=headers).json()
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity_badge(severity: Severity) -> str:
    color, _ = _SEVERITY_COLOR[severity]
    icon      = _SEVERITY_ICON[severity]
    label     = severity.value.upper()
    return (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};border-radius:4px;padding:2px 8px;'
        f'font-size:.75rem;font-weight:600;">{icon} {label}</span>'
    )


def _render_summary_banner(result: SanitizerResult) -> None:
    if result.is_clean:
        st.success("**No secrets detected.** The code is safe to send to an LLM.", icon="✅")
        return

    sev   = result.highest_severity
    color = _SEVERITY_COLOR[sev][0]
    icon  = _SEVERITY_ICON[sev]

    counts: dict[Severity, int] = {}
    for f in result.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    breakdown = "  ·  ".join(
        f"{_SEVERITY_ICON[s]} {n} {s.value}"
        for s, n in counts.items()
    )

    if sev == Severity.CRITICAL:
        st.error(
            f"**{result.number_of_secrets} secret(s) detected** — {breakdown}. "
            "CRITICAL findings present. Do NOT commit this code.",
            icon="🚨",
        )
    elif sev == Severity.HIGH:
        st.error(
            f"**{result.number_of_secrets} secret(s) detected** — {breakdown}.",
            icon="🔴",
        )
    elif sev == Severity.MEDIUM:
        st.warning(
            f"**{result.number_of_secrets} secret(s) detected** — {breakdown}.",
            icon="⚠️",
        )
    else:
        st.info(
            f"**{result.number_of_secrets} low-confidence finding(s)** — {breakdown}.",
            icon="ℹ️",
        )


def _render_findings_table(result: SanitizerResult) -> None:
    st.markdown("#### Findings")

    if result.is_clean:
        st.caption("No findings to display.")
        return

    for f in result.findings:
        color, bg = _SEVERITY_COLOR[f.severity]

        st.markdown(
            f"""
            <div style="background:{bg};border:1px solid {color}40;
                        border-left:4px solid {color};border-radius:8px;
                        padding:.85rem 1.1rem;margin-bottom:.6rem;">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:.4rem;">
                    <span style="font-weight:600;font-size:.9rem;
                                 color:#e6edf3;">{f.description}</span>
                    {_severity_badge(f.severity)}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                            gap:.5rem;font-size:.8rem;color:#8b949e;">
                    <span>📍 Line <strong style="color:#e6edf3;">{f.line_number}</strong></span>
                    <span>🔍 Detector: <strong style="color:#e6edf3;">{f.detector}</strong></span>
                    <span>🔑 Type: <strong style="color:#e6edf3;">{f.secret_type.value}</strong></span>
                </div>
                <div style="margin-top:.4rem;font-size:.8rem;color:#8b949e;">
                    Masked value: <code style="color:{color};">{f.masked_value}</code>
                    &nbsp;→&nbsp;
                    Placeholder: <code style="color:#3b82f6;">{f.placeholder}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_diff_view(original: str, sanitized: str) -> None:
    """Show original and sanitized side-by-side with line-level highlighting."""
    col_orig, col_san = st.columns(2)

    with col_orig:
        st.markdown("#### Original Code")
        st.code(original, language="python", line_numbers=True)

    with col_san:
        st.markdown("#### Sanitized Code")
        st.code(sanitized, language="python", line_numbers=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public render function
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Secret Sanitization")
    st.caption(
        "Paste or upload Python source code. "
        "The sanitizer will detect and redact secrets before any code touches an LLM."
    )

    # ── Input panel ───────────────────────────────────────────────────────────
    with st.expander("📥  Input", expanded=True):
        input_method = st.radio(
            "Input method",
            ["Paste code", "Upload file", "Load example"],
            horizontal=True,
            key="san_input_method",
        )

        source_code = ""
        filename    = "pasted_code.py"

        if input_method == "Paste code":
            source_code = st.text_area(
                "Python source code",
                height=280,
                placeholder="Paste your Python code here…",
                key="san_paste_area",
            )

        elif input_method == "Upload file":
            uploaded = st.file_uploader(
                "Upload a .py file",
                type=["py", "txt"],
                key="san_uploader",
            )
            if uploaded:
                filename    = uploaded.name
                source_code = io.StringIO(uploaded.read().decode("utf-8", errors="replace")).read()
                st.success(f"Loaded **{filename}** ({len(source_code):,} bytes)")
                st.code(source_code[:500] + ("…" if len(source_code) > 500 else ""),
                        language="python")

        else:  # Load example
            example_name = st.selectbox(
                "Choose an example",
                list(_EXAMPLE_SNIPPETS.keys()),
                key="san_example_select",
            )
            source_code = _EXAMPLE_SNIPPETS[example_name]
            filename    = example_name.lower().replace(" ", "_") + ".py"
            st.code(source_code, language="python", line_numbers=True)

        run_btn = st.button(
            "🔍  Run Sanitizer",
            key="san_run_btn",
            disabled=not source_code.strip(),
            type="primary",
        )

    # ── Results ───────────────────────────────────────────────────────────────
    if run_btn and source_code.strip():
        with st.spinner("Scanning for secrets…"):
            result = sanitize(SanitizerInput(source_code=source_code, filename=filename))

        # Store in session state so the results persist across re-renders
        st.session_state["san_result"] = result

    result: SanitizerResult | None = st.session_state.get("san_result")

    if result is None:
        st.info("Run the sanitizer to see results here.", icon="👆")
        return

    st.divider()

    # ── Summary banner ────────────────────────────────────────────────────────
    _render_summary_banner(result)

    # ── Metrics strip ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Secrets Found",     result.number_of_secrets)
    m2.metric(
        "Highest Severity",
        result.highest_severity.value.capitalize() if result.highest_severity else "None",
    )
    critical_count = sum(1 for f in result.findings if f.severity == Severity.CRITICAL)
    high_count     = sum(1 for f in result.findings if f.severity == Severity.HIGH)
    m3.metric("Critical",  critical_count)
    m4.metric("High",      high_count)

    # ── Warnings from sanitizer ────────────────────────────────────────────────
    for w in result.warnings:
        st.warning(w, icon="⚠️")

    st.divider()

    # ── Tabbed view ───────────────────────────────────────────────────────────
    tab_findings, tab_diff, tab_sanitized = st.tabs([
        f"🔎  Findings ({result.number_of_secrets})",
        "⬛  Side-by-Side Diff",
        "✅  Sanitized Code",
    ])

    with tab_findings:
        _render_findings_table(result)

    with tab_diff:
        _render_diff_view(source_code, result.sanitized_code)

    with tab_sanitized:
        st.markdown("#### Sanitized Code")
        st.caption(
            "All detected secrets have been replaced with structured placeholders. "
            "This is safe to send to an LLM."
        )
        st.code(result.sanitized_code, language="python", line_numbers=True)

        st.download_button(
            "⬇  Download sanitized code",
            data=result.sanitized_code,
            file_name=f"sanitized_{result.filename}",
            mime="text/plain",
            key="san_download_btn",
        )
