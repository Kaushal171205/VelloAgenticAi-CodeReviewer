"""
ui/security_scanner_page.py — Interactive RAG-based Security Scanner Page.

Allows developers to audit code using local ChromaDB vector retrieval augmented
with LLM reasoning. Displays retrieved security references alongside findings.
"""

from __future__ import annotations

import json
from typing import Dict

import streamlit as st

from agents.security_scanner_agent import SecurityScannerAgent, SecurityScannerInput
from agents.static_analysis_agent import FindingSeverity
from config import settings
from tools.language_utils import detect_language

SAMPLE_SECURITY_SNIPPETS: Dict[str, Dict[str, str]] = {
    "💉 SQL Injection (Formatted String in SQLite)": {
        "filename": "user_repo.py",
        "description": "Dynamic SQL query formed with user input without parameterization.",
        "code": '''import sqlite3

def find_user_by_name(username: str):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # Vulnerable query concatenation
    query = f"SELECT id, username, email FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchall()''',
    },
    "🐚 OS Command Injection (os.system ping)": {
        "filename": "diagnostics.py",
        "description": "Accepts user domain and executes shell command without sanitization.",
        "code": '''import os

def ping_host(target_domain: str):
    # Vulnerable to shell injection (e.g. "google.com; cat /etc/passwd")
    command = f"ping -c 1 {target_domain}"
    exit_code = os.system(command)
    return {"status": exit_code == 0}''',
    },
    "🌐 SSRF (Unchecked Webhook Fetch)": {
        "filename": "webhook_caller.py",
        "description": "Fetches user supplied URL directly, permitting access to internal metadata endpoints.",
        "code": '''import requests

def trigger_webhook(callback_url: str, payload: dict):
    # SSRF vulnerability: can target 169.254.169.254 or localhost
    resp = requests.post(callback_url, json=payload, timeout=5)
    return resp.status_code''',
    },
    "📦 Insecure Pickle Deserialization": {
        "filename": "session_store.py",
        "description": "Deserializes raw session tokens using pickle.loads without validation.",
        "code": '''import pickle
import base64

def restore_session(session_cookie: str):
    # Insecure deserialization leading to Remote Code Execution (RCE)
    raw_bytes = base64.b64decode(session_cookie)
    user_session = pickle.loads(raw_bytes)
    return user_session''',
    },
}


def _severity_color(sev: FindingSeverity) -> str:
    palette = {
        FindingSeverity.CRITICAL: "#ef4444",
        FindingSeverity.HIGH:     "#f97316",
        FindingSeverity.MEDIUM:   "#eab308",
        FindingSeverity.LOW:      "#3b82f6",
        FindingSeverity.INFO:     "#64748b",
    }
    return palette.get(sev, "#64748b")


def render() -> None:
    """Render the Security Scanner RAG page."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ RAG Security Scanner</h1>
            <p>Audits code for vulnerabilities using local ChromaDB vector search + sentence-transformers embeddings grounded with LLM security analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; font-size: 0.9rem;">
            📚 <strong>Retrieval-Augmented Generation (RAG):</strong> Relevant CVE/CWE reference documents are retrieved from the local ChromaDB vector store and injected into the security analysis prompt.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_samples, col_opts = st.columns([2, 1], gap="medium")
    with col_samples:
        st.markdown("**Load Vulnerable Security Sample:**")
        selected_sample = st.selectbox(
            "Select Vulnerability Pattern",
            options=list(SAMPLE_SECURITY_SNIPPETS.keys()),
            label_visibility="collapsed",
        )

    with col_opts:
        st.caption(f"Provider: **{settings.llm_provider.value.upper()}** (`{settings.gemini_model}`)")
        if st.button("📋 Load Sample into Editor", use_container_width=True):
            st.session_state["sec_code_input"] = SAMPLE_SECURITY_SNIPPETS[selected_sample]["code"]
            st.session_state["sec_filename"] = SAMPLE_SECURITY_SNIPPETS[selected_sample]["filename"]

    if "sec_code_input" not in st.session_state:
        st.session_state["sec_code_input"] = SAMPLE_SECURITY_SNIPPETS[list(SAMPLE_SECURITY_SNIPPETS.keys())[0]]["code"]
        st.session_state["sec_filename"] = "user_repo.py"

    code_input = st.text_area(
        "Source Code to Scan",
        value=st.session_state.get("sec_code_input", ""),
        height=260,
    )

    col_btn, col_k = st.columns([1, 2])
    with col_btn:
        run_scan = st.button("🔍 Scan for Vulnerabilities", type="primary", use_container_width=True)

    if run_scan:
        if not code_input.strip():
            st.warning("Please supply code to scan.")
            return

        with st.spinner("Sanitizing code, retrieving knowledge from ChromaDB & running security scan..."):
            agent = SecurityScannerAgent()
            result = agent.scan(
                SecurityScannerInput(
                    source_code=code_input,
                    filename=st.session_state.get("sec_filename", "snippet.py"),
                    top_k_context=3,
                )
            )

        if result.status != "success":
            st.error(f"Security scan encountered an error: {', '.join(result.errors)}")
            return

        st.markdown("---")
        st.subheader("📊 Security Vulnerability Report")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vulnerabilities Found", len(result.findings))
        
        highest_sev = result.highest_severity
        sev_label = highest_sev.value.upper() if highest_sev else "CLEAN"
        m2.metric("Highest Severity", sev_label)
        
        m3.metric("RAG Docs Retrieved", len(result.retrieved_contexts))
        m4.metric("Latency", f"{result.latency_ms:.0f} ms")

        st.info(f"**Audit Summary:** {result.summary}")

        # ── Retrieved Context Accordion ─────────────────────────────────────
        if result.retrieved_contexts:
            with st.expander(f"📚 Retrieved RAG Security Context ({len(result.retrieved_contexts)} documents)", expanded=False):
                for doc in result.retrieved_contexts:
                    st.markdown(f"**{doc.title}** (`{doc.cwe}`) — Relevance: `{doc.relevance_score:.2f}`")
                    st.text(doc.content)
                    st.divider()

        if not result.findings:
            st.success("🎉 No exploitable security vulnerabilities detected in this snippet.")
            return

        # ── Findings Cards ──────────────────────────────────────────────────
        st.markdown("### ⚠️ Identified Vulnerabilities")

        for i, finding in enumerate(result.findings, 1):
            sev_color = _severity_color(finding.severity)
            cwe_tag = f"[{finding.cwe_id}] " if finding.cwe_id else ""

            with st.expander(
                f"**#{i} [{finding.severity.value.upper()}] {cwe_tag}{finding.title}**",
                expanded=True,
            ):
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {sev_color}; padding-left: 12px; margin-bottom: 12px;">
                        <p style="margin: 0; font-weight: 500;">{finding.description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_left, col_right = st.columns([1, 1], gap="medium")
                with col_left:
                    st.markdown("**Vulnerable Code:**")
                    lang_name, lang_id = detect_language(
                        st.session_state.get("sec_filename", "snippet.txt")
                    )
                    st.code(finding.affected_code, language=lang_id)
                    st.markdown(f"**Line Number:** `{finding.line_number or 'N/A'}`")
                    st.markdown(f"**Confidence:** `{finding.confidence.value.upper()}`")
                    if finding.matched_rule_id:
                        st.markdown(f"**Knowledge Rule ID:** `{finding.matched_rule_id}`")

                with col_right:
                    st.markdown("**Exploit Reasoning:**")
                    st.write(finding.reasoning)

                st.markdown("**Secure Remediation Fix:**")
                st.code(finding.suggested_remediation, language=lang_id)

        # ── Export ──────────────────────────────────────────────────────────
        export_payload = {
            "summary": result.summary,
            "findings_count": len(result.findings),
            "findings": [f.to_dict() for f in result.findings],
            "retrieved_context_ids": [d.doc_id for d in result.retrieved_contexts],
        }
        st.download_button(
            "📥 Download Security Scan Report (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name="security_scan_report.json",
            mime="application/json",
        )


render_security_scanner_page = render
