"""
ui/quick_review_page.py — Streamlit UI for the Quick Review workflow.

Allows the user to paste code or upload a file and run the full LangGraph 
review pipeline (Sanitizer -> Context (if applicable) -> Agents -> Aggregator 
-> Classifier -> Report).
"""

from __future__ import annotations

import io
import logging

import streamlit as st

from graph.workflow import build_review_graph, run_review
from tools.language_utils import detect_language

logger = logging.getLogger(__name__)

def render() -> None:
    st.title("⚡ Quick Review")
    st.markdown(
        """
        Run a full AI-powered code review on a single file or pasted code snippet.
        The code will go through secret sanitization, static analysis, security scanning, 
        and logic review, producing a comprehensive final report.
        """
    )

    # ── Input panel ───────────────────────────────────────────────────────────
    with st.expander("📥  Input Source", expanded=True):
        input_method = st.radio(
            "Input method",
            ["Paste code", "Upload file"],
            horizontal=True,
            key="quick_input_method",
        )

        source_code = ""
        filename = "snippet.txt"

        if input_method == "Paste code":
            source_code = st.text_area(
                "Source code",
                height=250,
                placeholder="Paste your code here (Python, JavaScript, Java, Go, etc.)…",
                key="quick_paste_area",
            )
        elif input_method == "Upload file":
            uploaded = st.file_uploader(
                "Upload a source file",
                type=None,  # accept any file type
                key="quick_uploader",
            )
            if uploaded:
                filename = uploaded.name
                source_code = io.StringIO(uploaded.read().decode("utf-8", errors="replace")).read()
                lang_name, _ = detect_language(filename)
                st.success(f"Loaded **{filename}** ({lang_name}, {len(source_code):,} bytes)")

        enable_critic = st.checkbox(
            "Enable Critic & Self-Correction (Fix Suggester)",
            value=True,
            help="If enabled, the pipeline will generate patches for findings and iteratively validate them.",
        )
        
        run_btn = st.button(
            "▶️ Run Code Review",
            key="quick_run_btn",
            disabled=not source_code.strip(),
            type="primary",
            use_container_width=True,
        )

    # ── Execution & Results ───────────────────────────────────────────────────
    if run_btn and source_code.strip():
        # Initialize graph
        app = build_review_graph(enable_critic=enable_critic, enable_human_review=False)
        
        with st.spinner("Pipeline running: Sanitizing, analyzing, and reviewing code..."):
            try:
                # We do not pass changed_functions here since it's a full file review
                final_state = run_review(
                    app=app,
                    raw_code=source_code,
                    filename=filename,
                    enable_critic=enable_critic
                )
                st.session_state["quick_review_result"] = final_state
                st.success("Review pipeline completed successfully!")
            except Exception as e:
                logger.exception("Error running quick review pipeline")
                st.error(f"Pipeline execution failed: {e}")
                return

    final_state = st.session_state.get("quick_review_result")
    if not final_state:
        st.info("Provide code and click **Run Code Review** to see results.", icon="ℹ️")
        return

    st.divider()

    # ── Summary Metrics ───────────────────────────────────────────────────────
    total = final_state.get("total_findings", 0)
    highest = (final_state.get("highest_severity") or "none").upper()
    sev_summary = final_state.get("severity_summary") or {}
    
    st.subheader("📊 Review Summary")
    
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    col_m1.metric("Total Findings", total)
    col_m2.metric("Highest Severity", highest)
    col_m3.metric("🔴 Critical", sev_summary.get("critical", 0))
    col_m4.metric("🟠 High", sev_summary.get("high", 0))
    col_m5.metric("🟡 Medium", sev_summary.get("medium", 0))
    col_m6.metric("🔵 Low", sev_summary.get("low", 0))

    st.divider()

    # ── Final Report ──────────────────────────────────────────────────────────
    st.subheader("📄 Final Review Report")
    report = final_state.get("final_report", "*No report generated.*")
    
    with st.container(border=True):
        st.markdown(report)
