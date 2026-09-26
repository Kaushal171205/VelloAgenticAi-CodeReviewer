"""
ui/signoff_page.py — Human-in-the-Loop Sign-off & Review Approval Portal.

Allows security engineers and code reviewers to inspect paused LangGraph
review runs, verify AI findings, explanations, and suggested fixes, and issue
an authoritative APPROVE or REJECT decision to resume the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from graph.workflow import (
    build_review_graph,
    resume_review,
)
from storage.checkpoint_db import (
    get_default_db_path,
    get_sqlite_checkpointer,
    list_checkpoint_threads,
)

logger = logging.getLogger(__name__)

_SEVERITY_EMOJIS = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🔵",
    "info":     "⚪",
}


def _severity_badge(sev: str) -> str:
    s = str(sev).lower()
    emoji = _SEVERITY_EMOJIS.get(s, "⚫")
    return f"{emoji} **{s.upper()}**"


def render_signoff_page() -> None:
    """Render the Human Sign-off UI page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:1.8rem;font-weight:700;color:#e6edf3;display:flex;align-items:center;gap:.6rem;">
                <span>✅</span> Human-in-the-Loop Approval Portal
            </div>
            <div style="font-size:.92rem;color:#8b949e;margin-top:.3rem;">
                Review AI-generated security findings, logic analysis, and proposed remediations.
                Provide human sign-off (APPROVE or REJECT) to resume paused LangGraph workflows.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db_path = get_default_db_path()
    saver = get_sqlite_checkpointer(db_path)
    app = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver)

    # ── Thread Selection ──────────────────────────────────────────────────────
    threads = list_checkpoint_threads(db_path)
    thread_ids = [t["thread_id"] for t in threads]

    # Session state tracking for current thread
    default_thread = st.session_state.get("current_review_thread_id", "")
    if default_thread and default_thread not in thread_ids:
        thread_ids.insert(0, default_thread)

    col_select, col_refresh = st.columns([4, 1])
    with col_select:
        if thread_ids:
            selected_thread = st.selectbox(
                "Select Review Thread ID to Inspect / Sign-off:",
                options=thread_ids,
                index=0,
                help="Threads are stored in the SQLite checkpointer database.",
            )
        else:
            selected_thread = st.text_input(
                "Enter Review Thread ID:",
                value=default_thread,
                placeholder="e.g. rev-20260922-1234",
            )
    with col_refresh:
        st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    if not selected_thread:
        st.info(
            "No active review threads found in checkpoint database. "
            "Run a code review with human approval enabled to generate a paused checkpoint.",
            icon="ℹ️",
        )
        return

    # ── Fetch Thread State from Checkpointer ──────────────────────────────────
    config = {"configurable": {"thread_id": selected_thread}}
    try:
        snapshot = app.get_state(config)
    except Exception as exc:
        st.error(f"Failed to load checkpoint state for thread `{selected_thread}`: {exc}")
        return

    values: Dict[str, Any] = snapshot.values if snapshot else {}
    next_nodes = snapshot.next if snapshot else ()
    is_paused_at_human = "human_node" in next_nodes

    # ── Status Banner ─────────────────────────────────────────────────────────
    if is_paused_at_human:
        st.warning(
            f"⏸️ **Workflow Paused**: Thread `{selected_thread}` is currently awaiting human sign-off at `human_node`.",
            icon="⏸️",
        )
    elif values.get("human_decision") == "approve":
        st.success(f"✅ **Workflow Approved**: Thread `{selected_thread}` has been approved and completed.", icon="✅")
    elif values.get("human_decision") == "reject":
        st.error(f"❌ **Workflow Rejected**: Thread `{selected_thread}` was rejected by a human reviewer.", icon="❌")
    elif not values:
        st.warning(f"No checkpoint state found for thread `{selected_thread}`.", icon="⚠️")
        return
    else:
        st.info(f"Thread `{selected_thread}` status: {', '.join(next_nodes) if next_nodes else 'Completed'}")

    st.divider()

    # ── 1. Summary & Severity Counts ──────────────────────────────────────────
    st.markdown("### 📊 Review Summary")
    total = values.get("total_findings") or 0
    highest = values.get("highest_severity") or "none"
    filename = values.get("filename") or "snippet.py"
    sev_summary: Dict[str, int] = values.get("severity_summary") or {}

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric("Total Findings", total)
    with col_m2:
        st.metric("Highest Severity", highest.upper())
    with col_m3:
        st.metric("🔴 Critical", sev_summary.get("critical", 0))
    with col_m4:
        st.metric("🟠 High", sev_summary.get("high", 0))
    with col_m5:
        st.metric("🟡 Medium", sev_summary.get("medium", 0))
    with col_m6:
        st.metric("🔵 Low", sev_summary.get("low", 0))

    st.markdown(f"**Target File**: `{filename}`")

    st.divider()

    # ── 2. Findings & Affected Lines ──────────────────────────────────────────
    st.markdown("### 🚨 Detected Findings")
    raw_findings = values.get("aggregated_findings") or []
    if not raw_findings:
        classified = values.get("classified_findings") or {}
        for bucket in classified.values():
            if isinstance(bucket, list):
                raw_findings.extend(bucket)

    if not raw_findings:
        st.markdown("*No security or logic findings detected in this code review.*")
    else:
        for idx, f in enumerate(raw_findings, 1):
            def _get(k: str, d: Any = "") -> Any:
                if isinstance(f, dict):
                    return f.get(k, d)
                return getattr(f, k, d)

            title = _get("title", "Untitled")
            sev = _get("severity", "medium")
            confidence = _get("confidence", "medium")
            line = _get("line_number")
            desc = _get("description", "")
            affected = _get("affected_code", "")
            reasoning = _get("reasoning", "")
            tools = _get("source_tools", [_get("primary_tool", "analyzer")])
            tool_str = ", ".join(f"`{t}`" for t in tools) if isinstance(tools, list) else f"`{tools}`"

            with st.expander(f"{idx}. [{str(sev).upper()}] {title} (Line {line or 'N/A'})", expanded=(idx <= 2)):
                st.markdown(f"**Severity**: {_severity_badge(sev)} | **Confidence**: `{confidence}` | **Tool(s)**: {tool_str}")
                st.markdown(f"**Description**: {desc}")
                if reasoning:
                    st.markdown(f"**AI Reasoning**: {reasoning}")
                if affected:
                    st.markdown("**Affected Code:**")
                    st.code(affected, language="python")

    st.divider()

    # ── 3. Suggested Fixes & Explanations ─────────────────────────────────────
    st.markdown("### 🛠️ Suggested Remediations & Critic Review")
    fixes: List[Dict[str, Any]] = values.get("suggested_fixes") or []

    if not fixes:
        st.markdown("*No concrete patch suggestions were generated for this review.*")
    else:
        for idx, fix in enumerate(fixes, 1):
            f_title = fix.get("finding_title", f"Finding #{idx}")
            proposed = fix.get("proposed_fix", "")
            explanation = fix.get("explanation", "")
            ast_valid = fix.get("ast_valid")
            critic_accepted = fix.get("critic_accepted")
            critic_fb = fix.get("critic_feedback", "")
            attempt_num = fix.get("attempt_number", 1)

            ast_badge = "✅ Valid Python (AST)" if ast_valid else ("❌ Syntax Error" if ast_valid is False else "⚪ Pending AST")
            critic_badge = "✅ Approved by Critic" if critic_accepted else ("⚠️ Revision Requested" if critic_accepted is False else "⚪ Pending Review")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;margin-bottom:1rem;">
                        <div style="font-weight:600;font-size:1rem;color:#e6edf3;margin-bottom:.4rem;">
                            Fix #{idx}: {f_title}
                        </div>
                        <div style="font-size:.82rem;color:#8b949e;margin-bottom:.6rem;">
                            <b>AST Validation:</b> {ast_badge} &nbsp;|&nbsp;
                            <b>Critic Verdict:</b> {critic_badge} &nbsp;|&nbsp;
                            <b>Attempts:</b> #{attempt_num}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f"**AI Explanation of What Changed:**\n\n> {explanation}")
                if proposed:
                    st.markdown("**Proposed Replacement Code:**")
                    st.code(proposed, language="python")
                if critic_fb:
                    st.markdown(f"**Critic Notes:**\n> {critic_fb}")

    st.divider()

    # ── 4. Human Decision Action Bar ──────────────────────────────────────────
    st.markdown("### ✍️ Human Decision")

    if is_paused_at_human:
        st.markdown(
            "Review the above findings, explanations, and proposed fixes. "
            "Choose an action to resume the workflow:"
        )

        col_approve, col_reject, _ = st.columns([2, 2, 4])
        with col_approve:
            if st.button("✅ APPROVE REVIEW", type="primary", use_container_width=True):
                with st.spinner("Resuming workflow with APPROVE decision..."):
                    try:
                        res = resume_review(app, selected_thread, decision="approve")
                        st.success("Workflow resumed and successfully approved!", icon="🎉")
                        st.session_state[f"decision_{selected_thread}"] = "approved"
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error resuming workflow: {err}")

        with col_reject:
            if st.button("❌ REJECT REVIEW", type="secondary", use_container_width=True):
                with st.spinner("Resuming workflow with REJECT decision..."):
                    try:
                        res = resume_review(app, selected_thread, decision="reject")
                        st.error("Workflow resumed with REJECT decision. Pipeline terminated without approving fixes.", icon="🛑")
                        st.session_state[f"decision_{selected_thread}"] = "rejected"
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error resuming workflow: {err}")
    else:
        decision = values.get("human_decision", "none")
        if decision == "approve":
            st.success("This review was **APPROVED** by the human reviewer.", icon="✅")
            report = values.get("final_report", "")
            if report:
                with st.expander("📄 View Final Code Review Report", expanded=True):
                    st.markdown(report)
        elif decision == "reject":
            st.error("This review was **REJECTED** by the human reviewer. No remediations were accepted.", icon="❌")
        else:
            st.info(f"Review thread is currently in state: {values.get('human_decision', 'completed')}.")


def render() -> None:
    """Entry point for Streamlit page routing."""
    render_signoff_page()
