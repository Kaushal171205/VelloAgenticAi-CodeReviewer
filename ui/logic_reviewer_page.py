"""
ui/logic_reviewer_page.py — Interactive Business Logic & Correctness Review Page.

Allows users to analyze code for semantic, correctness, and business logic flaws
(such as double refunds, inverted conditions, and missing authorizations)
using the Logic Reviewer Agent with automatic secret sanitization.
"""

from __future__ import annotations

import json
from typing import Dict

import streamlit as st

from agents.logic_reviewer_agent import LogicReviewerAgent, LogicReviewerInput
from agents.static_analysis_agent import FindingSeverity
from config import settings
from tools.language_utils import detect_language
PAYMENT_REFUND_SAMPLE = """class PaymentService:
    def __init__(self, db, payment_gateway):
        self.db = db
        self.gateway = payment_gateway

    def refund_order(self, user_id: str, order_id: str, refund_amount: float, target_account: str):
        # 1. Authorization flaw: user_id is never checked against order.owner_id (IDOR)
        order = self.db.find_order(order_id)

        # 2. Missing validation: Negative or zero refund amounts are not rejected
        # 3. Incorrect condition: Allows refunding more than order.amount
        if refund_amount > order.total_amount:
            print(f"Notice: Refund {refund_amount} exceeds order total {order.total_amount}")

        # 4. Invalid state transition: No check if order is already REFUNDED (Double Refund Flaw)
        # 5. Missing idempotency: Can be called multiple times concurrently

        # 6. Unsafe assumption: External transfer invoked without verifying gateway status
        transfer_result = self.gateway.send_payout(
            amount=refund_amount,
            destination=target_account
        )

        # 7. Inconsistent state: Updates DB without verifying transfer_result.success
        order.status = "REFUNDED"
        order.total_refunded += refund_amount
        self.db.save(order)

        return {"success": True, "refunded": refund_amount}
"""

# Pre-defined sample snippets
SAMPLE_SNIPPETS: Dict[str, Dict[str, str]] = {
    "💳 Vulnerable Payment & Refund Service": {
        "filename": "payment_service.py",
        "description": "Contains double refund flaw, inverted limits, missing auth (IDOR), and unhandled gateway errors.",
        "code": PAYMENT_REFUND_SAMPLE.strip(),
    },
    "📦 Order Cancellation & Inventory Double-Restore": {
        "filename": "order_manager.py",
        "description": "Restores warehouse stock on canceled orders repeatedly without status check.",
        "code": '''class OrderManager:
    def cancel_order(self, user_id, order_id):
        order = self.db.get_order(order_id)
        # Missing check: order might already be CANCELED or DELIVERED!
        # Double inventory restore hazard:
        for item in order.items:
            self.inventory.restock(item.sku, item.quantity)
        
        order.status = "CANCELED"
        self.db.update(order)
        return True''',
    },
    "🔐 Broken Password Reset Token Verification": {
        "filename": "auth_service.py",
        "description": "Inverted condition allows expired tokens to reset user passwords.",
        "code": '''import time

def verify_and_reset_password(user, token, new_password):
    token_record = db.lookup_token(token)
    # INVERTED CONDITION: Allows token only if current time is past expiration!
    if time.time() < token_record.expires_at:
        return {"error": "Token is still active"}
    
    # State update without invalidating the token
    user.set_password(new_password)
    db.save_user(user)
    return {"status": "password_updated"}''',
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
    """Render the Logic Reviewer Streamlit page."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🧠 Business Logic & Correctness Reviewer</h1>
            <p>Audits code for subtle semantic flaws, broken business rules, inverted conditions, double refunds, and authorization vulnerabilities that linters miss.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; font-size: 0.9rem;">
            🛡️ <strong>Pre-Execution Sanitization:</strong> All code is automatically scanned and scrubbed by the Sanitizer Agent before being passed to the LLM.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_samples, col_opts = st.columns([2, 1], gap="medium")
    with col_samples:
        st.markdown("**Load Vulnerable Sample:**")
        selected_sample = st.selectbox(
            "Select Vulnerable Architecture Pattern",
            options=list(SAMPLE_SNIPPETS.keys()),
            label_visibility="collapsed",
        )

    with col_opts:
        st.caption(f"Provider: **{settings.llm_provider.value.upper()}** (`{settings.gemini_model}`)")
        if st.button("📋 Load Sample into Editor", use_container_width=True):
            st.session_state["logic_code_input"] = SAMPLE_SNIPPETS[selected_sample]["code"]
            st.session_state["logic_filename"] = SAMPLE_SNIPPETS[selected_sample]["filename"]

    # Initialise session state code if empty
    if "logic_code_input" not in st.session_state:
        st.session_state["logic_code_input"] = SAMPLE_SNIPPETS[list(SAMPLE_SNIPPETS.keys())[0]]["code"]
        st.session_state["logic_filename"] = "payment_service.py"

    code_input = st.text_area(
        "Source Code for Logic Audit",
        value=st.session_state.get("logic_code_input", ""),
        height=280,
    )

    col_btn, col_blank = st.columns([1, 2])
    with col_btn:
        run_review = st.button("🚀 Analyze Business Logic", type="primary", use_container_width=True)

    if run_review:
        if not code_input.strip():
            st.warning("Please supply source code to analyze.")
            return

        with st.spinner("Sanitizing code & performing deep business-logic analysis with LLM..."):
            agent = LogicReviewerAgent()
            result = agent.review(
                LogicReviewerInput(
                    source_code=code_input,
                    filename=st.session_state.get("logic_filename", "snippet.py"),
                )
            )

        if result.status != "success":
            st.error(f"Logic review encountered an error: {', '.join(result.errors)}")
            return

        # ── Results Header & Metrics ────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Logic Review Report")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Logic Flaws Found", len(result.findings))
        
        highest_sev = result.highest_severity
        sev_label = highest_sev.value.upper() if highest_sev else "CLEAN"
        m2.metric("Highest Severity", sev_label)
        
        m3.metric("Secrets Scrubbed", result.secrets_detected)
        m4.metric("Latency", f"{result.latency_ms:.0f} ms")

        st.info(f"**Executive Summary:** {result.summary}")

        if not result.findings:
            st.success("🎉 No business logic or semantic flaws detected in this snippet.")
            return

        # ── Findings Cards ──────────────────────────────────────────────────
        st.markdown("### ⚠️ Detailed Findings")

        for i, finding in enumerate(result.findings, 1):
            sev_color = _severity_color(finding.severity)
            cat_display = finding.category.replace("_", " ").title()

            with st.expander(
                f"**#{i} [{finding.severity.value.upper()}] {finding.title}** (Category: {cat_display})",
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
                    st.markdown("**Affected Code Expression:**")
                    _, lang_id = detect_language(
                        st.session_state.get("logic_filename", "snippet.txt")
                    )
                    st.code(finding.affected_code, language=lang_id)
                    st.markdown(f"**Line Number:** `{finding.line_number or 'N/A'}`")
                    st.markdown(f"**Confidence:** `{finding.confidence.value.upper()}`")

                with col_right:
                    st.markdown("**Reasoning & Attack Vector:**")
                    st.write(finding.reasoning)

                st.markdown("**Suggested Remediation / Fix:**")
                st.code(finding.suggested_remediation, language=lang_id)

        # ── Export ──────────────────────────────────────────────────────────
        export_payload = {
            "summary": result.summary,
            "findings_count": len(result.findings),
            "findings": [f.to_dict() for f in result.findings],
        }
        st.download_button(
            "📥 Download Logic Review Report (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name="logic_review_report.json",
            mime="application/json",
        )


# Export alias
render_logic_reviewer_page = render
