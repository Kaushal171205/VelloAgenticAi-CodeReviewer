"""
main.py — Application entrypoint for the AI-Powered Enterprise Code Review System.

Run with:
    streamlit run main.py

Responsibilities (Phase 1):
  • Page layout and navigation shell
  • Sidebar with project identity and navigation
  • Route to individual UI page modules
  • Display key system status in the sidebar

Architecture note:
  Navigation is handled through Streamlit's built-in `st.navigation` /
  `st.Page` API (Streamlit ≥ 1.36). Each page is a standalone module in the
  `ui/` package, keeping main.py thin and free of business logic.
"""

from __future__ import annotations

import streamlit as st

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Code Reviewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**AI-Powered Enterprise Code Review System**\n\n"
            "Automatically detects security vulnerabilities, logic flaws, "
            "best-practice violations, and leaked secrets."
        ),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — design tokens and custom overrides
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Import typeface ────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Design tokens ──────────────────────────────────────────────────── */
    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #161b22;
        --bg-card:       #1c2230;
        --border:        #30363d;
        --accent:        #3b82f6;
        --accent-glow:   rgba(59,130,246,.18);
        --accent-hover:  #60a5fa;
        --text-primary:  #e6edf3;
        --text-muted:    #8b949e;
        --success:       #3fb950;
        --warning:       #d29922;
        --danger:        #f85149;
        --radius:        10px;
        --font-body:     'Inter', sans-serif;
        --font-mono:     'JetBrains Mono', monospace;
    }

    /* ── Base ───────────────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        font-family: var(--font-body) !important;
        color: var(--text-primary) !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* ── Main container ─────────────────────────────────────────────────── */
    [data-testid="stMain"] {
        background-color: var(--bg-primary) !important;
    }

    /* ── Cards / containers ─────────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Metrics ────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: .8rem; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 600; }

    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {
        background-color: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: background .2s ease, box-shadow .2s ease !important;
    }
    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
        box-shadow: 0 0 12px var(--accent-glow) !important;
    }

    /* ── Expander ───────────────────────────────────────────────────────── */
    details[data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }
    details[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background-color: var(--bg-card) !important;
    }

    /* ── Code blocks ────────────────────────────────────────────────────── */
    code, pre {
        font-family: var(--font-mono) !important;
        font-size: .85rem !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    [data-testid="stTab"] {
        font-weight: 500 !important;
    }

    /* ── Divider ────────────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }

    /* ── Table ──────────────────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; }
    th {
        text-align: left !important;
        color: var(--text-muted) !important;
        font-size: .75rem !important;
        text-transform: uppercase !important;
        letter-spacing: .06em !important;
        border-bottom: 1px solid var(--border) !important;
        padding: .5rem .75rem !important;
    }
    td {
        padding: .55rem .75rem !important;
        border-bottom: 1px solid var(--border) !important;
        font-size: .875rem !important;
    }

    /* ── Hide Streamlit chrome we don't need ────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — project identity + navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / wordmark
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem;">
            <span style="font-size:1.6rem;">🔍</span>
            <div>
                <div style="font-size:1rem;font-weight:700;line-height:1.2;
                            color:#e6edf3;letter-spacing:-.01em;">
                    Code Reviewer
                </div>
                <div style="font-size:.7rem;color:#8b949e;letter-spacing:.04em;
                            text-transform:uppercase;font-weight:500;">
                    Enterprise AI System
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Phase badge
    st.markdown(
        """
        <div style="background:#1c2230;border:1px solid #30363d;border-radius:8px;
                    padding:.6rem 1rem;margin-bottom:.5rem;">
            <div style="font-size:.7rem;color:#8b949e;text-transform:uppercase;
                        letter-spacing:.06em;font-weight:500;">Current Phase</div>
            <div style="font-size:.95rem;font-weight:600;color:#3b82f6;margin-top:.15rem;">
                Phase 8 — Git Diff Processing
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LLM status pill
    llm_ready = settings.is_llm_ready()
    pill_color = "#1a7f37" if llm_ready else "#6e3605"
    pill_text_color = "#3fb950" if llm_ready else "#d29922"
    pill_label = "LLM Ready" if llm_ready else "LLM Not Configured"
    pill_dot   = "●"

    st.markdown(
        f"""
        <div style="display:inline-flex;align-items:center;gap:.4rem;
                    background:{pill_color}33;border:1px solid {pill_color};
                    border-radius:999px;padding:.25rem .75rem;
                    font-size:.78rem;font-weight:500;color:{pill_text_color};
                    margin-bottom:1rem;">
            <span style="font-size:.6rem;">{pill_dot}</span> {pill_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LLM Provider Selection
    st.markdown(
        "<div style='font-size:.7rem;color:#8b949e;text-transform:uppercase;"
        "letter-spacing:.06em;font-weight:500;margin-bottom:.4rem;'>Active Provider</div>",
        unsafe_allow_html=True,
    )
    
    from config import LLMProvider
    provider_options = [p.value for p in LLMProvider]
    
    selected_provider_val = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=provider_options.index(settings.llm_provider.value) if settings.llm_provider.value in provider_options else 0,
        label_visibility="collapsed",
    )
    
    if selected_provider_val != settings.llm_provider.value:
        settings.llm_provider = LLMProvider(selected_provider_val)
        st.rerun()

    st.divider()

    # Navigation
    st.markdown(
        "<div style='font-size:.7rem;color:#8b949e;text-transform:uppercase;"
        "letter-spacing:.06em;font-weight:500;margin-bottom:.4rem;'>Navigation</div>",
        unsafe_allow_html=True,
    )

    pages = {
        "🏠  Overview":           "overview",
        "⚡  Quick Review":       "quick_review",
        "🌿  Git Diff Review":    "diff_review",
        "🗂️  Codebase Indexer":   "codebase_indexer",
        "✅  Human Sign-off":     "signoff",
        "📄  Final Reports":      "final_reports",
        "🔑  Sanitizer":          "sanitizer",
        "🛡️  Static Analysis":    "static_analysis",
        "🧠  Logic Reviewer":     "logic_reviewer",
        "🔍  RAG Security":      "security_scanner",
        "🤖  LLM Abstraction":   "llm_playground",
        "🩺  System Health":      "status",
        "⚙️  Configuration":      "settings",
        "🗺️  Roadmap":            "roadmap",
    }

    if "active_page" not in st.session_state:
        st.session_state.active_page = "overview"

    for label, key in pages.items():
        is_active = st.session_state.active_page == key
        btn_style = (
            "background:#3b82f620;border:1px solid #3b82f640;"
            if is_active else "background:transparent;border:1px solid transparent;"
        )
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
        ):
            st.session_state.active_page = key
            st.rerun()

    st.divider()
    st.caption(
        f"v0.7.0 — Phase 7  \n"
        f"Env: `{settings.app_env.value}`  \n"
        f"Provider: `{settings.llm_provider.value}`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page router
# ─────────────────────────────────────────────────────────────────────────────

page = st.session_state.get("active_page", "overview")

if page == "overview":
    # ── Overview / landing page ───────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="font-size:2rem;font-weight:700;letter-spacing:-.02em;
                   margin-bottom:.25rem;color:#e6edf3;">
            AI-Powered Enterprise Code Review
        </h1>
        <p style="color:#8b949e;font-size:1rem;margin-bottom:1.5rem;">
            Automated first-pass code review — security, correctness, best practices.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Capability cards
    capabilities = [
        ("🔐", "Security Analysis",   "Detects vulnerabilities, injection flaws, and insecure patterns using Bandit and RAG retrieval over a curated CVE knowledge base."),
        ("🧠", "Logic Review",        "An LLM agent reasons about business-logic correctness, edge-case handling, and algorithmic flaws across changed files."),
        ("🔑", "Secret Detection",    "Scans diffs for leaked API keys, tokens, and credentials before any code is sent to an external LLM."),
        ("🛠️", "Fix Suggestions",     "Generates concrete, validated patch suggestions for every finding — including a critic loop to verify correctness."),
        ("🗂️", "Severity Ranking",    "Aggregates and deduplicates findings from all agents and classifies them by severity (Critical → Info)."),
        ("👤", "Human-in-the-Loop",   "Routes high-severity findings to a human reviewer approval workflow before applying suggested fixes."),
    ]

    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(capabilities):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="background:#1c2230;border:1px solid #30363d;
                            border-radius:10px;padding:1.25rem;margin-bottom:.75rem;
                            height:100%;">
                    <div style="font-size:1.5rem;margin-bottom:.5rem;">{icon}</div>
                    <div style="font-weight:600;font-size:.95rem;color:#e6edf3;
                                margin-bottom:.4rem;">{title}</div>
                    <div style="font-size:.82rem;color:#8b949e;line-height:1.55;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Tech stack
    st.markdown("### Technology Stack")
    tech = {
        "Orchestration":   "LangGraph · Python",
        "LLM Providers":   "Google Gemini · Ollama",
        "Static Analysis": "Bandit · detect-secrets · AST",
        "Vector Store":    "ChromaDB · sentence-transformers",
        "Persistence":     "SQLite (checkpoints)",
        "UI":              "Streamlit",
        "Ingestion":       "GitPython · diff parsing",
    }
    t_cols = st.columns(4)
    for i, (layer, stack) in enumerate(tech.items()):
        with t_cols[i % 4]:
            st.markdown(
                f"""
                <div style="background:#161b22;border:1px solid #30363d;
                            border-radius:8px;padding:.75rem 1rem;margin-bottom:.5rem;">
                    <div style="font-size:.7rem;color:#8b949e;text-transform:uppercase;
                                letter-spacing:.06em;font-weight:500;">{layer}</div>
                    <div style="font-size:.82rem;color:#e6edf3;margin-top:.2rem;
                                font-weight:500;">{stack}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    with st.expander("🤖 View LangGraph Multi-Agent Architecture Graph", expanded=False):
        st.markdown(
            "This interactive graph illustrates the entire multi-agent code review workflow — from "
            "secret sanitization, static security analysis, and logic review, to severity classification, "
            "critic verification loops, and human-in-the-loop sign-off."
        )
        try:
            from graph.workflow import build_review_graph
            agent_graph = build_review_graph(enable_critic=True, enable_human_review=False)
            st.markdown(f"```mermaid\n{agent_graph.get_graph().draw_mermaid()}\n```")
        except Exception as exc:
            st.caption(f"Unable to render agent workflow graph: {exc}")

    st.divider()

    # Quick-start CTA
    st.info(
        "**Getting started:**  \n"
        "1. Copy `.env.example` → `.env` and fill in your API key  \n"
        "2. Head to **⚙️ Configuration** to verify your LLM provider  \n"
        "3. Check **🩺 System Health** to confirm all Phase 1 checks pass",
        icon="🚀",
    )

elif page == "status":
    from ui.status_page import render
    render()

elif page == "settings":
    from ui.settings_page import render
    render()

elif page == "roadmap":
    from ui.roadmap_page import render
    render()

elif page == "sanitizer":
    from ui.sanitizer_page import render
    render()

elif page == "static_analysis":
    from ui.static_analysis_page import render
    render()

elif page == "logic_reviewer":
    from ui.logic_reviewer_page import render
    render()

elif page == "security_scanner":
    from ui.security_scanner_page import render
    render()

elif page == "codebase_indexer":
    from ui.codebase_indexer_page import render
    render()

elif page == "diff_review":
    from ui.diff_review_page import render
    render()

elif page == "signoff":
    from ui.signoff_page import render
    render()

elif page == "final_reports":
    from ui.final_report_page import render
    render()

elif page == "quick_review":
    from ui.quick_review_page import render
    render()

elif page == "llm_playground":
    from ui.llm_playground_page import render_llm_playground_page
    render_llm_playground_page()

else:
    st.error(f"Unknown page: '{page}'. This is a bug.")
