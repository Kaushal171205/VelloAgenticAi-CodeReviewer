"""
ui/roadmap_page.py — Visual build roadmap showing which phases are
complete, in-progress, and upcoming.
"""

from __future__ import annotations

import streamlit as st


PHASES = [
    {
        "number": 1,
        "title": "Project Foundation",
        "status": "complete",
        "features": [
            "Configuration management (pydantic-settings)",
            "Environment variable loading (.env)",
            "Basic Streamlit application shell",
            "System health / status dashboard",
            "LLM provider configuration interface",
            "Clean module initialisation",
        ],
    },
    {
        "number": 2,
        "title": "Secret Sanitization",
        "status": "complete",
        "features": [
            "Regex-based secret detection (15+ patterns)",
            "detect-secrets library integration",
            "API keys, passwords, tokens, DB URLs, private keys",
            "Placeholder substitution with structured findings",
            "Severity classification (Critical → Low)",
            "Sanitizer UI with paste / upload / examples",
        ],
    },
    {
        "number": 3,
        "title": "Static Analysis",
        "status": "complete",
        "features": [
            "Bandit security linting wrapper",
            "Programmatic JSON capture & normalisation",
            "Common finding schema (title, severity, confidence, CWE)",
            "Graceful error handling for all edge cases",
            "Static analysis Streamlit UI page",
            "Intentionally vulnerable test suite",
        ],
    },
    {
        "number": 4,
        "title": "LLM Provider Abstraction",
        "status": "complete",
        "features": [
            "Unified BaseLLMProvider abstract interface",
            "GeminiProvider using official google-genai SDK",
            "OllamaProvider using local REST endpoint",
            "Factory function (get_llm_provider)",
            "Secret masking & zero key exposure in logs/UI",
            "Pydantic structured output validation",
            "Latency & token usage metrics tracking",
            "Interactive Streamlit testing playground",
        ],
    },
    {
        "number": 5,
        "title": "Logic Reviewer Agent",
        "status": "complete",
        "features": [
            "Business logic, correctness & semantic flaw auditing",
            "Detection of inverted conditions & off-by-one errors",
            "Identification of missing validation on monetary/critical inputs",
            "Unsafe assumptions and unhandled external failure detection",
            "Invalid state transitions & double refund/spending hazards",
            "Broken authorization (IDOR) & business-rule violation checks",
            "Pre-execution automatic secret sanitization guarantee",
            "Structured schema: title, severity, code, reasoning, remediation",
        ],
    },
    {
        "number": 6,
        "title": "RAG Security Scanner",
        "status": "complete",
        "features": [
            "Local ChromaDB vector database with persistent storage",
            "Local sentence-transformers embedding generation (all-MiniLM-L6-v2)",
            "Curated local Python security knowledge base (SQLi, Command Injection, SSRF, Deserialization, etc.)",
            "Semantic similarity retrieval and top-k context extraction",
            "Retrieval-Augmented Generation (RAG) security scanning agent",
            "Pre-execution secret sanitization on changed code",
            "Structured vulnerability findings with CWE mapping & verified remediations",
            "Interactive Streamlit RAG audit UI with context inspection",
        ],
    },
    {
        "number": 7,
        "title": "Shared Codebase Indexing",
        "status": "complete",
        "features": [
            "Recursive Python repository scanning (honours .git, venv, __pycache__, node_modules exclusions)",
            "AST-based import resolution and bidirectional dependency graph",
            "Function, class, and method extraction with stable deterministic IDs",
            "Fallback summaries (signature + docstring) with optional LLM enhancement",
            "ChromaDB upsert — fully repeatable / idempotent indexing",
            "Semantic search over the codebase index via sentence-transformers",
            "Syntax-error tolerance: bad files captured in errors, rest proceeds",
            "Streamlit Codebase Indexer UI with stats, semantic search, and dependency graph inspector",
        ],
    },
    {
        "number": 8,
        "title": "Git Diff Processing & AST Scope Identification",
        "status": "complete",
        "features": [
            "Branch & commit diff extraction (supports git diff main...feature-branch)",
            "Automatic detection and filtering of changed Python files",
            "AST line mapping to enclosing functions and classes",
            "Exact preservation of old/new line numbers, scopes, and unified diff content",
            "Token-bounded review chunking (never transmits full codebase to LLM)",
            "Adjacent hunk grouping by function boundary",
            "Clean public API: extract_review_changes(repo_path, base_ref, head_ref)",
            "Streamlit Git Diff Review page with interactive scope & chunk preview",
        ],
    },
    {
        "number": 9,
        "title": "Multi-Agent Orchestration",
        "status": "upcoming",
        "features": [
            "LangGraph review pipeline",
            "Specialised agents: security, logic, style",
            "Dynamic routing",
            "Finding aggregation & deduplication",
        ],
    },
    {
        "number": 10,
        "title": "Fix Generation & Validation",
        "status": "upcoming",
        "features": [
            "AI-generated fixes",
            "Fix validation pipeline",
            "Critic / self-correction loop",
        ],
    },
    {
        "number": 11,
        "title": "Human-in-the-Loop & Persistence",
        "status": "upcoming",
        "features": [
            "Human approval workflow",
            "SQLite checkpoint persistence",
            "Severity classification UI",
        ],
    },
    {
        "number": 12,
        "title": "Production Hardening",
        "status": "upcoming",
        "features": [
            "Gemini ↔ Ollama provider switching",
            "Full Streamlit UI with history",
            "Performance optimisation",
        ],
    },
]

STATUS_CONFIG = {
    "complete":     ("✅", "Complete",     "#1a7f37"),
    "in_progress":  ("🔄", "In Progress",  "#9a6700"),
    "upcoming":     ("⏳", "Upcoming",     "#57606a"),
}


def render() -> None:
    st.markdown("## Build Roadmap")
    st.caption(
        "The system is built incrementally across 8 phases. "
        "Each phase adds a functional layer without breaking previous ones."
    )

    # ── Summary row ───────────────────────────────────────────────────────────
    complete = sum(1 for p in PHASES if p["status"] == "complete")
    in_prog  = sum(1 for p in PHASES if p["status"] == "in_progress")
    upcoming = sum(1 for p in PHASES if p["status"] == "upcoming")

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Complete",     complete)
    c2.metric("🔄 In Progress",  in_prog)
    c3.metric("⏳ Upcoming",     upcoming)

    st.divider()

    # ── Phase cards ───────────────────────────────────────────────────────────
    for phase in PHASES:
        icon, label, _ = STATUS_CONFIG[phase["status"]]
        is_done = phase["status"] == "complete"

        with st.expander(
            f"Phase {phase['number']} — {phase['title']}   {icon} *{label}*",
            expanded=is_done,
        ):
            for feat in phase["features"]:
                prefix = "✔" if is_done else "◦"
                st.markdown(f"{prefix} {feat}")
