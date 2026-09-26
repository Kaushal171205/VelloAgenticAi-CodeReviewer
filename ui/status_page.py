"""
ui/status_page.py — System health and project status dashboard.

Renders a real-time health check of every subsystem so the team can
verify which capabilities are available before submitting a review.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Callable

import streamlit as st

from config import LLMProvider, settings


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthCheck:
    name: str
    description: str
    check_fn: Callable[[], tuple[bool, str]]
    phase: int = 1          # which phase introduces this subsystem
    critical: bool = False  # critical checks block all review functionality


@dataclass
class CheckResult:
    check: HealthCheck
    passed: bool
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
# Individual health-check functions
# ─────────────────────────────────────────────────────────────────────────────

def _check_python_version() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return ok, f"Python {major}.{minor} {'✓' if ok else '— requires ≥ 3.10'}"


def _check_config_loads() -> tuple[bool, str]:
    try:
        _ = settings.llm_provider
        return True, "Settings loaded successfully from environment."
    except Exception as exc:
        return False, f"Settings error: {exc}"


def _check_llm_credentials() -> tuple[bool, str]:
    ready = settings.is_llm_ready()
    if ready:
        return True, f"{settings.active_model_name()} — credentials present."
    provider = settings.llm_provider.value
    missing = "GEMINI_API_KEY" if provider == "gemini" else "OLLAMA_BASE_URL"
    return False, f"Provider '{provider}' selected but {missing} is not set."


def _check_storage_dirs() -> tuple[bool, str]:
    try:
        settings.ensure_storage_dirs()
        return True, (
            f"SQLite → {settings.sqlite_db_path}  |  "
            f"ChromaDB → {settings.chroma_persist_dir}"
        )
    except Exception as exc:
        return False, f"Could not create storage directories: {exc}"


def _check_optional_import(module: str) -> Callable[[], tuple[bool, str]]:
    """Factory: returns a health-check fn that tests whether `module` is importable."""
    def _inner() -> tuple[bool, str]:
        spec = importlib.util.find_spec(module)
        if spec is not None:
            return True, f"'{module}' is installed and importable."
        return False, f"'{module}' not found — install it when Phase introduces it."
    return _inner


# ─────────────────────────────────────────────────────────────────────────────
# Health-check registry
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_CHECKS: list[HealthCheck] = [
    # ── Phase 1 (foundation) ─────────────────────────────────────────────────
    HealthCheck(
        name="Python Version",
        description="Requires Python 3.10+",
        check_fn=_check_python_version,
        phase=1,
        critical=True,
    ),
    HealthCheck(
        name="Configuration",
        description="Environment variables load without error",
        check_fn=_check_config_loads,
        phase=1,
        critical=True,
    ),
    HealthCheck(
        name="LLM Credentials",
        description="Active LLM provider has the required credentials",
        check_fn=_check_llm_credentials,
        phase=1,
        critical=False,
    ),
    HealthCheck(
        name="Storage Directories",
        description="SQLite and ChromaDB directories are writable",
        check_fn=_check_storage_dirs,
        phase=1,
        critical=False,
    ),
    # ── Phase 2 (diff & indexing) ─────────────────────────────────────────────
    HealthCheck(
        name="GitPython",
        description="Git diff processing (Phase 2)",
        check_fn=_check_optional_import("git"),
        phase=2,
    ),
    # ── Phase 3 (static analysis) ─────────────────────────────────────────────
    HealthCheck(
        name="Bandit",
        description="Static security linting (Phase 3)",
        check_fn=_check_optional_import("bandit"),
        phase=3,
    ),
    HealthCheck(
        name="detect-secrets",
        description="Secret detection (Phase 3)",
        check_fn=_check_optional_import("detect_secrets"),
        phase=3,
    ),
    # ── Phase 4 (RAG / embeddings) ────────────────────────────────────────────
    HealthCheck(
        name="ChromaDB",
        description="Vector store for RAG (Phase 4)",
        check_fn=_check_optional_import("chromadb"),
        phase=4,
    ),
    HealthCheck(
        name="sentence-transformers",
        description="Embedding model (Phase 4)",
        check_fn=_check_optional_import("sentence_transformers"),
        phase=4,
    ),
    # ── Phase 5 (orchestration) ───────────────────────────────────────────────
    HealthCheck(
        name="LangGraph",
        description="Agent orchestration graph (Phase 5)",
        check_fn=_check_optional_import("langgraph"),
        phase=5,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_checks() -> list[CheckResult]:
    results = []
    for hc in HEALTH_CHECKS:
        try:
            passed, detail = hc.check_fn()
        except Exception as exc:
            passed, detail = False, f"Unexpected error: {exc}"
        results.append(CheckResult(check=hc, passed=passed, detail=detail))
    return results


def _phase_label(phase: int) -> str:
    labels = {1: "Foundation", 2: "Diff & Index", 3: "Static Analysis",
               4: "RAG & Embeddings", 5: "Orchestration"}
    return labels.get(phase, f"Phase {phase}")


# ─────────────────────────────────────────────────────────────────────────────
# Public render function (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## System Health")
    st.caption(
        "Live readiness check for every subsystem. "
        "Green = ready. Yellow = optional dependency for a future phase. "
        "Red = required component missing."
    )

    if st.button("↻  Refresh", key="btn_refresh_status"):
        st.cache_data.clear()

    results = _run_checks()

    # ── Summary banner ────────────────────────────────────────────────────────
    phase1_results = [r for r in results if r.check.phase == 1]
    critical_failures = [r for r in phase1_results if r.check.critical and not r.passed]
    phase1_ok = sum(1 for r in phase1_results if r.passed)

    if critical_failures:
        st.error(
            f"**{len(critical_failures)} critical check(s) failed.** "
            "The application cannot function correctly. See details below.",
            icon="🚨",
        )
    elif all(r.passed for r in phase1_results):
        st.success(
            f"**Phase 1 fully operational** — all {phase1_ok} foundation checks passed.",
            icon="✅",
        )
    else:
        warnings = [r for r in phase1_results if not r.passed and not r.check.critical]
        st.warning(
            f"Phase 1 core is healthy, but {len(warnings)} non-critical check(s) need "
            "attention. Review the details below.",
            icon="⚠️",
        )

    st.divider()

    # ── Per-phase check table ─────────────────────────────────────────────────
    from itertools import groupby
    sorted_results = sorted(results, key=lambda r: r.check.phase)

    for phase, group in groupby(sorted_results, key=lambda r: r.check.phase):
        group_list = list(group)
        is_current = phase == 1

        with st.expander(
            f"Phase {phase} — {_phase_label(phase)}"
            + ("  *(current)*" if is_current else ""),
            expanded=is_current,
        ):
            for res in group_list:
                icon = "✅" if res.passed else ("🔴" if res.check.critical else "🟡")
                badge = " `CRITICAL`" if res.check.critical else ""
                st.markdown(f"**{icon} {res.check.name}**{badge}")
                st.caption(f"{res.check.description} — {res.detail}")

    # ── Active configuration summary ──────────────────────────────────────────
    st.divider()
    st.markdown("### Active Configuration")

    col1, col2, col3 = st.columns(3)
    col1.metric("LLM Provider", settings.llm_provider.value.capitalize())
    col2.metric("Active Model", settings.active_model_name())
    col3.metric("Environment", settings.app_env.value.capitalize())

    col4, col5 = st.columns(2)
    col4.metric("Log Level", settings.log_level.value)
    col5.metric("Max File Size", f"{settings.max_file_size_bytes // 1024} KB")
