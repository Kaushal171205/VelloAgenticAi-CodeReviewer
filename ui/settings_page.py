"""
ui/settings_page.py — LLM provider configuration interface.

Lets the operator view and understand how to configure the LLM provider
through environment variables. We intentionally do not write .env files
from the UI (to avoid accidental overwrites and keep secrets out of the
Streamlit session state). Instead we show a copy-ready snippet.
"""

from __future__ import annotations

import streamlit as st

from config import LLMProvider, settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_DOCS = {
    LLMProvider.GEMINI: {
        "title": "Google Gemini",
        "doc_url": "https://ai.google.dev/gemini-api/docs",
        "get_key_url": "https://aistudio.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"],
        "description": (
            "Gemini is Google's multimodal LLM family, accessible via the "
            "Gemini API. A free tier is available; production workloads should "
            "use a paid plan."
        ),
    },
    LLMProvider.OLLAMA: {
        "title": "Ollama (local)",
        "doc_url": "https://ollama.com/library",
        "get_key_url": "https://ollama.com/download",
        "models": ["llama3.2", "llama3.1", "codellama", "mistral", "deepseek-coder"],
        "description": (
            "Ollama runs open-weight models locally. No API key required — "
            "just install Ollama and pull a model. Ideal for air-gapped or "
            "privacy-sensitive environments."
        ),
    },
}


def _env_snippet(provider: LLMProvider) -> str:
    if provider == LLMProvider.GEMINI:
        return (
            "LLM_PROVIDER=gemini\n"
            "GEMINI_API_KEY=<your-key-here>\n"
            f"GEMINI_MODEL={settings.gemini_model}"
        )
    return (
        "LLM_PROVIDER=ollama\n"
        f"OLLAMA_BASE_URL={settings.ollama_base_url}\n"
        f"OLLAMA_MODEL={settings.ollama_model}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public render function
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## LLM Provider Configuration")
    st.caption(
        "Configuration is managed through environment variables loaded from "
        "your `.env` file. Edit `.env` and restart the app to apply changes."
    )

    # ── Current status card ───────────────────────────────────────────────────
    ready = settings.is_llm_ready()
    status_icon = "🟢" if ready else "🔴"
    status_text = "Connected & Ready" if ready else "Not Ready — credentials missing"

    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"### {status_icon}")
        c2.markdown(f"**{settings.active_model_name()}**  \n{status_text}")

    st.divider()

    # ── Provider selector (view-only — actual config via .env) ────────────────
    st.markdown("### Provider Overview")
    st.info(
        "Select a provider below to see its setup instructions. "
        "To switch providers, update `LLM_PROVIDER` in your `.env` file.",
        icon="ℹ️",
    )

    tab_gemini, tab_ollama = st.tabs(["☁️  Google Gemini", "🖥️  Ollama (local)"])

    for tab, provider in [(tab_gemini, LLMProvider.GEMINI), (tab_ollama, LLMProvider.OLLAMA)]:
        doc = _PROVIDER_DOCS[provider]
        is_active = settings.llm_provider == provider

        with tab:
            if is_active:
                st.success(f"**Active provider** — currently selected in your `.env`.")

            st.markdown(f"#### {doc['title']}")
            st.markdown(doc["description"])
            st.markdown(
                f"📖 [Documentation]({doc['doc_url']})   |   "
                f"🔑 [{'Download Ollama' if provider == LLMProvider.OLLAMA else 'Get API Key'}]({doc['get_key_url']})"
            )

            st.markdown("**Available models:**")
            st.markdown(
                "  ".join(f"`{m}`" for m in doc["models"])
            )

            st.markdown("**`.env` snippet:**")
            st.code(_env_snippet(provider), language="bash")

    st.divider()

    # ── Advanced settings reference ───────────────────────────────────────────
    st.markdown("### Full Settings Reference")
    st.caption("All recognised environment variables and their current resolved values.")

    rows = [
        ("LLM_PROVIDER", settings.llm_provider.value, "gemini | ollama"),
        ("GEMINI_API_KEY", "●●●●●●●●" if settings.gemini_api_key else "*(not set)*", "Required for Gemini"),
        ("GEMINI_MODEL", settings.gemini_model, "Gemini model tag"),
        ("OLLAMA_BASE_URL", settings.ollama_base_url, "Ollama server URL"),
        ("OLLAMA_MODEL", settings.ollama_model, "Ollama model tag"),
        ("APP_ENV", settings.app_env.value, "development | staging | production"),
        ("LOG_LEVEL", settings.log_level.value, "DEBUG | INFO | WARNING | ERROR"),
        ("SQLITE_DB_PATH", str(settings.sqlite_db_path), "Path to SQLite database"),
        ("CHROMA_PERSIST_DIR", str(settings.chroma_persist_dir), "ChromaDB data directory"),
        ("GITHUB_TOKEN", "●●●●●●●●" if getattr(settings, "github_token", None) else "*(not set — 60 req/hr)*", "Optional GitHub Token (5,000 req/hr)"),
        ("MAX_FILE_SIZE_BYTES", str(settings.max_file_size_bytes), "Max source file size (bytes)"),
    ]

    st.table(
        {
            "Variable": [r[0] for r in rows],
            "Current Value": [r[1] for r in rows],
            "Description": [r[2] for r in rows],
        }
    )

    st.caption(
        "ℹ️  `GEMINI_API_KEY` is masked for security. "
        "All values are read-only here — edit `.env` to change them."
    )
