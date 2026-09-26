"""
tests/test_config.py — Phase 1 unit tests for the configuration module.

Run with:
    pytest tests/test_config.py -v
"""

from __future__ import annotations

import os

import pytest

from config import AppEnvironment, LLMProvider, LogLevel, Settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_settings(**overrides) -> Settings:
    """Instantiate Settings with specific env vars, bypassing the .env file."""
    env = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_MODEL": "gemini-2.0-flash",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "APP_ENV": "development",
        "LOG_LEVEL": "INFO",
        "SQLITE_DB_PATH": "./storage/reviews.db",
        "CHROMA_PERSIST_DIR": "./storage/chroma",
        "MAX_FILE_SIZE_BYTES": "524288",
    }
    env.update(overrides)
    # Patch environment so pydantic-settings picks up our values
    original = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        s = Settings(_env_file=None)  # skip .env file lookup
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_provider_is_gemini(self):
        s = make_settings()
        assert s.llm_provider == LLMProvider.GEMINI

    def test_default_gemini_model(self):
        s = make_settings()
        assert s.gemini_model == "gemini-2.0-flash"

    def test_default_env_is_development(self):
        s = make_settings()
        assert s.app_env == AppEnvironment.DEVELOPMENT

    def test_default_log_level(self):
        s = make_settings()
        assert s.log_level == LogLevel.INFO


class TestProviderReadiness:
    def test_gemini_not_ready_without_key(self):
        s = make_settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="")
        assert not s.is_llm_ready()

    def test_gemini_ready_with_key(self):
        s = make_settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="AIza_test_key")
        assert s.is_llm_ready()

    def test_ollama_ready_with_url(self):
        s = make_settings(LLM_PROVIDER="ollama", OLLAMA_BASE_URL="http://localhost:11434")
        assert s.is_llm_ready()

    def test_active_model_name_gemini(self):
        s = make_settings(LLM_PROVIDER="gemini", GEMINI_MODEL="gemini-2.0-flash")
        assert "Gemini" in s.active_model_name()
        assert "gemini-2.0-flash" in s.active_model_name()

    def test_active_model_name_ollama(self):
        s = make_settings(LLM_PROVIDER="ollama", OLLAMA_MODEL="llama3.2")
        assert "Ollama" in s.active_model_name()
        assert "llama3.2" in s.active_model_name()


class TestValidation:
    def test_invalid_provider_raises(self):
        with pytest.raises(Exception):
            make_settings(LLM_PROVIDER="unknown_provider")

    def test_max_file_size_is_int(self):
        s = make_settings(MAX_FILE_SIZE_BYTES="1048576")
        assert isinstance(s.max_file_size_bytes, int)
        assert s.max_file_size_bytes == 1_048_576

    def test_empty_api_key_treated_as_none(self):
        s = make_settings(GEMINI_API_KEY="")
        assert s.gemini_api_key is None
