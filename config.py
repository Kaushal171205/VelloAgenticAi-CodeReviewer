"""
config.py — Centralised configuration management for the Code Review System.

All runtime settings are loaded from environment variables (via .env).
Downstream modules should import the singleton `settings` object rather than
reading os.environ directly, so we have one place to add validation, defaults,
and documentation.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Settings model
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Application-wide settings.

    Pydantic-settings automatically reads values from environment variables
    (case-insensitive) and from the .env file located at the project root.
    Fields without defaults MUST be provided through the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unrecognised env vars
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="Which LLM backend to use: 'gemini' or 'ollama'.",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key. Required when llm_provider=gemini.",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Gemini model identifier.",
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model tag to use for reviews.",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        description="Runtime environment.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging verbosity.",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    sqlite_db_path: Path = Field(
        default=Path("./storage/reviews.db"),
        description="Path to the SQLite checkpoint database.",
    )
    chroma_persist_dir: Path = Field(
        default=Path("./storage/chroma"),
        description="Directory where ChromaDB persists its data.",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    max_file_size_bytes: int = Field(
        default=524_288,   # 512 KB
        description="Maximum source-file size accepted for review (bytes).",
    )

    # ── Git & Remote Integration ──────────────────────────────────────────────
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub Personal Access Token for higher API rate limits (5,000 req/hr) and private repos.",
    )

    # ── Derived / computed ────────────────────────────────────────────────────

    @field_validator("gemini_api_key", "github_token", mode="before")
    @classmethod
    def _mask_empty_string(cls, v: Optional[str]) -> Optional[str]:
        """Treat empty strings as unset so callers only check for None."""
        return v if v else None

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> "Settings":
        """
        Warn (don't raise) if the active provider's credentials are absent.
        We warn rather than error so the UI can still launch and show the
        configuration page even before the user adds credentials.
        """
        if self.llm_provider == LLMProvider.GEMINI and not self.gemini_api_key:
            logging.getLogger(__name__).warning(
                "LLM provider is 'gemini' but GEMINI_API_KEY is not set. "
                "Review functionality will be unavailable until you add the key."
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_llm_ready(self) -> bool:
        """Return True if the configured LLM provider has all required credentials."""
        if self.llm_provider == LLMProvider.GEMINI:
            return self.gemini_api_key is not None
        if self.llm_provider == LLMProvider.OLLAMA:
            return bool(self.ollama_base_url)
        return False

    def active_model_name(self) -> str:
        """Human-readable name of the currently configured model."""
        if self.llm_provider == LLMProvider.GEMINI:
            return f"Gemini / {self.gemini_model}"
        return f"Ollama / {self.ollama_model}"

    def ensure_storage_dirs(self) -> None:
        """Create storage directories if they do not already exist."""
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — import this from other modules
# ─────────────────────────────────────────────────────────────────────────────

settings = Settings()

# Configure the root logger once, when config.py is first imported.
logging.basicConfig(
    level=settings.log_level.value,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
