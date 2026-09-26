"""
tools/language_utils.py — Language detection and syntax validation utilities.

Provides:
  - detect_language(filename): maps file extension → (language_name, syntax_id)
  - is_python(filename): returns True when the file is Python
  - validate_syntax(code, filename): validates Python-only; always passes for other languages
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Extension → (pretty name, Pygments/st.code language id)
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, tuple[str, str]] = {
    # Python
    ".py":   ("Python",      "python"),
    ".pyw":  ("Python",      "python"),
    ".pyi":  ("Python",      "python"),
    # JavaScript / TypeScript
    ".js":   ("JavaScript",  "javascript"),
    ".jsx":  ("JavaScript",  "javascript"),
    ".mjs":  ("JavaScript",  "javascript"),
    ".ts":   ("TypeScript",  "typescript"),
    ".tsx":  ("TypeScript",  "typescript"),
    # Java
    ".java": ("Java",        "java"),
    # C / C++
    ".c":    ("C",           "c"),
    ".h":    ("C",           "c"),
    ".cpp":  ("C++",         "cpp"),
    ".cc":   ("C++",         "cpp"),
    ".cxx":  ("C++",         "cpp"),
    ".hpp":  ("C++",         "cpp"),
    # C#
    ".cs":   ("C#",          "csharp"),
    # Go
    ".go":   ("Go",          "go"),
    # Rust
    ".rs":   ("Rust",        "rust"),
    # Ruby
    ".rb":   ("Ruby",        "ruby"),
    # PHP
    ".php":  ("PHP",         "php"),
    # Swift
    ".swift":("Swift",       "swift"),
    # Kotlin
    ".kt":   ("Kotlin",      "kotlin"),
    ".kts":  ("Kotlin",      "kotlin"),
    # Scala
    ".scala":("Scala",       "scala"),
    # Shell
    ".sh":   ("Shell",       "bash"),
    ".bash": ("Shell",       "bash"),
    ".zsh":  ("Shell",       "bash"),
    # SQL
    ".sql":  ("SQL",         "sql"),
    # Markdown / text
    ".md":   ("Markdown",    "markdown"),
    ".txt":  ("Text",        "text"),
    # YAML / TOML / JSON
    ".yml":  ("YAML",        "yaml"),
    ".yaml": ("YAML",        "yaml"),
    ".toml": ("TOML",        "toml"),
    ".json": ("JSON",        "json"),
    # Terraform / HCL
    ".tf":   ("Terraform",   "hcl"),
    ".hcl":  ("HCL",         "hcl"),
}

_DEFAULT_LANGUAGE = ("Code", "text")


def detect_language(filename: str) -> tuple[str, str]:
    """Return (pretty_name, pygments_id) from file extension.

    Falls back to ("Code", "text") for unknown extensions.
    """
    ext = Path(filename).suffix.lower()
    return _EXT_MAP.get(ext, _DEFAULT_LANGUAGE)


def is_python(filename: str) -> bool:
    """Return True if the given filename has a Python extension."""
    return Path(filename).suffix.lower() in (".py", ".pyw", ".pyi")


# ---------------------------------------------------------------------------
# Prompt formatting for review agents
# ---------------------------------------------------------------------------

def is_diff_payload(text: str) -> bool:
    """Detect a pre-formatted markdown diff payload.

    ``diff_processing.chunker.create_review_prompt_payload`` emits a markdown
    document (``## Changed Code for Review`` header, ``### File:`` sections and
    embedded ````` ```diff ````` fenced hunks). Such a payload is already
    prompt-ready: re-wrapping it in line numbers and an outer code fence
    corrupts it and degrades LLM detection.
    """
    if not text:
        return False
    head = text.lstrip()[:400]
    return (
        head.startswith("## Changed Code for Review")
        or "```diff" in text
        or ("### File:" in text and "**Change Type**" in text)
    )


def format_code_for_prompt(source_code: str, filename: str) -> tuple[str, bool]:
    """Build a clean, LLM-ready code block for review prompts.

    Returns ``(block, is_diff)`` where:
      - ``is_diff`` is True when *source_code* is a pre-formatted markdown diff
        payload. In that case ``block`` is the payload verbatim (no line
        numbers, no outer fence) so the embedded ````` ```diff ````` fences stay
        intact.
      - Otherwise ``block`` is the line-numbered source wrapped in a fenced code
        block for the language detected from *filename*.
    """
    text = source_code or ""
    if is_diff_payload(text):
        return text.rstrip(), True

    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1:4d} | {ln}" for i, ln in enumerate(lines))
    _, lang_id = detect_language(filename)
    return f"```{lang_id}\n{numbered}\n```", False


# ---------------------------------------------------------------------------
# Language-agnostic syntax validation
# ---------------------------------------------------------------------------

@dataclass
class SyntaxValidationResult:
    """Result of generic syntax validation."""
    is_valid: bool
    language: str             # detected language name
    errors: list[str] = field(default_factory=list)
    error_line: Optional[int] = None
    error_col: Optional[int] = None
    node_count: int = 0
    normalized_code: str = ""
    note: str = ""            # e.g. "Syntax validation only supported for Python"


def validate_syntax(code: str, filename: str = "snippet.txt") -> SyntaxValidationResult:
    """Validate syntax for any file type.

    For Python files: performs full AST parse (fast, zero-cost rejection).
    For all other languages: always returns is_valid=True (LLM handles semantic checks).

    Parameters
    ----------
    code:
        Source code to validate.
    filename:
        Used to determine language.

    Returns
    -------
    SyntaxValidationResult
    """
    lang_name, _ = detect_language(filename)

    if not code or not code.strip():
        return SyntaxValidationResult(
            is_valid=True,
            language=lang_name,
            normalized_code=code or "",
        )

    if not is_python(filename):
        return SyntaxValidationResult(
            is_valid=True,
            language=lang_name,
            normalized_code=code,
            note=f"Syntax validation not available for {lang_name}; semantic review will be performed by LLM.",
        )

    # ── Python: use ast.parse ──────────────────────────────────────────────
    try:
        tree = ast.parse(code, filename=filename)
        node_count = sum(1 for _ in ast.walk(tree))
        return SyntaxValidationResult(
            is_valid=True,
            language=lang_name,
            node_count=node_count,
            normalized_code=code,
        )
    except (SyntaxError, IndentationError) as initial_err:
        dedented = textwrap.dedent(code)
        if dedented != code:
            try:
                tree = ast.parse(dedented, filename=filename)
                node_count = sum(1 for _ in ast.walk(tree))
                return SyntaxValidationResult(
                    is_valid=True,
                    language=lang_name,
                    node_count=node_count,
                    normalized_code=dedented,
                )
            except (SyntaxError, IndentationError):
                pass

        line_info = f"line {initial_err.lineno}" if initial_err.lineno else "unknown line"
        col_info = f"col {initial_err.offset}" if initial_err.offset else ""
        loc = f" ({line_info}{', ' + col_info if col_info else ''})"
        msg = f"SyntaxError: {initial_err.msg}{loc}"

        return SyntaxValidationResult(
            is_valid=False,
            language=lang_name,
            errors=[msg],
            error_line=initial_err.lineno,
            error_col=initial_err.offset,
            normalized_code=code,
        )
    except Exception as exc:
        return SyntaxValidationResult(
            is_valid=False,
            language=lang_name,
            errors=[f"Unexpected validation error: {exc}"],
            normalized_code=code,
        )
