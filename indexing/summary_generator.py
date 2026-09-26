"""
indexing/summary_generator.py — LLM-backed Code Unit Summary Generator.

Parses Python files with the `ast` module, extracts function and class
definitions, then optionally generates concise natural-language summaries
using the configured LLM.  Summaries are combined with AST-derived metadata
to produce rich ChromaDB documents.

Design rules
────────────
• Gracefully degrades when the LLM is unavailable (falls back to a
  docstring-only or signature-only summary so indexing still works).
• No I/O side-effects beyond logging.
• All identifiers are deterministic (hash of path + unit name) so indexing
  is fully repeatable / idempotent when used with ChromaDB ``upsert``.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.language_utils import detect_language, is_python

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeUnit:
    """
    Represents one indexable code unit (a function or class).

    ``stable_id`` is deterministic so repeated indexing is idempotent.
    """
    stable_id: str          # SHA256-prefix of "<rel_path>:<unit_name>"
    file_path: str          # POSIX path relative to the repository root
    unit_name: str          # function or class name
    unit_type: str          # "function" | "class" | "method"
    start_line: int
    end_line: int
    signature: str          # def/class line(s) without body
    docstring: str          # first docstring if present, else ""
    source_snippet: str     # up to 40 lines of source (truncated for embedding)
    summary: str            # LLM-generated or fallback human-readable summary
    decorators: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None   # Set for methods

    def to_chroma_document(self) -> Dict[str, Any]:
        """Flatten into a dict suitable for ChromaDB ``upsert``."""
        return {
            "id": self.stable_id,
            "content": self.summary or self.source_snippet,
            "title": self.unit_name,
            "file_path": self.file_path,
            "unit_type": self.unit_type,
            "start_line": str(self.start_line),
            "end_line": str(self.end_line),
            "signature": self.signature[:500],
            "docstring": self.docstring[:500],
        }


# ─────────────────────────────────────────────────────────────────────────────
# AST extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable_id(file_rel: str, unit_name: str) -> str:
    """Return a 16-character hex prefix of the SHA-256 of ``<path>:<name>``."""
    raw = f"{file_rel}:{unit_name}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _get_docstring(node: ast.AST) -> str:
    """Extract the first docstring from a function or class body."""
    try:
        return ast.get_docstring(node) or ""  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return ""


def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: List[str]) -> str:
    """
    Extract the ``def`` line(s) without the body.
    Handles multi-line signatures and decorators.
    """
    lines = source_lines[node.lineno - 1: node.body[0].lineno - 1]
    return "".join(lines).rstrip()


def _get_class_signature(node: ast.ClassDef, source_lines: List[str]) -> str:
    """Extract the ``class`` line without the body."""
    lines = source_lines[node.lineno - 1: node.body[0].lineno - 1]
    return "".join(lines).rstrip()


def _get_decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> List[str]:
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(f"{ast.unparse(dec)}")
        else:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                decorators.append("@<decorator>")
    return decorators


def _snippet(source_lines: List[str], start: int, end: int, max_lines: int = 40) -> str:
    """Return up to *max_lines* lines from [start, end] (1-indexed, inclusive)."""
    lines = source_lines[start - 1: end]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)\n"]
    return textwrap.dedent("".join(lines))


def extract_code_units(source: str, file_rel: str) -> List[CodeUnit]:
    """
    Parse *source* and return one :class:`CodeUnit` per top-level function/class
    (and one per method within each class).

    Returns an empty list (and logs a warning) on syntax errors.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning(f"Syntax error while extracting units from '{file_rel}': {exc}")
        return []

    source_lines = source.splitlines(keepends=True)
    units: List[CodeUnit] = []

    def _process_func(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: Optional[str] = None,
    ) -> CodeUnit:
        unit_type = "method" if parent_class else "function"
        end_line = node.end_lineno or node.lineno
        return CodeUnit(
            stable_id=_stable_id(file_rel, f"{parent_class}.{node.name}" if parent_class else node.name),
            file_path=file_rel,
            unit_name=node.name,
            unit_type=unit_type,
            start_line=node.lineno,
            end_line=end_line,
            signature=_get_signature(node, source_lines),
            docstring=_get_docstring(node),
            source_snippet=_snippet(source_lines, node.lineno, end_line),
            summary="",  # Filled in later
            decorators=_get_decorator_names(node),
            parent_class=parent_class,
        )

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(_process_func(node))

        elif isinstance(node, ast.ClassDef):
            end_line = node.end_lineno or node.lineno
            class_unit = CodeUnit(
                stable_id=_stable_id(file_rel, node.name),
                file_path=file_rel,
                unit_name=node.name,
                unit_type="class",
                start_line=node.lineno,
                end_line=end_line,
                signature=_get_class_signature(node, source_lines),
                docstring=_get_docstring(node),
                source_snippet=_snippet(source_lines, node.lineno, end_line),
                summary="",
                decorators=_get_decorator_names(node),
            )
            units.append(class_unit)

            # Extract methods
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    units.append(_process_func(child, parent_class=node.name))

    return units


# ─────────────────────────────────────────────────────────────────────────────
# Fallback summary (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_summary(unit: CodeUnit) -> str:
    """
    Construct a deterministic summary without an LLM.
    Uses signature + docstring (first sentence) + decorator hints.
    """
    parts = []

    if unit.unit_type == "chunk":
        # Non-Python line-chunk — summary is already set by _chunk_source_file
        return unit.summary or f"Code chunk `{unit.unit_name}` in `{unit.file_path}` at line {unit.start_line}."
    elif unit.unit_type == "class":
        parts.append(f"Class `{unit.unit_name}`.")
    elif unit.unit_type == "method":
        parts.append(f"Method `{unit.parent_class}.{unit.unit_name}`.")
    else:
        parts.append(f"Function `{unit.unit_name}`.")

    if unit.docstring:
        first_sentence = unit.docstring.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence + ".")

    if unit.decorators:
        parts.append(f"Decorators: {', '.join('@' + d for d in unit.decorators)}.")

    parts.append(f"Defined in `{unit.file_path}` at line {unit.start_line}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-backed summary (optional)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SUMMARY_PROMPT = """\
You are a senior software engineer. Write a ONE-sentence concise summary \
(max 25 words) of the following {unit_type}.

File: {file_path}
Signature: {signature}
Docstring: {docstring}

Code:
{snippet}

Respond with ONLY the summary sentence. No preamble, no markdown.
"""


def generate_summary_with_llm(unit: CodeUnit, llm_client: Any) -> str:
    """
    Use the LLM to generate a one-sentence summary for *unit*.

    Falls back to :func:`_build_fallback_summary` if the LLM call fails.

    Args:
        unit:       The :class:`CodeUnit` to summarise.
        llm_client: An instance of the project's ``LLMClient`` abstraction.

    Returns:
        Summary string (never empty).
    """
    prompt = _LLM_SUMMARY_PROMPT.format(
        unit_type=unit.unit_type,
        file_path=unit.file_path,
        signature=unit.signature[:300],
        docstring=unit.docstring[:200] or "No docstring.",
        snippet=unit.source_snippet[:800],
    )
    try:
        response = llm_client.generate(prompt)
        text = response.text.strip() if hasattr(response, "text") else str(response).strip()
        if text:
            # Truncate to avoid accidentally storing huge responses
            return text[:300]
    except Exception as exc:
        logger.warning(f"LLM summary failed for '{unit.unit_name}': {exc}")

    return _build_fallback_summary(unit)


# ─────────────────────────────────────────────────────────────────────────────
# High-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_summaries(
    units: List[CodeUnit],
    llm_client: Optional[Any] = None,
    use_llm: bool = False,
) -> List[CodeUnit]:
    """
    Attach summaries to each :class:`CodeUnit`.

    Args:
        units:      Code units extracted from a file.
        llm_client: Optional LLM client.  Required when ``use_llm=True``.
        use_llm:    If True and ``llm_client`` is provided, call the LLM for
                    each unit.  Otherwise use the fast fallback summary.

    Returns:
        The same list with ``unit.summary`` filled in.
    """
    for unit in units:
        if use_llm and llm_client is not None:
            unit.summary = generate_summary_with_llm(unit, llm_client)
        else:
            unit.summary = _build_fallback_summary(unit)
    return units


# ─────────────────────────────────────────────────────────────────────────────
# Language-agnostic file chunker (for non-Python source files)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 60   # lines per chunk for non-Python files


def _chunk_source_file(
    source: str,
    file_rel: str,
    chunk_size: int = _CHUNK_SIZE,
) -> List[CodeUnit]:
    """
    Split *source* into fixed-size line chunks and return one :class:`CodeUnit`
    per chunk.  Used for non-Python files where AST parsing is not available.

    Each chunk is indexed as a ``unit_type="chunk"`` unit so it is still
    semantically searchable.
    """
    lang_name, _ = detect_language(file_rel)
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    units: List[CodeUnit] = []
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    padding = len(str(total_chunks))

    for idx, start in enumerate(range(0, len(lines), chunk_size), start=1):
        chunk_lines = lines[start : start + chunk_size]
        end = start + len(chunk_lines)
        chunk_text = "".join(chunk_lines)

        chunk_name = f"chunk_{idx:0{padding}d}"
        stable = f"{file_rel}:{chunk_name}".encode()
        sid = hashlib.sha256(stable).hexdigest()[:16]

        summary = (
            f"{lang_name} code block {idx}/{total_chunks} "
            f"in `{file_rel}` (lines {start + 1}–{end})."
        )

        units.append(CodeUnit(
            stable_id=sid,
            file_path=file_rel,
            unit_name=chunk_name,
            unit_type="chunk",
            start_line=start + 1,
            end_line=end,
            signature="",
            docstring="",
            source_snippet=chunk_text[:2000],  # cap for embedding
            summary=summary,
        ))

    return units


def extract_code_units_for_file(
    source: str,
    file_rel: str,
    llm_client: Optional[Any] = None,
    use_llm: bool = False,
) -> List[CodeUnit]:
    """
    Language-aware dispatcher.

    * **Python (.py)**: uses full AST extraction → function/class/method units.
    * **Other languages**: falls back to fixed-size line chunking.

    After extraction, summaries are generated (LLM or fallback) and attached.

    Args:
        source:     Raw source code.
        file_rel:   Relative file path (used for language detection & IDs).
        llm_client: Optional LLM client for enhanced summaries.
        use_llm:    Whether to invoke the LLM for summaries.

    Returns:
        List of :class:`CodeUnit` with summaries attached.
    """
    if is_python(file_rel):
        units = extract_code_units(source, file_rel)
    else:
        units = _chunk_source_file(source, file_rel)

    return generate_summaries(units, llm_client=llm_client, use_llm=use_llm)
