"""
diff_processing/chunker.py — LLM-focused Diff Chunking & Code Packaging.

Transforms extracted Git diff changes into compact, token-bounded ReviewChunk
objects. Ensures that only relevant changed code (never the complete repository)
is transmitted to LLM review agents.

Key capabilities:
- Groups or splits diff hunks by function/file boundary.
- Preserves file paths, line ranges, function scopes, and diff content.
- Generates clean, markdown-formatted code snippets tailored for prompt injection.
- Estimates token costs to prevent context window saturation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from diff_processing.git_diff_extractor import (
    DiffExtractionResult,
    DiffHunk,
    ReviewChange,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReviewChunk:
    """
    A compact, self-contained unit of changed code prepared for LLM review.

    Guarantees:
    - Never contains irrelevant files or the full codebase.
    - Preserves file path, old/new line numbers, function scope, and diff content.
    """
    chunk_id: str
    file_path: str
    change_type: str
    is_python: bool
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    old_start: int = 0
    old_lines: int = 0
    new_start: int = 0
    new_lines: int = 0
    diff_content: str = ""
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    char_count: int = 0
    estimated_tokens: int = 0

    @property
    def display_scope(self) -> str:
        """Human-readable scope description (e.g. 'Class.method' or 'func_name')."""
        if self.class_name and self.function_name:
            if self.function_name.startswith(self.class_name + "."):
                return self.function_name
            return f"{self.class_name}.{self.function_name}"
        if self.function_name:
            return self.function_name
        return "<module>"

    def format_for_llm(self) -> str:
        """
        Format this chunk as a clean markdown snippet ready for an LLM prompt.

        Returns only the relevant diff content with context metadata.
        """
        lines_info = (
            f"Lines: -{self.old_start},{self.old_lines} +{self.new_start},{self.new_lines}"
            if (self.old_start or self.new_start)
            else "Lines: N/A"
        )
        return (
            f"### File: `{self.file_path}`\n"
            f"- **Scope**: `{self.display_scope}`\n"
            f"- **Change Type**: `{self.change_type}`\n"
            f"- **{lines_info}** (+{len(self.added_lines)} / -{len(self.removed_lines)} lines)\n\n"
            f"```diff\n"
            f"{self.diff_content.strip()}\n"
            f"```\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "display_scope": self.display_scope,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "change_type": self.change_type,
            "is_python": self.is_python,
            "old_start": self.old_start,
            "old_lines": self.old_lines,
            "new_start": self.new_start,
            "new_lines": self.new_lines,
            "diff_content": self.diff_content,
            "additions": len(self.added_lines),
            "deletions": len(self.removed_lines),
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk_id(file_path: str, scope: str, old_start: int, new_start: int) -> str:
    """Generate a deterministic 12-char hex ID for a review chunk."""
    seed = f"{file_path}:{scope}:{old_start}:{new_start}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Chunking API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_review_changes(
    changes_or_result: Union[List[ReviewChange], DiffExtractionResult],
    max_chars_per_chunk: int = 4000,
    python_only: bool = False,
    merge_same_function: bool = True,
) -> List[ReviewChunk]:
    """
    Convert diff changes into bounded, LLM-ready :class:`ReviewChunk` instances.

    Guarantees:
    - Filters out non-Python changes if `python_only=True`.
    - Merges consecutive hunks within the same function or file when they fit
      inside `max_chars_per_chunk`.
    - Never includes unedited codebase files.

    Args:
        changes_or_result: List of ReviewChange objects, or a DiffExtractionResult.
        max_chars_per_chunk: Approximate maximum characters per chunk (default: 4000).
        python_only: If True, filters out non-.py changes (default: True).
        merge_same_function: If True, collapses adjacent hunks affecting the same
                             function/file into one chunk when within budget.

    Returns:
        A list of :class:`ReviewChunk` objects.
    """
    raw_changes: List[ReviewChange] = (
        changes_or_result.changes
        if isinstance(changes_or_result, DiffExtractionResult)
        else list(changes_or_result)
    )

    if python_only:
        raw_changes = [c for c in raw_changes if c.is_python]

    if not raw_changes:
        return []

    chunks: List[ReviewChunk] = []

    if not merge_same_function:
        for c in raw_changes:
            content = c.diff_content
            char_count = len(content)
            chunk_id = _make_chunk_id(c.file_path, c.display_scope, c.old_start, c.new_start)
            chunks.append(
                ReviewChunk(
                    chunk_id=chunk_id,
                    file_path=c.file_path,
                    change_type=c.change_type,
                    is_python=c.is_python,
                    function_name=c.function_name,
                    class_name=c.class_name,
                    old_start=c.old_start,
                    old_lines=c.old_lines,
                    new_start=c.new_start,
                    new_lines=c.new_lines,
                    diff_content=content,
                    added_lines=c.added_lines,
                    removed_lines=c.removed_lines,
                    char_count=char_count,
                    estimated_tokens=max(1, char_count // 4),
                )
            )
        return chunks

    # Group adjacent changes by (file_path, display_scope) when within max_chars_per_chunk
    current_group: List[ReviewChange] = []
    current_key: Optional[tuple] = None
    current_size = 0

    def flush_group():
        nonlocal current_group, current_key, current_size
        if not current_group:
            return

        first = current_group[0]
        combined_diff = "\n\n".join(c.diff_content.strip() for c in current_group if c.diff_content.strip())
        all_added = [line for c in current_group for line in c.added_lines]
        all_removed = [line for c in current_group for line in c.removed_lines]

        min_old = min(c.old_start for c in current_group)
        sum_old = sum(c.old_lines for c in current_group)
        min_new = min(c.new_start for c in current_group)
        sum_new = sum(c.new_lines for c in current_group)

        char_count = len(combined_diff)
        chunk_id = _make_chunk_id(first.file_path, first.display_scope, min_old, min_new)

        chunks.append(
            ReviewChunk(
                chunk_id=chunk_id,
                file_path=first.file_path,
                change_type=first.change_type,
                is_python=first.is_python,
                function_name=first.function_name,
                class_name=first.class_name,
                old_start=min_old,
                old_lines=sum_old,
                new_start=min_new,
                new_lines=sum_new,
                diff_content=combined_diff,
                added_lines=all_added,
                removed_lines=all_removed,
                char_count=char_count,
                estimated_tokens=max(1, char_count // 4),
            )
        )
        current_group = []
        current_key = None
        current_size = 0

    for change in raw_changes:
        key = (change.file_path, change.display_scope)
        diff_len = len(change.diff_content)

        if current_key is not None and (
            key != current_key or (current_size + diff_len > max_chars_per_chunk)
        ):
            flush_group()

        current_group.append(change)
        current_key = key
        current_size += diff_len

    flush_group()
    return chunks


def create_review_prompt_payload(
    chunks: List[ReviewChunk],
    max_tokens: int = 4000,
) -> str:
    """
    Consolidate a sequence of :class:`ReviewChunk` objects into a single markdown
    payload ready to inject into an LLM prompt.

    Enforces token bounding to ensure prompt limits are respected.
    """
    if not chunks:
        return "No code changes identified for review."

    parts: List[str] = [
        "## Changed Code for Review\n",
        "The following changes represent only the modified hunks and functions "
        "extracted from the Git diff. Review them for logic errors, vulnerabilities, "
        "and code quality.\n\n",
    ]

    total_tokens = 0
    included_count = 0

    for chunk in chunks:
        snippet = chunk.format_for_llm()
        chunk_tokens = chunk.estimated_tokens

        if total_tokens + chunk_tokens > max_tokens and included_count > 0:
            remaining = len(chunks) - included_count
            parts.append(
                f"\n> [!NOTE]\n"
                f"> Truncated {remaining} additional review chunk(s) to stay within the "
                f"token budget ({max_tokens} tokens).\n"
            )
            break

        parts.append(snippet)
        parts.append("\n---\n\n")
        total_tokens += chunk_tokens
        included_count += 1

    return "".join(parts)
