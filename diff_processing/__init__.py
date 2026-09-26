"""
diff_processing — Git diff extraction, function-level identification, and LLM chunking.
"""

from __future__ import annotations

from diff_processing.chunker import (
    ReviewChunk,
    chunk_review_changes,
    create_review_prompt_payload,
)
from diff_processing.git_diff_extractor import (
    ChangedFile,
    DiffExtractionResult,
    DiffHunk,
    ReviewChange,
    extract_review_changes,
)

__all__ = [
    "ChangedFile",
    "DiffExtractionResult",
    "DiffHunk",
    "ReviewChange",
    "ReviewChunk",
    "chunk_review_changes",
    "create_review_prompt_payload",
    "extract_review_changes",
]
