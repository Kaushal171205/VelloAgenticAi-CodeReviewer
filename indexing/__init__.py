"""
indexing — Shared codebase indexing pipeline.

Public API
──────────
    from indexing import index_codebase, search_codebase, get_indexed_count
    from indexing import DependencyGraph, build_dependency_graph
    from indexing import CodeUnit, extract_code_units, generate_summaries
    from indexing import IndexingResult
"""

from indexing.dependency_graph_builder import DependencyGraph, build_dependency_graph
from indexing.incremental_updater import (
    IndexingResult,
    get_indexed_count,
    index_codebase,
    search_codebase,
)
from indexing.summary_generator import CodeUnit, extract_code_units, extract_code_units_for_file, generate_summaries

__all__ = [
    # Main entry-point
    "index_codebase",
    "search_codebase",
    "get_indexed_count",
    "IndexingResult",
    # Dependency graph
    "DependencyGraph",
    "build_dependency_graph",
    # Summary generator
    "CodeUnit",
    "extract_code_units",
    "extract_code_units_for_file",
    "generate_summaries",
]
