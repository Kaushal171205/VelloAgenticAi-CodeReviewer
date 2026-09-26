"""
indexing/incremental_updater.py — Incremental Codebase Indexer.

High-level orchestrator that ties together:
  • :mod:`indexing.dependency_graph_builder` — filesystem traversal + import
    graph
  • :mod:`indexing.summary_generator`        — AST extraction + summaries
  • :class:`storage.vector_store.SecurityVectorStore` (reused for code units
    with a dedicated collection)

Exposes a single public entry-point::

    result = index_codebase(path="/path/to/repo")

The function is *repeatable*: ChromaDB ``upsert`` semantics mean re-running
on the same codebase updates changed units and leaves unchanged ones in place.

Skipped directories
───────────────────
Same set as :mod:`indexing.dependency_graph_builder` (SKIP_DIRS).

Error handling
──────────────
Syntax errors and file-read failures in individual files are captured in
``IndexingResult.errors`` — they never abort the entire run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from indexing.dependency_graph_builder import (
    SKIP_DIRS,
    DependencyGraph,
    build_dependency_graph,
)
from indexing.summary_generator import (
    CodeUnit,
    extract_code_units_for_file,
    generate_summaries,
)
from tools.language_utils import _EXT_MAP

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Collection name for the codebase index
# ─────────────────────────────────────────────────────────────────────────────

CODEBASE_COLLECTION_NAME = "codebase_index"


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexingResult:
    """Statistics and metadata returned by :func:`index_codebase`."""

    repository_path: str
    total_files_scanned: int = 0
    total_units_indexed: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_methods: int = 0
    skipped_files: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    dependency_graph_stats: Dict[str, int] = field(default_factory=dict)
    mermaid_graph: str = ""

    def summary(self) -> str:
        lines = [
            f"Repository : {self.repository_path}",
            f"Files scanned : {self.total_files_scanned}",
            f"Code units indexed : {self.total_units_indexed} "
            f"(functions={self.total_functions}, "
            f"classes={self.total_classes}, "
            f"methods={self.total_methods})",
            f"Files skipped (errors) : {self.skipped_files}",
            f"Duration : {self.duration_seconds:.2f}s",
        ]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}) : {', '.join(self.errors[:5])}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Codebase vector store (lazy singleton per process)
# ─────────────────────────────────────────────────────────────────────────────

_codebase_store: Optional[Any] = None


def _get_codebase_store(persist_dir: Optional[Path] = None) -> Any:
    """
    Return (or lazily create) the ChromaDB-backed code unit store.

    Uses a *separate* collection (``codebase_index``) from the security
    knowledge base so the two do not interfere.
    """
    global _codebase_store
    if _codebase_store is None:
        # Import here to avoid circular imports at module load time
        from config import settings
        from storage.vector_store import SecurityVectorStore  # reused wrapper

        store_dir = persist_dir or settings.chroma_persist_dir
        _codebase_store = SecurityVectorStore(
            persist_directory=store_dir,
            collection_name=CODEBASE_COLLECTION_NAME,
            auto_preload=False,   # codebase index is populated by index_codebase()
        )
    return _codebase_store


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Supported source file extensions (mirrors tools/language_utils._EXT_MAP)
_SOURCE_EXTENSIONS: frozenset[str] = frozenset(_EXT_MAP.keys())


def _collect_source_files(root: Path) -> List[Path]:
    """Recursively collect all supported source files, honouring SKIP_DIRS."""
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _SOURCE_EXTENSIONS:
            if not any(part in SKIP_DIRS for part in path.parts):
                files.append(path)
    return sorted(files)


def _units_to_chroma_docs(units: List[CodeUnit]) -> List[Dict[str, Any]]:
    """
    Convert a list of code units into the batch-upsert format expected by
    :meth:`storage.vector_store.SecurityVectorStore.add_documents`.
    """
    docs = []
    for u in units:
        # We embed the summary (or source_snippet) as the main content
        content = u.summary or u.source_snippet
        docs.append({
            "id": u.stable_id,
            "content": content,
            "title": u.unit_name,
            # ChromaDB metadata values must be str/int/float/bool
            "file_path": u.file_path,
            "unit_type": u.unit_type,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "signature": u.signature[:400],
            "docstring": u.docstring[:400],
            # Reuse the generic metadata fields recognised by SecurityVectorStore
            "cwe": "",
            "severity": "INFO",
            "category": "codebase",
        })
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def index_codebase(
    path: str | Path,
    llm_client: Optional[Any] = None,
    use_llm_summaries: bool = False,
    persist_dir: Optional[Path] = None,
    batch_size: int = 50,
) -> IndexingResult:
    """
    Index a Python repository into ChromaDB.

    Args:
        path:              Root directory of the repository to index.
        llm_client:        Optional LLM client for rich summaries.  When
                           ``None`` or ``use_llm_summaries=False`` the fast
                           fallback summary is used.
        use_llm_summaries: Pass ``True`` to call the LLM for each unit.
                           Slower but richer.
        persist_dir:       Override the ChromaDB persistence directory.
        batch_size:        Number of code units to upsert per ChromaDB call.

    Returns:
        :class:`IndexingResult` with statistics about the run.
    """
    path_str = str(path).strip()
    start_time = time.monotonic()
    result = IndexingResult(repository_path=path_str)
    store = _get_codebase_store(persist_dir)

    # ── Remote GitHub Repository (Zero-Clone) ─────────────────────────────────
    from tools.github_client import GitHubAPIError, GitHubClient, is_github_url, parse_github_url

    if is_github_url(path_str):
        parsed = parse_github_url(path_str)
        if not parsed:
            raise ValueError(f"Invalid GitHub repository URL: '{path_str}'")
        owner, repo_name = parsed
        client = GitHubClient()
        logger.info(f"[Indexing] Fetching remote file tree for GitHub repo '{owner}/{repo_name}'…")

        try:
            repo_info = client.get_repo_info(owner, repo_name)
            default_branch = repo_info.get("default_branch", "main")
            tree_files = client.get_file_tree(owner, repo_name, branch=default_branch)
        except GitHubAPIError as exc:
            result.errors.append(str(exc))
            result.duration_seconds = time.monotonic() - start_time
            return result
        except Exception as exc:
            result.errors.append(f"GitHub API error: {exc}")
            result.duration_seconds = time.monotonic() - start_time
            return result

        py_files = [
            f for f in tree_files
            if Path(f).suffix.lower() in _SOURCE_EXTENSIONS
            and not any(part in SKIP_DIRS for part in Path(f).parts)
        ]
        result.total_files_scanned = len(py_files)

        if not py_files:
            lang = repo_info.get("language") or "unknown"
            result.errors.append(
                f"No supported source files found in remote repository '{owner}/{repo_name}'. "
                f"Primary language: {lang} (total scanned files: {len(tree_files)})."
            )
            result.duration_seconds = time.monotonic() - start_time
            return result

        logger.info(f"[Indexing] {len(py_files)} source file(s) found in '{owner}/{repo_name}'. Fetching in memory…")
        pending_docs: List[Dict[str, Any]] = []
        fetched_sources: Dict[str, str] = {}

        for py_rel in py_files:
            source = client.get_file_content(owner, repo_name, py_rel, ref=default_branch)
            if source is None:
                result.skipped_files += 1
                result.errors.append(f"Failed to fetch content for '{py_rel}'")
                continue

            fetched_sources[py_rel] = source

            units = extract_code_units_for_file(
                source, py_rel,
                llm_client=llm_client,
                use_llm=use_llm_summaries,
            )
            if not units:
                continue

            for u in units:
                if u.unit_type == "function":
                    result.total_functions += 1
                elif u.unit_type == "class":
                    result.total_classes += 1
                elif u.unit_type == "method":
                    result.total_methods += 1

            pending_docs.extend(_units_to_chroma_docs(units))
            if len(pending_docs) >= batch_size:
                try:
                    store.add_documents(pending_docs)
                    result.total_units_indexed += len(pending_docs)
                except Exception as exc:
                    result.errors.append(f"batch_upsert_error: {exc}")
                pending_docs = []

        if pending_docs:
            try:
                store.add_documents(pending_docs)
                result.total_units_indexed += len(pending_docs)
            except Exception as exc:
                result.errors.append(f"final_batch_error: {exc}")

        try:
            from indexing.dependency_graph_builder import build_dependency_graph_from_sources
            dep_graph = build_dependency_graph_from_sources(fetched_sources, root=f"{owner}/{repo_name}")
            result.dependency_graph_stats = dep_graph.stats()
            result.mermaid_graph = dep_graph.to_mermaid()
        except Exception as exc:
            logger.warning(f"[Indexing] Remote dependency graph build failed: {exc}")

        result.duration_seconds = time.monotonic() - start_time
        logger.info(f"[Indexing] Remote GitHub indexing complete.\n{result.summary()}")
        return result

    # ── Local Repository ──────────────────────────────────────────────────────
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"'{root}' is not a directory.")

    result.repository_path = str(root)

    # ── Step 1: Build dependency graph ────────────────────────────────────────
    logger.info(f"[Indexing] Building dependency graph for '{root}'…")
    try:
        dep_graph: DependencyGraph = build_dependency_graph(root)
        result.dependency_graph_stats = dep_graph.stats()
        result.mermaid_graph = dep_graph.to_mermaid()
        result.errors.extend(dep_graph.errors)
    except Exception as exc:
        logger.error(f"[Indexing] Dependency graph build failed: {exc}")
        dep_graph = DependencyGraph(root)

    # ── Step 2: Collect source files ──────────────────────────────────────────
    source_files_local = _collect_source_files(root)
    result.total_files_scanned = len(source_files_local)
    logger.info(f"[Indexing] {len(source_files_local)} source file(s) to process.")

    # ── Step 3: Extract + summarise + batch-upsert ────────────────────────────
    pending_docs = []

    for src_file in source_files_local:
        rel_path = dep_graph.to_rel(src_file)

        try:
            source = src_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"[Indexing] Cannot read '{rel_path}': {exc}")
            result.skipped_files += 1
            result.errors.append(rel_path)
            continue

        units = extract_code_units_for_file(
            source, rel_path,
            llm_client=llm_client,
            use_llm=use_llm_summaries,
        )

        if not units:
            # File is empty or has no extractable content
            continue

        # Count unit types
        for u in units:
            if u.unit_type == "function":
                result.total_functions += 1
            elif u.unit_type == "class":
                result.total_classes += 1
            elif u.unit_type == "method":
                result.total_methods += 1

        pending_docs.extend(_units_to_chroma_docs(units))

        # Flush batch
        if len(pending_docs) >= batch_size:
            try:
                store.add_documents(pending_docs)
                result.total_units_indexed += len(pending_docs)
                logger.debug(f"[Indexing] Flushed {len(pending_docs)} docs.")
            except Exception as exc:
                logger.error(f"[Indexing] Batch upsert failed: {exc}")
                result.errors.append(f"batch_upsert_error: {exc}")
            pending_docs = []

    # Final flush
    if pending_docs:
        try:
            store.add_documents(pending_docs)
            result.total_units_indexed += len(pending_docs)
        except Exception as exc:
            logger.error(f"[Indexing] Final batch upsert failed: {exc}")
            result.errors.append(f"final_batch_error: {exc}")

    result.duration_seconds = time.monotonic() - start_time
    logger.info(f"[Indexing] Done.\n{result.summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Incremental update helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_indexed_count(persist_dir: Optional[Path] = None) -> int:
    """Return the number of code units currently in the codebase index."""
    try:
        store = _get_codebase_store(persist_dir)
        return store.count()
    except Exception:
        return 0


def search_codebase(
    query: str,
    top_k: int = 5,
    persist_dir: Optional[Path] = None,
) -> List[Any]:
    """
    Semantic search over the codebase index.

    Args:
        query:  Natural-language or code snippet query.
        top_k:  Number of results to return.

    Returns:
        List of :class:`storage.vector_store.RetrievedDocument`.
    """
    store = _get_codebase_store(persist_dir)
    return store.search(query, top_k=top_k, category="codebase")
