"""
context/context_retriever.py — Context Retriever Agent for Review Workflows.

Retrieves semantically related functions and dependency context from the shared
ChromaDB codebase index for functions modified in a Git diff.

Core Workflow:
  Changed Functions from Diff
              │
              ▼
  1. Generate query per function (name, signature, diff context)
              │
              ▼
  2. Query shared ChromaDB index (codebase_index collection)
              │
              ▼
  3. Filter & rank semantically related functions (excluding self)
              │
              ▼
  4. Enrich with dependency graph relations (imports, callers)
              │
              ▼
  5. Assemble token-bounded, compact context package for LLM review

Ensures the LLM receives only relevant surrounding architectural and functional
context without sending the entire repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from diff_processing.git_diff_extractor import DiffExtractionResult, ReviewChange
from indexing.dependency_graph_builder import DependencyGraph, build_dependency_graph
from indexing.incremental_updater import _get_codebase_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelatedFunction:
    """A semantically related function retrieved from the codebase index."""
    name: str
    file_path: str
    unit_type: str = "function"
    signature: str = ""
    summary: str = ""
    docstring: str = ""
    relevance_score: float = 0.0
    distance: float = 0.0
    is_direct_dependency: bool = False
    is_dependent: bool = False

    @property
    def relationship_badge(self) -> str:
        tags = []
        if self.is_direct_dependency:
            tags.append("Imported Dependency")
        if self.is_dependent:
            tags.append("Caller / Dependent")
        return ", ".join(tags) if tags else "Semantic Match"

    def format_snippet(self) -> str:
        """Compact markdown representation of this related function."""
        rel_tag = f" ({self.relationship_badge})" if (self.is_direct_dependency or self.is_dependent) else ""
        lines = [
            f"- **`{self.name}`** in `{self.file_path}`{rel_tag} — Relevance: {self.relevance_score:.0%}",
        ]
        if self.signature:
            lines.append(f"  - Signature: `{self.signature.strip()}`")
        if self.summary:
            lines.append(f"  - Summary: {self.summary.strip()}")
        elif self.docstring:
            first_doc = self.docstring.strip().split("\n")[0]
            lines.append(f"  - Doc: {first_doc}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "unit_type": self.unit_type,
            "signature": self.signature,
            "summary": self.summary,
            "docstring": self.docstring,
            "relevance_score": round(self.relevance_score, 4),
            "distance": round(self.distance, 4),
            "is_direct_dependency": self.is_direct_dependency,
            "is_dependent": self.is_dependent,
        }


@dataclass
class FunctionDependencyInfo:
    """Dependency graph connections for a file containing changed code."""
    file_path: str
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    external_imports: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "imports": self.imports,
            "imported_by": self.imported_by,
            "external_imports": self.external_imports,
        }


@dataclass
class FunctionContext:
    """Retrieved context package for a single changed function."""
    changed_function: str
    file_path: str
    query_used: str = ""
    diff_summary: str = ""
    related_functions: List[RelatedFunction] = field(default_factory=list)
    dependency_info: Optional[FunctionDependencyInfo] = None

    def format_for_prompt(self) -> str:
        """Format single function context into clean markdown for LLM injection."""
        lines = [
            f"#### Context for Changed Function: `{self.changed_function}` (`{self.file_path}`)",
        ]
        if self.diff_summary:
            lines.append(f"*Change*: {self.diff_summary}")

        # Dependency links
        if self.dependency_info:
            dep_parts = []
            if self.dependency_info.imports:
                dep_parts.append(f"Imports: {', '.join(f'`{m}`' for m in self.dependency_info.imports[:4])}")
            if self.dependency_info.imported_by:
                dep_parts.append(f"Imported by: {', '.join(f'`{m}`' for m in self.dependency_info.imported_by[:4])}")
            if dep_parts:
                lines.append(f"*Dependencies*: {' | '.join(dep_parts)}")

        # Related functions
        if self.related_functions:
            lines.append("*Related Codebase Units*:")
            for rf in self.related_functions:
                lines.append(rf.format_snippet())
        else:
            lines.append("*Related Codebase Units*: None retrieved above relevance threshold.")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_function": self.changed_function,
            "file_path": self.file_path,
            "query_used": self.query_used,
            "diff_summary": self.diff_summary,
            "related_functions": [rf.to_dict() for rf in self.related_functions],
            "dependency_info": self.dependency_info.to_dict() if self.dependency_info else None,
        }


@dataclass
class RetrievedContextPackage:
    """Consolidated context package covering all changed functions in a diff."""
    contexts: List[FunctionContext] = field(default_factory=list)
    total_related_functions: int = 0
    repository_path: Optional[str] = None
    estimated_tokens: int = 0

    def to_llm_prompt_snippet(self, max_tokens: int = 2500) -> str:
        """
        Produce a compact, token-bounded markdown section ready for LLM agent prompts.
        Guarantees that token limits are respected.
        """
        if not self.contexts:
            return ""

        parts = [
            "### Relevant Existing Codebase Context (Retrieved via Shared Index)\n",
            "The following functions and dependencies were retrieved as semantically relevant "
            "to the changed code. Use this architectural context when evaluating logic and security:\n\n",
        ]

        total_tokens = 0
        for ctx in self.contexts:
            snippet = ctx.format_for_prompt() + "\n\n"
            tokens = max(1, len(snippet) // 4)
            if total_tokens + tokens > max_tokens:
                parts.append("> [!NOTE]\n> Surrounding context truncated to respect prompt token budget.\n")
                break
            parts.append(snippet)
            total_tokens += tokens

        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "total_related_functions": self.total_related_functions,
            "estimated_tokens": self.estimated_tokens,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Query Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_retrieval_query(
    function_name: str,
    file_path: str,
    diff_content: str = "",
    added_lines: Optional[List[str]] = None,
    summary: str = "",
) -> str:
    """
    Construct a dense semantic query for ChromaDB retrieval from function metadata.
    """
    terms = [function_name]

    # Add meaningful terms from file path
    path_stem = Path(file_path).stem
    if path_stem not in ("__init__", "utils", "main", "service", "model"):
        terms.append(path_stem)

    # Add added code context
    if added_lines:
        clean_lines = [l.strip() for l in added_lines if l.strip() and not l.strip().startswith(("#", '"""'))]
        if clean_lines:
            terms.append(" ".join(clean_lines[:3]))
    elif diff_content:
        # Extract added lines from diff
        plus_lines = [
            l[1:].strip()
            for l in diff_content.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        ]
        if plus_lines:
            terms.append(" ".join(plus_lines[:3]))

    if summary:
        terms.append(summary)

    query = " ".join(terms).strip()
    return query[:400]  # Bound query length


# ─────────────────────────────────────────────────────────────────────────────
# Context Retriever Engine
# ─────────────────────────────────────────────────────────────────────────────

class ContextRetriever:
    """
    Agent that queries the shared ChromaDB index to enrich diff reviews
    with semantically related codebase functions and dependency information.
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        dependency_graph: Optional[DependencyGraph] = None,
        repo_path: Optional[str | Path] = None,
        persist_dir: Optional[Path] = None,
        max_related_per_function: int = 3,
        max_total_related: int = 8,
        min_relevance_score: float = 0.35,
    ) -> None:
        self.vector_store = vector_store or _get_codebase_store(persist_dir)
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.dependency_graph = dependency_graph
        self.max_related_per_function = max_related_per_function
        self.max_total_related = max_total_related
        self.min_relevance_score = min_relevance_score

        # Lazy initialize dependency graph if repo_path provided but graph is None
        if self.dependency_graph is None and self.repo_path and self.repo_path.is_dir():
            try:
                self.dependency_graph = build_dependency_graph(self.repo_path)
            except Exception as exc:
                logger.warning(f"Could not build dependency graph for {self.repo_path}: {exc}")

    def retrieve_for_function(
        self,
        function_name: str,
        file_path: str,
        diff_content: str = "",
        added_lines: Optional[List[str]] = None,
        summary: str = "",
        seen_function_keys: Optional[Set[str]] = None,
    ) -> FunctionContext:
        """
        Retrieve context for a single changed function.

        Excludes the function itself from results and filters by relevance threshold.
        """
        if seen_function_keys is None:
            seen_function_keys = set()

        query = _generate_retrieval_query(
            function_name=function_name,
            file_path=file_path,
            diff_content=diff_content,
            added_lines=added_lines,
            summary=summary,
        )

        # 1. Dependency information
        dep_info: Optional[FunctionDependencyInfo] = None
        direct_deps: Set[str] = set()
        dependents: Set[str] = set()

        if self.dependency_graph:
            rel_path = self.dependency_graph.to_rel(Path(file_path))
            imports = self.dependency_graph.direct_dependencies(rel_path)
            callers = self.dependency_graph.dependents(rel_path)
            ext_imports = self.dependency_graph.external_imports.get(rel_path, [])
            direct_deps = set(imports)
            dependents = set(callers)

            dep_info = FunctionDependencyInfo(
                file_path=file_path,
                imports=imports,
                imported_by=callers,
                external_imports=ext_imports,
            )

        # 2. Semantic search in ChromaDB
        related_functions: List[RelatedFunction] = []
        try:
            # Request top_k = max_related + 3 to allow for filtering out self & duplicates
            search_k = min(15, self.max_related_per_function + 4)
            retrieved_docs = self.vector_store.search(
                query=query,
                top_k=search_k,
                category="codebase",
            )
        except Exception as exc:
            logger.warning(f"ChromaDB search failed for query '{query}': {exc}")
            retrieved_docs = []

        # 3. Filter & map results
        normalized_target_path = Path(file_path).as_posix()

        for doc in retrieved_docs:
            doc_file = doc.metadata.get("file_path", "")
            doc_norm_path = Path(doc_file).as_posix() if doc_file else ""
            doc_name = doc.title or doc.metadata.get("title", "")

            # Exclude self (same function in the same file)
            if doc_name == function_name and (doc_norm_path == normalized_target_path or not doc_norm_path):
                continue

            # Exclude already included functions across changes (deduplication)
            dedup_key = f"{doc_norm_path}:{doc_name}"
            if dedup_key in seen_function_keys:
                continue

            # Check relevance score threshold
            if doc.relevance_score < self.min_relevance_score:
                continue

            is_direct_dep = doc_norm_path in direct_deps
            is_caller = doc_norm_path in dependents

            rf = RelatedFunction(
                name=doc_name,
                file_path=doc_file,
                unit_type=doc.metadata.get("unit_type", "function"),
                signature=doc.metadata.get("signature", ""),
                summary=doc.content if doc.metadata.get("unit_type") else "",
                docstring=doc.metadata.get("docstring", ""),
                relevance_score=doc.relevance_score,
                distance=doc.distance,
                is_direct_dependency=is_direct_dep,
                is_dependent=is_caller,
            )
            related_functions.append(rf)
            seen_function_keys.add(dedup_key)

            if len(related_functions) >= self.max_related_per_function:
                break

        return FunctionContext(
            changed_function=function_name,
            file_path=file_path,
            query_used=query,
            diff_summary=summary,
            related_functions=related_functions,
            dependency_info=dep_info,
        )

    def retrieve_for_changes(
        self,
        changes: Union[List[ReviewChange], DiffExtractionResult],
    ) -> RetrievedContextPackage:
        """
        Process a collection of diff review changes and assemble a consolidated
        context package respecting overall budget limits.
        """
        raw_changes: List[ReviewChange] = (
            changes.changes
            if isinstance(changes, DiffExtractionResult)
            else list(changes)
        )

        # Target changes with identifiable functions across any language
        target_changes = [c for c in raw_changes if c.function_name]
        if not target_changes:
            target_changes = raw_changes

        seen_functions: Set[str] = set()
        seen_keys: Set[str] = set()
        contexts: List[FunctionContext] = []
        total_related = 0

        for change in target_changes:
            if isinstance(change, str):
                fn_name = change
                file_path = ""
                diff_content = ""
                added_lines = None
                summary = ""
            else:
                fn_name = getattr(change, "function_name", None) or Path(getattr(change, "file_path", "unknown")).stem
                file_path = getattr(change, "file_path", "")
                diff_content = getattr(change, "diff_content", "")
                added_lines = getattr(change, "added_lines", None)
                summary = getattr(change, "summary", "")

            # Deduplicate by (file_path, fn_name) so multiple hunks in one function don't duplicate queries
            change_key = f"{file_path}:{fn_name}"
            if change_key in seen_functions:
                continue
            seen_functions.add(change_key)

            if total_related >= self.max_total_related:
                # Add context without additional vector queries if budget exhausted
                contexts.append(
                    FunctionContext(
                        changed_function=fn_name,
                        file_path=file_path,
                        diff_summary=summary,
                    )
                )
                continue

            ctx = self.retrieve_for_function(
                function_name=fn_name,
                file_path=file_path,
                diff_content=diff_content,
                added_lines=added_lines,
                summary=summary,
                seen_function_keys=seen_keys,
            )
            contexts.append(ctx)
            total_related += len(ctx.related_functions)

        # Estimate tokens
        package_snippet = "".join(ctx.format_for_prompt() for ctx in contexts)
        estimated_tokens = max(1, len(package_snippet) // 4)

        return RetrievedContextPackage(
            contexts=contexts,
            total_related_functions=total_related,
            repository_path=str(self.repo_path) if self.repo_path else None,
            estimated_tokens=estimated_tokens,
        )

    # Alias for pipeline compatibility
    retrieve_context = retrieve_for_changes


# ─────────────────────────────────────────────────────────────────────────────
# Clean Functional Public API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context_for_changes(
    changes: Union[List[ReviewChange], DiffExtractionResult],
    repo_path: Optional[str | Path] = None,
    vector_store: Optional[Any] = None,
    dependency_graph: Optional[DependencyGraph] = None,
    persist_dir: Optional[Path] = None,
    max_related_per_function: int = 3,
    max_total_related: int = 8,
    min_relevance_score: float = 0.35,
) -> RetrievedContextPackage:
    """
    Primary functional entry-point to retrieve surrounding codebase context
    for a set of changed functions from a Git diff.
    """
    retriever = ContextRetriever(
        vector_store=vector_store,
        dependency_graph=dependency_graph,
        repo_path=repo_path,
        persist_dir=persist_dir,
        max_related_per_function=max_related_per_function,
        max_total_related=max_total_related,
        min_relevance_score=min_relevance_score,
    )
    return retriever.retrieve_for_changes(changes)


def retrieve_context_for_function(
    function_name: str,
    file_path: str,
    diff_content: str = "",
    added_lines: Optional[List[str]] = None,
    summary: str = "",
    repo_path: Optional[str | Path] = None,
    vector_store: Optional[Any] = None,
    dependency_graph: Optional[DependencyGraph] = None,
    persist_dir: Optional[Path] = None,
    max_related: int = 3,
    min_relevance_score: float = 0.35,
) -> FunctionContext:
    """
    Retrieve surrounding codebase context for a single function.
    """
    retriever = ContextRetriever(
        vector_store=vector_store,
        dependency_graph=dependency_graph,
        repo_path=repo_path,
        persist_dir=persist_dir,
        max_related_per_function=max_related,
        min_relevance_score=min_relevance_score,
    )
    return retriever.retrieve_for_function(
        function_name=function_name,
        file_path=file_path,
        diff_content=diff_content,
        added_lines=added_lines,
        summary=summary,
    )
