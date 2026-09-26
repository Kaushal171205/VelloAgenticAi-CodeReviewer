"""
indexing/dependency_graph_builder.py — Python Dependency Graph Builder.

Recursively scans a Python repository, parses each file with the AST module,
and resolves import statements to local module paths.  The resulting graph
records:

  • Which modules each file imports (edges: importer → imported)
  • Which files import a given module (reverse index)

The graph is stored as plain dicts so it can be serialised to JSON / stored
in ChromaDB metadata without extra dependencies.

Design rules
────────────
• No network calls.
• Syntax errors in individual files are silently skipped (logged at WARNING).
• Skips irrelevant directories: .git, venv, __pycache__, node_modules, etc.
• All paths are stored as POSIX strings relative to the repository root.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "site-packages",
    "dist",
    "build",
    "eggs",
    ".eggs",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

class DependencyGraph:
    """
    Bidirectional dependency graph for a codebase repository.

    Attributes:
        root:        Path or identifier of the repository root.
        edges:       Mapping ``{file_posix_path: [imported_module_posix_path, ...]}``.
                     Only *local* modules resolved to actual files are included;
                     third-party / external imports are stored in ``external_imports``.
        reverse:     Mapping ``{module_posix_path: [files_that_import_it, ...]}``.
        external_imports: Mapping ``{file_posix_path: [raw_import_name, ...]}``.
        errors:      Files that could not be parsed (syntax errors, encoding issues).
    """

    def __init__(self, root: Path | str) -> None:
        if isinstance(root, Path):
            self.root: Path | str = root.resolve()
        else:
            self.root = root
        self.edges: Dict[str, List[str]] = {}
        self.reverse: Dict[str, List[str]] = {}
        self.external_imports: Dict[str, List[str]] = {}
        self.errors: List[str] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_rel(self, path: Path | str) -> str:
        """Return a POSIX path relative to the repository root."""
        if isinstance(path, str):
            return path.replace("\\", "/")
        if isinstance(self.root, Path):
            try:
                return path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                return path.as_posix()
        return path.as_posix()

    def all_files(self) -> List[str]:
        """Return all files that appear as graph nodes (keys in ``edges``)."""
        return list(self.edges.keys())

    def direct_dependencies(self, file_rel: str) -> List[str]:
        """Local modules imported by *file_rel*."""
        return self.edges.get(file_rel, [])

    def dependents(self, file_rel: str) -> List[str]:
        """Files that import *file_rel*."""
        return self.reverse.get(file_rel, [])

    def stats(self) -> Dict[str, int]:
        return {
            "total_files": len(self.edges),
            "total_edges": sum(len(v) for v in self.edges.values()),
            "files_with_errors": len(self.errors),
        }

    def to_mermaid(self, max_edges: int = 50) -> str:
        """Render the dependency graph as a clean Mermaid diagram."""
        import posixpath
        import re

        if not self.edges:
            return "graph TD;\n    empty[No files found in repository]"

        lines = ["graph TD;"]
        edge_count = 0
        nodes_with_edges: Set[str] = set()

        def clean_id(p: str) -> str:
            return "node_" + re.sub(r"[^a-zA-Z0-9_]", "_", p)

        # 1. Add direct dependency edges
        for src, deps in self.edges.items():
            src_id = clean_id(src)
            for dep in deps:
                if edge_count >= max_edges:
                    break
                dep_id = clean_id(dep)
                lines.append(f'    {src_id}["{src}"] --> {dep_id}["{dep}"]')
                nodes_with_edges.add(src)
                nodes_with_edges.add(dep)
                edge_count += 1

        # 2. Add unlinked / remaining files clustered by directory
        dirs: Dict[str, List[str]] = {}
        for f in list(self.edges.keys())[:35]:
            if f not in nodes_with_edges:
                d = posixpath.dirname(f) or "root"
                dirs.setdefault(d, []).append(f)

        subgraph_idx = 0
        for d, files in dirs.items():
            lines.append(f'    subgraph dir_{subgraph_idx}["📁 {d}"]')
            for f in files:
                f_id = clean_id(f)
                lines.append(f'        {f_id}["{posixpath.basename(f)}"]')
            lines.append("    end")
            subgraph_idx += 1

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

import re

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:(?:[\w*\s{},]+)\s+from\s+)?['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\)|export\s+(?:(?:[\w*\s{},]+)\s+from\s+)?['"]([^'"]+)['"])"""
)


def _parse_js_imports(source: str) -> List[str]:
    """Extract raw module import strings from JavaScript/TypeScript source."""
    imports = []
    for match in _JS_IMPORT_RE.finditer(source):
        imp = match.group(1) or match.group(2) or match.group(3)
        if imp:
            imports.append(imp.strip())
    return imports


def _should_skip(path: Path) -> bool:
    """Return True if this directory/file should be excluded from scanning."""
    return any(part in SKIP_DIRS for part in path.parts)


def _collect_all_source_files(root: Path) -> List[Path]:
    """Recursively find all supported source files under *root*, skipping irrelevant dirs."""
    from tools.language_utils import _EXT_MAP
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _EXT_MAP:
            if not _should_skip(path):
                files.append(path)
    return sorted(files)


def _parse_imports(source: str) -> tuple[List[str], List[tuple[Optional[str], str]]]:
    """
    Parse a Python source string and extract import names.

    Returns:
        plain_imports:    ``["os", "sys", "json"]``        — from ``import X``
        from_imports:     ``[("module", "name"), ...]``    — from ``from X import Y``
                          module may be ``None`` for relative imports
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise

    plain_imports: List[str] = []
    from_imports: List[tuple[Optional[str], str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                plain_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module  # may be None for ``from . import X``
            for alias in node.names:
                from_imports.append((module, alias.name))

    return plain_imports, from_imports


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_dependency_graph_from_sources(
    sources: Dict[str, str],
    root: Path | str = "repo",
) -> DependencyGraph:
    """
    Build dependency graph from an in-memory dictionary of {posix_rel_path: source_code}.
    Supports Python, JavaScript, TypeScript, and multi-language files.
    """
    import posixpath
    from tools.language_utils import detect_language

    graph = DependencyGraph(root)

    for rel_path, source in sources.items():
        _, lang_id = detect_language(rel_path)
        local_deps: List[str] = []
        external_deps: List[str] = []

        if lang_id == "python":
            try:
                plain_imports, from_imports = _parse_imports(source)
                all_module_names = plain_imports + [
                    m for m, _ in from_imports if m is not None
                ]
                for module_name in all_module_names:
                    mod_path = "/".join(module_name.split("."))
                    candidates = [
                        f"{mod_path}.py",
                        f"{mod_path}/__init__.py",
                    ]
                    importer_dir = posixpath.dirname(rel_path)
                    if importer_dir:
                        candidates.extend([
                            f"{importer_dir}/{mod_path}.py",
                            f"{importer_dir}/{mod_path}/__init__.py",
                        ])
                    resolved = None
                    for c in candidates:
                        norm = posixpath.normpath(c)
                        if norm in sources:
                            resolved = norm
                            break
                    if resolved:
                        if resolved not in local_deps:
                            local_deps.append(resolved)
                        graph.reverse.setdefault(resolved, [])
                        if rel_path not in graph.reverse[resolved]:
                            graph.reverse[resolved].append(rel_path)
                    else:
                        if module_name not in external_deps:
                            external_deps.append(module_name)
            except Exception as exc:
                graph.errors.append(f"{rel_path}: {exc}")

        elif lang_id in ("javascript", "typescript"):
            try:
                raw_imports = _parse_js_imports(source)
                for raw_imp in raw_imports:
                    if raw_imp.startswith("."):
                        importer_dir = posixpath.dirname(rel_path)
                        target = posixpath.normpath(posixpath.join(importer_dir, raw_imp))
                        resolved = None
                        if target in sources:
                            resolved = target
                        else:
                            for ext in [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".json"]:
                                if target + ext in sources:
                                    resolved = target + ext
                                    break
                                if posixpath.join(target, "index" + ext) in sources:
                                    resolved = posixpath.join(target, "index" + ext)
                                    break
                        if resolved:
                            if resolved not in local_deps:
                                local_deps.append(resolved)
                            graph.reverse.setdefault(resolved, [])
                            if rel_path not in graph.reverse[resolved]:
                                graph.reverse[resolved].append(rel_path)
                    else:
                        if raw_imp not in external_deps:
                            external_deps.append(raw_imp)
            except Exception as exc:
                graph.errors.append(f"{rel_path}: {exc}")

        graph.edges[rel_path] = local_deps
        graph.external_imports[rel_path] = external_deps

    logger.info(
        f"Dependency graph built: {len(graph.edges)} nodes, "
        f"{sum(len(v) for v in graph.edges.values())} local edges, "
        f"{len(graph.errors)} error(s)."
    )
    return graph


def build_dependency_graph(root: Path | str) -> DependencyGraph:
    """
    Build a dependency graph for a local directory or remote GitHub repository URL.

    Args:
        root: Local directory Path or GitHub repository URL string.

    Returns:
        A populated :class:`DependencyGraph`.
    """
    from tools.github_client import GitHubClient, is_github_url, parse_github_url
    from tools.language_utils import _EXT_MAP

    if isinstance(root, str) and is_github_url(root):
        parsed = parse_github_url(root)
        if not parsed:
            raise ValueError(f"Invalid GitHub URL: {root}")
        owner, repo_name = parsed
        client = GitHubClient()
        repo_info = client.get_repo_info(owner, repo_name)
        default_branch = repo_info.get("default_branch", "main")
        tree_files = client.get_file_tree(owner, repo_name, branch=default_branch)

        source_files = [
            f for f in tree_files
            if Path(f).suffix.lower() in _EXT_MAP
            and not any(part in SKIP_DIRS for part in Path(f).parts)
        ]

        sources: Dict[str, str] = {}
        for f in source_files:
            content = client.get_file_content(owner, repo_name, f, ref=default_branch)
            if content is not None:
                sources[f] = content

        return build_dependency_graph_from_sources(sources, root=f"{owner}/{repo_name}")

    # Local Directory
    local_path = Path(root).resolve()
    if not local_path.is_dir():
        raise ValueError(f"'{local_path}' is not a directory.")

    files = _collect_all_source_files(local_path)
    sources_dict: Dict[str, str] = {}
    for f in files:
        try:
            rel = f.relative_to(local_path).as_posix()
            sources_dict[rel] = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

    return build_dependency_graph_from_sources(sources_dict, root=local_path)
