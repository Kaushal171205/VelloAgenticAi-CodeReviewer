"""
ui/codebase_indexer_page.py — Streamlit UI for the Shared Codebase Indexer.

Allows a user to:
  1. Enter (or browse to) a local repository path.
  2. Trigger indexing (no LLM summaries by default for speed).
  3. View indexing statistics.
  4. Run a semantic search over the indexed codebase.
"""

from __future__ import annotations

import logging

import streamlit as st

from tools.language_utils import detect_language

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Codebase Indexer page."""

    st.title("🗂️ Shared Codebase Indexer")
    st.markdown(
        """
        Index any local or remote repository **once** and let all review agents
        reuse the indexed knowledge base. Supports Python, JavaScript, TypeScript,
        Java, Go, Rust, C/C++, and [many more languages](https://github.com).
        The pipeline runs:

        ```
        Repository → Language Detection → Code Unit Extraction
                   → Summaries → Embeddings → ChromaDB
        ```

        > **Python files** get full AST extraction (function/class/method units).  
        > **Other languages** are indexed as 60-line semantic chunks.
        """
    )

    # ── Sidebar: configuration ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Indexer Settings")
        use_llm = st.toggle(
            "LLM-enhanced summaries",
            value=False,
            help="Use the configured LLM to write richer one-sentence summaries. "
                 "Slower but improves semantic search quality.",
        )
        batch_size = st.slider("Batch size (ChromaDB upsert)", 10, 200, 50, step=10)

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab_index, tab_search, tab_graph = st.tabs(
        ["📥 Index Repository", "🔍 Semantic Search", "🕸️ Dependency Graph"]
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Index Repository
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_index:
        st.subheader("Index a Repository")

        repo_path = st.text_input(
            "Repository path or GitHub URL",
            placeholder="/path/to/my-project or https://github.com/owner/repo",
            help="Local folder path or remote GitHub URL (e.g. https://github.com/owner/repo). Fetches directly via GitHub API without cloning.",
            key="indexer_repo_path",
        )

        col_btn, col_count = st.columns([1, 3])
        with col_btn:
            run_index = st.button("▶ Run Indexer", type="primary", use_container_width=True)
        with col_count:
            _show_current_count()

        if run_index:
            if not repo_path or not repo_path.strip():
                st.error("Please enter a repository path.")
            else:
                _run_indexing(repo_path.strip(), use_llm=use_llm, batch_size=batch_size)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Semantic Search
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_search:
        st.subheader("Semantic Code Search")
        st.caption(
            "Search the indexed codebase using natural language or code snippets."
        )

        query = st.text_area(
            "Search query",
            placeholder="e.g. 'authenticate a user with JWT token' or 'calculate order total'",
            height=100,
            key="codebase_search_query",
        )
        top_k = st.slider("Results to return", 1, 20, 5, key="codebase_top_k")

        if st.button("🔍 Search", type="primary", key="codebase_search_btn"):
            _run_search(query, top_k)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Dependency Graph
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_graph:
        st.subheader("Dependency & Architecture Graph")
        st.caption(
            "Build and inspect the module dependency & architecture graph of any repository "
            "(Python, JavaScript, TypeScript, etc., local directory or remote GitHub URL)."
        )

        default_graph_path = st.session_state.get("last_indexed_repo", "")
        graph_path = st.text_input(
            "Repository path or GitHub URL",
            value=default_graph_path,
            placeholder="/path/to/my-project or https://github.com/owner/repo",
            key="graph_repo_path",
        )

        if st.button("🕸️ Build Graph", type="primary", key="graph_build_btn"):
            _build_and_show_graph(graph_path.strip() if graph_path else "")
        elif "last_indexing_result" in st.session_state and getattr(st.session_state["last_indexing_result"], "mermaid_graph", None):
            res = st.session_state["last_indexing_result"]
            st.info(f"Showing graph from recent indexing of: `{res.repository_path}`")
            st.subheader("🕸️ Visual Graph Diagram")
            st.markdown(f"```mermaid\n{res.mermaid_graph}\n```")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_current_count() -> None:
    """Display how many units are currently indexed."""
    try:
        from indexing import get_indexed_count
        count = get_indexed_count()
        if count:
            st.info(f"📦 Currently indexed: **{count:,}** code units.")
        else:
            st.caption("No units indexed yet.")
    except Exception:
        st.caption("Unable to read current index count.")


def _run_indexing(repo_path: str, use_llm: bool, batch_size: int) -> None:
    """Execute `index_codebase` and display the results."""
    from pathlib import Path
    from tools.github_client import is_github_url

    is_remote = is_github_url(repo_path)
    if not is_remote:
        path = Path(repo_path)
        if not path.exists():
            st.error(f"Path does not exist: `{repo_path}`")
            return
        if not path.is_dir():
            st.error(f"Path is not a directory: `{repo_path}`")
            return
        target: str | Path = path
    else:
        target = repo_path
        st.info("🌐 Remote GitHub repository detected. Accessing files via GitHub API (zero-clone)...")

    llm_client = None
    if use_llm:
        try:
            from llm.llm_client import LLMClient
            llm_client = LLMClient()
            st.info("🤖 LLM-enhanced summaries enabled.")
        except Exception as e:
            st.warning(f"Could not initialise LLM client — falling back to fast summaries. ({e})")

    with st.spinner(f"Indexing `{repo_path}` …  This may take a moment."):
        try:
            from indexing import index_codebase
            result = index_codebase(
                target,
                llm_client=llm_client,
                use_llm_summaries=use_llm and llm_client is not None,
                batch_size=batch_size,
            )
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            logger.exception("index_codebase raised an exception")
            return

    st.session_state["last_indexing_result"] = result
    st.session_state["last_indexed_repo"] = repo_path.strip()

    st.success("✅ Indexing complete!")

    # Statistics cards
    cols = st.columns(4)
    metrics = [
        ("Files Scanned", result.total_files_scanned),
        ("Units Indexed", result.total_units_indexed),
        ("Functions", result.total_functions),
        ("Classes", result.total_classes),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{value:,}")

    # Visual Graph Diagram
    if getattr(result, "mermaid_graph", None):
        st.subheader("🕸️ Codebase Dependency & Architecture Graph")
        st.caption("Visual representation of modules, import relationships, and structure:")
        st.markdown(f"```mermaid\n{result.mermaid_graph}\n```")

    with st.expander("📊 Full indexing statistics"):
        st.code(result.summary(), language="text")

        if result.errors:
            st.subheader(f"⚠️ Errors ({len(result.errors)})")
            for err in result.errors[:20]:
                st.markdown(f"- `{err}`")
            if len(result.errors) > 20:
                st.caption(f"… and {len(result.errors) - 20} more.")

        if result.dependency_graph_stats:
            st.subheader("Dependency graph stats")
            st.json(result.dependency_graph_stats)


def _run_search(query: str, top_k: int) -> None:
    """Execute semantic search and render results."""
    if not query or not query.strip():
        st.warning("Please enter a search query.")
        return

    with st.spinner("Searching codebase index…"):
        try:
            from indexing import search_codebase
            results = search_codebase(query.strip(), top_k=top_k)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            logger.exception("search_codebase raised an exception")
            return

    if not results:
        st.info("No results found. Make sure you have indexed a repository first.")
        return

    st.success(f"Found **{len(results)}** result(s).")

    for i, doc in enumerate(results, start=1):
        relevance_pct = int(doc.relevance_score * 100)
        label = f"{i}. `{doc.title}` — {doc.metadata.get('unit_type', 'code unit').capitalize()} in `{doc.metadata.get('file_path', '?')}`"

        with st.expander(label, expanded=(i == 1)):
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"**File:** `{doc.metadata.get('file_path', '—')}`")
                st.markdown(
                    f"**Lines:** {doc.metadata.get('start_line', '?')} – {doc.metadata.get('end_line', '?')}"
                )
                if doc.metadata.get("signature"):
                    _, lang_id = detect_language(doc.metadata.get('file_path', 'snippet.txt'))
                    st.code(doc.metadata["signature"], language=lang_id)
                if doc.metadata.get("docstring"):
                    st.caption(f"📝 {doc.metadata['docstring']}")
                st.markdown("**Summary / content:**")
                st.markdown(f"> {doc.content[:400]}")
            with col_r:
                st.metric("Relevance", f"{relevance_pct}%")
                st.caption(f"Unit type: `{doc.metadata.get('unit_type', '?')}`")


def _build_and_show_graph(repo_path: str) -> None:
    """Build dependency graph and display stats + visual diagram + adjacency table."""
    if not repo_path:
        st.warning("Please enter a repository path or GitHub URL.")
        return

    from tools.github_client import is_github_url
    from pathlib import Path

    is_remote = is_github_url(repo_path)
    if not is_remote:
        path = Path(repo_path)
        if not path.exists() or not path.is_dir():
            st.error(f"Not a valid directory: `{repo_path}`")
            return
        target: str | Path = path
    else:
        target = repo_path
        st.info("🌐 Fetching repository structure from GitHub…")

    with st.spinner("Building dependency graph…"):
        try:
            from indexing import build_dependency_graph
            graph = build_dependency_graph(target)
        except Exception as exc:
            st.error(f"Failed to build graph: {exc}")
            return

    stats = graph.stats()
    cols = st.columns(3)
    cols[0].metric("Files Indexed", stats["total_files"])
    cols[1].metric("Local Import Edges", stats["total_edges"])
    cols[2].metric("Files with Errors", stats["files_with_errors"])

    # Visual Graph Diagram
    mermaid_code = graph.to_mermaid()
    if mermaid_code:
        st.subheader("🕸️ Visual Graph Diagram")
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

    if graph.errors:
        with st.expander(f"⚠️ Syntax errors ({len(graph.errors)})"):
            for err in graph.errors:
                st.markdown(f"- `{err}`")

    # Show top 30 edges
    with st.expander("📋 Local dependency edges (up to 30)", expanded=True):
        rows = []
        for src, deps in graph.edges.items():
            for dep in deps:
                rows.append({"Importer": src, "Imported module": dep})
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows[:30]), use_container_width=True)
        else:
            st.info("No local import edges found.")

    with st.expander("🔁 Most-imported modules (reverse index)"):
        reverse_counts = [
            (mod, len(importers))
            for mod, importers in graph.reverse.items()
        ]
        reverse_counts.sort(key=lambda x: x[1], reverse=True)
        if reverse_counts:
            import pandas as pd
            df = pd.DataFrame(reverse_counts[:20], columns=["Module", "Imported By (count)"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No shared modules found.")
