"""
ui/diff_review_page.py — Streamlit UI for Git Diff Processing & Chunking.

Allows users to:
1. Specify a repository path, base ref, and head ref (e.g. `main...feature-branch`).
2. Extract git diffs and automatically identify modified Python functions via AST.
3. Inspect atomic ReviewChanges with line numbers and diff content.
4. Preview token-bounded ReviewChunks ready for LLM agent prompts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from diff_processing import (
    DiffExtractionResult,
    ReviewChange,
    ReviewChunk,
    chunk_review_changes,
    create_review_prompt_payload,
    extract_review_changes,
)
from graph.workflow import build_review_graph, run_review

logger = logging.getLogger(__name__)


def _execute_review_pipeline(
    result: DiffExtractionResult,
    max_chars_chunk: int = 4000,
    merge_hunks: bool = True,
    enable_human: bool = False,
    enable_critic: bool = True,
    display_ref: str = "diff",
) -> Optional[dict]:
    """Execute the LangGraph code review pipeline on extracted diff changes."""
    chunks = chunk_review_changes(
        result,
        max_chars_per_chunk=max_chars_chunk,
        python_only=False,  # Multi-language review
        merge_same_function=merge_hunks,
    )
    if not chunks:
        st.warning("No review chunks found to process.")
        return None

    payload = create_review_prompt_payload(chunks)
    app = build_review_graph(
        enable_critic=enable_critic,
        enable_human_review=enable_human,
    )

    with st.spinner("Pipeline running: Processing Diff Chunks through AI Review..."):
        try:
            import uuid
            thread_id = f"diff-review-{uuid.uuid4().hex[:8]}"
            final_state = run_review(
                app=app,
                raw_code=payload,
                filename=f"Diff: {display_ref}",
                changed_functions=result.changes,
                enable_critic=enable_critic,
                thread_id=thread_id if enable_human else None,
            )
            # Ensure final_report is populated
            if not final_state.get("final_report"):
                from graph.workflow import report_node
                try:
                    rep_res = report_node(final_state)
                    final_state["final_report"] = rep_res.get("final_report")
                except Exception as rep_err:
                    logger.warning("Could not generate fallback report: %s", rep_err)

            st.session_state["diff_review_final_state"] = final_state
            st.success("Review pipeline completed successfully!")
            return final_state
        except Exception as e:
            logger.exception("Error running diff review pipeline")
            st.error(f"Pipeline execution failed: {e}")
            return None


def render() -> None:
    """Render the Git Diff Processing page."""
    st.title("🌿 Git Diff Processor")
    st.markdown(
        """
        Compare two Git branches, commits, or tags (e.g., `git diff HEAD~1...HEAD` or `main...feature-branch`),
        isolate changed source files across any programming language, and automatically detect modified functions.
        
        **Rule:** Never send the complete repository to the LLM — only focused, token-bounded diff chunks.
        """
    )

    # ── Sidebar Settings ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Diff Extraction Settings")
        triple_dot = st.checkbox(
            "Use triple-dot diff (base...head)",
            value=True,
            help="Triple-dot compares HEAD against the common merge-base with base (standard GitHub PR behavior).",
        )
        context_lines = st.slider(
            "Context lines (-U)",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of unchanged context lines surrounding each diff hunk.",
        )
        max_chars_chunk = st.slider(
            "Max characters per chunk",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=500,
            help="Maximum size before splitting or grouping hunks into LLM review chunks.",
        )
        merge_hunks = st.checkbox(
            "Merge same-function hunks",
            value=True,
            help="Combine adjacent hunks in the same function into a single coherent review chunk.",
        )

    # ── Form Inputs ────────────────────────────────────────────────────────────
    st.subheader("Repository & Branch Selection")

    # Quick comparison presets
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("🔄 Latest Commit (HEAD~1...HEAD)", use_container_width=True):
        st.session_state["diff_base_ref"] = "HEAD~1"
        st.session_state["diff_head_ref"] = "HEAD"
        st.rerun()
    if col_p2.button("🌿 Feature vs Main (main...HEAD)", use_container_width=True):
        st.session_state["diff_base_ref"] = "main"
        st.session_state["diff_head_ref"] = "HEAD"
        st.rerun()
    if col_p3.button("⏪ Last 3 Commits (HEAD~3...HEAD)", use_container_width=True):
        st.session_state["diff_base_ref"] = "HEAD~3"
        st.session_state["diff_head_ref"] = "HEAD"
        st.rerun()

    current_base = st.session_state.get("diff_base_ref", "HEAD~1")
    current_head = st.session_state.get("diff_head_ref", "HEAD")

    col_repo, col_base, col_head = st.columns([2, 1, 1])

    from tools.github_client import is_github_pr_url, is_github_url, parse_github_pr_url

    with col_repo:
        repo_path_input = st.text_input(
            "Repository Path, GitHub URL, or Pull Request (PR) URL",
            value=os.getcwd(),
            placeholder="/path/to/repo, https://github.com/owner/repo, or https://github.com/owner/repo/pull/1",
            help="Local directory path, remote GitHub repository URL, or direct GitHub PR URL (e.g. https://github.com/owner/repo/pull/1).",
        )

    is_pr = is_github_pr_url(repo_path_input)
    is_remote_repo = is_pr or is_github_url(repo_path_input)

    if is_pr:
        parsed_pr = parse_github_pr_url(repo_path_input)
        if parsed_pr:
            owner, repo_name, pr_num = parsed_pr
            st.info(
                f"🔀 **GitHub Pull Request #{pr_num} detected** on `{owner}/{repo_name}`. "
                "The exact diff between the PR branch and the target branch will be fetched automatically via GitHub API."
            )
    elif is_remote_repo:
        st.info("🌐 Remote GitHub repository detected — diffs and code will be fetched via GitHub REST API without cloning.")

    with col_base:
        base_ref = st.text_input(
            "Base Ref (Target)",
            value=current_base,
            placeholder="HEAD~1 / main / commit / tag",
            disabled=is_pr,
            help="Base commit or branch to compare from (auto-detected for PRs).",
        )

    with col_head:
        head_ref = st.text_input(
            "Head Ref (Source)",
            value=current_head,
            placeholder="HEAD / feature-branch / commit",
            disabled=is_pr,
            help="Head commit or branch containing new changes (auto-detected for PRs).",
        )

    if is_remote_repo and not is_pr and base_ref.strip() in ("main", "master") and head_ref.strip() in ("main", "master", "HEAD"):
        st.warning(
            "💡 **Tip**: On GitHub, `HEAD` points to `main`, so comparing `main...HEAD` compares the branch against itself (0 changes). "
            "To review the latest commit on main, set Base Ref to **HEAD~1** and Head Ref to **HEAD**.",
            icon="💡",
        )

    col_run1, col_run2 = st.columns([1, 1])
    with col_run1:
        btn_extract = st.button("🔍 Extract Changes Only", use_container_width=True)
    with col_run2:
        btn_label_full = "🚀 Review Pull Request Now" if is_pr else "🚀 Run Full AI Code Review"
        btn_run_full = st.button(btn_label_full, type="primary", use_container_width=True)

    if btn_extract or btn_run_full:
        if not repo_path_input.strip() or (not is_pr and (not base_ref.strip() or not head_ref.strip())):
            st.error("Please provide repository path, base ref, and head ref.")
            return

        # Reset previous review state when requesting fresh extraction
        st.session_state["diff_review_final_state"] = None

        with st.spinner(f"Extracting diff for `{base_ref}...{head_ref}`..."):
            result = extract_review_changes(
                repo_path=repo_path_input.strip(),
                base_ref=base_ref.strip(),
                head_ref=head_ref.strip(),
                triple_dot=triple_dot,
                context_lines=context_lines,
            )
            st.session_state["diff_result"] = result

        if btn_run_full and result and result.changes:
            display_ref = f"{result.base_ref}...{result.head_ref}" if result.base_ref else f"{base_ref}...{head_ref}"
            _execute_review_pipeline(
                result=result,
                max_chars_chunk=max_chars_chunk,
                merge_hunks=merge_hunks,
                enable_human=False,
                enable_critic=True,
                display_ref=display_ref,
            )

    # ── Render Results ─────────────────────────────────────────────────────────
    result: Optional[DiffExtractionResult] = st.session_state.get("diff_result")
    if not result:
        st.info("Enter repository details and click **Extract Changes** or **Run Full AI Code Review** to begin.")
        return

    if result.errors:
        for err in result.errors:
            st.error(f"❌ {err}")
        return

    # Metric summary row
    st.divider()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Files Changed", result.total_files)
    with m_col2:
        st.metric("Changes / Hunks", len(result.changes))
    with m_col3:
        st.metric("Lines Added", f"+{result.total_additions}")
    with m_col4:
        st.metric("Lines Deleted", f"-{result.total_deletions}")
    with m_col5:
        st.metric("Functions Identified", len(result.get_changed_functions()))

    if not result.changes:
        st.warning(
            f"No code changes detected between `{result.base_ref}` and `{result.head_ref}`. "
            "If reviewing recent commits on the main branch, try setting Base Ref to **HEAD~1** and Head Ref to **HEAD**."
        )
        return

    # ── Render Review Report (if generated) ──────────────────────────────────
    final_state = st.session_state.get("diff_review_final_state")
    if final_state:
        st.divider()
        st.subheader("📊 Review Summary")
        
        total = final_state.get("total_findings", 0)
        highest = (final_state.get("highest_severity") or "none").upper()
        sev_summary = final_state.get("severity_summary") or {}
        
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        col_m1.metric("Total Findings", total)
        col_m2.metric("Highest Severity", highest)
        col_m3.metric("🔴 Critical", sev_summary.get("critical", 0))
        col_m4.metric("🟠 High", sev_summary.get("high", 0))
        col_m5.metric("🟡 Medium", sev_summary.get("medium", 0))
        col_m6.metric("🔵 Low", sev_summary.get("low", 0))

        report = final_state.get("final_report")
        if not report:
            from graph.workflow import report_node
            try:
                rep_res = report_node(final_state)
                report = rep_res.get("final_report")
                final_state["final_report"] = report
                st.session_state["diff_review_final_state"] = final_state
            except Exception as e:
                logger.warning("Failed to render fallback report: %s", e)
                report = None

        if not report:
            report = "*No report content generated.*"

        if final_state.get("human_decision") == "none" or "__interrupt__" in final_state:
            st.info("⏸️ **Human Sign-off Active:** You can inspect the findings below and approve or reject suggested fixes.")
            col_app, col_rej = st.columns(2)
            if col_app.button("✅ Approve Fixes & Finalize", type="primary", use_container_width=True, key="diff_approve_btn"):
                if not final_state.get("final_report"):
                    from graph.workflow import report_node
                    try:
                        rep_res = report_node(final_state)
                        final_state["final_report"] = rep_res.get("final_report")
                    except Exception:
                        pass
                final_state["human_decision"] = "approve"
                st.session_state["diff_review_final_state"] = final_state
                st.rerun()
            if col_rej.button("❌ Reject Fixes", use_container_width=True, key="diff_reject_btn"):
                final_state["human_decision"] = "reject"
                st.session_state["diff_review_final_state"] = final_state
                st.rerun()

        st.subheader("📄 Final Review Report")
        with st.container(border=True):
            st.markdown(report)

        if is_pr and report and report != "*No report content generated.*":
            st.divider()
            st.subheader("💬 Post Review to GitHub Pull Request")
            parsed_pr = parse_github_pr_url(repo_path_input)
            if parsed_pr:
                p_owner, p_repo, p_num = parsed_pr
                if st.button(f"🚀 Post Review Comment to PR #{p_num}", type="secondary", key="post_pr_comment_btn"):
                    from tools.github_client import GitHubAPIError, GitHubClient
                    client = GitHubClient()
                    try:
                        comment_header = (
                            f"### 🤖 AI Code Review Summary — PR #{p_num}\n\n"
                            f"> **Status:** {total} finding(s) detected (Highest severity: **{highest}**)\n\n"
                            "---\n\n"
                        )
                        client.post_pull_request_comment(p_owner, p_repo, p_num, comment_header + report)
                        st.success(f"✅ Successfully posted review comment to Pull Request #{p_num} on GitHub!")
                    except GitHubAPIError as api_err:
                        st.error(f"GitHub API Error: {api_err}")
                    except Exception as post_err:
                        st.error(f"Failed to post comment: {post_err}")

    # Tab navigation for detailed hunk and chunk inspection
    st.divider()
    st.subheader("🔍 Detailed Diff & Context Inspection")
    st.caption("Inspect individual atomic hunks, LLM chunk groupings, and RAG semantic context.")
    tab_changes, tab_chunks, tab_context, tab_files = st.tabs(
        ["📋 Review Changes", "🧩 LLM Review Chunks", "🧠 Retrieved Context (RAG)", "📁 Changed Files Overview"]
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: Review Changes
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_changes:
        st.subheader("Atomic Review Changes")
        st.caption("Each hunk preserves file path, line numbers, enclosing function, and diff content.")

        # Filters
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            all_files = sorted({c.file_path for c in result.changes})
            sel_file = st.selectbox("Filter by file:", ["All files"] + all_files)
        with f_col2:
            all_scopes = sorted({c.display_scope for c in result.changes})
            sel_scope = st.selectbox("Filter by function / scope:", ["All scopes"] + all_scopes)

        filtered_changes = [
            c for c in result.changes
            if (sel_file == "All files" or c.file_path == sel_file)
            and (sel_scope == "All scopes" or c.display_scope == sel_scope)
        ]

        st.write(f"Showing **{len(filtered_changes)}** of **{len(result.changes)}** review change(s):")

        for idx, change in enumerate(filtered_changes):
            type_color = "#3fb950" if change.change_type == "added" else ("#f85149" if change.change_type == "deleted" else "#d29922")
            scope_badge = f"`{change.display_scope}`" if change.display_scope != "<module>" else "*<module-level>*"
            py_badge = "🐍 Python" if change.is_python else "📄 Non-Python"

            with st.expander(
                f"#{idx + 1} | {change.file_path} → {change.display_scope} (+{len(change.added_lines)} / -{len(change.removed_lines)})",
                expanded=(idx == 0),
            ):
                st.markdown(
                    f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:.5rem .75rem;margin-bottom:.5rem;">
                        <span style="font-weight:600;color:#e6edf3;">File:</span> <code>{change.file_path}</code> &nbsp;|&nbsp;
                        <span style="font-weight:600;color:#e6edf3;">Function:</span> {scope_badge} &nbsp;|&nbsp;
                        <span style="font-weight:600;color:#e6edf3;">Type:</span> <span style="color:{type_color};font-weight:600;">{change.change_type.upper()}</span> &nbsp;|&nbsp;
                        <span style="color:#8b949e;">Lines: -{change.old_start},{change.old_lines} +{change.new_start},{change.new_lines}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if change.diff_content:
                    st.code(change.diff_content, language="diff")
                else:
                    st.caption("No diff content for this hunk.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: LLM Review Chunks
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_chunks:
        st.subheader("LLM Prompt-Ready Chunks")
        st.caption(
            "Review chunks packaged specifically for LLM context windows. "
            "Unrelated files are filtered out, and changes are bounded."
        )

        chunks = chunk_review_changes(
            result,
            max_chars_per_chunk=max_chars_chunk,
            python_only=False,
            merge_same_function=merge_hunks,
        )

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.metric("Total Generated Chunks", len(chunks))
        with c_col2:
            est_total_tokens = sum(c.estimated_tokens for c in chunks)
            st.metric("Estimated Total Tokens", f"~{est_total_tokens:,}")

        for i, chunk in enumerate(chunks):
            with st.expander(
                f"Chunk [{chunk.chunk_id}] — {chunk.file_path} ({chunk.display_scope}) | ~{chunk.estimated_tokens} tokens",
                expanded=(i == 0),
            ):
                st.markdown(chunk.format_for_llm())

        st.divider()
        st.subheader("Consolidated Prompt Payload Preview")
        payload = create_review_prompt_payload(chunks, max_tokens=3000)
        st.text_area(
            "Formatted Payload (Ready for Agent Prompt)",
            value=payload,
            height=250,
            disabled=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: Retrieved Context (RAG)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_context:
        from context import retrieve_context_for_changes
        from indexing import get_indexed_count

        st.subheader("Shared Codebase Context Retrieval")
        st.caption(
            "Enriches diff review changes with semantically related functions and dependency "
            "links from the shared ChromaDB index."
        )

        indexed_count = get_indexed_count()
        if indexed_count == 0:
            st.warning(
                "⚠️ The shared codebase index is currently empty. "
                "Go to **🗂️ Codebase Indexer** to index this repository first so semantic search can find related code.",
                icon="⚠️",
            )
        else:
            st.success(f"Shared Codebase Index active: **{indexed_count}** code unit(s) available.", icon="✅")

        rc_col1, rc_col2 = st.columns([1, 1])
        with rc_col1:
            min_score = st.slider("Min relevance score", 0.1, 0.9, 0.35, step=0.05)
        with rc_col2:
            max_related = st.slider("Max related units per function", 1, 5, 3)

        btn_get_context = st.button("🧠 Retrieve Related Context", type="primary", use_container_width=True)

        if btn_get_context:
            with st.spinner("Searching shared index and building dependency graph..."):
                ctx_package = retrieve_context_for_changes(
                    changes=result,
                    repo_path=repo_path_input.strip(),
                    min_relevance_score=min_score,
                    max_related_per_function=max_related,
                )
                st.session_state["context_package"] = ctx_package

        ctx_pkg = st.session_state.get("context_package")
        if ctx_pkg:
            st.divider()
            stat_c1, stat_c2 = st.columns(2)
            with stat_c1:
                st.metric("Total Related Units Retrieved", ctx_pkg.total_related_functions)
            with stat_c2:
                st.metric("Estimated Context Tokens", f"~{ctx_pkg.estimated_tokens}")

            for ctx in ctx_pkg.contexts:
                with st.expander(
                    f"Context: `{ctx.changed_function}` ({ctx.file_path}) — {len(ctx.related_functions)} related unit(s)",
                    expanded=True,
                ):
                    if ctx.dependency_info:
                        d_parts = []
                        if ctx.dependency_info.imports:
                            d_parts.append(f"**Imports:** {', '.join(f'`{m}`' for m in ctx.dependency_info.imports[:4])}")
                        if ctx.dependency_info.imported_by:
                            d_parts.append(f"**Imported By:** {', '.join(f'`{m}`' for m in ctx.dependency_info.imported_by[:4])}")
                        if d_parts:
                            st.markdown(" • ".join(d_parts))

                    if ctx.related_functions:
                        st.markdown("**Related Functions / Classes:**")
                        for rf in ctx.related_functions:
                            badge_color = "#3b82f6" if rf.is_direct_dependency else ("#a371f7" if rf.is_dependent else "#8b949e")
                            st.markdown(
                                f"""
                                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:.5rem .75rem;margin-bottom:.5rem;">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <span style="font-weight:600;color:#e6edf3;">{rf.name}</span>
                                        <span style="background:{badge_color}22;border:1px solid {badge_color};color:{badge_color};border-radius:999px;padding:.1rem .5rem;font-size:.75rem;">
                                            {rf.relationship_badge} ({rf.relevance_score:.0%})
                                        </span>
                                    </div>
                                    <div style="font-size:.8rem;color:#8b949e;margin-top:.2rem;">File: <code>{rf.file_path}</code></div>
                                    <div style="font-size:.85rem;color:#c9d1d9;margin-top:.3rem;">{rf.summary or rf.docstring or 'No summary'}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No related functions found above the relevance threshold.")

            st.subheader("Context Prompt Section Preview")
            st.text_area(
                "Injected Markdown for LLM Agent",
                value=ctx_pkg.to_llm_prompt_snippet(),
                height=200,
                disabled=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: Changed Files Overview
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_files:
        st.subheader("Changed Files Summary")
        file_data = []
        for f in result.files:
            file_data.append({
                "File Path": f.path,
                "Change": f.change_type,
                "Python?": "✅" if f.is_python else "❌",
                "Hunks": len(f.hunks),
                "Additions": f"+{f.additions}",
                "Deletions": f"-{f.deletions}",
                "Identified Functions": ", ".join(f.changed_functions) or "None",
            })
        st.dataframe(file_data, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION: Re-run or Customize Review
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("⚙️ Run or Re-run AI Review with Custom Settings")
    st.caption("Adjust review flags (human sign-off, critic) and execute the review pipeline on the extracted changes.")

    col_btn, col_opt1, col_opt2 = st.columns([2, 1, 1])
    with col_opt1:
        enable_human = st.checkbox("Require Human Sign-off", value=False, key="diff_enable_human_cb")
    with col_opt2:
        enable_critic = st.checkbox("Enable Critic (Fixes)", value=True, key="diff_enable_critic_cb")

    with col_btn:
        run_review_btn = st.button("▶️ Run AI Review on Extracted Changes", type="primary", use_container_width=True, key="diff_run_review_btn")

    if run_review_btn:
        display_ref = f"{result.base_ref}...{result.head_ref}" if result.base_ref else f"{base_ref}...{head_ref}"
        _execute_review_pipeline(
            result=result,
            max_chars_chunk=max_chars_chunk,
            merge_hunks=merge_hunks,
            enable_human=enable_human,
            enable_critic=enable_critic,
            display_ref=display_ref,
        )
        st.rerun()

    st.divider()
    with st.expander("🤖 Automated CI/CD: Run AI Review Automatically on Every GitHub PR"):
        st.markdown(
            """
            ### How to run this pipeline automatically whenever a PR is raised:
            
            1. Copy the workflow file `.github/workflows/ai_code_review.yml` into your GitHub repository.
            2. Add your `GEMINI_API_KEY` to GitHub Repository Secrets (`Settings ➔ Secrets and variables ➔ Actions`).
            3. Whenever a developer opens or updates a Pull Request:
               - GitHub Actions triggers automatically.
               - Runs `python -m tools.pr_review_bot --pr-url <url> --post-comment`.
               - Automatically posts the full review report and suggested fixes directly as a comment on the PR!
            """
        )

