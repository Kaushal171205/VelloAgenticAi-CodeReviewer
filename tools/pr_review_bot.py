"""
tools/pr_review_bot.py — Automated GitHub Pull Request AI Review Bot.

Allows running the AI code review pipeline automatically against a GitHub Pull Request:
1. Interactive / Manual CLI:
       python -m tools.pr_review_bot --pr-url https://github.com/owner/repo/pull/123 --post-comment

2. CI/CD / GitHub Actions:
   Can be triggered automatically on 'pull_request: [opened, synchronize]'.
   Automatically fetches the diff between the PR branch and the target branch,
   executes the multi-agent review pipeline, and optionally posts the report as a PR comment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from diff_processing import (
    chunk_review_changes,
    create_review_prompt_payload,
    extract_review_changes,
)
from graph.workflow import build_review_graph, run_review
from tools.github_client import (
    GitHubAPIError,
    GitHubClient,
    is_github_pr_url,
    parse_github_pr_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PRReviewBot")


def review_pull_request(
    pr_url: str,
    post_comment: bool = False,
    enable_critic: bool = True,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch PR diff from GitHub, run multi-agent code review, and optionally post report.

    Args:
        pr_url: Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/123)
        post_comment: Whether to post the final report as a PR comment.
        enable_critic: Whether to run the fix suggester and critic self-correction loop.
        token: Optional GitHub personal access token or actions token.

    Returns:
        The final LangGraph review state dictionary.
    """
    parsed = parse_github_pr_url(pr_url)
    if not parsed:
        raise ValueError(f"Invalid GitHub PR URL: '{pr_url}'. Expected format: https://github.com/owner/repo/pull/123")

    owner, repo, pr_num = parsed
    logger.info(f"Analyzing GitHub Pull Request #{pr_num} on '{owner}/{repo}'...")

    client = GitHubClient(token=token or os.environ.get("GITHUB_TOKEN"))
    try:
        pr_info = client.get_pull_request(owner, repo, pr_num)
        pr_title = pr_info.get("title", f"PR #{pr_num}")
        pr_author = pr_info.get("user", {}).get("login", "unknown")
        base_ref = pr_info.get("base", {}).get("ref", "main")
        head_ref = pr_info.get("head", {}).get("ref", "HEAD")
        logger.info(f"PR Title: '{pr_title}' by @{pr_author} ({base_ref} <--- {head_ref})")
    except Exception as exc:
        logger.warning(f"Could not fetch full PR metadata: {exc}")
        pr_title = f"PR #{pr_num}"
        base_ref, head_ref = "main", "HEAD"

    # Step 1: Extract review changes from GitHub PR diff
    logger.info("Fetching unified diff from GitHub API...")
    diff_result = extract_review_changes(pr_url)

    if diff_result.errors:
        for err in diff_result.errors:
            logger.error(f"Diff error: {err}")
        raise RuntimeError(f"Failed to extract diff: {'; '.join(diff_result.errors)}")

    if not diff_result.changes:
        logger.info("No code changes found in this Pull Request.")
        return {"total_findings": 0, "final_report": "No changes found."}

    logger.info(
        f"Extracted {len(diff_result.changes)} change hunk(s) across {diff_result.total_files} file(s) "
        f"(+{diff_result.total_additions} / -{diff_result.total_deletions} lines)."
    )

    # Step 2: Package bounded chunks for multi-agent LLM prompts
    chunks = chunk_review_changes(
        diff_result,
        max_chars_per_chunk=4000,
        python_only=False,  # Support all programming languages!
        merge_same_function=True,
    )

    if not chunks:
        logger.warning("No review chunks generated from changes.")
        return {"total_findings": 0, "final_report": "No reviewable code found."}

    logger.info(f"Packaged {len(chunks)} review chunk(s) for the AI pipeline.")
    payload = create_review_prompt_payload(chunks, max_tokens=4000)

    # Step 3: Run the LangGraph Multi-Agent Pipeline
    logger.info("Running LangGraph Multi-Agent Review Pipeline...")
    app = build_review_graph(enable_critic=enable_critic, enable_human_review=False)

    final_state = run_review(
        app=app,
        raw_code=payload,
        filename=f"PR #{pr_num}: {pr_title}",
        changed_functions=diff_result.changes,
        enable_critic=enable_critic,
    )

    findings_count = final_state.get("total_findings", 0)
    highest_sev = (final_state.get("highest_severity") or "none").upper()
    logger.info(f"Review completed! Findings: {findings_count} (Highest severity: {highest_sev})")

    report = final_state.get("final_report", "")

    # Step 4: Optionally post comment back to GitHub Pull Request
    if post_comment and report:
        comment_header = (
            f"### 🤖 AI Code Review Summary — PR #{pr_num}\n\n"
            f"> **Status:** {findings_count} finding(s) detected (Highest severity: **{highest_sev}**)\n\n"
            "---\n\n"
        )
        full_comment = comment_header + report
        try:
            logger.info(f"Posting AI Review comment to PR #{pr_num}...")
            client.post_pull_request_comment(owner, repo, pr_num, full_comment)
            logger.info("Successfully posted review comment to GitHub PR!")
        except Exception as exc:
            logger.error(f"Failed to post PR comment: {exc}")

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated GitHub PR AI Code Reviewer")
    parser.add_argument("--pr-url", type=str, help="GitHub Pull Request URL (e.g. https://github.com/owner/repo/pull/123)")
    parser.add_argument("--post-comment", action="store_true", help="Post review report as a comment on the PR")
    parser.add_argument("--no-critic", action="store_true", help="Disable the fix suggester / critic loop")

    args = parser.parse_args()

    pr_url = args.pr_url
    if not pr_url:
        # Check GitHub Actions environment variables
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path, "r", encoding="utf-8") as f:
                    event_data = json.load(f)
                    pr_url = event_data.get("pull_request", {}).get("html_url")
            except Exception as e:
                logger.warning(f"Could not read GITHUB_EVENT_PATH: {e}")

    if not pr_url:
        print("Error: Please provide a PR URL via --pr-url <url> or run inside a GitHub Actions pull_request event.")
        sys.exit(1)

    try:
        state = review_pull_request(
            pr_url=pr_url,
            post_comment=args.post_comment,
            enable_critic=not args.no_critic,
        )
        print("\n" + "=" * 60)
        print("REVIEW REPORT:")
        print("=" * 60)
        print(state.get("final_report", "No report generated."))
    except Exception as exc:
        logger.exception(f"PR Review failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
