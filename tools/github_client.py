"""
tools/github_client.py — Zero-Clone GitHub REST API Client.

Enables direct access to GitHub repositories (file trees, code contents, and git diffs)
via the GitHub REST API without requiring git clone or local repository storage.

Features:
- URL parsing for https://github.com/owner/repo formats.
- Recursive git tree traversal (files discovery).
- Direct in-memory file content fetching.
- Full unified diff fetching via the /compare endpoint.
- Support for unauthenticated access (60 req/hr) and token authentication (5,000 req/hr).
- Rate limit detection with informative user guidance.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Matches GitHub repo URLs:
# - https://github.com/owner/repo
# - http://github.com/owner/repo.git
# - github.com/owner/repo
# - git@github.com:owner/repo.git
_GITHUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com[:/]([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?(?:/.*)?$"
)
_GITHUB_SSH_RE = re.compile(
    r"^git@github\.com:([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?$"
)


# Matches GitHub PR URLs:
# - https://github.com/owner/repo/pull/123
_GITHUB_PR_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com[:/]([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)/pull/(\d+)(?:/.*)?$"
)


def is_github_url(url_or_path: str) -> bool:
    """Return True if *url_or_path* appears to be a remote GitHub repository URL."""
    s = str(url_or_path).strip()
    return bool(_GITHUB_URL_RE.match(s) or _GITHUB_SSH_RE.match(s))


def is_github_pr_url(url_or_path: str) -> bool:
    """Return True if *url_or_path* is a direct GitHub Pull Request URL."""
    s = str(url_or_path).strip()
    return bool(_GITHUB_PR_URL_RE.match(s))


def parse_github_url(url_or_path: str) -> Optional[Tuple[str, str]]:
    """
    Extract (owner, repo) from a GitHub URL or string.

    Examples:
        'https://github.com/Kaushal171205/globetrotter-travel-planner'
        -> ('Kaushal171205', 'globetrotter-travel-planner')
    """
    s = str(url_or_path).strip()
    m = _GITHUB_URL_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    m_ssh = _GITHUB_SSH_RE.match(s)
    if m_ssh:
        return m_ssh.group(1), m_ssh.group(2)
    return None


def parse_github_pr_url(url_or_path: str) -> Optional[Tuple[str, str, int]]:
    """
    Extract (owner, repo, pull_number) from a GitHub Pull Request URL.

    Example:
        'https://github.com/Kaushal171205/globetrotter-travel-planner/pull/1'
        -> ('Kaushal171205', 'globetrotter-travel-planner', 1)
    """
    s = str(url_or_path).strip()
    m = _GITHUB_PR_URL_RE.match(s)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


class GitHubAPIError(Exception):
    """Raised when a GitHub API request fails or is rate-limited."""
    pass


class GitHubClient:
    """Client for querying GitHub repositories without cloning them locally."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token
        if not self.token:
            try:
                from config import settings
                self.token = getattr(settings, "github_token", None)
            except Exception:
                self.token = None

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Enterprise-Code-Reviewer/1.0",
            "Accept": "application/vnd.github.v3+json",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _check_rate_limit(self, response: requests.Response) -> None:
        """Inspect response headers and raise GitHubAPIError if rate-limited."""
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_time = response.headers.get("X-RateLimit-Reset", "soon")
            auth_msg = (
                "You are currently unauthenticated (limit: 60 req/hr). "
                "Add a GITHUB_TOKEN in .env or Settings to increase your limit to 5,000 req/hr."
                if not self.token
                else "Your authenticated rate limit (5,000 req/hr) has been exceeded."
            )
            raise GitHubAPIError(f"GitHub API rate limit exceeded. {auth_msg}")

    def get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch repository metadata (default branch, description, etc.)."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        resp = self.session.get(url, timeout=15)
        self._check_rate_limit(resp)

        if resp.status_code == 404:
            raise GitHubAPIError(f"Repository '{owner}/{repo}' not found or is private.")
        if not resp.ok:
            raise GitHubAPIError(f"GitHub API error ({resp.status_code}): {resp.text[:200]}")

        return resp.json()

    def get_file_tree(self, owner: str, repo: str, branch: Optional[str] = None) -> List[str]:
        """
        Recursively fetch all file paths in the repository using the Git Trees API.
        Does not download file contents.
        """
        ref = branch
        if not ref:
            info = self.get_repo_info(owner, repo)
            ref = info.get("default_branch", "main")

        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
        resp = self.session.get(url, timeout=20)
        self._check_rate_limit(resp)

        if not resp.ok:
            raise GitHubAPIError(f"Failed to fetch file tree for '{owner}/{repo}@{ref}': {resp.text[:200]}")

        data = resp.json()
        tree_items = data.get("tree", [])
        return [item["path"] for item in tree_items if item.get("type") == "blob"]

    def get_file_content(
        self, owner: str, repo: str, file_path: str, ref: str = "HEAD"
    ) -> Optional[str]:
        """
        Fetch the text content of a single file in memory without saving to disk.
        """
        # Try raw.githubusercontent.com first for speed
        raw_headers = {}
        if self.token:
            raw_headers["Authorization"] = f"token {self.token}"

        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
        try:
            r = requests.get(raw_url, headers=raw_headers, timeout=15)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass

        # Fallback to contents API
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={ref}"
        resp = self.session.get(api_url, timeout=15)
        self._check_rate_limit(resp)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("encoding") == "base64" and data.get("content"):
                try:
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception:
                    return None
            return data.get("content")

        return None

    def get_diff(self, owner: str, repo: str, base_ref: str, head_ref: str) -> str:
        """
        Fetch the unified git diff between two branches or commits via GitHub Compare API.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_ref}...{head_ref}"
        diff_headers = dict(self.session.headers)
        diff_headers["Accept"] = "application/vnd.github.v3.diff"

        resp = self.session.get(url, headers=diff_headers, timeout=25)
        self._check_rate_limit(resp)

        if resp.status_code == 404:
            raise GitHubAPIError(f"Comparison '{base_ref}...{head_ref}' not found in '{owner}/{repo}'.")
        if not resp.ok:
            raise GitHubAPIError(f"GitHub compare failed ({resp.status_code}): {resp.text[:200]}")

        return resp.text

    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> Dict[str, Any]:
        """Fetch Pull Request details (title, branches, author, state, etc.)."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
        resp = self.session.get(url, timeout=15)
        self._check_rate_limit(resp)

        if resp.status_code == 404:
            raise GitHubAPIError(f"Pull Request #{pull_number} not found in '{owner}/{repo}'.")
        if not resp.ok:
            raise GitHubAPIError(f"GitHub API error ({resp.status_code}): {resp.text[:200]}")

        return resp.json()

    def get_pull_request_diff(self, owner: str, repo: str, pull_number: int) -> str:
        """Fetch the unified git diff for a Pull Request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
        diff_headers = dict(self.session.headers)
        diff_headers["Accept"] = "application/vnd.github.v3.diff"

        resp = self.session.get(url, headers=diff_headers, timeout=25)
        self._check_rate_limit(resp)

        if resp.status_code == 404:
            raise GitHubAPIError(f"Pull Request #{pull_number} not found in '{owner}/{repo}'.")
        if not resp.ok:
            raise GitHubAPIError(f"Failed to fetch PR diff ({resp.status_code}): {resp.text[:200]}")

        return resp.text

    def post_pull_request_comment(self, owner: str, repo: str, pull_number: int, body: str) -> Dict[str, Any]:
        """Post a comment to a Pull Request or issue."""
        if not self.token:
            raise GitHubAPIError("A GITHUB_TOKEN is required to post comments to GitHub Pull Requests.")

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
        resp = self.session.post(url, json={"body": body}, timeout=20)
        self._check_rate_limit(resp)

        if not resp.ok:
            raise GitHubAPIError(f"Failed to post comment ({resp.status_code}): {resp.text[:200]}")

        return resp.json()

