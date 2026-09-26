"""
tests/test_github_client.py — Unit and integration tests for GitHub zero-clone client.

Validates GitHub URL parsing, API calls, rate-limit error handling, and zero-clone
diff extraction / indexing integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from tools.github_client import (
    GitHubAPIError,
    GitHubClient,
    is_github_url,
    parse_github_url,
)
from diff_processing.git_diff_extractor import extract_review_changes
from indexing.incremental_updater import index_codebase


# ─────────────────────────────────────────────────────────────────────────────
# URL Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGitHubURLParsing:
    def test_is_github_url(self):
        assert is_github_url("https://github.com/Kaushal171205/globetrotter-travel-planner")
        assert is_github_url("http://github.com/Kaushal171205/globetrotter-travel-planner.git")
        assert is_github_url("github.com/Kaushal171205/globetrotter-travel-planner")
        assert is_github_url("git@github.com:Kaushal171205/globetrotter-travel-planner.git")

        # Negative checks
        assert not is_github_url("/Users/username/Desktop/my-project")
        assert not is_github_url("./relative/folder")
        assert not is_github_url("https://gitlab.com/owner/repo")

    def test_parse_github_url(self):
        parsed = parse_github_url("https://github.com/Kaushal171205/globetrotter-travel-planner")
        assert parsed == ("Kaushal171205", "globetrotter-travel-planner")

        parsed_git = parse_github_url("https://github.com/Kaushal171205/globetrotter-travel-planner.git")
        assert parsed_git == ("Kaushal171205", "globetrotter-travel-planner")

        parsed_ssh = parse_github_url("git@github.com:Kaushal171205/globetrotter-travel-planner.git")
        assert parsed_ssh == ("Kaushal171205", "globetrotter-travel-planner")

        assert parse_github_url("/not/a/url") is None


# ─────────────────────────────────────────────────────────────────────────────
# GitHubClient API Mock Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGitHubClientMocked:
    @patch("requests.Session.get")
    def test_get_repo_info(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "name": "globetrotter-travel-planner",
            "full_name": "Kaushal171205/globetrotter-travel-planner",
            "default_branch": "main",
            "language": "JavaScript",
        }
        mock_get.return_value = mock_resp

        client = GitHubClient()
        info = client.get_repo_info("Kaushal171205", "globetrotter-travel-planner")
        assert info["default_branch"] == "main"
        assert info["language"] == "JavaScript"

    @patch("requests.Session.get")
    def test_get_file_tree(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "tree": [
                {"path": "package.json", "type": "blob"},
                {"path": "src/App.jsx", "type": "blob"},
                {"path": "src", "type": "tree"},
            ]
        }
        mock_get.return_value = mock_resp

        client = GitHubClient()
        tree = client.get_file_tree("Kaushal171205", "globetrotter-travel-planner", branch="main")
        assert tree == ["package.json", "src/App.jsx"]

    @patch("requests.Session.get")
    def test_rate_limit_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.ok = False
        mock_resp.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"}
        mock_get.return_value = mock_resp

        client = GitHubClient(token=None)
        with pytest.raises(GitHubAPIError, match="rate limit exceeded"):
            client.get_repo_info("owner", "repo")


# ─────────────────────────────────────────────────────────────────────────────
# Integration with extract_review_changes & index_codebase
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroCloneIntegration:
    @patch("tools.github_client.GitHubClient.get_diff")
    @patch("tools.github_client.GitHubClient.get_file_content")
    def test_extract_review_changes_github_url(self, mock_content, mock_diff):
        sample_diff = (
            "diff --git a/service.py b/service.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/service.py\n"
            "+++ b/service.py\n"
            "@@ -1,4 +1,5 @@\n"
            " def process_payment(amount):\n"
            "+    if amount <= 0: raise ValueError()\n"
            "     return True\n"
        )
        mock_diff.return_value = sample_diff
        mock_content.return_value = (
            "def process_payment(amount):\n"
            "    if amount <= 0: raise ValueError()\n"
            "    return True\n"
        )

        result = extract_review_changes(
            repo_path="https://github.com/Kaushal171205/globetrotter-travel-planner",
            base_ref="main",
            head_ref="feature",
        )

        assert not result.errors
        assert result.total_files == 1
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.file_path == "service.py"
        assert change.function_name == "process_payment"
        assert "ValueError" in change.diff_content

    @patch("tools.github_client.GitHubClient.get_repo_info")
    @patch("tools.github_client.GitHubClient.get_file_tree")
    def test_index_codebase_github_url_non_python(self, mock_tree, mock_info):
        mock_info.return_value = {
            "name": "globetrotter-travel-planner",
            "default_branch": "main",
            "language": "JavaScript",
        }
        mock_tree.return_value = [
            "package.json",
            "src/App.jsx",
            "src/main.jsx",
            "index.html",
        ]

        result = index_codebase("https://github.com/Kaushal171205/globetrotter-travel-planner")
        assert result.total_files_scanned == 0
        assert len(result.errors) > 0
        assert "No Python files found" in result.errors[0]
        assert "JavaScript" in result.errors[0]
