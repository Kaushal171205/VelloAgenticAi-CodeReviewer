"""
tests/test_diff_processing.py — Unit and integration tests for Git diff processing.

Validates git diff extraction, AST-based function identification, line tracking,
and token-bounded chunking using temporary Git repositories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest

from diff_processing import (
    DiffExtractionResult,
    ReviewChange,
    ReviewChunk,
    chunk_review_changes,
    create_review_prompt_payload,
    extract_review_changes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _run_git_cmd(repo_dir: Path, args: list[str]) -> str:
    """Helper to run git commands in a temporary repository."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """
    Creates a temporary git repository with a configured user,
    an initial 'main' branch, and baseline Python files.
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    _run_git_cmd(repo_dir, ["init", "-b", "main"])
    _run_git_cmd(repo_dir, ["config", "user.name", "Test Reviewer"])
    _run_git_cmd(repo_dir, ["config", "user.email", "reviewer@enterprise.ai"])

    # Initial math_utils.py on main
    math_code = (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    return a - b\n"
    )
    (repo_dir / "math_utils.py").write_text(math_code, encoding="utf-8")

    # Initial service.py on main
    service_code = (
        "class PaymentService:\n"
        "    def __init__(self, key: str):\n"
        "        self.key = key\n\n"
        "    def process_refund(self, tx_id: str, amount: float) -> bool:\n"
        "        if amount == 0:\n"
        "            return False\n"
        "        return True\n"
    )
    (repo_dir / "service.py").write_text(service_code, encoding="utf-8")

    # Initial README.md
    (repo_dir / "README.md").write_text("# Enterprise System\n", encoding="utf-8")

    _run_git_cmd(repo_dir, ["add", "."])
    _run_git_cmd(repo_dir, ["commit", "-m", "Initial commit on main"])

    return repo_dir


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffExtraction:
    def test_extract_review_changes_basic_function(self, temp_git_repo: Path):
        """Verify modifying a function correctly identifies its name and line numbers."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-math"])

        # Modify only the `add` function
        new_math = (
            "def add(a: int, b: int) -> int:\n"
            "    if not isinstance(a, int) or not isinstance(b, int):\n"
            "        raise TypeError('Expected ints')\n"
            "    return a + b\n\n\n"
            "def subtract(a: int, b: int) -> int:\n"
            "    return a - b\n"
        )
        (temp_git_repo / "math_utils.py").write_text(new_math, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Add input validation to add"])

        result = extract_review_changes(temp_git_repo, "main", "feature-math")

        assert not result.errors
        assert result.total_files == 1
        assert result.python_files == 1
        assert len(result.changes) >= 1

        change = result.changes[0]
        assert change.file_path == "math_utils.py"
        assert change.is_python is True
        assert change.function_name == "add"
        assert change.class_name is None
        assert change.display_scope == "add"
        assert change.old_start == 1
        assert change.new_start == 1
        assert "Expected ints" in change.diff_content
        assert any("Expected ints" in line for line in change.added_lines)

    def test_extract_review_changes_class_method(self, temp_git_repo: Path):
        """Verify modifying a method inside a class captures both class and method name."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-refund"])

        updated_service = (
            "class PaymentService:\n"
            "    def __init__(self, key: str):\n"
            "        self.key = key\n\n"
            "    def process_refund(self, tx_id: str, amount: float) -> bool:\n"
            "        if amount <= 0:\n"
            "            raise ValueError('Refund amount must be positive')\n"
            "        return True\n"
        )
        (temp_git_repo / "service.py").write_text(updated_service, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Fix non-positive refund flaw"])

        result = extract_review_changes(temp_git_repo, "main", "feature-refund")

        assert not result.errors
        assert result.total_files == 1
        assert len(result.changes) >= 1

        change = result.changes[0]
        assert change.file_path == "service.py"
        assert change.function_name == "process_refund"
        assert change.class_name == "PaymentService"
        assert change.display_scope == "PaymentService.process_refund"
        assert "Refund amount must be positive" in change.diff_content
        assert result.get_changed_functions() == ["PaymentService.process_refund"]

    def test_extract_review_changes_new_file(self, temp_git_repo: Path):
        """Verify adding a new Python file identifies functions in the new file."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-new-module"])

        new_file_content = (
            "def calculate_tax(subtotal: float, rate: float = 0.08) -> float:\n"
            "    return round(subtotal * rate, 2)\n"
        )
        (temp_git_repo / "tax.py").write_text(new_file_content, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["add", "tax.py"])
        _run_git_cmd(temp_git_repo, ["commit", "-m", "Add tax calculator"])

        result = extract_review_changes(temp_git_repo, "main", "feature-new-module")

        assert not result.errors
        assert result.total_files == 1
        change = result.changes[0]
        assert change.file_path == "tax.py"
        assert change.change_type == "added"
        assert change.function_name == "calculate_tax"
        assert "calculate_tax" in change.diff_content

    def test_extract_review_changes_deleted_file(self, temp_git_repo: Path):
        """Verify deleting a file identifies the deleted function from base_ref."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-delete-math"])

        _run_git_cmd(temp_git_repo, ["rm", "math_utils.py"])
        _run_git_cmd(temp_git_repo, ["commit", "-m", "Remove math_utils"])

        result = extract_review_changes(temp_git_repo, "main", "feature-delete-math")

        assert not result.errors
        assert result.total_files == 1
        change = result.changes[0]
        assert change.file_path == "math_utils.py"
        assert change.change_type == "deleted"
        # Should identify the function that was in main
        assert change.function_name in ("add", "subtract")

    def test_extract_review_changes_non_python_file(self, temp_git_repo: Path):
        """Verify non-Python files are tracked with is_python=False without errors."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-docs"])

        (temp_git_repo / "README.md").write_text("# Enterprise Code Review System\nUpdated docs.", encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Update README"])

        result = extract_review_changes(temp_git_repo, "main", "feature-docs")

        assert not result.errors
        assert result.total_files == 1
        assert result.python_files == 0
        change = result.changes[0]
        assert change.file_path == "README.md"
        assert change.is_python is False
        assert change.function_name is None
        assert "Updated docs." in change.diff_content

    def test_syntax_error_fallback(self, temp_git_repo: Path):
        """Verify files with syntax errors fall back to hunk headers gracefully."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-wip-syntax"])

        broken_code = (
            "def broken_function(x, y:\n"  # Missing closing parenthesis -> SyntaxError
            "    return x + y\n"
        )
        (temp_git_repo / "broken.py").write_text(broken_code, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["add", "broken.py"])
        _run_git_cmd(temp_git_repo, ["commit", "-m", "WIP broken code"])

        result = extract_review_changes(temp_git_repo, "main", "feature-wip-syntax")

        assert not result.errors
        assert result.total_files == 1
        change = result.changes[0]
        # Should gracefully extract function name from hunk header or line regex
        assert change.function_name == "broken_function"

    def test_error_handling_nonexistent_repo(self, tmp_path: Path):
        """Verify non-existent repository path produces clean error without raising."""
        fake_path = tmp_path / "does_not_exist"
        result = extract_review_changes(fake_path, "main", "HEAD")
        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0]

    def test_error_handling_invalid_refs(self, temp_git_repo: Path):
        """Verify comparing invalid branches produces clean error in result.errors."""
        result = extract_review_changes(temp_git_repo, "main", "branch-does-not-exist")
        assert len(result.errors) > 0
        assert "git diff failed" in result.errors[0]

    def test_result_iteration_and_methods(self, temp_git_repo: Path):
        """Verify DiffExtractionResult iterable protocol and convenience helpers."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-test-iter"])
        (temp_git_repo / "math_utils.py").write_text("def add(): pass\n", encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Change"])

        result = extract_review_changes(temp_git_repo, "main", "feature-test-iter")

        # __len__ and __iter__
        assert len(result) == len(result.changes)
        items = list(result)
        assert len(items) == len(result.changes)

        # to_dict
        d = result.to_dict()
        assert "repo_path" in d
        assert "changed_functions" in d
        assert d["total_files"] == 1

    def test_extract_review_changes_renamed_file(self, temp_git_repo: Path):
        """Verify renamed Python file preserves old_path and identifies functions."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-rename"])
        _run_git_cmd(temp_git_repo, ["mv", "math_utils.py", "calculator.py"])
        # Also edit calculator.py
        new_text = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b + 0\n\n\n"
            "def subtract(a: int, b: int) -> int:\n"
            "    return a - b\n"
        )
        (temp_git_repo / "calculator.py").write_text(new_text, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Rename and tweak"])

        result = extract_review_changes(temp_git_repo, "main", "feature-rename")
        assert not result.errors
        assert result.total_files == 1
        f = result.files[0]
        assert f.path == "calculator.py"
        assert f.old_path == "math_utils.py"
        assert f.change_type == "renamed"

    def test_extract_review_changes_multiple_functions_same_file(self, temp_git_repo: Path):
        """Verify editing both add and subtract in math_utils identifies both functions."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-multi-func"])
        new_math = (
            "def add(a: int, b: int) -> int:\n"
            "    return int(a + b)\n\n\n"
            "def subtract(a: int, b: int) -> int:\n"
            "    return int(a - b)\n"
        )
        (temp_git_repo / "math_utils.py").write_text(new_math, encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["commit", "-am", "Cast both to int"])

        result = extract_review_changes(temp_git_repo, "main", "feature-multi-func")
        assert not result.errors
        funcs = result.get_changed_functions()
        assert "add" in funcs
        assert "subtract" in funcs

    def test_triple_dot_merge_base_workflow(self, temp_git_repo: Path):
        """Verify git diff main...feature-branch compares against the common merge-base."""
        _run_git_cmd(temp_git_repo, ["checkout", "-b", "feature-diverged"])
        (temp_git_repo / "feature.py").write_text("def feat(): return 42\n", encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["add", "feature.py"])
        _run_git_cmd(temp_git_repo, ["commit", "-m", "Add feature"])

        # Switch back to main and commit something else
        _run_git_cmd(temp_git_repo, ["checkout", "main"])
        (temp_git_repo / "main_only.txt").write_text("main update\n", encoding="utf-8")
        _run_git_cmd(temp_git_repo, ["add", "main_only.txt"])
        _run_git_cmd(temp_git_repo, ["commit", "-m", "Commit on main"])

        # Triple dot diff main...feature-diverged should ONLY show feature.py, not main_only.txt!
        result = extract_review_changes(temp_git_repo, "main", "feature-diverged", triple_dot=True)
        assert not result.errors
        file_paths = [f.path for f in result.files]
        assert "feature.py" in file_paths
        assert "main_only.txt" not in file_paths


# ─────────────────────────────────────────────────────────────────────────────
# Test Chunker
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffChunker:
    def test_chunk_review_changes_basic(self):
        """Verify chunker creates ReviewChunk instances from changes."""
        change = ReviewChange(
            file_path="src/payments.py",
            change_type="modified",
            is_python=True,
            function_name="refund",
            class_name="PaymentManager",
            old_start=10,
            old_lines=4,
            new_start=10,
            new_lines=6,
            diff_content="@@ -10,4 +10,6 @@\n- return False\n+ if amount <= 0:\n+   return False\n+ return True\n",
            added_lines=["if amount <= 0:", "  return False", "return True"],
            removed_lines=["return False"],
        )

        chunks = chunk_review_changes([change])
        assert len(chunks) == 1

        chunk = chunks[0]
        assert chunk.file_path == "src/payments.py"
        assert chunk.display_scope == "PaymentManager.refund"
        assert chunk.estimated_tokens > 0

        # Verify formatted snippet does not send entire repo
        snippet = chunk.format_for_llm()
        assert "### File: `src/payments.py`" in snippet
        assert "`PaymentManager.refund`" in snippet
        assert "+ if amount <= 0:" in snippet

    def test_chunker_filters_non_python_when_requested(self):
        """Verify non-Python changes are filtered out when python_only=True."""
        py_change = ReviewChange(
            file_path="app.py",
            is_python=True,
            diff_content="@@ -1,1 +1,1 @@\n+print('hi')",
        )
        md_change = ReviewChange(
            file_path="README.md",
            is_python=False,
            diff_content="@@ -1,1 +1,1 @@\n+# Title",
        )

        chunks_py = chunk_review_changes([py_change, md_change], python_only=True)
        assert len(chunks_py) == 1
        assert chunks_py[0].file_path == "app.py"

        chunks_all = chunk_review_changes([py_change, md_change], python_only=False)
        assert len(chunks_all) == 2

    def test_create_review_prompt_payload_token_limit(self):
        """Verify create_review_prompt_payload adheres to max token constraints."""
        changes = [
            ReviewChange(
                file_path=f"file_{i}.py",
                function_name=f"func_{i}",
                diff_content=f"@@ -1,2 +1,3 @@\n+ # Line {i} change in function {i}\n" * 10,
                added_lines=[f"Line {i}"] * 10,
            )
            for i in range(10)
        ]
        chunks = chunk_review_changes(changes, merge_same_function=False)
        assert len(chunks) == 10

        # Request small token limit
        payload = create_review_prompt_payload(chunks, max_tokens=100)
        assert "Changed Code for Review" in payload
        assert "Truncated" in payload

    def test_chunk_merging_same_function(self):
        """Verify multiple adjacent hunks within the same function are merged into one chunk."""
        change1 = ReviewChange(
            file_path="src/service.py",
            function_name="process_order",
            class_name="OrderService",
            old_start=10,
            old_lines=3,
            new_start=10,
            new_lines=4,
            diff_content="@@ -10,3 +10,4 @@\n+ # Hunk 1\n",
        )
        change2 = ReviewChange(
            file_path="src/service.py",
            function_name="process_order",
            class_name="OrderService",
            old_start=25,
            old_lines=2,
            new_start=26,
            new_lines=3,
            diff_content="@@ -25,2 +26,3 @@\n+ # Hunk 2\n",
        )

        merged = chunk_review_changes([change1, change2], merge_same_function=True)
        assert len(merged) == 1
        assert "Hunk 1" in merged[0].diff_content
        assert "Hunk 2" in merged[0].diff_content
        assert merged[0].display_scope == "OrderService.process_order"

