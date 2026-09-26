"""
diff_processing/git_diff_extractor.py — Git Diff Extraction & Function Identification.

Extracts unified diffs between Git branches or commits (e.g., `git diff main...feature-branch`),
parses file- and hunk-level changes, and uses Python AST analysis to map modified lines to their
enclosing functions and classes.

Design rules:
- Subprocess-based git execution (zero external git C-extension dependencies).
- Preserves file path, old/new line ranges, function name, class name, and diff content.
- Graceful degradation: syntax errors in WIP code or deleted files fall back to hunk headers and line regex.
- Never crashes on bad repositories or invalid refs; errors are captured in result objects.
- Does not expose secrets or send unnecessary repository code to callers.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Hunk header regex: @@ -old_start[,old_lines] +new_start[,new_lines] @@ [hint]
_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(?:\s*(.*))?$"
)

# Regex to detect function or class definition in code or hunk header (multi-language)
_FUNC_DEF_RE = re.compile(
    r"""(?:def|function|func|fn)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)|(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>|([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*(?:\{|throws)"""
)
_CLASS_DEF_RE = re.compile(r"""(?:class|struct|interface|type)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)""")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffHunk:
    """Represents a single unified diff hunk within a file."""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    diff_content: str
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    context_lines: List[str] = field(default_factory=list)
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    all_functions: List[str] = field(default_factory=list)

    @property
    def additions_count(self) -> int:
        return len(self.added_lines)

    @property
    def deletions_count(self) -> int:
        return len(self.removed_lines)


@dataclass
class ChangedFile:
    """Represents a single file modified in the diff."""
    path: str
    old_path: Optional[str] = None
    change_type: str = "modified"  # "modified" | "added" | "deleted" | "renamed"
    is_python: bool = False
    hunks: List[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0

    @property
    def changed_functions(self) -> List[str]:
        funcs = []
        for hunk in self.hunks:
            if hunk.function_name and hunk.function_name not in funcs:
                funcs.append(hunk.function_name)
        return funcs


@dataclass
class ReviewChange:
    """
    An atomic unit of code change for review.

    Preserves:
    - file path
    - old/new line information
    - function name (and class name if applicable)
    - diff content
    """
    file_path: str
    old_path: Optional[str] = None
    change_type: str = "modified"
    is_python: bool = True
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    all_functions: List[str] = field(default_factory=list)
    old_start: int = 0
    old_lines: int = 0
    new_start: int = 0
    new_lines: int = 0
    diff_content: str = ""
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    hunk_header: str = ""
    summary: str = ""

    @property
    def display_scope(self) -> str:
        """Return 'ClassName.function_name', 'function_name', or '<module>'."""
        if self.class_name and self.function_name:
            if self.function_name.startswith(self.class_name + "."):
                return self.function_name
            return f"{self.class_name}.{self.function_name}"
        if self.function_name:
            return self.function_name
        return "<module>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "old_path": self.old_path,
            "change_type": self.change_type,
            "is_python": self.is_python,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "all_functions": self.all_functions,
            "display_scope": self.display_scope,
            "old_start": self.old_start,
            "old_lines": self.old_lines,
            "new_start": self.new_start,
            "new_lines": self.new_lines,
            "diff_content": self.diff_content,
            "additions": len(self.added_lines),
            "deletions": len(self.removed_lines),
            "hunk_header": self.hunk_header,
            "summary": self.summary,
        }


@dataclass
class DiffExtractionResult:
    """Overall outcome of extracting review changes between two refs."""
    repo_path: str
    base_ref: str
    head_ref: str
    files: List[ChangedFile] = field(default_factory=list)
    changes: List[ReviewChange] = field(default_factory=list)
    total_files: int = 0
    python_files: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    errors: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[ReviewChange]:
        return iter(self.changes)

    def __len__(self) -> int:
        return len(self.changes)

    def get_python_changes(self) -> List[ReviewChange]:
        """Return only changes belonging to Python (.py) files."""
        return [c for c in self.changes if c.is_python]

    def get_changed_functions(self) -> List[str]:
        """Return unique function scopes identified across all changes."""
        seen = set()
        result = []
        for c in self.changes:
            candidates = c.all_functions if c.all_functions else ([c.display_scope] if c.display_scope != "<module>" else [])
            for fn in candidates:
                if fn not in seen and fn != "<module>":
                    seen.add(fn)
                    result.append(fn)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "total_files": self.total_files,
            "python_files": self.python_files,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "changed_functions": self.get_changed_functions(),
            "errors": self.errors,
            "changes_count": len(self.changes),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AST Scope Extractor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CodeScope:
    name: str
    qualname: str
    parent_class: Optional[str]
    start_line: int
    end_line: int
    is_class: bool


def _extract_ast_scopes(source_code: str) -> List[_CodeScope]:
    """
    Parse Python code and return all function, method, and class definitions
    with their 1-indexed line spans [start_line, end_line].
    """
    scopes: List[_CodeScope] = []
    if not source_code.strip():
        return scopes

    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError) as exc:
        logger.debug(f"AST parsing failed for code snippet: {exc}")
        return scopes

    def visit_node(node: ast.AST, parent_class: Optional[str] = None, prefix: str = ""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                c_name = child.name
                c_qualname = f"{prefix}{c_name}" if prefix else c_name
                end_lineno = getattr(child, "end_lineno", child.lineno)
                scopes.append(
                    _CodeScope(
                        name=c_name,
                        qualname=c_qualname,
                        parent_class=parent_class,
                        start_line=child.lineno,
                        end_line=end_lineno,
                        is_class=True,
                    )
                )
                visit_node(child, parent_class=c_name, prefix=f"{c_qualname}.")

            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_name = child.name
                f_qualname = f"{prefix}{f_name}" if prefix else f_name
                end_lineno = getattr(child, "end_lineno", child.lineno)
                scopes.append(
                    _CodeScope(
                        name=f_name,
                        qualname=f_qualname,
                        parent_class=parent_class,
                        start_line=child.lineno,
                        end_line=end_lineno,
                        is_class=False,
                    )
                )
                visit_node(child, parent_class=parent_class, prefix=f"{f_qualname}.")

    visit_node(tree)
    return scopes


def _find_enclosing_scope(
    scopes: List[_CodeScope],
    target_lines: List[int],
    fallback_start: int,
    fallback_end: int,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Given the exact lines that were added/modified, find the innermost enclosing
    function/method and its parent class.

    If target_lines is empty, uses [fallback_start, fallback_end].
    """
    if not scopes:
        return None, None, []

    if target_lines:
        # Score each function by how many target lines fall within its bounds
        func_scores: Dict[_CodeScope, int] = {}
        class_scores: Dict[_CodeScope, int] = {}

        for line in target_lines:
            for s in scopes:
                if s.start_line <= line <= s.end_line:
                    if s.is_class:
                        class_scores[s] = class_scores.get(s, 0) + 1
                    else:
                        func_scores[s] = func_scores.get(s, 0) + 1

        if func_scores:
            # Pick function with most hits; break ties by innermost (smallest span)
            best_func = max(
                func_scores.keys(),
                key=lambda s: (func_scores[s], -(s.end_line - s.start_line)),
            )
            all_names = [f.qualname for f in func_scores.keys()]
            return best_func.name, best_func.parent_class, all_names

        if class_scores:
            best_cls = max(
                class_scores.keys(),
                key=lambda s: (class_scores[s], -(s.end_line - s.start_line)),
            )
            return None, best_cls.name, [best_cls.qualname]

    # Fallback to hunk span overlap
    eff_start = max(1, fallback_start)
    eff_end = max(eff_start, fallback_end)
    matching_funcs: List[_CodeScope] = []
    matching_classes: List[_CodeScope] = []

    for scope in scopes:
        if not (scope.end_line < eff_start or scope.start_line > eff_end):
            if scope.is_class:
                matching_classes.append(scope)
            else:
                matching_funcs.append(scope)

    if matching_funcs:
        best_func = min(matching_funcs, key=lambda s: (s.end_line - s.start_line))
        return best_func.name, best_func.parent_class, [f.qualname for f in matching_funcs]

    if matching_classes:
        best_class = min(matching_classes, key=lambda s: (s.end_line - s.start_line))
        return None, best_class.name, [best_class.qualname]

    return None, None, []


# ─────────────────────────────────────────────────────────────────────────────
# Git Subprocess Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(repo_path: Path, args: List[str]) -> Tuple[int, str, str]:
    """Execute a git command within *repo_path* and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", "git executable not found in PATH"
    except Exception as exc:
        return 1, "", str(exc)


def _get_git_file_content(repo_path: Path, ref: str, file_path: str) -> Optional[str]:
    """Fetch content of a file at a specific git ref using `git show <ref>:<path>`."""
    rc, stdout, stderr = _run_git(repo_path, ["show", f"{ref}:{file_path}"])
    if rc == 0:
        return stdout

    try:
        local_path = repo_path / file_path
        if local_path.is_file():
            return local_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified Diff Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_diff_blocks(diff_text: str) -> List[str]:
    """Split full unified diff into per-file diff blocks."""
    blocks: List[str] = []
    current_block: List[str] = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_block:
                blocks.append("".join(current_block))
                current_block = []
        current_block.append(line)

    if current_block:
        blocks.append("".join(current_block))

    return blocks


def _parse_file_block(
    repo_path: Optional[Path],
    base_ref: str,
    head_ref: str,
    block_text: str,
    content_fetcher: Optional[Any] = None,
) -> Optional[ChangedFile]:
    """Parse one per-file diff block into a :class:`ChangedFile` with hunks and AST scopes."""
    lines = block_text.splitlines()
    if not lines:
        return None

    first_line = lines[0]
    match_diff = re.match(r"^diff --git a/(.*?) b/(.*?)$", first_line)
    if not match_diff:
        return None

    path_a = match_diff.group(1).strip()
    path_b = match_diff.group(2).strip()

    change_type = "modified"
    old_path = None
    target_path = path_b

    # Inspect file metadata headers
    for line in lines[1:]:
        if line.startswith("@@"):
            break
        if line.startswith("new file mode"):
            change_type = "added"
        elif line.startswith("deleted file mode"):
            change_type = "deleted"
            target_path = path_a
        elif line.startswith("rename from "):
            change_type = "renamed"
            old_path = line[len("rename from "):].strip()
        elif line.startswith("rename to "):
            change_type = "renamed"
            target_path = line[len("rename to "):].strip()

    is_python = target_path.endswith(".py")

    # If Python, attempt to fetch AST scopes
    ast_scopes: List[_CodeScope] = []
    if is_python:
        ref_for_ast = base_ref if change_type == "deleted" else head_ref
        if content_fetcher:
            py_content = content_fetcher(ref_for_ast, target_path)
        elif repo_path:
            py_content = _get_git_file_content(repo_path, ref_for_ast, target_path)
        else:
            py_content = None

        if py_content:
            ast_scopes = _extract_ast_scopes(py_content)

    # Parse hunks
    hunks: List[DiffHunk] = []
    total_adds = 0
    total_dels = 0

    current_hunk_lines: List[str] = []
    current_hunk_header = ""
    current_hunk_meta: Optional[Tuple[int, int, int, int, str]] = None

    def flush_hunk():
        nonlocal current_hunk_lines, current_hunk_header, current_hunk_meta, total_adds, total_dels
        if not current_hunk_meta or not current_hunk_lines:
            return

        old_start, old_lines, new_start, new_lines, hint = current_hunk_meta
        added: List[str] = []
        removed: List[str] = []
        context: List[str] = []
        modified_new_lines: List[int] = []
        modified_old_lines: List[int] = []

        cur_old = old_start
        cur_new = new_start

        for h_line in current_hunk_lines[1:]:
            if h_line.startswith("+"):
                added.append(h_line[1:])
                modified_new_lines.append(cur_new)
                cur_new += 1
            elif h_line.startswith("-"):
                removed.append(h_line[1:])
                modified_old_lines.append(cur_old)
                cur_old += 1
            elif h_line.startswith(" "):
                context.append(h_line[1:])
                cur_old += 1
                cur_new += 1

        total_adds += len(added)
        total_dels += len(removed)

        # Identify function / class
        fn_name: Optional[str] = None
        cls_name: Optional[str] = None
        all_funcs: List[str] = []

        if is_python and ast_scopes:
            target_lines = modified_old_lines if change_type == "deleted" else modified_new_lines
            fallback_start = old_start if change_type == "deleted" else new_start
            fallback_count = old_lines if change_type == "deleted" else new_lines
            fallback_end = fallback_start + max(1, fallback_count) - 1
            fn_name, cls_name, all_funcs = _find_enclosing_scope(
                ast_scopes, target_lines, fallback_start, fallback_end
            )

        # Fallback 1: Hunk header hint
        if not fn_name and hint:
            m_fn = _FUNC_DEF_RE.search(hint)
            if m_fn:
                fn_name = next((g for g in m_fn.groups() if g), None)
            m_cls = _CLASS_DEF_RE.search(hint)
            if m_cls:
                cls_name = m_cls.group(1)
                if not fn_name:
                    fn_name = cls_name

        # Fallback 2: Regex across added/removed/diff lines (handles syntax errors in WIP code)
        if not fn_name:
            for l_candidate in added + removed + current_hunk_lines:
                m_fn = _FUNC_DEF_RE.search(l_candidate)
                if m_fn:
                    fn_name = next((g for g in m_fn.groups() if g), None)
                    if fn_name:
                        break
            for l_candidate in added + removed + current_hunk_lines:
                m_cls = _CLASS_DEF_RE.search(l_candidate)
                if m_cls:
                    cls_name = m_cls.group(1)
                    break

        hunk_obj = DiffHunk(
            old_start=old_start,
            old_lines=old_lines,
            new_start=new_start,
            new_lines=new_lines,
            header=current_hunk_header,
            diff_content="\n".join(current_hunk_lines),
            added_lines=added,
            removed_lines=removed,
            context_lines=context,
            function_name=fn_name,
            class_name=cls_name,
            all_functions=all_funcs,
        )
        hunks.append(hunk_obj)
        current_hunk_lines = []
        current_hunk_meta = None

    for line in lines:
        if line.startswith("@@"):
            flush_hunk()
            m = _HUNK_HEADER_RE.match(line)
            if m:
                old_start = int(m.group(1))
                old_lines = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_lines = int(m.group(4)) if m.group(4) is not None else 1
                hint = (m.group(5) or "").strip()
                current_hunk_meta = (old_start, old_lines, new_start, new_lines, hint)
                current_hunk_header = line
                current_hunk_lines = [line]
        elif current_hunk_meta is not None:
            current_hunk_lines.append(line)

    flush_hunk()

    return ChangedFile(
        path=target_path,
        old_path=old_path,
        change_type=change_type,
        is_python=is_python,
        hunks=hunks,
        additions=total_adds,
        deletions=total_dels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Primary Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_review_changes(
    repo_path: str | Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    triple_dot: bool = True,
    context_lines: int = 3,
) -> DiffExtractionResult:
    """
    Extract changed files and functions between two Git branches or commits.

    Supports:
        `git diff base_ref...head_ref` (triple-dot merge-base comparison)
        `git diff base_ref..head_ref` (double-dot direct comparison)

    Preserves:
    - file path
    - old/new line numbers
    - function and class names (via AST parsing where possible)
    - diff content (unified diff hunks)

    Args:
        repo_path: Path to the root of the Git repository.
        base_ref: Base branch or commit (e.g. 'main', 'master', 'v1.0.0').
        head_ref: Head branch or commit (e.g. 'feature-branch', 'HEAD').
        triple_dot: If True, uses 'base_ref...head_ref' (standard PR diff).
                    If False, uses 'base_ref..head_ref'.
        context_lines: Number of unified diff context lines (default: 3).

    Returns:
        A :class:`DiffExtractionResult` object containing changed files,
        atomic review changes, statistics, and any errors encountered.
    """
    from tools.github_client import (
        GitHubAPIError,
        GitHubClient,
        is_github_pr_url,
        is_github_url,
        parse_github_pr_url,
        parse_github_url,
    )

    repo_str = str(repo_path).strip()
    is_pr = is_github_pr_url(repo_str)
    is_remote = is_pr or is_github_url(repo_str)

    result = DiffExtractionResult(
        repo_path=repo_str,
        base_ref=base_ref,
        head_ref=head_ref,
    )

    diff_blocks: List[str] = []
    content_fetcher: Optional[Any] = None
    repo_obj: Optional[Path] = None

    if is_pr:
        parsed_pr = parse_github_pr_url(repo_str)
        if not parsed_pr:
            result.errors.append(f"Invalid GitHub PR URL: '{repo_str}'")
            return result
        owner, repo_name, pr_num = parsed_pr
        client = GitHubClient()
        try:
            pr_info = client.get_pull_request(owner, repo_name, pr_num)
            base_ref = pr_info.get("base", {}).get("ref", base_ref)
            head_ref = pr_info.get("head", {}).get("ref", head_ref)
            result.base_ref = base_ref
            result.head_ref = head_ref
            diff_text = client.get_pull_request_diff(owner, repo_name, pr_num)
        except GitHubAPIError as exc:
            result.errors.append(str(exc))
            return result
        except Exception as exc:
            result.errors.append(f"Failed to fetch PR diff: {exc}")
            return result

        diff_blocks = _parse_diff_blocks(diff_text)
        content_fetcher = lambda ref, fpath: client.get_file_content(owner, repo_name, fpath, ref=ref)

    elif is_remote:
        parsed = parse_github_url(repo_str)
        if not parsed:
            result.errors.append(f"Invalid GitHub URL: '{repo_str}'")
            return result
        owner, repo_name = parsed
        client = GitHubClient()
        try:
            diff_text = client.get_diff(owner, repo_name, base_ref, head_ref)
        except GitHubAPIError as exc:
            result.errors.append(str(exc))
            return result
        except Exception as exc:
            result.errors.append(f"Failed to fetch diff from GitHub API: {exc}")
            return result

        diff_blocks = _parse_diff_blocks(diff_text)
        content_fetcher = lambda ref, fpath: client.get_file_content(owner, repo_name, fpath, ref=ref)
    else:
        repo = Path(repo_str).resolve()
        repo_obj = repo
        result.repo_path = str(repo)

        if not repo.exists():
            result.errors.append(f"Repository directory does not exist: {repo}")
            return result

        # Verify git repository
        rc, _, err = _run_git(repo, ["rev-parse", "--is-inside-work-tree"])
        if rc != 0:
            result.errors.append(f"'{repo}' is not a valid git repository: {err.strip()}")
            return result

        # Construct diff specifier
        separator = "..." if triple_dot else ".."
        diff_spec = f"{base_ref}{separator}{head_ref}"

        diff_args = [
            "diff",
            f"--unified={context_lines}",
            "--no-color",
            "--find-renames",
            diff_spec,
        ]

        rc, stdout, stderr = _run_git(repo, diff_args)
        if rc != 0:
            # If triple-dot failed (e.g. no merge base), try fallback to double-dot
            if triple_dot:
                fallback_spec = f"{base_ref}..{head_ref}"
                logger.info(f"Triple-dot diff '{diff_spec}' failed; falling back to '{fallback_spec}'")
                diff_args[-1] = fallback_spec
                rc, stdout, stderr = _run_git(repo, diff_args)

            if rc != 0:
                result.errors.append(
                    f"git diff failed for '{diff_spec}': {stderr.strip() or 'Unknown error'}"
                )
                return result

        diff_blocks = _parse_diff_blocks(stdout)

    changed_files: List[ChangedFile] = []
    review_changes: List[ReviewChange] = []

    for block in diff_blocks:
        file_obj = _parse_file_block(
            repo_obj,
            base_ref,
            head_ref,
            block,
            content_fetcher=content_fetcher,
        )
        if not file_obj:
            continue

        changed_files.append(file_obj)

        # Create atomic ReviewChange objects for each hunk
        for hunk in file_obj.hunks:
            scope_desc = (
                f"{hunk.class_name}.{hunk.function_name}"
                if (hunk.class_name and hunk.function_name and not hunk.function_name.startswith(hunk.class_name + "."))
                else (hunk.function_name or hunk.class_name or "<module>")
            )
            summary = (
                f"{file_obj.change_type.capitalize()} {scope_desc} in "
                f"{file_obj.path} (+{hunk.additions_count} -{hunk.deletions_count})"
            )

            rc_item = ReviewChange(
                file_path=file_obj.path,
                old_path=file_obj.old_path,
                change_type=file_obj.change_type,
                is_python=file_obj.is_python,
                function_name=hunk.function_name,
                class_name=hunk.class_name,
                all_functions=hunk.all_functions,
                old_start=hunk.old_start,
                old_lines=hunk.old_lines,
                new_start=hunk.new_start,
                new_lines=hunk.new_lines,
                diff_content=hunk.diff_content,
                added_lines=hunk.added_lines,
                removed_lines=hunk.removed_lines,
                hunk_header=hunk.header,
                summary=summary,
            )
            review_changes.append(rc_item)

    # Compute overall statistics
    result.files = changed_files
    result.changes = review_changes
    result.total_files = len(changed_files)
    result.python_files = sum(1 for f in changed_files if f.is_python)
    result.total_additions = sum(f.additions for f in changed_files)
    result.total_deletions = sum(f.deletions for f in changed_files)

    return result
