"""
tests/test_indexing.py — Unit tests for the shared codebase indexing system.

Coverage:
  • dependency_graph_builder — file collection, import resolution, error handling
  • summary_generator        — code unit extraction, stable IDs, fallback summaries
  • incremental_updater      — index_codebase() statistics, search_codebase()
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_py_file(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# dependency_graph_builder
# ─────────────────────────────────────────────────────────────────────────────

class TestDependencyGraphBuilder:

    def test_discovers_python_files(self, tmp_path):
        make_py_file(tmp_path, "a.py", "x = 1")
        make_py_file(tmp_path, "pkg/b.py", "y = 2")
        # Hidden dirs should be skipped
        make_py_file(tmp_path, "__pycache__/c.pyc", "z = 3")
        make_py_file(tmp_path, ".git/config.py", "w = 4")

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)

        files = graph.all_files()
        assert "a.py" in files
        assert "pkg/b.py" in files
        # Skipped dirs must NOT appear
        assert not any("__pycache__" in f for f in files)
        assert not any(".git" in f for f in files)

    def test_resolves_local_imports(self, tmp_path):
        make_py_file(tmp_path, "utils.py", "def helper(): pass")
        make_py_file(tmp_path, "main.py", "import utils\nfrom utils import helper")

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)

        assert "utils.py" in graph.direct_dependencies("main.py")

    def test_reverse_index(self, tmp_path):
        make_py_file(tmp_path, "shared.py", "CONST = 42")
        make_py_file(tmp_path, "a.py", "import shared")
        make_py_file(tmp_path, "b.py", "import shared")

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)

        dependents = graph.dependents("shared.py")
        assert "a.py" in dependents
        assert "b.py" in dependents

    def test_third_party_imports_not_in_edges(self, tmp_path):
        make_py_file(tmp_path, "app.py", "import os\nimport json\nimport requests")

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)

        # Local edges should be empty (os / json / requests are stdlib/third-party)
        assert graph.edges["app.py"] == []
        # But external_imports should record them
        ext = graph.external_imports["app.py"]
        assert "os" in ext
        assert "json" in ext

    def test_syntax_error_file_is_captured(self, tmp_path):
        make_py_file(tmp_path, "good.py", "x = 1")
        make_py_file(tmp_path, "bad.py", "def broken(\n")   # intentional syntax error

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)

        assert "bad.py" in graph.errors
        assert "good.py" in graph.edges   # good file still indexed

    def test_stats(self, tmp_path):
        make_py_file(tmp_path, "x.py", "import y")
        make_py_file(tmp_path, "y.py", "pass")

        from indexing.dependency_graph_builder import build_dependency_graph
        graph = build_dependency_graph(tmp_path)
        stats = graph.stats()

        assert stats["total_files"] == 2
        assert stats["total_edges"] == 1   # x.py → y.py
        assert stats["files_with_errors"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# summary_generator
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SOURCE = '''\
"""Module docstring."""

CONSTANT = 42


def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


async def fetch(url: str) -> bytes:
    """Fetch a URL asynchronously."""
    ...


class Calculator:
    """A simple calculator class."""

    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    def divide(self, x: float, y: float) -> float:
        return x / y
'''


class TestSummaryGenerator:

    def test_extract_functions(self):
        from indexing.summary_generator import extract_code_units
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        names = [u.unit_name for u in units]
        assert "add" in names
        assert "fetch" in names

    def test_extract_class_and_methods(self):
        from indexing.summary_generator import extract_code_units
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        names = [u.unit_name for u in units]
        assert "Calculator" in names
        assert "multiply" in names
        assert "divide" in names

    def test_unit_types(self):
        from indexing.summary_generator import extract_code_units
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        by_name = {u.unit_name: u for u in units}

        assert by_name["add"].unit_type == "function"
        assert by_name["fetch"].unit_type == "function"
        assert by_name["Calculator"].unit_type == "class"
        assert by_name["multiply"].unit_type == "method"
        assert by_name["divide"].unit_type == "method"

    def test_docstrings_captured(self):
        from indexing.summary_generator import extract_code_units
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        by_name = {u.unit_name: u for u in units}

        assert "Return the sum" in by_name["add"].docstring
        assert "simple calculator" in by_name["Calculator"].docstring

    def test_stable_ids_are_deterministic(self):
        from indexing.summary_generator import extract_code_units
        units1 = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        units2 = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        ids1 = {u.unit_name: u.stable_id for u in units1}
        ids2 = {u.unit_name: u.stable_id for u in units2}
        assert ids1 == ids2

    def test_stable_ids_differ_across_files(self):
        from indexing.summary_generator import extract_code_units
        units_a = extract_code_units(SAMPLE_SOURCE, "file_a.py")
        units_b = extract_code_units(SAMPLE_SOURCE, "file_b.py")
        ids_a = {u.unit_name: u.stable_id for u in units_a}
        ids_b = {u.unit_name: u.stable_id for u in units_b}
        # Same unit names but different paths → different stable IDs
        assert ids_a["add"] != ids_b["add"]

    def test_syntax_error_returns_empty_list(self):
        from indexing.summary_generator import extract_code_units
        units = extract_code_units("def broken(\n", "bad.py")
        assert units == []

    def test_fallback_summary_not_empty(self):
        from indexing.summary_generator import extract_code_units, generate_summaries
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        units = generate_summaries(units, llm_client=None, use_llm=False)
        for u in units:
            assert u.summary, f"Empty summary for '{u.unit_name}'"

    def test_fallback_summary_contains_unit_name(self):
        from indexing.summary_generator import extract_code_units, generate_summaries
        units = extract_code_units(SAMPLE_SOURCE, "math/calc.py")
        units = generate_summaries(units)
        by_name = {u.unit_name: u for u in units}
        assert "add" in by_name["add"].summary
        assert "Calculator" in by_name["Calculator"].summary

    def test_to_chroma_document_has_required_keys(self):
        from indexing.summary_generator import extract_code_units, generate_summaries
        units = generate_summaries(extract_code_units(SAMPLE_SOURCE, "math/calc.py"))
        doc = units[0].to_chroma_document()
        for key in ("id", "content", "title", "file_path", "unit_type", "start_line"):
            assert key in doc, f"Missing key '{key}' in chroma document"


# ─────────────────────────────────────────────────────────────────────────────
# incremental_updater (integration — hits ChromaDB in-memory)
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementalUpdater:

    @pytest.fixture(autouse=True)
    def _reset_store(self, tmp_path, monkeypatch):
        """Each test gets a fresh in-process store singleton."""
        import indexing.incremental_updater as iu
        monkeypatch.setattr(iu, "_codebase_store", None)
        # Store the tmp chroma dir so test methods can reference it
        self._chroma_dir = tmp_path / "chroma"

    def _make_repo(self, tmp_path: Path) -> Path:
        """Create a minimal Python repository for integration testing."""
        make_py_file(tmp_path, "utils.py", """\
            def greet(name: str) -> str:
                \"\"\"Return a personalised greeting.\"\"\"
                return f"Hello, {name}!"
        """)
        make_py_file(tmp_path, "math_utils.py", """\
            def factorial(n: int) -> int:
                \"\"\"Compute factorial recursively.\"\"\"
                if n <= 1:
                    return 1
                return n * factorial(n - 1)

            class Statistics:
                \"\"\"Basic statistical operations.\"\"\"
                def mean(self, data):
                    \"\"\"Compute arithmetic mean.\"\"\"
                    return sum(data) / len(data)
        """)
        make_py_file(tmp_path, "bad_syntax.py", "def (broken):\n")
        return tmp_path

    def test_index_codebase_returns_result(self, tmp_path):
        from indexing import index_codebase
        repo = self._make_repo(tmp_path / "repo")
        result = index_codebase(repo, use_llm_summaries=False, persist_dir=self._chroma_dir)

        assert result.total_files_scanned >= 2
        assert result.total_units_indexed > 0
        assert result.duration_seconds > 0

    def test_index_codebase_counts_unit_types(self, tmp_path):
        from indexing import index_codebase
        repo = self._make_repo(tmp_path / "repo")
        result = index_codebase(repo, use_llm_summaries=False, persist_dir=self._chroma_dir)

        # greet + factorial = 2 functions; Statistics = 1 class; mean = 1 method
        assert result.total_functions >= 2
        assert result.total_classes >= 1
        assert result.total_methods >= 1

    def test_syntax_error_captured_not_crashed(self, tmp_path):
        from indexing import index_codebase
        repo = self._make_repo(tmp_path / "repo")
        result = index_codebase(repo, use_llm_summaries=False, persist_dir=self._chroma_dir)

        # The run must succeed even with bad_syntax.py
        assert result is not None
        # bad_syntax.py should appear in errors
        assert any("bad_syntax.py" in e for e in result.errors)

    def test_repeatable_indexing_idempotent(self, tmp_path):
        from indexing import index_codebase, get_indexed_count
        repo = self._make_repo(tmp_path / "repo")
        d = self._chroma_dir

        r1 = index_codebase(repo, use_llm_summaries=False, persist_dir=d)
        count_after_first = get_indexed_count(persist_dir=d)

        # Reset singleton so second call creates a fresh store on same dir
        import indexing.incremental_updater as iu
        iu._codebase_store = None

        r2 = index_codebase(repo, use_llm_summaries=False, persist_dir=d)
        count_after_second = get_indexed_count(persist_dir=d)

        # ChromaDB upsert — count should not grow on re-index
        assert count_after_first == count_after_second
        assert r1.total_units_indexed == r2.total_units_indexed

    def test_search_returns_relevant_results(self, tmp_path):
        from indexing import index_codebase, search_codebase
        repo = self._make_repo(tmp_path / "repo")
        d = self._chroma_dir
        index_codebase(repo, use_llm_summaries=False, persist_dir=d)

        results = search_codebase("compute factorial recursively", top_k=3, persist_dir=d)
        assert len(results) > 0
        # The most relevant result should mention factorial
        top = results[0]
        assert "factorial" in (top.title + top.content).lower()

    def test_search_on_greeting_function(self, tmp_path):
        from indexing import index_codebase, search_codebase
        repo = self._make_repo(tmp_path / "repo")
        d = self._chroma_dir
        index_codebase(repo, use_llm_summaries=False, persist_dir=d)

        results = search_codebase("personalised greeting for a user", top_k=3, persist_dir=d)
        assert any("greet" in r.title.lower() or "greet" in r.content.lower() for r in results)

    def test_invalid_path_raises(self, tmp_path):
        from indexing import index_codebase
        with pytest.raises(ValueError, match="is not a directory"):
            index_codebase(tmp_path / "nonexistent_repo")

    def test_result_summary_string(self, tmp_path):
        from indexing import index_codebase
        repo = self._make_repo(tmp_path / "repo")
        result = index_codebase(repo, persist_dir=self._chroma_dir)
        summary = result.summary()
        assert "Files scanned" in summary
        assert "Code units indexed" in summary
