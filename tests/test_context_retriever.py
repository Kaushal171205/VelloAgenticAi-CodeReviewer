"""
tests/test_context_retriever.py — Unit and integration tests for Context Retriever Agent.

Validates semantic query construction, ChromaDB index querying, self-exclusion,
relevance thresholds, dependency graph enrichment, and token budgeting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from context import (
    ContextRetriever,
    FunctionContext,
    RelatedFunction,
    RetrievedContextPackage,
    retrieve_context_for_changes,
    retrieve_context_for_function,
)
from context.context_retriever import _generate_retrieval_query
from diff_processing.git_diff_extractor import ReviewChange
from indexing.dependency_graph_builder import DependencyGraph
from storage.vector_store import RetrievedDocument


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests (Mocked Store)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryGeneration:
    def test_query_includes_function_and_path(self):
        query = _generate_retrieval_query(
            function_name="process_refund",
            file_path="src/billing/payments.py",
            diff_content="+ raise ValueError('Negative amount')\n",
        )
        assert "process_refund" in query
        assert "payments" in query
        assert "ValueError" in query

    def test_query_handles_empty_diff(self):
        query = _generate_retrieval_query(
            function_name="calculate_tax",
            file_path="tax.py",
        )
        assert "calculate_tax" in query


class TestContextRetrieverWithMockStore:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        # Simulated ChromaDB search results
        store.search.return_value = [
            # 1. The function itself (should be excluded)
            RetrievedDocument(
                doc_id="self_1",
                title="process_refund",
                content="Processes customer refunds.",
                cwe="",
                severity="INFO",
                category="codebase",
                distance=0.05,
                relevance_score=0.95,
                metadata={"file_path": "payments.py", "unit_type": "function", "signature": "def process_refund(tx_id, amount):"},
            ),
            # 2. Highly relevant helper in another file
            RetrievedDocument(
                doc_id="doc_2",
                title="get_transaction_gateway",
                content="Fetches payment gateway client.",
                cwe="",
                severity="INFO",
                category="codebase",
                distance=0.18,
                relevance_score=0.82,
                metadata={"file_path": "gateway.py", "unit_type": "function", "signature": "def get_transaction_gateway():"},
            ),
            # 3. Related fee calculator
            RetrievedDocument(
                doc_id="doc_3",
                title="calculate_refund_fee",
                content="Computes processing fee on refund.",
                cwe="",
                severity="INFO",
                category="codebase",
                distance=0.25,
                relevance_score=0.75,
                metadata={"file_path": "fees.py", "unit_type": "function", "signature": "def calculate_refund_fee(amt):"},
            ),
            # 4. Irrelevant / low score unit (should be filtered out by min_relevance_score)
            RetrievedDocument(
                doc_id="doc_4",
                title="unrelated_logging_util",
                content="Formats log messages.",
                cwe="",
                severity="INFO",
                category="codebase",
                distance=0.80,
                relevance_score=0.20,
                metadata={"file_path": "logging_util.py", "unit_type": "function", "signature": "def format_log():"},
            ),
        ]
        return store

    def test_retrieve_for_function_excludes_self_and_filters_low_scores(self, mock_store):
        retriever = ContextRetriever(
            vector_store=mock_store,
            min_relevance_score=0.35,
            max_related_per_function=3,
        )

        ctx = retriever.retrieve_for_function(
            function_name="process_refund",
            file_path="payments.py",
            diff_content="+ # change\n",
        )

        assert ctx.changed_function == "process_refund"
        assert ctx.file_path == "payments.py"
        # Should contain get_transaction_gateway and calculate_refund_fee
        assert len(ctx.related_functions) == 2
        names = [rf.name for rf in ctx.related_functions]
        assert "process_refund" not in names  # Self excluded!
        assert "get_transaction_gateway" in names
        assert "calculate_refund_fee" in names
        assert "unrelated_logging_util" not in names  # Low score excluded!

    def test_retrieve_includes_dependency_graph(self, mock_store, tmp_path):
        # Create a mock DependencyGraph
        dep_graph = DependencyGraph(tmp_path)
        dep_graph.edges["payments.py"] = ["gateway.py", "fees.py"]
        dep_graph.reverse["payments.py"] = ["api/routes.py"]
        dep_graph.external_imports["payments.py"] = ["stripe"]

        retriever = ContextRetriever(
            vector_store=mock_store,
            dependency_graph=dep_graph,
            min_relevance_score=0.35,
        )

        ctx = retriever.retrieve_for_function(
            function_name="process_refund",
            file_path="payments.py",
        )

        assert ctx.dependency_info is not None
        assert "gateway.py" in ctx.dependency_info.imports
        assert "api/routes.py" in ctx.dependency_info.imported_by
        assert "stripe" in ctx.dependency_info.external_imports

        # Verify badges on related functions
        for rf in ctx.related_functions:
            if rf.name == "get_transaction_gateway":
                assert rf.is_direct_dependency is True
                assert "Imported Dependency" in rf.relationship_badge

    def test_deduplication_across_multiple_changes(self, mock_store):
        retriever = ContextRetriever(
            vector_store=mock_store,
            min_relevance_score=0.35,
            max_related_per_function=2,
            max_total_related=5,
        )

        changes = [
            ReviewChange(file_path="payments.py", function_name="process_refund", is_python=True),
            ReviewChange(file_path="checkout.py", function_name="process_checkout", is_python=True),
        ]

        package = retriever.retrieve_for_changes(changes)

        # Both changes would have retrieved get_transaction_gateway,
        # but deduplication ensures it only appears once across all contexts.
        all_related_names = [rf.name for ctx in package.contexts for rf in ctx.related_functions]
        assert all_related_names.count("get_transaction_gateway") == 1
        assert package.total_related_functions == len(all_related_names)

    def test_prompt_snippet_formatting(self, mock_store):
        retriever = ContextRetriever(vector_store=mock_store)
        ctx = retriever.retrieve_for_function(
            function_name="process_refund",
            file_path="payments.py",
            summary="Added currency check",
        )
        snippet = ctx.format_for_prompt()
        assert "Context for Changed Function: `process_refund`" in snippet
        assert "get_transaction_gateway" in snippet

        package = RetrievedContextPackage(contexts=[ctx])
        prompt_str = package.to_llm_prompt_snippet(max_tokens=2000)
        assert "Relevant Existing Codebase Context" in prompt_str
        assert "get_transaction_gateway" in prompt_str


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Integration Test with ChromaDB & Indexer
# ─────────────────────────────────────────────────────────────────────────────

class TestContextRetrieverIntegration:
    def test_e2e_indexing_and_context_retrieval(self, tmp_path: Path):
        """Index a sample repository and verify context retriever retrieves real units."""
        from indexing import index_codebase

        repo_dir = tmp_path / "sample_app"
        repo_dir.mkdir()
        chroma_dir = tmp_path / "chroma_store"

        # Create two interconnected files
        (repo_dir / "auth.py").write_text(
            "def verify_jwt_token(token: str) -> bool:\n"
            "    '''Verify a cryptographic user token.'''\n"
            "    return len(token) > 10\n",
            encoding="utf-8",
        )
        (repo_dir / "user_service.py").write_text(
            "from auth import verify_jwt_token\n\n"
            "def get_user_profile(user_id: str, token: str):\n"
            "    '''Fetch user profile if token is valid.'''\n"
            "    if not verify_jwt_token(token):\n"
            "        raise PermissionError('Unauthorized')\n"
            "    return {'id': user_id, 'name': 'Alice'}\n",
            encoding="utf-8",
        )

        # Index codebase into local chroma
        idx_result = index_codebase(path=repo_dir, persist_dir=chroma_dir)
        assert idx_result.total_units_indexed >= 2

        # Create review change for user_service.py
        change = ReviewChange(
            file_path="user_service.py",
            function_name="get_user_profile",
            is_python=True,
            diff_content="+ if not verify_jwt_token(token):\n+ raise PermissionError('Unauthorized')\n",
            summary="Added JWT token authorization check",
        )

        # Retrieve context
        package = retrieve_context_for_changes(
            changes=[change],
            repo_path=repo_dir,
            persist_dir=chroma_dir,
            min_relevance_score=0.20,
        )

        assert len(package.contexts) == 1
        ctx = package.contexts[0]
        assert ctx.changed_function == "get_user_profile"

        # Check dependency info was picked up from real repo
        if ctx.dependency_info:
            assert any("auth" in dep for dep in ctx.dependency_info.imports)

        # Check that verify_jwt_token was retrieved from semantic search
        retrieved_names = [rf.name for rf in ctx.related_functions]
        assert "get_user_profile" not in retrieved_names  # Excluded self!
        assert "verify_jwt_token" in retrieved_names  # Retrieved related function!
