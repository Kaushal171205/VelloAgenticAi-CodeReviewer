"""
tests/test_vector_store.py — Unit & Integration tests for ChromaDB Security Vector Store.

Verifies:
  • Local collection creation & initialization
  • Embedding computation using sentence-transformers
  • Document indexing and retrieval
  • Semantic search ranking (e.g. SQLi query matches SQLi knowledge, not crypto)
  • Top-k context bounds and relevance scoring
"""

import shutil
from pathlib import Path

import pytest

from storage.vector_store import RetrievedDocument, SecurityVectorStore

TEST_DB_PATH = Path("./storage/test_chroma_db")


@pytest.fixture(scope="module")
def vector_store():
    """Fixture providing an initialized test vector store."""
    # Clean up test path if exists
    if TEST_DB_PATH.exists():
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)

    store = SecurityVectorStore(
        persist_directory=TEST_DB_PATH,
        collection_name="test_security_knowledge",
        auto_preload=True,
    )
    yield store

    # Cleanup after test suite
    if TEST_DB_PATH.exists():
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


class TestVectorStoreInitialization:
    """Test vector store startup and document counts."""

    def test_collection_preloaded_documents(self, vector_store):
        count = vector_store.count()
        assert count >= 8, f"Expected at least 8 documents, found {count}"

    def test_add_custom_document(self, vector_store):
        initial_count = vector_store.count()
        vector_store.add_document(
            doc_id="CUSTOM-TEST-001",
            title="GraphQL Depth Limit Missing",
            content="GraphQL endpoints vulnerable to deeply nested query denial of service attacks.",
            cwe="CWE-400",
            severity="HIGH",
            category="dos",
        )
        assert vector_store.count() == initial_count + 1


class TestSemanticRetrieval:
    """Prove semantic retrieval matches relevant vulnerabilities over irrelevant ones."""

    def test_semantic_match_sql_injection(self, vector_store):
        code_query = 'cursor.execute(f"SELECT * FROM accounts WHERE id = \'{user_id}\'")'
        results = vector_store.search(code_query, top_k=2)

        assert len(results) > 0
        top_match = results[0]
        assert "SQL" in top_match.title or top_match.cwe == "CWE-89"
        assert top_match.relevance_score > 0.6

    def test_semantic_match_command_injection(self, vector_store):
        code_query = 'os.system(f"ping -c 1 {user_host}")'
        results = vector_store.search(code_query, top_k=2)

        assert len(results) > 0
        titles = [r.title for r in results]
        cwes = [r.cwe for r in results]
        assert "CWE-78" in cwes or any("Command" in t for t in titles)
        assert results[0].relevance_score > 0.5

    def test_semantic_match_ssrf(self, vector_store):
        code_query = 'requests.get(request.args["webhook_url"])'
        results = vector_store.search(code_query, top_k=2)

        assert len(results) > 0
        # Check that SSRF is in top results
        cwes = [r.cwe for r in results]
        assert "CWE-918" in cwes or any("SSRF" in r.title for r in results)

    def test_semantic_match_insecure_deserialization(self, vector_store):
        code_query = 'obj = pickle.loads(raw_user_payload)'
        results = vector_store.search(code_query, top_k=2)

        assert len(results) > 0
        top_match = results[0]
        assert "Deserialization" in top_match.title or top_match.cwe == "CWE-502"

    def test_top_k_limit_respected(self, vector_store):
        results = vector_store.search("general python code", top_k=3)
        assert len(results) <= 3

    def test_empty_query_returns_empty(self, vector_store):
        assert vector_store.search("") == []
        assert vector_store.search("   ") == []
