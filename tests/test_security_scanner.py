"""
tests/test_security_scanner.py — Tests for RAG-based Security Scanner Agent.

Tests:
  • Pre-execution secret sanitization before calling LLM
  • Context retrieval from ChromaDB vector store
  • Prompt augmentation with security references
  • Structured output parsing into SecurityFinding instances
  • Error handling and graceful fallbacks
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.security_scanner_agent import (
    FindingConfidence,
    FindingSeverity,
    SecurityFinding,
    SecurityScanReport,
    SecurityScannerAgent,
    SecurityScannerInput,
    SecurityScannerResult,
    VulnerabilityIssue,
    scan_security,
)
from llm.llm_client import BaseLLMProvider, LLMResponse, LLMStructuredResponse
from storage.vector_store import RetrievedDocument, SecurityVectorStore

VULNERABLE_SQLI_CODE = """
import sqlite3

def get_user_profile(user_id):
    # Injected raw SQL query
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
"""

CODE_WITH_API_KEY = """
OPENAI_KEY = "sk-proj-11223344556677889900aabbccddeeff"

def fetch_data():
    pass
"""


class TestSecurityScannerSanitization:
    """Ensure secrets are never passed to the LLM during security scans."""

    def test_secrets_sanitized_before_rag_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "mock_gemini"
        mock_provider.model_name = "gemini-3.5-flash-lite"

        mock_report = SecurityScanReport(
            summary="Clean report",
            findings=[]
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_report,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="mock",
                provider="mock",
                status="success"
            )
        )

        mock_vector_store = MagicMock(spec=SecurityVectorStore)
        mock_vector_store.search.return_value = []

        agent = SecurityScannerAgent(vector_store=mock_vector_store, provider=mock_provider)
        result = agent.scan_code(CODE_WITH_API_KEY, filename="auth.py")

        assert result.status == "success"
        assert result.sanitization_applied is True
        assert result.secrets_detected >= 1
        assert "sk-proj-11223344556677889900aabbccddeeff" not in result.sanitized_code
        assert "<REDACTED_" in result.sanitized_code

        # Verify LLM prompt received sanitized code
        call_kwargs = mock_provider.generate_structured.call_args.kwargs
        prompt = call_kwargs.get("prompt", "")
        assert "sk-proj-11223344556677889900aabbccddeeff" not in prompt
        assert "<REDACTED_" in prompt


class TestSecurityScannerRAGIntegration:
    """Test RAG context retrieval and prompt construction."""

    def test_retrieval_and_finding_mapping(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "gemini"
        mock_provider.model_name = "gemini-3.5-flash-lite"

        # Mock structured LLM response
        mock_finding = VulnerabilityIssue(
            title="SQL Injection via Formatted Query",
            description="Direct string interpolation of user_id permits SQL Injection.",
            severity="critical",
            confidence="high",
            affected_code='query = f"SELECT * FROM users WHERE id = \'{user_id}\'"',
            cwe_id="CWE-89",
            reasoning="Unsanitized user_id alters query syntax.",
            suggested_remediation='cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
            line_number=8,
            category="injection"
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=SecurityScanReport(
                summary="Critical SQL Injection flaw identified.",
                findings=[mock_finding]
            ),
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="gemini",
                provider="gemini",
                status="success"
            )
        )

        mock_doc = RetrievedDocument(
            doc_id="SEC-001-SQLI",
            title="SQL Injection",
            content="Use parameterized queries.",
            cwe="CWE-89",
            severity="CRITICAL",
            category="injection",
            distance=0.1,
            relevance_score=0.95
        )
        mock_vector_store = MagicMock(spec=SecurityVectorStore)
        mock_vector_store.search.return_value = [mock_doc]

        agent = SecurityScannerAgent(vector_store=mock_vector_store, provider=mock_provider)
        result = agent.scan_code(VULNERABLE_SQLI_CODE, filename="profile.py")

        assert result.status == "success"
        assert len(result.findings) == 1
        assert len(result.retrieved_contexts) == 1

        finding = result.findings[0]
        assert isinstance(finding, SecurityFinding)
        assert finding.cwe_id == "CWE-89"
        assert finding.severity == FindingSeverity.CRITICAL
        assert finding.matched_rule_id == "SEC-001-SQLI"
        assert finding.tool == "security_scanner_rag"

        # Check prompt contained retrieved context
        call_kwargs = mock_provider.generate_structured.call_args.kwargs
        prompt = call_kwargs.get("prompt", "")
        assert "RELEVANT SECURITY KNOWLEDGE CONTEXT" in prompt
        assert "SEC-001-SQLI" not in prompt  # Doc title is present
        assert "SQL Injection (CWE-89)" in prompt


class TestSecurityScannerErrorHandling:
    """Verify scanner handles edge cases without crashing."""

    def test_empty_code_returns_empty_result(self):
        result = scan_security(SecurityScannerInput(source_code=""))
        assert result.status == "success"
        assert len(result.findings) == 0

    def test_llm_exception_handled_gracefully(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.generate_structured.side_effect = RuntimeError("Provider down")

        mock_vector_store = MagicMock(spec=SecurityVectorStore)
        mock_vector_store.search.return_value = []

        result = scan_security(
            SecurityScannerInput(source_code="import os"),
            vector_store=mock_vector_store,
            provider=mock_provider,
        )

        assert result.status == "error"
        assert len(result.errors) > 0
        assert "Provider down" in result.errors[0]
