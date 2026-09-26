"""
agents/security_scanner_agent.py — RAG-based Security Scanner Agent.

Combines semantic similarity retrieval from a local ChromaDB security knowledge base
with LLM reasoning to identify vulnerabilities, classify severity, reference CWE IDs,
and provide verified remediations.

Workflow:
  1. Receive source code (or diff snippet).
  2. Enforce secret sanitization before calling LLM.
  3. Query local ChromaDB vector store for top-k relevant vulnerability documents.
  4. Augment prompt with retrieved security reference context (RAG).
  5. Call LLM abstraction layer with structured output schema.
  6. Return normalized SecurityFinding list adhering to common finding schema.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.sanitizer import SanitizerInput, sanitize
from agents.static_analysis_agent import FindingConfidence, FindingSeverity
from llm.llm_client import BaseLLMProvider, get_llm_provider
from storage.vector_store import RetrievedDocument, SecurityVectorStore
from tools.language_utils import detect_language, format_code_for_prompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for LLM Structured Output
# ─────────────────────────────────────────────────────────────────────────────

class VulnerabilityIssue(BaseModel):
    """Pydantic model representing a single security vulnerability identified by RAG."""
    title: str = Field(..., description="Concise title of the security vulnerability.")
    description: str = Field(..., description="Detailed technical description of the vulnerability and exploit scenario.")
    severity: str = Field(..., description="Severity rating: 'critical', 'high', 'medium', or 'low'.")
    confidence: str = Field(..., description="Confidence rating: 'high', 'medium', or 'low'.")
    affected_code: str = Field(..., description="Exact code snippet or line containing the flaw.")
    cwe_id: Optional[str] = Field(default=None, description="CWE identifier (e.g. 'CWE-89', 'CWE-78', 'CWE-22').")
    reasoning: str = Field(..., description="Explanation of why this constitutes a security vulnerability referencing provided knowledge.")
    suggested_remediation: str = Field(..., description="Actionable, safe replacement code fixing the issue.")
    line_number: Optional[int] = Field(default=None, description="Line number in the submitted code if identifiable.")
    category: Optional[str] = Field(default="security", description="Vulnerability category (e.g. injection, file_access, deserialization).")


class SecurityScanReport(BaseModel):
    """Pydantic container for complete security review report."""
    summary: str = Field(..., description="Executive summary of the security audit.")
    findings: List[VulnerabilityIssue] = Field(default_factory=list, description="List of identified security vulnerabilities.")


# ─────────────────────────────────────────────────────────────────────────────
# Agent Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SecurityFinding:
    """Normalised security finding conforming to the project's common finding schema."""
    title: str
    description: str
    severity: FindingSeverity
    confidence: FindingConfidence
    affected_code: str
    reasoning: str
    suggested_remediation: str
    tool: str = "security_scanner_rag"
    cwe_id: Optional[str] = None
    line_number: Optional[int] = None
    category: str = "security"
    file: str = "snippet.py"
    matched_rule_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "affected_code": self.affected_code,
            "reasoning": self.reasoning,
            "suggested_remediation": self.suggested_remediation,
            "tool": self.tool,
            "cwe_id": self.cwe_id,
            "line_number": self.line_number,
            "category": self.category,
            "file": self.file,
            "matched_rule_id": self.matched_rule_id,
        }


@dataclass
class SecurityScannerInput:
    """Input payload for the Security Scanner Agent."""
    source_code: str
    filename: str = "snippet.py"
    context_notes: Optional[str] = None
    top_k_context: int = 3
    is_pre_sanitized: bool = False


@dataclass
class SecurityScannerResult:
    """Complete output produced by the Security Scanner Agent."""
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: str = ""
    sanitized_code: str = ""
    sanitization_applied: bool = False
    secrets_detected: int = 0
    retrieved_contexts: List[RetrievedDocument] = field(default_factory=list)
    raw_llm_response: str = ""
    provider_name: str = ""
    model_name: str = ""
    latency_ms: float = 0.0
    status: str = "success"  # "success" | "error"
    errors: List[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    @property
    def highest_severity(self) -> Optional[FindingSeverity]:
        if not self.findings:
            return None
        order = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
            FindingSeverity.INFO,
        ]
        present = {f.severity for f in self.findings}
        for s in order:
            if s in present:
                return s
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _map_severity(raw: str) -> FindingSeverity:
    clean = (raw or "").strip().lower()
    mapping = {
        "critical": FindingSeverity.CRITICAL,
        "high": FindingSeverity.HIGH,
        "medium": FindingSeverity.MEDIUM,
        "low": FindingSeverity.LOW,
        "info": FindingSeverity.INFO,
    }
    return mapping.get(clean, FindingSeverity.MEDIUM)


def _map_confidence(raw: str) -> FindingConfidence:
    clean = (raw or "").strip().lower()
    mapping = {
        "high": FindingConfidence.HIGH,
        "medium": FindingConfidence.MEDIUM,
        "low": FindingConfidence.LOW,
    }
    return mapping.get(clean, FindingConfidence.HIGH)


SECURITY_SCANNER_SYSTEM_PROMPT = """
You are a Senior Application Security Engineer performing static source code security audits.
You analyze source code for software vulnerabilities using both your security expertise and
the provided reference security documents retrieved from an enterprise vulnerability knowledge base.

Guidelines:
1. Ground your vulnerability assessment in the provided Security Knowledge Context when applicable.
2. Provide precise CWE classifications when identifying vulnerabilities.
3. Distinguish actual exploitable security vulnerabilities from general code smells.
4. Output concise, production-ready remediation code blocks that completely fix the vulnerability.
5. Strict adherence to the requested JSON format is mandatory.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Core Scanner Function
# ─────────────────────────────────────────────────────────────────────────────

def scan_security(
    input_data: SecurityScannerInput,
    vector_store: Optional[SecurityVectorStore] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> SecurityScannerResult:
    """
    Execute RAG-based security scan on the input code.

    Args:
        input_data: SecurityScannerInput containing code and options.
        vector_store: Optional SecurityVectorStore instance.
        provider: Optional BaseLLMProvider instance.

    Returns:
        SecurityScannerResult containing findings and retrieved context.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    if not input_data.source_code or not input_data.source_code.strip():
        return SecurityScannerResult(
            findings=[],
            summary="No code provided for security scan.",
            sanitized_code="",
            status="success",
        )

    # 1. Enforce Secret Sanitization
    sanitized_code = input_data.source_code
    secrets_count = 0
    sanitization_applied = False

    if not input_data.is_pre_sanitized:
        try:
            sanitizer_res = sanitize(
                SanitizerInput(
                    source_code=input_data.source_code,
                    filename=input_data.filename,
                )
            )
            sanitized_code = sanitizer_res.sanitized_code
            secrets_count = sanitizer_res.number_of_secrets
            sanitization_applied = True
        except Exception as e:
            err = f"Pre-scan secret sanitization failed: {e}"
            logger.error(err)
            errors.append(err)
            return SecurityScannerResult(
                status="error",
                errors=errors,
            )

    # 2. Retrieve Relevant Context from Vector Store
    v_store = vector_store
    retrieved_docs: List[RetrievedDocument] = []
    if v_store is None:
        try:
            v_store = SecurityVectorStore()
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            errors.append(f"Vector store unavailable: {e}")

    if v_store is not None:
        try:
            # Query vector store using sanitized code snippet
            retrieved_docs = v_store.search(
                query=sanitized_code[:2000],  # Use first 2000 chars for semantic matching
                top_k=input_data.top_k_context,
            )
            logger.info(f"Retrieved {len(retrieved_docs)} relevant context document(s) for {input_data.filename}.")
        except Exception as e:
            logger.warning(f"Vector store retrieval error: {e}")
            errors.append(f"Context retrieval error: {e}")

    # 3. Resolve LLM Provider
    llm = provider
    if llm is None:
        try:
            llm = get_llm_provider()
        except Exception as e:
            err = f"Failed to initialize LLM provider: {e}"
            logger.error(err)
            errors.append(err)
            return SecurityScannerResult(
                status="error",
                errors=errors,
                sanitized_code=sanitized_code,
                sanitization_applied=sanitization_applied,
                secrets_detected=secrets_count,
                retrieved_contexts=retrieved_docs,
            )

    # 4. Build RAG Augmented Prompt
    code_block, is_diff = format_code_for_prompt(sanitized_code, input_data.filename)

    context_str = ""
    if retrieved_docs:
        context_str = "\n--- RELEVANT SECURITY KNOWLEDGE CONTEXT (Retrieved from Local Knowledge Base) ---\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context_str += (
                f"\n[Document {i}]: {doc.title} ({doc.cwe}) - Severity: {doc.severity}\n"
                f"Content:\n{doc.content}\n"
            )
        context_str += "--- END CONTEXT ---\n"

    if is_diff:
        header = (
            "Perform a comprehensive security vulnerability scan on the code changes below.\n\n"
            "The input is a unified Git diff packaged for review: each section describes a "
            "changed file with metadata, followed by diff hunks where '+' marks added lines "
            "and '-' marks removed lines. Analyze the added/changed code for vulnerabilities.\n\n"
            f"Change set: {input_data.filename}\n\n"
            f"{code_block}\n"
        )
    else:
        header = (
            "Perform a comprehensive security vulnerability scan on the following code:\n\n"
            f"File: {input_data.filename}\n"
            f"{code_block}\n"
        )

    prompt = (
        f"{header}"
        f"{context_str}\n"
        f"Analyze the source code thoroughly. Cross-reference with the retrieved security knowledge "
        f"and return a complete list of vulnerabilities adhering to the schema."
    )

    if input_data.context_notes:
        prompt += f"\nAdditional Context / Requirements:\n{input_data.context_notes}\n"

    # 5. Call LLM with Structured Output Schema
    try:
        struct_resp = llm.generate_structured(
            prompt=prompt,
            schema=SecurityScanReport,
            system_prompt=SECURITY_SCANNER_SYSTEM_PROMPT,
            temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not struct_resp.is_valid or not struct_resp.parsed_data:
            err_msg = struct_resp.error_message or "Structured output generation failed."
            logger.warning(f"Security scanner structured parsing failed: {err_msg}")
            errors.append(err_msg)
            return SecurityScannerResult(
                findings=[],
                summary="Security scan analysis could not be parsed into structured format.",
                sanitized_code=sanitized_code,
                sanitization_applied=sanitization_applied,
                secrets_detected=secrets_count,
                retrieved_contexts=retrieved_docs,
                raw_llm_response=struct_resp.raw_text,
                provider_name=llm.provider_name,
                model_name=llm.model_name,
                latency_ms=round(elapsed_ms, 2),
                status="error",
                errors=errors,
            )

        report_data: SecurityScanReport = struct_resp.parsed_data

        # 6. Map to SecurityFinding instances
        findings: List[SecurityFinding] = []
        for issue in report_data.findings:
            # Check if this issue maps to any retrieved document
            matched_rule = None
            if issue.cwe_id:
                for doc in retrieved_docs:
                    if doc.cwe.lower() in issue.cwe_id.lower() or issue.cwe_id.lower() in doc.cwe.lower():
                        matched_rule = doc.doc_id
                        break

            findings.append(
                SecurityFinding(
                    title=issue.title,
                    description=issue.description,
                    severity=_map_severity(issue.severity),
                    confidence=_map_confidence(issue.confidence),
                    affected_code=issue.affected_code,
                    cwe_id=issue.cwe_id,
                    reasoning=issue.reasoning,
                    suggested_remediation=issue.suggested_remediation,
                    tool="security_scanner_rag",
                    line_number=issue.line_number,
                    category=issue.category or "security",
                    file=input_data.filename,
                    matched_rule_id=matched_rule,
                )
            )

        return SecurityScannerResult(
            findings=findings,
            summary=report_data.summary,
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            retrieved_contexts=retrieved_docs,
            raw_llm_response=struct_resp.raw_text,
            provider_name=llm.provider_name,
            model_name=llm.model_name,
            latency_ms=round(elapsed_ms, 2),
            status="success",
            errors=[],
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Unexpected error during security scan: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return SecurityScannerResult(
            findings=[],
            summary="Security scan failed due to unexpected error.",
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            retrieved_contexts=retrieved_docs,
            provider_name=llm.provider_name if llm else "unknown",
            model_name=llm.model_name if llm else "unknown",
            latency_ms=round(elapsed_ms, 2),
            status="error",
            errors=errors,
        )


class SecurityScannerAgent:
    """Agent class wrapping RAG-based security scanning."""

    def __init__(
        self,
        vector_store: Optional[SecurityVectorStore] = None,
        provider: Optional[BaseLLMProvider] = None,
    ):
        self._vector_store = vector_store
        self._provider = provider

    def scan(self, input_data: SecurityScannerInput) -> SecurityScannerResult:
        """Scan input source code for security vulnerabilities using RAG."""
        return scan_security(
            input_data=input_data,
            vector_store=self._vector_store,
            provider=self._provider,
        )

    def scan_code(
        self,
        code: str,
        filename: str = "snippet.py",
        top_k_context: int = 3,
        context_notes: Optional[str] = None,
    ) -> SecurityScannerResult:
        """Convenience method accepting raw code string."""
        return self.scan(
            SecurityScannerInput(
                source_code=code,
                filename=filename,
                top_k_context=top_k_context,
                context_notes=context_notes,
            )
        )
