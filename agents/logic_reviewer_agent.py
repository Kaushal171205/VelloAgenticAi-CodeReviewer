"""
agents/logic_reviewer_agent.py — Business Logic & Correctness Reviewer Agent.

Analyzes source code for semantic, correctness, and business-logic flaws that
static analysis linters (e.g. Bandit, flake8) miss.

Key Focus Areas:
  • Incorrect conditions and inverted checks
  • Missing input and boundary validation (negative amounts, zero values, overflow)
  • Unsafe assumptions about object presence or external API states
  • Invalid state transitions (e.g. double refund, operating on cancelled entities)
  • Duplicate operations and missing idempotency
  • Authorization and business-rule mistakes (privilege escalation, tenant leakage)
  • Edge cases (empty payloads, extreme values, concurrent access hazards)
  • Suspicious control flow (early exits leaving inconsistent state, missing rollbacks)

Design:
  • Guarantees secret sanitization prior to LLM interaction.
  • Uses the project's LLM abstraction layer (Gemini / Ollama).
  • Emits structured findings conforming to the common finding schema.
  • Robust error handling: never crashes the caller.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.sanitizer import SanitizerInput, sanitize
from agents.static_analysis_agent import FindingConfidence, FindingSeverity
from llm.llm_client import BaseLLMProvider, get_llm_provider
from tools.language_utils import detect_language

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Schema for LLM Output
# ─────────────────────────────────────────────────────────────────────────────

class LogicIssueSchema(BaseModel):
    """Pydantic model representing a single logic finding parsed from the LLM."""
    title: str = Field(..., description="Concise, descriptive title of the logic/business issue.")
    description: str = Field(..., description="Detailed description of the vulnerability or flaw and its real-world impact.")
    severity: str = Field(..., description="Severity level: 'critical', 'high', 'medium', or 'low'.")
    confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'.")
    affected_code: str = Field(..., description="Specific code snippet or expression containing the flaw.")
    reasoning: str = Field(..., description="Step-by-step reasoning explaining why the logic fails or is unsafe.")
    suggested_remediation: str = Field(..., description="Concrete, actionable code fix or corrected implementation.")
    line_number: Optional[int] = Field(default=None, description="Starting line number of the affected code if identifiable.")
    category: Optional[str] = Field(
        default="business_logic",
        description="Category: 'incorrect_condition', 'missing_validation', 'unsafe_assumption', 'state_transition', 'duplicate_operation', 'authorization_flaw', 'edge_case', or 'control_flow'."
    )


class LogicReviewSchema(BaseModel):
    """Pydantic container for complete review analysis."""
    summary: str = Field(..., description="Executive summary of the logic review findings.")
    findings: List[LogicIssueSchema] = Field(default_factory=list, description="List of detected logic and correctness flaws.")


# ─────────────────────────────────────────────────────────────────────────────
# Agent Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogicFinding:
    """Normalised finding conforming to the project's common finding schema."""
    title: str
    description: str
    severity: FindingSeverity
    confidence: FindingConfidence
    affected_code: str
    reasoning: str
    suggested_remediation: str
    tool: str = "logic_reviewer"
    line_number: Optional[int] = None
    category: str = "business_logic"
    file: str = "snippet.py"

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to standard dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "affected_code": self.affected_code,
            "reasoning": self.reasoning,
            "suggested_remediation": self.suggested_remediation,
            "tool": self.tool,
            "line_number": self.line_number,
            "category": self.category,
            "file": self.file,
        }


@dataclass
class LogicReviewerInput:
    """Input payload for the Logic Reviewer Agent."""
    source_code: str
    filename: str = "snippet.py"
    context_notes: Optional[str] = None
    is_pre_sanitized: bool = False


@dataclass
class LogicReviewResult:
    """Complete structured output from the Logic Reviewer Agent."""
    findings: List[LogicFinding] = field(default_factory=list)
    summary: str = ""
    sanitized_code: str = ""
    sanitization_applied: bool = False
    secrets_detected: int = 0
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
        priority = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
            FindingSeverity.INFO,
        ]
        present = {f.severity for f in self.findings}
        for level in priority:
            if level in present:
                return level
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

LOGIC_REVIEW_SYSTEM_PROMPT = """
You are a Principal Software Engineer and Enterprise Application Security Auditor specializing in deep business logic, concurrency, and correctness review.

Your objective is to identify subtle semantic, architectural, and business logic flaws across any programming language that static linters CANNOT detect.

Audit rigorously for:
1. Incorrect Conditions & Logic Inversion:
   - Inverted conditionals (e.g. `<` when it should be `>`, or `==` instead of `!=`).
   - Off-by-one errors and boundary mistakes.
   - Faulty boolean expressions (`or` vs `and`).
2. Missing Validation & Sanitization:
   - Missing checks for negative amounts, zero, null/None, or exceeding limits.
   - Missing format or type verification.
3. Unsafe Assumptions:
   - Assuming database query, user lookup, or external API call succeeds without checking return status.
   - Assuming records belong to the requesting user without tenant validation.
4. Incorrect State Transitions & Idempotency:
   - Operating on already finalized entities (e.g. refunding an already refunded or cancelled order).
   - Double-spending or duplicate execution hazards.
   - Lack of idempotency keys in financial/critical transactions.
5. Authorization & Business Rule Violations:
   - Horizontal authorization bypass (IDOR).
   - Inconsistent state updates (updating balance after partial failure, missing database rollback/commit boundaries).
6. Edge Cases & Numerical Issues:
   - Floating-point inaccuracy in financial calculations.
   - Empty lists/collections handling.
7. Suspicious Control Flow:
   - Early returns leaving system state inconsistent.
   - Swallowed exceptions or missing compensation logic on failures.

Output Format:
You must provide a clear summary and an array of findings conforming strictly to the requested JSON schema.
For each finding, provide actionable, precise, and verified reasoning with clear remediation code.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _map_severity(raw_severity: str) -> FindingSeverity:
    """Map arbitrary LLM severity string to canonical FindingSeverity."""
    clean = (raw_severity or "").strip().lower()
    mapping = {
        "critical": FindingSeverity.CRITICAL,
        "high": FindingSeverity.HIGH,
        "medium": FindingSeverity.MEDIUM,
        "low": FindingSeverity.LOW,
        "info": FindingSeverity.INFO,
    }
    return mapping.get(clean, FindingSeverity.MEDIUM)


def _map_confidence(raw_confidence: str) -> FindingConfidence:
    """Map arbitrary LLM confidence string to canonical FindingConfidence."""
    clean = (raw_confidence or "").strip().lower()
    mapping = {
        "high": FindingConfidence.HIGH,
        "medium": FindingConfidence.MEDIUM,
        "low": FindingConfidence.LOW,
    }
    return mapping.get(clean, FindingConfidence.HIGH)


# ─────────────────────────────────────────────────────────────────────────────
# Core Review Logic
# ─────────────────────────────────────────────────────────────────────────────

def review_logic(
    input_data: LogicReviewerInput,
    provider: Optional[BaseLLMProvider] = None,
) -> LogicReviewResult:
    """
    Execute business-logic and correctness review on the provided source code.

    Enforces secret sanitization before calling the LLM.

    Args:
        input_data: LogicReviewerInput containing source code and metadata.
        provider: Optional BaseLLMProvider. If None, resolves from application settings.

    Returns:
        LogicReviewResult with structured findings and metrics.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    if not input_data.source_code or not input_data.source_code.strip():
        return LogicReviewResult(
            findings=[],
            summary="No source code provided for logic review.",
            sanitized_code="",
            sanitization_applied=False,
            secrets_detected=0,
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
            if secrets_count > 0:
                logger.info(
                    f"Sanitized {secrets_count} secret(s) from {input_data.filename} before LLM review."
                )
        except Exception as e:
            err = f"Secret sanitization failed: {e}"
            logger.error(err)
            errors.append(err)
            return LogicReviewResult(
                status="error",
                errors=errors,
                sanitized_code="",
                sanitization_applied=False,
            )

    # 2. Resolve LLM Provider
    llm = provider
    if llm is None:
        try:
            llm = get_llm_provider()
        except Exception as e:
            err = f"Failed to initialize LLM provider: {e}"
            logger.error(err)
            errors.append(err)
            return LogicReviewResult(
                status="error",
                errors=errors,
                sanitized_code=sanitized_code,
                sanitization_applied=sanitization_applied,
                secrets_detected=secrets_count,
            )

    # 3. Construct Review Prompt with Line Numbers for Precise Referencing
    lines = sanitized_code.splitlines()
    numbered_code = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))

    _, lang_id = detect_language(input_data.filename)
    fence = f"```{lang_id}"

    prompt = (
        f"Perform an exhaustive business logic and correctness review on the following code:\n\n"
        f"File: {input_data.filename}\n"
        f"{fence}\n{numbered_code}\n```\n"
    )

    if input_data.context_notes:
        prompt += f"\nAdditional Context / Requirements:\n{input_data.context_notes}\n"

    # 4. Invoke LLM with Structured Output Schema
    try:
        struct_resp = llm.generate_structured(
            prompt=prompt,
            schema=LogicReviewSchema,
            system_prompt=LOGIC_REVIEW_SYSTEM_PROMPT,
            temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not struct_resp.is_valid or not struct_resp.parsed_data:
            err_msg = struct_resp.error_message or "Structured output generation failed."
            logger.warning(f"Logic review structured parsing failed: {err_msg}")
            errors.append(err_msg)
            return LogicReviewResult(
                findings=[],
                summary="Analysis could not be parsed into structured format.",
                sanitized_code=sanitized_code,
                sanitization_applied=sanitization_applied,
                secrets_detected=secrets_count,
                raw_llm_response=struct_resp.raw_text,
                provider_name=llm.provider_name,
                model_name=llm.model_name,
                latency_ms=round(elapsed_ms, 2),
                status="error",
                errors=errors,
            )

        review_data: LogicReviewSchema = struct_resp.parsed_data

        # 5. Normalise into Canonical LogicFinding objects
        findings: List[LogicFinding] = []
        for issue in review_data.findings:
            findings.append(
                LogicFinding(
                    title=issue.title,
                    description=issue.description,
                    severity=_map_severity(issue.severity),
                    confidence=_map_confidence(issue.confidence),
                    affected_code=issue.affected_code,
                    reasoning=issue.reasoning,
                    suggested_remediation=issue.suggested_remediation,
                    tool="logic_reviewer",
                    line_number=issue.line_number,
                    category=issue.category or "business_logic",
                    file=input_data.filename,
                )
            )

        return LogicReviewResult(
            findings=findings,
            summary=review_data.summary,
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            raw_llm_response=struct_resp.raw_text,
            provider_name=llm.provider_name,
            model_name=llm.model_name,
            latency_ms=round(elapsed_ms, 2),
            status="success",
            errors=[],
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Unexpected error during logic review execution: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return LogicReviewResult(
            findings=[],
            summary="Review failed due to unexpected error.",
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            provider_name=llm.provider_name if llm else "unknown",
            model_name=llm.model_name if llm else "unknown",
            latency_ms=round(elapsed_ms, 2),
            status="error",
            errors=errors,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class Interface for Future LangGraph / Agent Workflows
# ─────────────────────────────────────────────────────────────────────────────

class LogicReviewerAgent:
    """Agent class wrapper for logic and business correctness analysis."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._provider = provider

    def review(self, input_data: LogicReviewerInput) -> LogicReviewResult:
        """Analyze code for business logic and correctness issues."""
        return review_logic(input_data, provider=self._provider)

    def review_code(
        self,
        code: str,
        filename: str = "snippet.py",
        context_notes: Optional[str] = None,
    ) -> LogicReviewResult:
        """Convenience method accepting raw code string."""
        return self.review(
            LogicReviewerInput(
                source_code=code,
                filename=filename,
                context_notes=context_notes,
            )
        )
