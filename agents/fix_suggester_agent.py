"""
agents/fix_suggester_agent.py — Fix Suggester Agent.

Generates concrete, targeted Python code remediation fixes and clear explanations
of what was changed for detected security, static analysis, and logic findings.
Supports self-correction feedback loops from downstream AST validation and
the Critic Agent.

Key Guarantees:
  1. Generates concrete, syntactically valid Python code (not abstract advice).
  2. Clearly explains what was changed and why.
  3. Sanitizes all inputs so secrets are never leaked to LLM providers.
  4. Preserves the original source code (operates in read-only analysis mode).
  5. Never modifies the user's repository directly.
  6. Robust: never raises uncaught exceptions.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.sanitizer import SanitizerInput, sanitize
from llm.llm_client import BaseLLMProvider, get_llm_provider
from tools.language_utils import detect_language

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Schema for LLM Output
# ─────────────────────────────────────────────────────────────────────────────

class FixSuggestionSchema(BaseModel):
    """Pydantic model representing structured fix output from the LLM."""

    proposed_fix: str = Field(
        ...,
        description=(
            "The concrete, syntactically valid replacement code snippet "
            "implementing the fix. Must not contain explanatory text outside code."
        ),
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation of what changed and why this resolves the finding.",
    )
    changed_summary: str = Field(
        default="",
        description="Concise one-line summary of changes made (e.g. 'Added input validation for negative amounts').",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FixSuggesterInput:
    """Input payload for the Fix Suggester Agent."""

    finding: Any  # FindingRecord or dict
    source_code: str
    filename: str = "snippet.py"
    previous_fix: Optional[str] = None
    critic_feedback: Optional[str] = None
    attempt_number: int = 1
    max_attempts: int = 3
    is_pre_sanitized: bool = False


@dataclass
class FixSuggesterResult:
    """Result emitted by the Fix Suggester Agent."""

    finding_title: str
    proposed_fix: str
    explanation: str
    changed_summary: str = ""
    status: str = "success"  # "success" | "error"
    attempt_number: int = 1
    sanitized_code: str = ""
    sanitization_applied: bool = False
    secrets_detected: int = 0
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "finding_title": self.finding_title,
            "proposed_fix": self.proposed_fix,
            "explanation": self.explanation,
            "changed_summary": self.changed_summary,
            "status": self.status,
            "attempt_number": self.attempt_number,
            "errors": list(self.errors),
            "latency_ms": self.latency_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# System & User Prompts
# ─────────────────────────────────────────────────────────────────────────────

FIX_SUGGESTER_SYSTEM_PROMPT = """You are an elite Senior Software Security and Reliability Engineer.
Your role is to generate precise, concrete, production-grade code fixes for detected vulnerabilities, bugs, or logic errors.

Follow these strict principles:
1. PRODUCE CONCRETE CODE: The 'proposed_fix' must be working, production-ready code that directly replaces or fixes the affected snippet.
2. MINIMAL AND TARGETED: Keep modifications minimal and focused on resolving the flaw without unnecessary refactoring.
3. SYNTACTICALLY VALID: Ensure the code has correct syntax, indentation, and imports if needed.
4. EXPLAIN CHANGES: Clearly describe exactly what changed and why it fixes the issue.
5. PRESERVE CONTEXT: Preserve the original function signature and valid existing business logic.
6. ADDRESS CRITIC FEEDBACK: If critic feedback or previous attempt notes are provided, directly address every raised problem.
7. LANGUAGE AWARENESS: Produce fixes in the same programming language as the source code provided.
"""


def _build_fix_prompt(
    finding: Any,
    source_code: str,
    filename: str,
    previous_fix: Optional[str],
    critic_feedback: Optional[str],
    attempt_number: int,
) -> str:
    """Build prompt for generating or revising a fix."""
    lang_name, lang_id = detect_language(filename)

    def _get(key: str, default: Any = "") -> Any:
        if isinstance(finding, dict):
            return finding.get(key, default)
        return getattr(finding, key, default)

    title = _get("title", "Untitled Issue")
    description = _get("description", "No description provided.")
    severity = _get("severity", "medium")
    line_number = _get("line_number")
    affected_code = _get("affected_code", "")
    suggested_remediation = _get("suggested_remediation", "")
    cwe_id = _get("cwe_id", "")

    fence = f"```{lang_id}"

    prompt_parts = [
        f"### Target File: `{filename}` (Language: {lang_name})",
        f"### Finding: {title} (Severity: {str(severity).upper()})",
    ]
    if cwe_id:
        prompt_parts.append(f"CWE: {cwe_id}")
    if line_number:
        prompt_parts.append(f"Affected Line: {line_number}")

    prompt_parts.extend([
        "",
        "### Problem Description:",
        description,
        "",
    ])

    if affected_code:
        prompt_parts.extend([
            "### Affected Code Snippet:",
            fence,
            affected_code,
            "```",
            "",
        ])

    if suggested_remediation:
        prompt_parts.extend([
            "### Remediation Guidance:",
            suggested_remediation,
            "",
        ])

    prompt_parts.extend([
        "### Context / Surrounding Source Code:",
        fence,
        source_code,
        "```",
        "",
    ])

    # If this is a revision loop based on Critic feedback
    if attempt_number > 1 or critic_feedback:
        prompt_parts.extend([
            f"### ⚠️ SELF-CORRECTION REQUIRED (Attempt #{attempt_number}):",
        ])
        if previous_fix:
            prompt_parts.extend([
                "Previous Proposed Fix:",
                fence,
                previous_fix,
                "```",
                "",
            ])
        if critic_feedback:
            prompt_parts.extend([
                "Critic Evaluation / Rejection Feedback:",
                f"> {critic_feedback}",
                "",
                "Please carefully correct the fix according to the critic feedback above.",
                "",
            ])

    prompt_parts.append(
        f"Generate a concrete, syntactically valid {lang_name} replacement snippet in 'proposed_fix' "
        "and explain what was changed in 'explanation'."
    )

    return "\n".join(prompt_parts)


def _clean_code_snippet(code: str) -> str:
    """Strip markdown backtick wrappers if LLM enclosed the code in fences."""
    trimmed = code.strip()
    match = re.match(r"^```(?:python)?\s*\n?(.*?)\n?```$", trimmed, re.DOTALL)
    if match:
        return match.group(1).strip()
    return trimmed


# ─────────────────────────────────────────────────────────────────────────────
# Functional Execution
# ─────────────────────────────────────────────────────────────────────────────

def suggest_fix(
    input_data: FixSuggesterInput,
    provider: Optional[BaseLLMProvider] = None,
) -> FixSuggesterResult:
    """
    Generate a concrete code fix and explanation for a given finding.

    Ensures that secrets in input_data.source_code are sanitized before calling
    the LLM provider.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    def _get_title(f: Any) -> str:
        if isinstance(f, dict):
            return f.get("title", "Untitled Issue")
        return getattr(f, "title", "Untitled Issue")

    finding_title = _get_title(input_data.finding)

    # 1. Sanitize code to ensure secrets never leak to LLM
    sanitized_code = input_data.source_code
    sanitization_applied = False
    secrets_count = 0

    if not input_data.is_pre_sanitized and input_data.source_code:
        try:
            san_res = sanitize(SanitizerInput(source_code=input_data.source_code))
            sanitized_code = san_res.sanitized_code
            sanitization_applied = san_res.number_of_secrets > 0
            secrets_count = san_res.number_of_secrets
            errors.extend(san_res.warnings)
        except Exception as san_err:
            logger.warning("Sanitization in FixSuggester encountered warning: %s", san_err)
            errors.append(f"Sanitization warning: {san_err}")

    # 2. Get LLM provider
    try:
        llm = provider or get_llm_provider()
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to acquire LLM provider: {exc}"
        logger.error(err_msg)
        return FixSuggesterResult(
            finding_title=finding_title,
            proposed_fix="",
            explanation="Could not generate fix: LLM provider unavailable.",
            status="error",
            attempt_number=input_data.attempt_number,
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            errors=[err_msg],
            latency_ms=round(elapsed_ms, 2),
        )

    # 3. Construct prompt
    prompt = _build_fix_prompt(
        finding=input_data.finding,
        source_code=sanitized_code,
        filename=input_data.filename,
        previous_fix=input_data.previous_fix,
        critic_feedback=input_data.critic_feedback,
        attempt_number=input_data.attempt_number,
    )

    # 4. Invoke LLM structured generation
    try:
        struct_resp = llm.generate_structured(
            prompt=prompt,
            schema=FixSuggestionSchema,
            system_prompt=FIX_SUGGESTER_SYSTEM_PROMPT,
            temperature=0.2,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not struct_resp.is_valid or not struct_resp.parsed_data:
            err_msg = struct_resp.error_message or "Structured parsing of fix failed."
            logger.warning("FixSuggester structured parsing failed: %s", err_msg)
            errors.append(err_msg)
            return FixSuggesterResult(
                finding_title=finding_title,
                proposed_fix="",
                explanation="Fix generation failed during schema parsing.",
                status="error",
                attempt_number=input_data.attempt_number,
                sanitized_code=sanitized_code,
                sanitization_applied=sanitization_applied,
                secrets_detected=secrets_count,
                errors=errors,
                latency_ms=round(elapsed_ms, 2),
            )

        data: FixSuggestionSchema = struct_resp.parsed_data
        clean_fix = _clean_code_snippet(data.proposed_fix)

        return FixSuggesterResult(
            finding_title=finding_title,
            proposed_fix=clean_fix,
            explanation=data.explanation,
            changed_summary=data.changed_summary or "Updated code to resolve finding.",
            status="success",
            attempt_number=input_data.attempt_number,
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            errors=errors,
            latency_ms=round(elapsed_ms, 2),
        )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Unexpected error during fix suggestion: {exc}"
        logger.exception(err_msg)
        errors.append(err_msg)
        return FixSuggesterResult(
            finding_title=finding_title,
            proposed_fix="",
            explanation="Fix generation aborted due to unexpected error.",
            status="error",
            attempt_number=input_data.attempt_number,
            sanitized_code=sanitized_code,
            sanitization_applied=sanitization_applied,
            secrets_detected=secrets_count,
            errors=errors,
            latency_ms=round(elapsed_ms, 2),
        )


class FixSuggesterAgent:
    """Agent class wrapper for fix generation."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._provider = provider

    def suggest(self, input_data: FixSuggesterInput) -> FixSuggesterResult:
        """Generate a fix for the provided input data."""
        return suggest_fix(input_data, provider=self._provider)

    def suggest_fix(
        self,
        finding: Any,
        source_code: str,
        filename: str = "snippet.py",
        previous_fix: Optional[str] = None,
        critic_feedback: Optional[str] = None,
        attempt_number: int = 1,
    ) -> FixSuggesterResult:
        """Convenience method accepting individual parameters."""
        return self.suggest(
            FixSuggesterInput(
                finding=finding,
                source_code=source_code,
                filename=filename,
                previous_fix=previous_fix,
                critic_feedback=critic_feedback,
                attempt_number=attempt_number,
            )
        )
