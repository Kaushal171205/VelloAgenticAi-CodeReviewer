"""
agents/critic_agent.py — Critic Agent for Proposed Fix Evaluation.

Evaluates proposed code remediation fixes against a strict correctness,
security, and reliability rubric:
  1. Syntax Validity: Evaluated first via AST Validator (zero-cost rejection).
  2. Finding Resolution: Does the fix genuinely eliminate the reported issue?
  3. No Regressions: Does it avoid breaking valid functionality or introducing
     new security flaws?
  4. Minimalism: Is the fix focused and proportionate?
  5. Explanation Accuracy: Is the explanation truthful and helpful?

If a proposed fix is deemed unacceptable, the Critic provides specific,
actionable feedback to guide the Fix Suggester in self-correction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from llm.llm_client import BaseLLMProvider, get_llm_provider
from tools.ast_validator import ASTValidationResult, validate_python_syntax
from tools.language_utils import detect_language, is_python, validate_syntax

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Schema for LLM Output
# ─────────────────────────────────────────────────────────────────────────────

class CriticEvaluationSchema(BaseModel):
    """Pydantic model representing structured critique from the LLM."""

    is_acceptable: bool = Field(
        ...,
        description="True if the fix is correct, safe, and ready to recommend; False if it requires revision.",
    )
    feedback: str = Field(
        ...,
        description="Detailed constructive critique. If rejected, clearly explain what is flawed and how to correct it.",
    )
    meets_rubric: bool = Field(
        ...,
        description="Whether the fix meets the core requirements of resolving the vulnerability cleanly.",
    )
    potential_regressions: List[str] = Field(
        default_factory=list,
        description="List of potential regressions, side effects, or new issues introduced by the fix.",
    )
    specific_issues: List[str] = Field(
        default_factory=list,
        description="Specific defects found in the proposed fix (e.g. 'Missing return statement', 'Does not validate zero').",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriticInput:
    """Input payload for the Critic Agent."""

    finding: Any  # FindingRecord or dict
    original_code: str
    proposed_fix: str
    explanation: str = ""
    ast_validation: Optional[ASTValidationResult] = None
    attempt_number: int = 1
    max_attempts: int = 3
    filename: str = "snippet.py"


@dataclass
class CriticResult:
    """Result emitted by the Critic Agent."""

    is_acceptable: bool
    feedback: str
    specific_issues: List[str] = field(default_factory=list)
    potential_regressions: List[str] = field(default_factory=list)
    meets_rubric: bool = False
    ast_valid: bool = True
    status: str = "success"  # "success" | "error"
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_acceptable": self.is_acceptable,
            "feedback": self.feedback,
            "specific_issues": list(self.specific_issues),
            "potential_regressions": list(self.potential_regressions),
            "meets_rubric": self.meets_rubric,
            "ast_valid": self.ast_valid,
            "status": self.status,
            "errors": list(self.errors),
            "latency_ms": self.latency_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# System & User Prompts
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a Principal Software Engineer and Security Auditor acting as an impartial code review critic.
Your role is to rigorously evaluate whether a proposed code fix genuinely resolves a reported security vulnerability, bug, or logic flaw without introducing new problems.

Evaluation Rubric:
1. DOES IT RESOLVE THE FLAW? The fix must address the core root cause of the finding. Partial or cosmetic changes are unacceptable.
2. REGRESSION HAZARDS: Does the fix break existing behavior, alter required return types, or drop essential edge-case handling?
3. NEW VULNERABILITIES: Does the fix introduce any new security vulnerabilities (e.g. injection, denial of service, race conditions)?
4. MINIMALITY & SAFETY: Is the change clean, readable, idiomatic, and scoped strictly to the problem?
5. ACCURACY: Does the explanation accurately match what the code changes accomplish?

Be demanding but constructive. If the fix is solid, mark is_acceptable=True. If it has flaws, mark is_acceptable=False and provide actionable feedback explaining precisely how to correct it.
"""


def _build_critic_prompt(
    finding: Any,
    original_code: str,
    proposed_fix: str,
    explanation: str,
    filename: str,
    attempt_number: int,
) -> str:
    """Construct evaluation prompt for the Critic LLM."""
    lang_name, lang_id = detect_language(filename)
    fence = f"```{lang_id}"

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

    prompt_parts = [
        f"### Target File: `{filename}` (Language: {lang_name})",
        f"### Finding Under Evaluation: {title} (Severity: {str(severity).upper()})",
    ]
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
            "### Original Vulnerable/Flawed Snippet:",
            fence,
            affected_code,
            "```",
            "",
        ])

    if suggested_remediation:
        prompt_parts.extend([
            "### Recommended Guidance:",
            suggested_remediation,
            "",
        ])

    prompt_parts.extend([
        "### Surrounding Code Context:",
        fence,
        original_code,
        "```",
        "",
        f"### Proposed Fix (Attempt #{attempt_number}):",
        fence,
        proposed_fix,
        "```",
        "",
        "### Author's Explanation:",
        explanation or "No explanation provided.",
        "",
        "### Task:",
        "Evaluate the proposed fix against the rubric. Set 'is_acceptable' to True only if the fix is correct, "
        "safe, and ready. If not acceptable, provide specific actionable critique in 'feedback'.",
    ])

    return "\n".join(prompt_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Functional Execution
# ─────────────────────────────────────────────────────────────────────────────

def critique_fix(
    input_data: CriticInput,
    provider: Optional[BaseLLMProvider] = None,
) -> CriticResult:
    """
    Critique a proposed fix.

    Step 1: Check Python syntax using AST validation. If invalid, reject immediately
    without an LLM call.
    Step 2: If syntax is valid, evaluate semantic correctness, security, and
    regressions via the Critic LLM.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    # 1. Syntax Validation Gate (fast rejection for Python only; other languages pass through)
    ast_val = input_data.ast_validation
    if ast_val is None:
        if is_python(input_data.filename):
            ast_val = validate_python_syntax(input_data.proposed_fix, filename=input_data.filename)
        else:
            # For non-Python languages, skip static syntax check — LLM handles it
            from tools.ast_validator import ASTValidationResult as _ASTVR  # noqa
            ast_val = _ASTVR(is_valid=True)

    if not ast_val.is_valid:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        lang_name, _ = detect_language(input_data.filename)
        syntax_err_msg = "; ".join(ast_val.errors) if ast_val.errors else f"Invalid {lang_name} syntax."
        feedback = (
            f"The proposed fix contains {lang_name} syntax errors: {syntax_err_msg}. "
            "Please fix the syntax and ensure all indentation, parentheses, and keywords are valid."
        )
        return CriticResult(
            is_acceptable=False,
            feedback=feedback,
            specific_issues=[syntax_err_msg],
            potential_regressions=["Syntax error prevents execution."],
            meets_rubric=False,
            ast_valid=False,
            status="success",
            errors=[],
            latency_ms=round(elapsed_ms, 2),
        )

    # If proposed fix is completely empty
    if not input_data.proposed_fix.strip():
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return CriticResult(
            is_acceptable=False,
            feedback="The proposed fix is empty. A concrete code replacement is required.",
            specific_issues=["Empty fix snippet."],
            meets_rubric=False,
            ast_valid=True,
            status="success",
            latency_ms=round(elapsed_ms, 2),
        )

    # 2. Acquire LLM provider
    try:
        llm = provider or get_llm_provider()
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to acquire LLM provider for Critic: {exc}"
        logger.error(err_msg)
        return CriticResult(
            is_acceptable=False,
            feedback="Critic evaluation unavailable (LLM connection error).",
            status="error",
            errors=[err_msg],
            latency_ms=round(elapsed_ms, 2),
        )

    # 3. Construct prompt
    prompt = _build_critic_prompt(
        finding=input_data.finding,
        original_code=input_data.original_code,
        proposed_fix=input_data.proposed_fix,
        explanation=input_data.explanation,
        filename=input_data.filename,
        attempt_number=input_data.attempt_number,
    )

    # 4. Invoke Critic LLM
    try:
        struct_resp = llm.generate_structured(
            prompt=prompt,
            schema=CriticEvaluationSchema,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not struct_resp.is_valid or not struct_resp.parsed_data:
            err_msg = struct_resp.error_message or "Structured parsing of critic response failed."
            logger.warning("Critic structured parsing failed: %s", err_msg)
            # Default to requiring revision if critique cannot be confirmed
            return CriticResult(
                is_acceptable=False,
                feedback="Could not verify fix correctness due to evaluation parsing error.",
                status="error",
                errors=[err_msg],
                latency_ms=round(elapsed_ms, 2),
            )

        data: CriticEvaluationSchema = struct_resp.parsed_data

        return CriticResult(
            is_acceptable=data.is_acceptable,
            feedback=data.feedback,
            specific_issues=data.specific_issues,
            potential_regressions=data.potential_regressions,
            meets_rubric=data.meets_rubric,
            ast_valid=True,
            status="success",
            errors=errors,
            latency_ms=round(elapsed_ms, 2),
        )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Unexpected error during critic evaluation: {exc}"
        logger.exception(err_msg)
        return CriticResult(
            is_acceptable=False,
            feedback="Critic evaluation failed due to unexpected error.",
            status="error",
            errors=[err_msg],
            latency_ms=round(elapsed_ms, 2),
        )


class CriticAgent:
    """Agent class wrapper for proposed fix evaluation."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._provider = provider

    def critique(self, input_data: CriticInput) -> CriticResult:
        """Evaluate a proposed fix."""
        return critique_fix(input_data, provider=self._provider)

    def evaluate_fix(
        self,
        finding: Any,
        original_code: str,
        proposed_fix: str,
        explanation: str = "",
        ast_validation: Optional[ASTValidationResult] = None,
        attempt_number: int = 1,
        filename: str = "snippet.py",
    ) -> CriticResult:
        """Convenience method accepting individual parameters."""
        return self.critique(
            CriticInput(
                finding=finding,
                original_code=original_code,
                proposed_fix=proposed_fix,
                explanation=explanation,
                ast_validation=ast_validation,
                attempt_number=attempt_number,
                filename=filename,
            )
        )
