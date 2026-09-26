"""
graph/workflow.py — LangGraph Code Review Pipeline.

Full execution graph:

  Input
    │
    ▼
  sanitizer_node          Strips secrets from raw_code → sanitized_code
    │
    ▼
  context_retriever_node  Retrieves semantically related functions from
    │                     the shared ChromaDB index (optional; skipped if
    │                     no changed_functions in state)
    ▼
  supervisor_node         Decides which review agents to run based on the
    │                     sanitized code and settings
    │
  ┌─────────────────────────────┐
  │          │                  │
  ▼          ▼                  ▼
static_node  security_node  logic_node
  │          │                  │
  └─────────────────────────────┘
    │
    ▼
  aggregator_node         Merges all findings into a uniform list
    │
    ▼
  classifier_node         Groups findings by severity; computes summary
    │
    ▼
  report_node             Renders the final markdown report

Optional critic/human-in-the-loop nodes are wired but do not block the
default linear flow unless `needs_revision` or `human_decision` is set.

Usage
-----
  from graph.workflow import build_review_graph, run_review

  # Build once (expensive: compiles the graph)
  app = build_review_graph()

  # Run a review
  result = run_review(app, raw_code="import os; os.system(input())")
  print(result["final_report"])
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from storage.checkpoint_db import get_sqlite_checkpointer

from agents.aggregator_agent import (
    AggregatorInput,
    AggregatorResult,
    FindingRecord,
    aggregate,
)
from agents.critic_agent import CriticInput, CriticResult, critique_fix
from agents.fix_suggester_agent import (
    FixSuggesterInput,
    FixSuggesterResult,
    suggest_fix,
)
from agents.logic_reviewer_agent import (
    LogicReviewResult,
    LogicReviewerInput,
    review_logic,
)
from agents.sanitizer import SanitizerInput, SanitizerResult, sanitize
from agents.security_scanner_agent import (
    SecurityScannerInput,
    SecurityScannerResult,
    scan_security,
)
from agents.severity_classifier_agent import (
    ClassifierInput,
    ClassifierResult,
    classify,
)
from agents.static_analysis_agent import (
    StaticAnalysisInput,
    StaticAnalysisResult,
    analyse,
)
from graph.state import AgentState
from tools.ast_validator import ASTValidationResult, validate_python_syntax
from tools.language_utils import detect_language, is_python, validate_syntax

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Severity ordering (higher index = higher priority)
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_ORDER: List[str] = ["info", "low", "medium", "high", "critical"]

_MAX_REVISIONS: int = 2  # guard against infinite critic loops


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _err(state: AgentState, msg: str) -> List[str]:
    """Return the updated errors list with a new message appended."""
    existing: List[str] = state.get("errors") or []
    return existing + [msg]


def _severity_rank(s: str) -> int:
    try:
        return _SEVERITY_ORDER.index(s.lower())
    except ValueError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Sanitizer
# ─────────────────────────────────────────────────────────────────────────────

def sanitizer_node(state: AgentState) -> AgentState:
    """
    Strips secrets from raw_code.

    Reads:  raw_code, filename
    Writes: sanitized_code, sanitization_applied, secrets_detected,
            sanitizer_errors
    """
    raw: str = state.get("raw_code") or ""
    filename: str = state.get("filename") or "input.txt"

    if not raw.strip():
        return {
            "sanitized_code": "",
            "sanitization_applied": False,
            "secrets_detected": 0,
            "sanitizer_errors": ["No input code provided to sanitizer."],
        }

    try:
        result: SanitizerResult = sanitize(
            SanitizerInput(source_code=raw, filename=filename)
        )
        return {
            "sanitized_code":       result.sanitized_code,
            "sanitization_applied": not result.is_clean,
            "secrets_detected":     result.number_of_secrets,
            "sanitizer_errors":     result.warnings,
        }
    except Exception as exc:
        logger.exception("sanitizer_node failed: %s", exc)
        return {
            "sanitized_code": raw,  # fallback — pass raw code through
            "sanitization_applied": False,
            "secrets_detected": 0,
            "sanitizer_errors": [f"Sanitizer error: {exc}"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Context Retriever (optional / best-effort)
# ─────────────────────────────────────────────────────────────────────────────

def context_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves semantically related functions from the shared ChromaDB index.

    This node is optional — if no changed_functions are present in state or
    the index is unavailable the node is a no-op and does not fail.

    Reads:  changed_functions
    Writes: related_context, context_prompt_snippet
    """
    changed_functions = state.get("changed_functions")

    if not changed_functions:
        logger.debug("context_retriever_node: no changed_functions, skipping.")
        return {
            "related_context": None,
            "context_prompt_snippet": "",
        }

    try:
        from context.context_retriever import ContextRetriever, RetrievedContextPackage  # noqa: PLC0415

        retriever = ContextRetriever()
        pkg: RetrievedContextPackage = retriever.retrieve_for_changes(changed_functions)
        snippet = pkg.to_llm_prompt_snippet()
        return {
            "related_context": pkg,
            "context_prompt_snippet": snippet,
        }
    except Exception as exc:
        logger.warning("context_retriever_node failed (non-fatal): %s", exc)
        return {
            "related_context": None,
            "context_prompt_snippet": "",
            "errors": _err(state, f"Context retrieval warning: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Supervisor
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    """
    Decides which review agents to activate.

    Rules:
    - Static analysis always runs (it is fast and local).
    - Security scanner runs when LLM is available (always for now).
    - Logic reviewer runs when LLM is available (always for now).

    The supervisor can be extended to accept policy flags (e.g.
    skip_security=True) via the initial state.

    Reads:  sanitized_code
    Writes: run_static, run_security, run_logic
    """
    code: str = state.get("sanitized_code") or ""
    has_code = bool(code.strip())

    return {
        "run_static":   has_code,
        "run_security": has_code,
        "run_logic":    has_code,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional router — after supervisor
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_router(state: AgentState) -> List[str]:
    """
    Returns the list of review nodes that should run in parallel.
    LangGraph supports returning a list from a conditional edge to fan out.
    """
    targets: List[str] = []
    if state.get("run_static"):
        targets.append("static_node")
    if state.get("run_security"):
        targets.append("security_node")
    if state.get("run_logic"):
        targets.append("logic_node")
    if not targets:
        # Nothing to run — jump straight to aggregator
        targets.append("aggregator_node")
    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Node 4a — Static Analysis
# ─────────────────────────────────────────────────────────────────────────────

def static_node(state: AgentState) -> AgentState:
    """
    Runs Bandit-based static analysis on sanitized_code.

    NOTE: Bandit is Python-only. For other languages this node returns an
    empty result so the LLM-based agents carry the full analysis load.

    Reads:  sanitized_code, filename
    Writes: static_findings
    """
    code: str = state.get("sanitized_code") or ""
    filename: str = state.get("filename") or "input.txt"

    if not code.strip():
        return {"static_findings": StaticAnalysisResult()}

    # Bandit only works on Python — skip silently for other languages
    if not is_python(filename):
        lang_name, _ = detect_language(filename)
        logger.info(
            "static_node: skipping Bandit for %s file (%s) — LLM agents handle security analysis.",
            lang_name, filename,
        )
        return {
            "static_findings": StaticAnalysisResult(
                filename=filename,
                errors=[],
            )
        }

    try:
        result: StaticAnalysisResult = analyse(
            StaticAnalysisInput(source_code=code, filename=filename)
        )
        return {"static_findings": result}
    except Exception as exc:
        logger.exception("static_node failed: %s", exc)
        return {
            "static_findings": StaticAnalysisResult(errors=[str(exc)]),
            "errors": _err(state, f"Static analysis error: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4b — Security Scanner (RAG)
# ─────────────────────────────────────────────────────────────────────────────

def security_node(state: AgentState) -> AgentState:
    """
    Runs RAG-based security scanning on sanitized_code.

    Reads:  sanitized_code, filename, context_prompt_snippet
    Writes: security_findings
    """
    code: str = state.get("sanitized_code") or ""
    filename: str = state.get("filename") or "input.py"
    context_notes: str = state.get("context_prompt_snippet") or ""

    if not code.strip():
        return {"security_findings": SecurityScannerResult()}

    try:
        result: SecurityScannerResult = scan_security(
            SecurityScannerInput(
                source_code=code,
                filename=filename,
                context_notes=context_notes or None,
                is_pre_sanitized=True,   # sanitizer already ran
            )
        )
        return {"security_findings": result}
    except Exception as exc:
        logger.exception("security_node failed: %s", exc)
        return {
            "security_findings": SecurityScannerResult(errors=[str(exc)]),
            "errors": _err(state, f"Security scanner error: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4c — Logic Reviewer
# ─────────────────────────────────────────────────────────────────────────────

def logic_node(state: AgentState) -> AgentState:
    """
    Runs LLM-based business logic review on sanitized_code.

    Reads:  sanitized_code, filename, context_prompt_snippet
    Writes: logic_findings
    """
    code: str = state.get("sanitized_code") or ""
    filename: str = state.get("filename") or "input.py"
    context_notes: str = state.get("context_prompt_snippet") or ""

    if not code.strip():
        return {"logic_findings": LogicReviewResult()}

    try:
        result: LogicReviewResult = review_logic(
            LogicReviewerInput(
                source_code=code,
                filename=filename,
                context_notes=context_notes or None,
                is_pre_sanitized=True,   # sanitizer already ran
            )
        )
        return {"logic_findings": result}
    except Exception as exc:
        logger.exception("logic_node failed: %s", exc)
        return {
            "logic_findings": LogicReviewResult(errors=[str(exc)]),
            "errors": _err(state, f"Logic reviewer error: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Aggregator (delegates to AggregatorAgent)
# ─────────────────────────────────────────────────────────────────────────────

def aggregator_node(state: AgentState) -> AgentState:
    """
    Merges, normalises, and deduplicates findings from all review agents
    using the standalone AggregatorAgent.

    Reads:  static_findings, security_findings, logic_findings, filename
    Writes: aggregated_findings (List[FindingRecord]), total_findings
    """
    filename: str = state.get("filename") or "input.py"

    try:
        agg_result: AggregatorResult = aggregate(
            AggregatorInput(
                static_result=state.get("static_findings"),
                security_result=state.get("security_findings"),
                logic_result=state.get("logic_findings"),
                filename=filename,
            )
        )
        errors = list(state.get("errors") or [])
        errors.extend(agg_result.errors)
        return {
            "aggregated_findings": agg_result.findings,
            "total_findings":      agg_result.total_output,
            "errors":              errors,
        }
    except Exception as exc:
        logger.exception("aggregator_node failed: %s", exc)
        return {
            "aggregated_findings": [],
            "total_findings":      0,
            "errors": _err(state, f"Aggregator error: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Severity Classifier (delegates to SeverityClassifierAgent)
# ─────────────────────────────────────────────────────────────────────────────

def classifier_node(state: AgentState) -> AgentState:
    """
    Re-classifies the severity of every FindingRecord using the rubric defined
    in SeverityClassifierAgent — does NOT blindly trust source-agent severity.

    Reads:  aggregated_findings
    Writes: classified_findings, highest_severity, severity_summary,
            total_findings (updated after dedup)
    """
    findings: List[FindingRecord] = state.get("aggregated_findings") or []

    try:
        clf_result: ClassifierResult = classify(ClassifierInput(findings=findings))
        errors = list(state.get("errors") or [])
        errors.extend(clf_result.errors)
        return {
            "classified_findings": clf_result.classified_findings,
            "severity_summary":    clf_result.severity_summary,
            "highest_severity":    clf_result.highest_severity,
            "total_findings":      clf_result.total,
            "errors":              errors,
        }
    except Exception as exc:
        logger.exception("classifier_node failed: %s", exc)
        return {
            "classified_findings": {},
            "severity_summary":    {},
            "highest_severity":    None,
            "errors": _err(state, f"Classifier error: {exc}"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def _severity_emoji(sev: str) -> str:
    return {
        "critical": "🔴",
        "high":     "🟠",
        "medium":   "🟡",
        "low":      "🔵",
        "info":     "⚪",
    }.get(sev, "⚫")


def report_node(state: AgentState) -> AgentState:
    """
    Renders the final markdown report combining all pipeline outputs.
    Now uses FindingRecord objects (with provenance / merge_notes) from
    the classifier_node output.

    Reads:  classified_findings (Dict[str, List[FindingRecord]]),
            severity_summary, highest_severity, context_prompt_snippet,
            sanitization_applied, secrets_detected, total_findings,
            errors, filename
    Writes: final_report
    """
    filename: str = state.get("filename") or "input.py"
    total: int = state.get("total_findings") or 0
    highest: str = state.get("highest_severity") or "none"
    summary: Dict[str, int] = state.get("severity_summary") or {}
    classified: Dict[str, Any] = state.get("classified_findings") or {}
    sanitized: bool = state.get("sanitization_applied") or False
    secrets: int = state.get("secrets_detected") or 0
    errors: List[str] = state.get("errors") or []
    context_snippet: str = state.get("context_prompt_snippet") or ""

    lines: List[str] = [
        f"# 🔍 Code Review Report — `{filename}`",
        "",
        "---",
        "",
        "## 📊 Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Findings | **{total}** |",
        f"| Highest Severity | **{_severity_emoji(highest)} {highest.upper()}** |",
        f"| Secrets Detected | `{secrets}` |",
        f"| Sanitization Applied | {'✅ Yes' if sanitized else '❌ No'} |",
    ]

    for level in reversed(_SEVERITY_ORDER):
        count = summary.get(level, 0)
        if count:
            lines.append(f"| {_severity_emoji(level)} {level.capitalize()} | {count} |")

    lines += ["", "---", ""]

    # ── Findings by severity ──────────────────────────────────────────────────
    if total == 0:
        lines += [
            "## ✅ No Issues Found",
            "",
            "The code review pipeline completed without detecting any findings.",
        ]
    else:
        lines += ["## 🚨 Findings", ""]
        for level in reversed(_SEVERITY_ORDER):
            bucket: List[Any] = classified.get(level, [])
            if not bucket:
                continue
            lines += [
                f"### {_severity_emoji(level)} {level.capitalize()} ({len(bucket)})",
                "",
            ]
            for idx, f in enumerate(bucket, 1):
                # Support both FindingRecord objects and plain dicts
                def _get(key: str, default: Any = "") -> Any:
                    if isinstance(f, dict):
                        return f.get(key, default)
                    return getattr(f, key, default)

                # Provenance: multi-tool label
                tools = _get("source_tools", None)
                if isinstance(tools, list) and len(tools) > 1:
                    tool_str = ", ".join(f"`{t}`" for t in tools)
                else:
                    tool_str = f"`{_get('primary_tool') or _get('tool', 'unknown')}`"

                # Reclassification note
                orig_sev = _get("reported_severity", level)
                reclass_note = ""
                if orig_sev and orig_sev != level:
                    reclass_note = f" *(reclassified from `{orig_sev}`)*"

                lines += [
                    f"#### {idx}. {_get('title', 'Untitled')}",
                    "",
                    f"**Tool(s)**: {tool_str} | "
                    f"**File**: `{_get('file')}` | "
                    f"**Line**: `{_get('line_number') or 'N/A'}` | "
                    f"**Confidence**: `{_get('confidence', '?')}`{reclass_note}",
                    "",
                    f"{_get('description')}",
                    "",
                ]

                cwe = _get("cwe_id")
                if cwe:
                    lines.append(f"**CWE**: [{cwe}](https://cwe.mitre.org/data/definitions/{cwe.replace('CWE-', '')}.html)")
                    lines.append("")

                affected = _get("affected_code", "")
                if affected:
                    lines += [
                        "**Affected Code:**",
                        "```python",
                        affected,
                        "```",
                        "",
                    ]

                remediation = _get("suggested_remediation", "")
                if remediation:
                    lines += [
                        "**Suggested Remediation:**",
                        f"> {remediation}",
                        "",
                    ]

                # Show merge notes if present
                merge_notes = _get("merge_notes", [])
                if merge_notes:
                    lines.append("<details><summary>Aggregation notes</summary>")
                    lines.append("")
                    for note in merge_notes:
                        lines.append(f"- {note}")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

                lines.append("---")
                lines.append("")

    # ── Proposed Fixes & Self-Correction Review ──────────────────────────────
    suggested_fixes: List[Dict[str, Any]] = state.get("suggested_fixes") or []
    if suggested_fixes:
        lines += [
            "## 🛠️ Proposed Remediations & Self-Correction Review",
            "",
            "> 🛡️ **Safety Guarantee**: Original repository files are untouched. "
            "All fixes below are suggested for developer review.",
            "",
        ]
        for idx, fix in enumerate(suggested_fixes, 1):
            title = fix.get("finding_title", f"Finding #{idx}")
            proposed = fix.get("proposed_fix", "")
            explanation = fix.get("explanation", "")
            ast_valid = fix.get("ast_valid")
            critic_accepted = fix.get("critic_accepted")
            critic_fb = fix.get("critic_feedback", "")
            attempt_num = fix.get("attempt_number", 1)

            if ast_valid is True:
                ast_badge = "✅ Valid Python (AST)"
            elif ast_valid is False:
                ast_badge = "❌ Syntax Error"
            else:
                ast_badge = "⚪ Pending AST"

            if critic_accepted is True:
                critic_badge = "✅ Approved by Critic"
            elif critic_accepted is False:
                critic_badge = "⚠️ Revision Requested"
            else:
                critic_badge = "⚪ Pending Review"

            lines += [
                f"### {idx}. Fix for: {title}",
                "",
                f"**AST Syntax**: {ast_badge} | **Critic Verdict**: {critic_badge} | **Attempt**: #{attempt_num}",
                "",
                "**What Changed:**",
                f"{explanation or 'No explanation provided.'}",
                "",
            ]
            if proposed:
                lines += [
                    "**Proposed Code:**",
                    "```python",
                    proposed,
                    "```",
                    "",
                ]
            if critic_fb:
                lines += [
                    "**Critic Feedback:**",
                    f"> {critic_fb}",
                    "",
                ]
            lines.append("---")
            lines.append("")

    # ── Codebase context ──────────────────────────────────────────────────────
    if context_snippet:
        lines += ["", context_snippet, ""]

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors:
        lines += [
            "## ⚠️ Pipeline Warnings",
            "",
        ]
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    return {"final_report": "\n".join(lines)}


# ─────────────────────────────────────────────────────────────────────────────
# Fix Suggester, AST Validator, and Critic Nodes (Self-Correction Loop)
# ─────────────────────────────────────────────────────────────────────────────

def fix_suggester_node(state: AgentState) -> AgentState:
    """
    Generates concrete code remediation fixes and explanations for findings.
    On retry (when needs_revision is True), it incorporates critic feedback
    to self-correct previous failed fixes.

    Safety: Operates strictly in read-only analysis mode. Never modifies
    the user's repository files.

    Reads:  aggregated_findings, classified_findings, sanitized_code,
            filename, suggested_fixes, revision_count, critic_feedback
    Writes: suggested_fixes
    """
    sanitized_code: str = state.get("sanitized_code") or state.get("raw_code") or ""
    filename: str = state.get("filename") or "snippet.py"
    revision_count: int = state.get("revision_count") or 0
    existing_fixes: List[Dict[str, Any]] = list(state.get("suggested_fixes") or [])

    # Identify findings to process
    raw_findings = list(state.get("aggregated_findings") or [])
    if not raw_findings:
        classified = state.get("classified_findings") or {}
        for level in reversed(_SEVERITY_ORDER):
            raw_findings.extend(classified.get(level, []))

    if not raw_findings:
        return {"suggested_fixes": []}

    def _title(finding: Any) -> str:
        if isinstance(finding, dict):
            return finding.get("title", "Untitled Finding")
        return getattr(finding, "title", "Untitled Finding")

    # Pass 1: generate initial fixes for each finding
    if not existing_fixes:
        new_fixes: List[Dict[str, Any]] = []
        for f in raw_findings:
            res = suggest_fix(
                FixSuggesterInput(
                    finding=f,
                    source_code=sanitized_code,
                    filename=filename,
                    attempt_number=1,
                    is_pre_sanitized=True,
                )
            )
            new_fixes.append({
                "finding_title": res.finding_title or _title(f),
                "proposed_fix": res.proposed_fix,
                "explanation": res.explanation,
                "changed_summary": res.changed_summary,
                "attempt_number": 1,
                "status": res.status,
                "ast_valid": None,
                "ast_errors": [],
                "critic_accepted": None,
                "critic_feedback": "",
                "finding_ref": f,
            })
        return {"suggested_fixes": new_fixes}

    # Pass 2+: self-correct fixes that were rejected by the critic
    updated_fixes: List[Dict[str, Any]] = []
    for fix_entry in existing_fixes:
        if fix_entry.get("critic_accepted") is True:
            updated_fixes.append(fix_entry)
            continue

        f = fix_entry.get("finding_ref")
        if not f:
            f = next(
                (item for item in raw_findings if _title(item) == fix_entry.get("finding_title")),
                raw_findings[0] if raw_findings else {},
            )

        curr_attempt = fix_entry.get("attempt_number", 1) + 1
        res = suggest_fix(
            FixSuggesterInput(
                finding=f,
                source_code=sanitized_code,
                filename=filename,
                previous_fix=fix_entry.get("proposed_fix"),
                critic_feedback=fix_entry.get("critic_feedback"),
                attempt_number=curr_attempt,
                is_pre_sanitized=True,
            )
        )
        updated_fixes.append({
            "finding_title": res.finding_title or fix_entry.get("finding_title", ""),
            "proposed_fix": res.proposed_fix,
            "explanation": res.explanation,
            "changed_summary": res.changed_summary,
            "attempt_number": curr_attempt,
            "status": res.status,
            "ast_valid": None,
            "ast_errors": [],
            "critic_accepted": None,
            "critic_feedback": "",
            "finding_ref": f,
        })

    return {"suggested_fixes": updated_fixes}


def ast_validator_node(state: AgentState) -> AgentState:
    """
    Validates syntax for each proposed fix.

    For Python files: performs full AST parse (fast, zero-cost rejection).
    For other languages: marks as valid so the Critic LLM handles semantic review.

    Reads:  suggested_fixes, filename
    Writes: suggested_fixes (annotated with ast_valid and ast_errors)
    """
    suggested_fixes: List[Dict[str, Any]] = list(state.get("suggested_fixes") or [])
    filename: str = state.get("filename") or "snippet.txt"

    if not suggested_fixes:
        return {"suggested_fixes": []}

    updated_fixes: List[Dict[str, Any]] = []
    for fix_entry in suggested_fixes:
        if fix_entry.get("critic_accepted") is True:
            updated_fixes.append(fix_entry)
            continue

        proposed = fix_entry.get("proposed_fix", "")
        val_res = validate_syntax(proposed, filename=filename)
        fix_entry["ast_valid"] = val_res.is_valid
        fix_entry["ast_errors"] = val_res.errors
        updated_fixes.append(fix_entry)

    return {"suggested_fixes": updated_fixes}


def critic_node(state: AgentState) -> AgentState:
    """
    Evaluates proposed fixes against correctness, safety, and regression rubrics.
    If AST syntax is invalid, rejects immediately without LLM invocation.
    If a fix fails evaluation and revision_count < _MAX_REVISIONS, triggers revision.

    Reads:  suggested_fixes, sanitized_code, filename, revision_count
    Writes: suggested_fixes, needs_revision, critic_feedback, revision_count
    """
    suggested_fixes: List[Dict[str, Any]] = list(state.get("suggested_fixes") or [])
    sanitized_code: str = state.get("sanitized_code") or state.get("raw_code") or ""
    filename: str = state.get("filename") or "snippet.py"
    revision_count: int = state.get("revision_count") or 0

    if not suggested_fixes:
        return {
            "needs_revision": False,
            "critic_feedback": "",
            "revision_count": revision_count,
        }

    # Guard against infinite loops: if max revisions reached, accept what we have
    if revision_count >= _MAX_REVISIONS:
        return {
            "suggested_fixes": suggested_fixes,
            "needs_revision": False,
            "critic_feedback": f"Maximum revisions ({_MAX_REVISIONS}) reached.",
            "revision_count": revision_count,
        }

    updated_fixes: List[Dict[str, Any]] = []
    any_rejected = False
    rejection_feedbacks: List[str] = []

    for fix_entry in suggested_fixes:
        if fix_entry.get("critic_accepted") is True:
            updated_fixes.append(fix_entry)
            continue

        # If syntax validation failed, reject without LLM call
        if fix_entry.get("ast_valid") is False:
            lang_name, _ = detect_language(filename)
            err_msg = "; ".join(fix_entry.get("ast_errors") or [f"Syntax error in {lang_name} code."])
            fb = f"Syntax error detected: {err_msg}. Please provide syntactically valid {lang_name} code."
            fix_entry["critic_accepted"] = False
            fix_entry["critic_feedback"] = fb
            any_rejected = True
            rejection_feedbacks.append(f"[{fix_entry.get('finding_title', 'Fix')}]: {fb}")
            updated_fixes.append(fix_entry)
            continue

        # Syntax is valid: call Critic Agent
        f_ref = fix_entry.get("finding_ref") or {"title": fix_entry.get("finding_title")}
        crit_res = critique_fix(
            CriticInput(
                finding=f_ref,
                original_code=sanitized_code,
                proposed_fix=fix_entry.get("proposed_fix", ""),
                explanation=fix_entry.get("explanation", ""),
                attempt_number=fix_entry.get("attempt_number", 1),
                filename=filename,
            )
        )
        fix_entry["critic_accepted"] = crit_res.is_acceptable
        fix_entry["critic_feedback"] = crit_res.feedback
        if not crit_res.is_acceptable:
            any_rejected = True
            rejection_feedbacks.append(f"[{fix_entry.get('finding_title', 'Fix')}]: {crit_res.feedback}")
        updated_fixes.append(fix_entry)

    if any_rejected and revision_count < _MAX_REVISIONS:
        combined_feedback = " | ".join(rejection_feedbacks)
        return {
            "suggested_fixes": updated_fixes,
            "needs_revision": True,
            "critic_feedback": combined_feedback,
            "revision_count": revision_count + 1,
        }

    return {
        "suggested_fixes": updated_fixes,
        "needs_revision": False,
        "critic_feedback": "",
        "revision_count": revision_count,
    }


def after_critic_router(state: AgentState) -> str:
    """Routes back to fix_suggester_node if revision needed, else to report_node."""
    if state.get("needs_revision"):
        return "fix_suggester_node"
    return "report_node"


# ─────────────────────────────────────────────────────────────────────────────
# Human-in-the-loop node (interrupt point)
# ─────────────────────────────────────────────────────────────────────────────

def human_node(state: AgentState) -> AgentState:
    """
    Interrupt node where execution pauses for human approval.
    Uses LangGraph's interrupt() to pause and commit intermediate state.
    The reviewer can inspect summary, findings, explanations, and fixes,
    and supply APPROVE or REJECT to resume.

    Reads:  severity_summary, highest_severity, total_findings,
            classified_findings, aggregated_findings, suggested_fixes, filename
    Writes: human_decision
    """
    review_payload = {
        "summary":             state.get("severity_summary") or {},
        "highest_severity":    state.get("highest_severity") or "none",
        "total_findings":      state.get("total_findings") or 0,
        "filename":            state.get("filename") or "snippet.py",
        "classified_findings": state.get("classified_findings") or {},
        "aggregated_findings": state.get("aggregated_findings") or [],
        "suggested_fixes":     state.get("suggested_fixes") or [],
    }

    # Genuine LangGraph interrupt: execution halts here and persists to checkpointer
    decision = interrupt(review_payload)
    decision_str = str(decision).strip().lower()

    return {"human_decision": decision_str}


def after_human_router(state: AgentState) -> str:
    """Routes based on human decision."""
    decision: str = (state.get("human_decision") or "approve").strip().lower()
    if decision == "reject":
        return END
    if decision == "revise":
        return "fix_suggester_node"
    return "report_node"   # approve → generate report


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_review_graph(
    *,
    enable_critic: bool = True,
    enable_human_review: bool = False,
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    Compile and return the LangGraph StateGraph for the Code Review Pipeline.

    Parameters
    ----------
    enable_critic:
        If True, the Fix Suggester, AST Validator, and Critic self-correction
        loop is inserted after the classifier.
    enable_human_review:
        If True, a Human node is inserted after the critic (or classifier)
        as an interrupt point backed by checkpointer persistence.
    checkpointer:
        Optional LangGraph checkpointer (e.g. SqliteSaver). If enable_human_review
        is True and checkpointer is None, defaults to get_sqlite_checkpointer().

    Returns
    -------
    A compiled LangGraph app (CompiledGraph) ready to be invoked with
    `app.invoke(initial_state)`.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("sanitizer_node",          sanitizer_node)
    builder.add_node("context_retriever_node",  context_retriever_node)
    builder.add_node("supervisor_node",         supervisor_node)
    builder.add_node("static_node",             static_node)
    builder.add_node("security_node",           security_node)
    builder.add_node("logic_node",              logic_node)
    builder.add_node("aggregator_node",         aggregator_node)
    builder.add_node("classifier_node",         classifier_node)
    builder.add_node("report_node",             report_node)

    if enable_critic:
        builder.add_node("fix_suggester_node",  fix_suggester_node)
        builder.add_node("ast_validator_node",  ast_validator_node)
        builder.add_node("critic_node",         critic_node)

    if enable_human_review:
        builder.add_node("human_node", human_node)

    # ── Linear entry flow ─────────────────────────────────────────────────────
    builder.add_edge(START, "sanitizer_node")
    builder.add_edge("sanitizer_node", "context_retriever_node")
    builder.add_edge("context_retriever_node", "supervisor_node")

    # ── Fan-out from supervisor (conditional parallel edges) ──────────────────
    builder.add_conditional_edges(
        "supervisor_node",
        supervisor_router,
        {
            "static_node":    "static_node",
            "security_node":  "security_node",
            "logic_node":     "logic_node",
            "aggregator_node": "aggregator_node",
        },
    )

    # ── Fan-in: all reviewer nodes → aggregator ───────────────────────────────
    builder.add_edge("static_node",   "aggregator_node")
    builder.add_edge("security_node", "aggregator_node")
    builder.add_edge("logic_node",    "aggregator_node")

    # ── Aggregator → classifier ───────────────────────────────────────────────
    builder.add_edge("aggregator_node", "classifier_node")

    # ── Classifier → fix suggester & critic self-correction loop ──────────────
    if enable_critic:
        builder.add_edge("classifier_node", "fix_suggester_node")
        builder.add_edge("fix_suggester_node", "ast_validator_node")
        builder.add_edge("ast_validator_node", "critic_node")

        if enable_human_review:
            builder.add_conditional_edges(
                "critic_node",
                after_critic_router,
                {
                    "fix_suggester_node": "fix_suggester_node",
                    "report_node": "human_node",
                },
            )
            builder.add_conditional_edges(
                "human_node",
                after_human_router,
                {
                    "report_node": "report_node",
                    "fix_suggester_node": "fix_suggester_node",
                    END: END,
                },
            )
        else:
            builder.add_conditional_edges(
                "critic_node",
                after_critic_router,
                {
                    "fix_suggester_node": "fix_suggester_node",
                    "report_node": "report_node",
                },
            )
    elif enable_human_review:
        builder.add_edge("classifier_node", "human_node")
        builder.add_conditional_edges(
            "human_node",
            after_human_router,
            {
                "report_node": "report_node",
                "fix_suggester_node": "report_node",  # fall through to report if critic not compiled
                END: END,
            },
        )
    else:
        builder.add_edge("classifier_node", "report_node")

    # ── Terminal ──────────────────────────────────────────────────────────────
    builder.add_edge("report_node", END)

    if enable_human_review and checkpointer is None:
        checkpointer = get_sqlite_checkpointer()

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runners
# ─────────────────────────────────────────────────────────────────────────────

def run_review(
    app: Any,
    *,
    raw_code: str,
    filename: str = "snippet.txt",
    changed_functions: Optional[List[Any]] = None,
    enable_critic: bool = True,
    thread_id: Optional[str] = None,
) -> AgentState:
    """
    Execute the compiled review graph and return the final state.

    Parameters
    ----------
    app:
        Compiled graph returned by ``build_review_graph()``.
    raw_code:
        Source code (any language) or diff snippet to review.
    filename:
        Display filename for report generation.
    changed_functions:
        Optional list of ReviewChange objects from the diff extractor.
    enable_critic:
        Forwarded flag (informational; graph already compiled with this flag).
    thread_id:
        Optional conversation/run ID for checkpoint tracking.

    Returns
    -------
    AgentState with all populated fields.
    """
    initial: AgentState = {
        "raw_code":          raw_code,
        "filename":          filename,
        "changed_functions": changed_functions,
        "errors":            [],
        "revision_count":    0,
        "needs_revision":    False,
        "run_static":        False,
        "run_security":      False,
        "run_logic":         False,
    }

    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    if config:
        final_state: AgentState = app.invoke(initial, config=config)
    else:
        final_state: AgentState = app.invoke(initial)
    return final_state


def resume_review(
    app: Any,
    thread_id: str,
    decision: str = "approve",
) -> AgentState:
    """
    Resume an interrupted review workflow from its SQLite checkpoint.

    Parameters
    ----------
    app:
        Compiled review graph.
    thread_id:
        Unique thread ID corresponding to the paused review.
    decision:
        Human decision: 'approve' or 'reject'.

    Returns
    -------
    AgentState after resuming and completing downstream nodes.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=decision.strip().lower()), config=config)

