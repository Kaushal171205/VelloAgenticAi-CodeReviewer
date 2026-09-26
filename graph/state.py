"""
graph/state.py — Typed AgentState for the LangGraph Code Review Pipeline.

Each field corresponds to a stage of the pipeline:

  raw_code           ← user / diff input (plain Python string or diff text)
  sanitized_code     ← sanitizer output (secrets redacted)
  changed_functions  ← functions identified in the diff (ReviewChange list)
  related_context    ← semantic context retrieved from the codebase index
  static_findings    ← Bandit-based findings (StaticAnalysisResult)
  security_findings  ← RAG-based security findings (SecurityScannerResult)
  logic_findings     ← LLM business-logic findings  (LogicReviewResult)
  aggregated_findings← unified list of all findings (list[dict])
  classified_findings← severity-classified view    (dict keyed by level)
  suggested_fixes    ← LLM-generated remediation suggestions (list[dict])
  needs_revision     ← supervisor flag: True if a re-run is warranted
  critic_feedback    ← free-text feedback from Critic node (str)
  human_decision     ← "approve" | "reject" | "revise" set by human node
  final_report       ← fully-rendered markdown report (str)
  errors             ← accumulated error messages across all nodes

All fields are Optional[...] and default to None so that the graph can run
partial pipelines without pre-populating the whole state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared mutable state passed through every node of the LangGraph
    Code Review Pipeline.

    Using TypedDict (instead of a dataclass) is idiomatic for LangGraph
    because it allows partial updates: each node only returns the keys it
    mutated and LangGraph merges them into the running state.
    """

    # ── Stage 0: raw input ────────────────────────────────────────────────────
    raw_code: Optional[str]
    """Original, unsanitized code submitted for review."""

    filename: Optional[str]
    """Display filename for the code under review."""

    # ── Stage 1: sanitizer ────────────────────────────────────────────────────
    sanitized_code: Optional[str]
    """Code after secret redaction by the Sanitizer node."""

    sanitization_applied: bool
    """True if the sanitizer redacted at least one secret."""

    secrets_detected: int
    """Number of secrets detected during sanitization."""

    sanitizer_errors: List[str]
    """Non-fatal errors emitted by the Sanitizer node."""

    # ── Stage 2: diff / context retriever ─────────────────────────────────────
    changed_functions: Optional[List[Any]]
    """
    List of ReviewChange objects from the diff extractor, describing each
    modified function / class with its file path and diff content.
    """

    related_context: Optional[Any]
    """
    RetrievedContextPackage produced by the Context Retriever — semantically
    related functions from the shared codebase index.
    """

    context_prompt_snippet: Optional[str]
    """
    Pre-rendered markdown snippet from related_context, ready to inject into
    agent prompts.  Generated once so every downstream agent can reuse it.
    """

    # ── Stage 3: supervisor routing ───────────────────────────────────────────
    run_static: bool
    """Supervisor decision: should the Static Analysis node execute?"""

    run_security: bool
    """Supervisor decision: should the Security Scanner node execute?"""

    run_logic: bool
    """Supervisor decision: should the Logic Reviewer node execute?"""

    # ── Stage 4a: static analysis ─────────────────────────────────────────────
    static_findings: Optional[Any]
    """StaticAnalysisResult from the Static Analysis (Bandit) node."""

    # ── Stage 4b: security scanner ────────────────────────────────────────────
    security_findings: Optional[Any]
    """SecurityScannerResult from the RAG-based Security Scanner node."""

    # ── Stage 4c: logic reviewer ──────────────────────────────────────────────
    logic_findings: Optional[Any]
    """LogicReviewResult from the Business Logic Reviewer node."""

    # ── Stage 5: aggregator ───────────────────────────────────────────────────
    aggregated_findings: Optional[List[Dict[str, Any]]]
    """
    Unified list of all findings from all agents, normalised to a common dict
    schema:
      {title, description, severity, confidence, tool, file, line_number,
       affected_code, suggested_remediation, category}
    """

    total_findings: int
    """Total count of aggregated findings."""

    # ── Stage 6: severity classifier ──────────────────────────────────────────
    classified_findings: Optional[Dict[str, List[Dict[str, Any]]]]
    """
    Findings grouped by severity level:
      {"critical": [...], "high": [...], "medium": [...], "low": [...], "info": [...]}
    """

    highest_severity: Optional[str]
    """The single highest severity level string across all findings."""

    severity_summary: Optional[Dict[str, int]]
    """Count of findings per severity level, e.g. {"critical": 1, "high": 3}."""

    # ── Stage 7: suggested fixes (optional) ───────────────────────────────────
    suggested_fixes: Optional[List[Dict[str, Any]]]
    """
    LLM-generated remediation suggestions keyed per finding title.
    Each entry: {title, fix_description, code_snippet}
    """

    # ── Stage 8: critic / revision loop ───────────────────────────────────────
    needs_revision: bool
    """
    Flag set by the Critic node.  If True the Supervisor routes back into the
    review agents for a second pass.
    """

    critic_feedback: Optional[str]
    """
    Free-text feedback from the Critic node explaining why revision is needed.
    """

    revision_count: int
    """Number of revision iterations completed (guards against infinite loops)."""

    # ── Stage 9: human-in-the-loop ────────────────────────────────────────────
    human_decision: Optional[str]
    """
    Decision provided by a human reviewer:
      "approve" — accept the review findings as-is.
      "reject"  — discard; do not produce a report.
      "revise"  — trigger another review pass with updated critic_feedback.
    """

    # ── Stage 10: final report ────────────────────────────────────────────────
    final_report: Optional[str]
    """
    Fully-rendered markdown report combining all findings, classification,
    dependency context, and suggested fixes.
    """

    # ── Cross-cutting ─────────────────────────────────────────────────────────
    errors: List[str]
    """Accumulated non-fatal error messages from any node in the pipeline."""
