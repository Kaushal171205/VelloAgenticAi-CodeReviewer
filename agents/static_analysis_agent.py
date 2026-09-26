"""
agents/static_analysis_agent.py — Static Security Analysis Agent (Phase 3).

Wraps the Bandit tool and normalises its output into the project's common
finding schema.  Designed to be called as a plain Python function now and
later plugged into a LangGraph node.

Design contract (for future LangGraph integration):
  - Input:  StaticAnalysisInput  (plain dataclass, no I/O)
  - Output: StaticAnalysisResult (plain dataclass, no I/O)
  - No LLM calls, no network calls.
  - Never raises — all errors are captured in ``result.errors``.

Usage:
    from agents.static_analysis_agent import analyse, StaticAnalysisInput

    result = analyse(StaticAnalysisInput(source_code=code, filename="app.py"))
    for f in result.findings:
        print(f.title, f.severity, f.line)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tools.bandit_wrapper import (
    BanditConfidence,
    BanditFinding,
    BanditInput,
    BanditResult,
    BanditSeverity,
    run_bandit,
)


# ─────────────────────────────────────────────────────────────────────────────
# Common finding schema (project-wide)
# ─────────────────────────────────────────────────────────────────────────────

class FindingSeverity(str, Enum):
    """Unified severity levels used across all agents."""
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


class FindingConfidence(str, Enum):
    """Confidence in the finding's accuracy."""
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


@dataclass
class StaticAnalysisFinding:
    """A single normalised finding from the static analysis agent.

    This schema is shared across all agents in the system so downstream
    consumers (aggregation, severity classification, UI) can process
    findings uniformly.
    """
    title:        str                          # short human-readable summary
    description:  str                          # detailed explanation
    severity:     FindingSeverity               # normalised severity
    confidence:   FindingConfidence             # how certain we are
    file:         str                          # filename
    line:         int                          # 1-indexed line number
    end_line:     int                          # ending line
    code_snippet: str                          # offending source lines
    tool:         str          = "bandit"       # always "bandit" for this agent
    test_id:      str          = ""             # e.g. "B102"
    test_name:    str          = ""             # e.g. "exec_used"
    cwe_id:       Optional[str] = None          # CWE identifier
    more_info:    str          = ""             # doc URL


# ─────────────────────────────────────────────────────────────────────────────
# Input / output data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaticAnalysisInput:
    """Input to the static analysis agent.

    Provide *either* ``source_code`` (raw Python string) or ``file_path``
    (path to an existing .py file).  When ``source_code`` is supplied,
    ``filename`` is used for display only.
    """
    source_code: Optional[str] = None
    file_path:   Optional[str] = None
    filename:    str           = "input.py"


@dataclass
class StaticAnalysisResult:
    """Complete output of the static analysis agent."""
    findings:      list[StaticAnalysisFinding] = field(default_factory=list)
    errors:        list[str]                   = field(default_factory=list)
    finding_count: int                         = 0
    highest_severity: Optional[FindingSeverity] = None
    filename:      str                         = ""
    bandit_metrics: dict                       = field(default_factory=dict)

    # Convenience helpers ─────────────────────────────────────────────────────

    @property
    def has_findings(self) -> bool:
        return self.finding_count > 0

    @property
    def is_clean(self) -> bool:
        return self.finding_count == 0

    def findings_by_severity(self) -> dict[FindingSeverity, list[StaticAnalysisFinding]]:
        out: dict[FindingSeverity, list[StaticAnalysisFinding]] = {s: [] for s in FindingSeverity}
        for f in self.findings:
            out[f.severity].append(f)
        return out

    def summary(self) -> str:
        if self.is_clean:
            if self.errors:
                return f"No findings — but {len(self.errors)} error(s) occurred."
            return "No security issues detected."
        sev = self.highest_severity.value.upper() if self.highest_severity else "?"
        return (
            f"{self.finding_count} security issue(s) found — "
            f"highest severity: {sev}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_MAP: dict[BanditSeverity, FindingSeverity] = {
    BanditSeverity.HIGH:   FindingSeverity.HIGH,
    BanditSeverity.MEDIUM: FindingSeverity.MEDIUM,
    BanditSeverity.LOW:    FindingSeverity.LOW,
}

_CONFIDENCE_MAP: dict[BanditConfidence, FindingConfidence] = {
    BanditConfidence.HIGH:   FindingConfidence.HIGH,
    BanditConfidence.MEDIUM: FindingConfidence.MEDIUM,
    BanditConfidence.LOW:    FindingConfidence.LOW,
}


def _normalise_finding(bf: BanditFinding) -> StaticAnalysisFinding:
    """Convert a raw BanditFinding to the project's common schema."""
    return StaticAnalysisFinding(
        title=bf.title,
        description=bf.description,
        severity=_SEVERITY_MAP.get(bf.severity, FindingSeverity.LOW),
        confidence=_CONFIDENCE_MAP.get(bf.confidence, FindingConfidence.LOW),
        file=bf.file,
        line=bf.line,
        end_line=bf.end_line,
        code_snippet=bf.code_snippet,
        tool=bf.tool,
        test_id=bf.test_id,
        test_name=bf.test_name,
        cwe_id=bf.cwe_id,
        more_info=bf.more_info,
    )


def _compute_highest_severity(
    findings: list[StaticAnalysisFinding],
) -> Optional[FindingSeverity]:
    if not findings:
        return None
    rank = {
        FindingSeverity.CRITICAL: 5,
        FindingSeverity.HIGH: 4,
        FindingSeverity.MEDIUM: 3,
        FindingSeverity.LOW: 2,
        FindingSeverity.INFO: 1,
    }
    return max(findings, key=lambda f: rank[f.severity]).severity


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(inp: StaticAnalysisInput) -> StaticAnalysisResult:
    """Run static security analysis on the given code.

    This function **never raises**.  All errors are captured inside the
    returned ``StaticAnalysisResult.errors`` list.
    """
    if inp.source_code is None and inp.file_path is None:
        return StaticAnalysisResult(
            errors=["No input provided: set source_code or file_path."],
            filename=inp.filename,
        )

    try:
        bandit_inp = BanditInput(
            source_code=inp.source_code,
            file_path=inp.file_path,
            filename=inp.filename,
        )
        bandit_result: BanditResult = run_bandit(bandit_inp)

        # Normalise findings
        findings = [_normalise_finding(bf) for bf in bandit_result.findings]
        highest = _compute_highest_severity(findings)

        return StaticAnalysisResult(
            findings=findings,
            errors=bandit_result.errors,
            finding_count=len(findings),
            highest_severity=highest,
            filename=inp.filename if inp.source_code else (inp.file_path or ""),
            bandit_metrics=bandit_result.metrics,
        )

    except Exception as exc:
        return StaticAnalysisResult(
            errors=[f"Static analysis failed: {exc}"],
            filename=inp.filename,
        )


# Aliases for convenience
analyze = analyse  # US spelling
