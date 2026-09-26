"""
agents/aggregator_agent.py — Finding Aggregator Agent.

Responsibilities
----------------
1. Accept raw findings from every review agent (static, security, logic).
2. Normalise all findings into a single, unified FindingRecord schema.
3. Deduplicate across tools using a multi-signal fingerprint:
     - exact duplicates   : same (title_key, file, line_number)
     - near-duplicates    : Jaccard token similarity > 0.85 on title + same file
     - line-range overlap : different titles but same file AND overlapping lines
       AND same category → merge, preserving both tools in provenance
4. When merging duplicates, keep the *highest* confidence finding as primary
   and record all tool names in the ``source_tools`` list.
5. Preserve all evidence: line numbers, affected code, reasoning, CWE IDs.
6. Never drop a finding silently — conflicts are recorded in ``merge_notes``.

Design contract
---------------
  - Input:  AggregatorInput  (plain dataclass, no I/O)
  - Output: AggregatorResult (plain dataclass, no I/O)
  - No LLM calls, no network calls, no filesystem side-effects.
  - Never raises — all errors are captured in ``result.errors``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.static_analysis_agent import (
    FindingConfidence,
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisResult,
)
from agents.logic_reviewer_agent import LogicFinding, LogicReviewResult
from agents.security_scanner_agent import SecurityFinding, SecurityScannerResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Unified finding record
# ─────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_RANK: Dict[str, int] = {
    "high":   3,
    "medium": 2,
    "low":    1,
    "":       0,
}

_SEVERITY_RANK: Dict[str, int] = {
    "critical": 5,
    "high":     4,
    "medium":   3,
    "low":      2,
    "info":     1,
    "":         0,
}


@dataclass
class FindingRecord:
    """
    Unified, normalised finding record produced by the Aggregator.

    This is the canonical representation used by the Severity Classifier,
    Report Generator, and UI — independent of which tool produced it.
    """

    # Core identity
    title: str
    description: str
    category: str              # e.g. "static" | "security" | "business_logic"

    # Severity / confidence as reported by the source agent (may be reclassified later)
    reported_severity: str     # lowercase: critical|high|medium|low|info
    reported_confidence: str   # lowercase: high|medium|low

    # Reclassified severity set by SeverityClassifierAgent (initially same as reported)
    severity: str              # the authoritative field used downstream
    confidence: str

    # Evidence
    file: str
    line_number: Optional[int]
    end_line: Optional[int]
    affected_code: str
    reasoning: str
    suggested_remediation: str

    # Provenance
    source_tools: List[str]    # all tools that flagged this (e.g. ["bandit", "security_scanner_rag"])
    primary_tool: str          # tool with the highest-confidence finding

    # Optional metadata
    cwe_id: Optional[str]
    test_id: Optional[str]
    more_info: str

    # Aggregator bookkeeping
    fingerprint: str           # hash used for deduplication
    merge_notes: List[str]     # human-readable notes about merges that occurred
    duplicate_count: int       # how many raw findings this record absorbed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title":                 self.title,
            "description":           self.description,
            "category":              self.category,
            "reported_severity":     self.reported_severity,
            "reported_confidence":   self.reported_confidence,
            "severity":              self.severity,
            "confidence":            self.confidence,
            "file":                  self.file,
            "line_number":           self.line_number,
            "end_line":              self.end_line,
            "affected_code":         self.affected_code,
            "reasoning":             self.reasoning,
            "suggested_remediation": self.suggested_remediation,
            "source_tools":          self.source_tools,
            "primary_tool":          self.primary_tool,
            "cwe_id":                self.cwe_id,
            "test_id":               self.test_id,
            "more_info":             self.more_info,
            "fingerprint":           self.fingerprint,
            "merge_notes":           self.merge_notes,
            "duplicate_count":       self.duplicate_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Input / output models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggregatorInput:
    """Raw findings collected from all review agents."""
    static_result:   Optional[StaticAnalysisResult]  = None
    security_result: Optional[SecurityScannerResult] = None
    logic_result:    Optional[LogicReviewResult]     = None
    filename:        str                             = "snippet.py"


@dataclass
class AggregatorResult:
    """Complete output of the Aggregator Agent."""
    findings:         List[FindingRecord] = field(default_factory=list)
    total_raw:        int = 0    # total findings before deduplication
    total_merged:     int = 0    # how many records were absorbed into others
    total_output:     int = 0    # findings after deduplication
    errors:           List[str]  = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    def findings_by_severity(self) -> Dict[str, List[FindingRecord]]:
        out: Dict[str, List[FindingRecord]] = {
            s: [] for s in ("critical", "high", "medium", "low", "info")
        }
        for f in self.findings:
            bucket = out.get(f.severity)
            if bucket is not None:
                bucket.append(f)
            else:
                out["low"].append(f)
        return out

    def summary(self) -> str:
        if not self.has_findings:
            return "No findings after aggregation."
        counts = ", ".join(
            f"{k}: {len(v)}"
            for k, v in self.findings_by_severity().items()
            if v
        )
        return (
            f"{self.total_output} unique finding(s) [{counts}] "
            f"from {self.total_raw} raw findings "
            f"({self.total_merged} duplicates removed)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_str(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _title_tokens(title: str) -> Set[str]:
    return set(_normalize_str(title).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _make_fingerprint(title: str, file: str, line: Optional[int]) -> str:
    """Create a stable, exact-match fingerprint."""
    import hashlib
    key = f"{_normalize_str(title)}|{file}|{line or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _lines_overlap(
    line_a: Optional[int],
    end_a: Optional[int],
    line_b: Optional[int],
    end_b: Optional[int],
) -> bool:
    """Return True if two line ranges overlap (both must be known)."""
    if line_a is None or line_b is None:
        return False
    a_start, a_end = line_a, end_a or line_a
    b_start, b_end = line_b, end_b or line_b
    return a_start <= b_end and b_start <= a_end


def _str_val(v: Any) -> str:
    if v is None:
        return ""
    if hasattr(v, "value"):
        return v.value
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Raw finding → FindingRecord
# ─────────────────────────────────────────────────────────────────────────────

def _from_static(f: StaticAnalysisFinding, filename: str) -> FindingRecord:
    file = f.file or filename
    sev  = _str_val(f.severity)
    conf = _str_val(f.confidence)
    fp   = _make_fingerprint(f.title, file, f.line)
    return FindingRecord(
        title=f.title,
        description=f.description,
        category="static",
        reported_severity=sev,
        reported_confidence=conf,
        severity=sev,
        confidence=conf,
        file=file,
        line_number=f.line,
        end_line=f.end_line,
        affected_code=f.code_snippet,
        reasoning="",
        suggested_remediation=f.more_info or "",
        source_tools=[f.tool or "bandit"],
        primary_tool=f.tool or "bandit",
        cwe_id=f.cwe_id,
        test_id=f.test_id,
        more_info=f.more_info or "",
        fingerprint=fp,
        merge_notes=[],
        duplicate_count=1,
    )


def _from_security(f: SecurityFinding, filename: str) -> FindingRecord:
    file = f.file or filename
    sev  = _str_val(f.severity)
    conf = _str_val(f.confidence)
    fp   = _make_fingerprint(f.title, file, f.line_number)
    return FindingRecord(
        title=f.title,
        description=f.description,
        category=f.category or "security",
        reported_severity=sev,
        reported_confidence=conf,
        severity=sev,
        confidence=conf,
        file=file,
        line_number=f.line_number,
        end_line=None,
        affected_code=f.affected_code,
        reasoning=f.reasoning,
        suggested_remediation=f.suggested_remediation,
        source_tools=[f.tool or "security_scanner_rag"],
        primary_tool=f.tool or "security_scanner_rag",
        cwe_id=f.cwe_id,
        test_id=getattr(f, "matched_rule_id", None),
        more_info="",
        fingerprint=fp,
        merge_notes=[],
        duplicate_count=1,
    )


def _from_logic(f: LogicFinding, filename: str) -> FindingRecord:
    file = f.file or filename
    sev  = _str_val(f.severity)
    conf = _str_val(f.confidence)
    fp   = _make_fingerprint(f.title, file, f.line_number)
    return FindingRecord(
        title=f.title,
        description=f.description,
        category=f.category or "business_logic",
        reported_severity=sev,
        reported_confidence=conf,
        severity=sev,
        confidence=conf,
        file=file,
        line_number=f.line_number,
        end_line=None,
        affected_code=f.affected_code,
        reasoning=f.reasoning,
        suggested_remediation=f.suggested_remediation,
        source_tools=[f.tool or "logic_reviewer"],
        primary_tool=f.tool or "logic_reviewer",
        cwe_id=None,
        test_id=None,
        more_info="",
        fingerprint=fp,
        merge_notes=[],
        duplicate_count=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication engine
# ─────────────────────────────────────────────────────────────────────────────

_JACCARD_THRESHOLD = 0.72      # title similarity for near-duplicate detection
_LINE_PROXIMITY    = 5         # lines within which same-category findings merge


def _is_exact_duplicate(a: FindingRecord, b: FindingRecord) -> bool:
    """Same fingerprint = same normalised title + file + line."""
    return a.fingerprint == b.fingerprint


def _is_near_duplicate(a: FindingRecord, b: FindingRecord) -> bool:
    """Similar title on same file (different tools often describe the same flaw)."""
    if a.file != b.file:
        return False
    tokens_a = _title_tokens(a.title)
    tokens_b = _title_tokens(b.title)
    return _jaccard(tokens_a, tokens_b) >= _JACCARD_THRESHOLD


def _is_line_overlap_duplicate(a: FindingRecord, b: FindingRecord) -> bool:
    """Same file + category + overlapping lines → likely the same issue."""
    if a.file != b.file:
        return False
    if a.category != b.category:
        return False
    la, ea = a.line_number, a.end_line
    lb, eb = b.line_number, b.end_line
    if la is None or lb is None:
        return False
    # Expand ranges by LINE_PROXIMITY to catch near-line duplicates
    ea_exp = (ea or la) + _LINE_PROXIMITY
    eb_exp = (eb or lb) + _LINE_PROXIMITY
    la_exp = la - _LINE_PROXIMITY
    lb_exp = lb - _LINE_PROXIMITY
    return la_exp <= eb_exp and lb_exp <= ea_exp


def _merge_into(primary: FindingRecord, absorbed: FindingRecord) -> None:
    """
    Merge `absorbed` into `primary` in-place.

    Keep primary's title/description (highest confidence).
    Append absorbed's tool to source_tools.
    Combine reasoning and remediation text if richer.
    Take the higher severity.
    """
    # Add tool provenance
    for t in absorbed.source_tools:
        if t not in primary.source_tools:
            primary.source_tools.append(t)

    # Take the higher severity
    if _SEVERITY_RANK.get(absorbed.severity, 0) > _SEVERITY_RANK.get(primary.severity, 0):
        primary.merge_notes.append(
            f"Severity elevated from {primary.severity} → {absorbed.severity} "
            f"by {absorbed.primary_tool}."
        )
        primary.severity = absorbed.severity
        primary.reported_severity = absorbed.reported_severity

    # Upgrade confidence if absorbed is higher
    if _CONFIDENCE_RANK.get(absorbed.confidence, 0) > _CONFIDENCE_RANK.get(primary.confidence, 0):
        primary.confidence = absorbed.confidence
        primary.reported_confidence = absorbed.reported_confidence
        primary.primary_tool = absorbed.primary_tool

    # Prefer non-empty CWE
    if not primary.cwe_id and absorbed.cwe_id:
        primary.cwe_id = absorbed.cwe_id

    # Append additional reasoning if present and different
    if absorbed.reasoning and absorbed.reasoning not in primary.reasoning:
        if primary.reasoning:
            primary.reasoning += f"\n\n[{absorbed.primary_tool}] {absorbed.reasoning}"
        else:
            primary.reasoning = absorbed.reasoning

    # Prefer richer remediation
    if len(absorbed.suggested_remediation) > len(primary.suggested_remediation):
        primary.suggested_remediation = absorbed.suggested_remediation

    primary.merge_notes.append(
        f"Absorbed duplicate from {absorbed.primary_tool} "
        f"(similarity/overlap detected)."
    )
    primary.duplicate_count += 1


def _deduplicate(records: List[FindingRecord]) -> Tuple[List[FindingRecord], int]:
    """
    Run multi-pass deduplication over a list of FindingRecords.

    Returns the deduplicated list and the number of records merged.
    """
    merged_count = 0
    output: List[FindingRecord] = []
    absorbed_indices: Set[int] = set()

    for i, record in enumerate(records):
        if i in absorbed_indices:
            continue

        primary = record

        for j, candidate in enumerate(records):
            if j <= i or j in absorbed_indices:
                continue

            if (
                _is_exact_duplicate(primary, candidate)
                or _is_near_duplicate(primary, candidate)
                or _is_line_overlap_duplicate(primary, candidate)
            ):
                # Merge candidate into primary
                _merge_into(primary, candidate)
                absorbed_indices.add(j)
                merged_count += 1

        output.append(primary)

    return output, merged_count


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(inp: AggregatorInput) -> AggregatorResult:
    """
    Aggregate, normalise, and deduplicate findings from all agents.

    This function **never raises**.  All errors are captured inside the
    returned ``AggregatorResult.errors`` list.
    """
    errors: List[str] = []
    raw_records: List[FindingRecord] = []

    # ── Collect static findings ───────────────────────────────────────────────
    if inp.static_result:
        for f in (inp.static_result.findings or []):
            try:
                raw_records.append(_from_static(f, inp.filename))
            except Exception as exc:
                errors.append(f"Static normalisation error: {exc}")
        errors.extend(inp.static_result.errors or [])

    # ── Collect security findings ─────────────────────────────────────────────
    if inp.security_result:
        for f in (inp.security_result.findings or []):
            try:
                raw_records.append(_from_security(f, inp.filename))
            except Exception as exc:
                errors.append(f"Security normalisation error: {exc}")
        errors.extend(inp.security_result.errors or [])

    # ── Collect logic findings ────────────────────────────────────────────────
    if inp.logic_result:
        for f in (inp.logic_result.findings or []):
            try:
                raw_records.append(_from_logic(f, inp.filename))
            except Exception as exc:
                errors.append(f"Logic normalisation error: {exc}")
        errors.extend(inp.logic_result.errors or [])

    total_raw = len(raw_records)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    try:
        deduped, merged_count = _deduplicate(raw_records)
    except Exception as exc:
        logger.exception("Deduplication failed: %s", exc)
        errors.append(f"Deduplication error: {exc}")
        deduped, merged_count = raw_records, 0

    # ── Sort: severity desc, then confidence desc ─────────────────────────────
    deduped.sort(
        key=lambda r: (
            _SEVERITY_RANK.get(r.severity, 0),
            _CONFIDENCE_RANK.get(r.confidence, 0),
        ),
        reverse=True,
    )

    return AggregatorResult(
        findings=deduped,
        total_raw=total_raw,
        total_merged=merged_count,
        total_output=len(deduped),
        errors=errors,
    )
