"""
agents/severity_classifier_agent.py — Rubric-Based Severity Classifier Agent.

Purpose
-------
Re-evaluate every finding's severity using a well-defined, deterministic rubric
**independent of the severity label supplied by the source agent**.

The source agents (Bandit, LLM security scanner, logic reviewer) all apply their
own heuristics.  The Classifier provides a single, consistent pass that overrides
those labels according to business-impact and exploitability criteria.

Rubric (applied in order of precedence)
----------------------------------------
CRITICAL — any ONE of the following is true:
  • Remote code / OS command execution possible
  • Authentication bypass (login without credentials, privilege escalation to root/admin)
  • SQL injection or NoSQL injection with data exfiltration potential
  • Hardcoded root / admin / production credentials detected
  • Complete access control bypass (any authenticated user = any user)
  • Deserialization of untrusted data leading to RCE
  • Server-side request forgery (SSRF) reaching internal metadata endpoints

HIGH — any ONE of the following is true:
  • Reflected / stored XSS on an authenticated endpoint
  • Path traversal / directory traversal to read arbitrary files
  • Weak cryptography used for authentication (MD5/SHA1 passwords without salt)
  • Missing rate limiting on authentication endpoints
  • Insecure direct object reference (IDOR) allowing cross-account data access
  • XML External Entity (XXE) injection
  • Open redirect with persistent session token exposure
  • Missing CSRF protection on state-changing endpoints
  • Sensitive data logged or returned in API response

MEDIUM — any ONE of the following is true:
  • Weak cryptography used for non-authentication purposes (checksums, MACs)
  • Missing input validation that could cause unexpected application behaviour
  • Information disclosure without direct data exfiltration
  • Race condition or TOCTOU with limited impact
  • Insufficient error handling leaking stack traces / internal paths
  • Logic flaw with limited blast radius (single-user impact)
  • Use of deprecated / insecure API that is not immediately exploitable

LOW — everything else that represents a real but minor improvement:
  • Verbose / overly permissive logging
  • Minor code quality issues with security implications
  • Stylistic / naming issues that could cause misuse
  • Informational notices with no direct security impact

Design contract
---------------
  - Input:  ClassifierInput  (plain dataclass, no I/O)
  - Output: ClassifierResult (plain dataclass, no I/O)
  - Does NOT call an LLM; rule-based so it is fast, deterministic, and offline.
  - Never raises — all errors are captured in ``result.errors``.
  - Preserves the original severity in ``original_severity`` for audit purposes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.aggregator_agent import FindingRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rubric definition
# ─────────────────────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"

    @classmethod
    def rank(cls, level: "SeverityLevel") -> int:
        return {
            cls.CRITICAL: 5,
            cls.HIGH:     4,
            cls.MEDIUM:   3,
            cls.LOW:      2,
            cls.INFO:     1,
        }.get(level, 0)

    @classmethod
    def from_str(cls, s: str) -> "SeverityLevel":
        mapping = {
            "critical": cls.CRITICAL,
            "high":     cls.HIGH,
            "medium":   cls.MEDIUM,
            "low":      cls.LOW,
            "info":     cls.INFO,
        }
        return mapping.get((s or "").strip().lower(), cls.LOW)


@dataclass(frozen=True)
class RubricRule:
    """A single rubric rule: a pattern + the severity it triggers."""
    severity:    SeverityLevel
    keywords:    Tuple[str, ...]        # any keyword match triggers this rule
    phrases:     Tuple[str, ...]        # any phrase match (substring) triggers this rule
    categories:  Tuple[str, ...]        # optional category filter (empty = match all)
    description: str                    # human-readable rationale


# Rules are evaluated in declaration order; the FIRST match wins.
# More specific (higher-severity) rules are declared first.
_RUBRIC: List[RubricRule] = [

    # ── CRITICAL ─────────────────────────────────────────────────────────────
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("rce", "remote code execution", "command injection", "os command",
                  "shell injection", "arbitrary code"),
        phrases=("shell=True", "os.system", "subprocess.run", "exec(", "eval("),
        categories=(),
        description="Remote code execution / OS command injection.",
    ),
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("authentication bypass", "auth bypass", "privilege escalation",
                  "unauthenticated access", "bypass authentication",
                  "admin without password", "root credential"),
        phrases=("bypass", "unauthenticated"),
        categories=(),
        description="Authentication bypass or privilege escalation.",
    ),
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("sql injection", "nosql injection", "sqli", "database injection",
                  "blind sql"),
        phrases=("sql injection", "execute(", "raw sql"),
        categories=(),
        description="SQL / NoSQL injection.",
    ),
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("hardcoded", "hardcoded credential", "hardcoded password",
                  "hardcoded secret", "hardcoded key", "embedded credential"),
        phrases=("hardcoded", "hard-coded"),
        categories=("static",),
        description="Hardcoded credentials in source code.",
    ),
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("deserialization", "pickle", "yaml.load", "marshal", "rce via deserializ"),
        phrases=("pickle.loads", "yaml.load(", "marshal.loads"),
        categories=(),
        description="Unsafe deserialization leading to RCE.",
    ),
    RubricRule(
        severity=SeverityLevel.CRITICAL,
        keywords=("ssrf", "server-side request forgery", "internal metadata"),
        phrases=("metadata.internal", "169.254.169.254"),
        categories=(),
        description="Server-Side Request Forgery targeting internal services.",
    ),

    # ── HIGH ──────────────────────────────────────────────────────────────────
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("xss", "cross-site scripting", "reflected xss", "stored xss",
                  "dom xss"),
        phrases=("script>", "xss"),
        categories=(),
        description="Cross-site scripting (XSS).",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("path traversal", "directory traversal", "lfi", "local file inclusion",
                  "arbitrary file read"),
        phrases=("../", "..\\"),
        categories=(),
        description="Path / directory traversal.",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("weak crypto", "md5", "sha1", "broken hash", "insecure hash",
                  "broken cryptographic", "no salt", "without salt", "rainbow table"),
        phrases=("md5(", "sha1(", "hashlib.md5", "hashlib.sha1"),
        categories=(),
        description="Weak or broken cryptographic hash used for authentication.",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("idor", "insecure direct object reference", "cross-account",
                  "unauthorised access to", "unauthorized access to"),
        phrases=("idor",),
        categories=(),
        description="Insecure direct object reference (IDOR).",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("xxe", "xml external entity", "xml injection"),
        phrases=("<!entity", "xxe"),
        categories=(),
        description="XML External Entity (XXE) injection.",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("csrf", "cross-site request forgery", "missing csrf"),
        phrases=("csrf",),
        categories=(),
        description="Missing CSRF protection.",
    ),
    RubricRule(
        severity=SeverityLevel.HIGH,
        keywords=("sensitive data in log", "password logged", "token logged",
                  "secret in response", "credential in response"),
        phrases=(),
        categories=(),
        description="Sensitive data logged or returned to caller.",
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("weak crypto", "md5", "sha1", "deprecated hash",
                  "checksum only", "non-authentication"),
        phrases=("hashlib.md5", "hashlib.sha1"),
        categories=("security",),
        description="Weak crypto for non-authentication purposes.",
    ),
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("missing input validation", "no validation", "unvalidated input",
                  "missing validation", "integer overflow", "negative value"),
        phrases=(),
        categories=(),
        description="Missing or insufficient input validation.",
    ),
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("information disclosure", "stack trace", "error message",
                  "internal path", "debug information"),
        phrases=("traceback", "stack trace"),
        categories=(),
        description="Information disclosure without direct exploitation.",
    ),
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("race condition", "toctou", "time of check", "concurrency flaw"),
        phrases=(),
        categories=(),
        description="Race condition or TOCTOU vulnerability.",
    ),
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("logic flaw", "incorrect condition", "business logic",
                  "double refund", "state transition", "incorrect state"),
        phrases=(),
        categories=("business_logic",),
        description="Business logic flaw with limited blast radius.",
    ),
    RubricRule(
        severity=SeverityLevel.MEDIUM,
        keywords=("deprecated api", "insecure api", "use of subprocess",
                  "use of exec", "use of eval"),
        phrases=(),
        categories=(),
        description="Use of deprecated / insecure API not immediately exploitable.",
    ),

    # ── LOW ───────────────────────────────────────────────────────────────────
    RubricRule(
        severity=SeverityLevel.LOW,
        keywords=("verbose logging", "excessive logging", "minor logic",
                  "informational", "code quality"),
        phrases=(),
        categories=(),
        description="Minor issue with limited security impact.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Matching logic
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return (s or "").lower()


def _match_rule(finding: FindingRecord, rule: RubricRule) -> bool:
    """Return True if this rule applies to the finding."""
    # Category filter (empty tuple = any)
    if rule.categories and finding.category not in rule.categories:
        return False

    # Build the search corpus: title + description + affected_code
    corpus = " ".join([
        _normalize(finding.title),
        _normalize(finding.description),
        _normalize(finding.affected_code),
        _normalize(finding.reasoning),
    ])

    # Keyword match (any full word or phrase)
    for kw in rule.keywords:
        if kw.lower() in corpus:
            return True

    # Phrase match in affected_code or title specifically
    for phrase in rule.phrases:
        if phrase.lower() in corpus:
            return True

    return False


def _classify_finding(finding: FindingRecord) -> Tuple[SeverityLevel, Optional[str]]:
    """
    Apply the rubric to a single finding.

    Returns (new_severity, matched_rule_description).
    If no rule matches, fall back to the finding's reported_severity, capped at HIGH
    (untrusted source).
    """
    for rule in _RUBRIC:
        if _match_rule(finding, rule):
            return rule.severity, rule.description

    # Fallback: cap unmatched findings at HIGH to prevent over-inflation
    fallback = SeverityLevel.from_str(finding.reported_severity)
    if SeverityLevel.rank(fallback) > SeverityLevel.rank(SeverityLevel.HIGH):
        fallback = SeverityLevel.HIGH
    return fallback, None


# ─────────────────────────────────────────────────────────────────────────────
# Input / output models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationDetail:
    """Detailed classification record for a single finding."""
    finding_title:      str
    original_severity:  str
    classified_severity: str
    changed:            bool
    matched_rule:       Optional[str]   # human-readable description of the matched rule
    justification:      str             # concise explanation of the classification decision


@dataclass
class ClassifierInput:
    """Input to the Severity Classifier Agent."""
    findings: List[FindingRecord]


@dataclass
class ClassifierResult:
    """Complete output of the Severity Classifier Agent."""
    findings:            List[FindingRecord]             = field(default_factory=list)
    details:             List[ClassificationDetail]      = field(default_factory=list)
    classified_findings: Dict[str, List[FindingRecord]]  = field(default_factory=dict)
    severity_summary:    Dict[str, int]                  = field(default_factory=dict)
    highest_severity:    Optional[str]                   = None
    reclassified_count:  int                             = 0
    errors:              List[str]                        = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "No findings to classify."
        counts = ", ".join(
            f"{k}: {len(v)}"
            for k, v in self.classified_findings.items()
            if v
        )
        changed = f", {self.reclassified_count} severity label(s) reclassified" if self.reclassified_count else ""
        return f"{self.total} finding(s) classified [{counts}]{changed}."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


def classify(inp: ClassifierInput) -> ClassifierResult:
    """
    Re-classify the severity of every FindingRecord using the built-in rubric.

    This function **never raises**.  All errors are captured inside the
    returned ``ClassifierResult.errors`` list.
    """
    errors: List[str] = []
    classified_findings: Dict[str, List[FindingRecord]] = {
        "critical": [],
        "high":     [],
        "medium":   [],
        "low":      [],
        "info":     [],
    }
    details: List[ClassificationDetail] = []
    reclassified_count = 0

    for record in inp.findings:
        try:
            new_severity, matched_rule = _classify_finding(record)
            original = record.severity
            changed = new_severity.value != original.lower()

            if changed:
                reclassified_count += 1
                record.merge_notes.append(
                    f"Severity reclassified: {original} → {new_severity.value} "
                    f"[rule: {matched_rule or 'fallback'}]"
                )

            record.severity = new_severity.value

            justification = (
                matched_rule or
                f"No rubric rule matched; retained source severity '{original}' "
                f"(capped at 'high')."
            )

            details.append(ClassificationDetail(
                finding_title=record.title,
                original_severity=original,
                classified_severity=new_severity.value,
                changed=changed,
                matched_rule=matched_rule,
                justification=justification,
            ))

            bucket = classified_findings.get(new_severity.value, classified_findings["low"])
            bucket.append(record)

        except Exception as exc:
            logger.exception("Classification error for '%s': %s", record.title, exc)
            errors.append(f"Classification error [{record.title}]: {exc}")
            # Place in low bucket on error
            classified_findings["low"].append(record)

    # Reorder within each bucket: confidence desc
    for bucket in classified_findings.values():
        bucket.sort(
            key=lambda r: _SEVERITY_ORDER.index(r.confidence) if r.confidence in _SEVERITY_ORDER else 99
        )

    summary = {k: len(v) for k, v in classified_findings.items() if v}

    highest: Optional[str] = None
    for level in _SEVERITY_ORDER:
        if classified_findings.get(level):
            highest = level
            break

    return ClassifierResult(
        findings=inp.findings,
        details=details,
        classified_findings=classified_findings,
        severity_summary=summary,
        highest_severity=highest,
        reclassified_count=reclassified_count,
        errors=errors,
    )
