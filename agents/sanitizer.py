"""
agents/sanitizer.py — Secret Sanitization Agent (Phase 2).

Scans Python source code for secrets before it is sent to any LLM.
Uses two complementary detection layers:

  1. Regex-based detection  — fast, deterministic, tuned for common patterns.
  2. detect-secrets library — broader entropy-based and pattern-based scanning.

Design contract (for future LangGraph integration):
  - Input:  SanitizerInput  (plain dataclass, no I/O)
  - Output: SanitizerResult (plain dataclass, no I/O)
  - No LLM calls, no network calls, no filesystem side-effects.
  - Idempotent: running sanitize() twice on already-sanitized code is safe.

Usage:
    from agents.sanitizer import sanitize, SanitizerInput

    result = sanitize(SanitizerInput(source_code=raw_code, filename="app.py"))
    print(result.sanitized_code)
    print(result.findings)
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations & constants
# ─────────────────────────────────────────────────────────────────────────────

class SecretType(str, Enum):
    API_KEY           = "api_key"
    PASSWORD          = "password"
    TOKEN             = "token"
    PRIVATE_KEY       = "private_key"
    DATABASE_URL      = "database_url"
    AWS_ACCESS_KEY    = "aws_access_key"
    AWS_SECRET_KEY    = "aws_secret_key"
    GITHUB_TOKEN      = "github_token"
    GOOGLE_API_KEY    = "google_api_key"
    OPENAI_KEY        = "openai_key"
    STRIPE_KEY        = "stripe_key"
    SLACK_TOKEN       = "slack_token"
    JWT_TOKEN         = "jwt_token"
    GENERIC_SECRET    = "generic_secret"
    DETECT_SECRETS    = "detect_secrets_finding"


class Severity(str, Enum):
    CRITICAL = "critical"   # private key, root credential
    HIGH     = "high"       # db URL with creds, cloud access key
    MEDIUM   = "medium"     # password, token, API key
    LOW      = "low"        # low-confidence / possibly false positive


# Map secret type → severity
_SEVERITY_MAP: dict[SecretType, Severity] = {
    SecretType.PRIVATE_KEY:    Severity.CRITICAL,
    SecretType.AWS_ACCESS_KEY: Severity.HIGH,
    SecretType.AWS_SECRET_KEY: Severity.HIGH,
    SecretType.DATABASE_URL:   Severity.HIGH,
    SecretType.GOOGLE_API_KEY: Severity.HIGH,
    SecretType.OPENAI_KEY:     Severity.HIGH,
    SecretType.GITHUB_TOKEN:   Severity.HIGH,
    SecretType.STRIPE_KEY:     Severity.HIGH,
    SecretType.SLACK_TOKEN:    Severity.MEDIUM,
    SecretType.API_KEY:        Severity.MEDIUM,
    SecretType.TOKEN:          Severity.MEDIUM,
    SecretType.JWT_TOKEN:      Severity.MEDIUM,
    SecretType.PASSWORD:       Severity.MEDIUM,
    SecretType.GENERIC_SECRET: Severity.LOW,
    SecretType.DETECT_SECRETS: Severity.MEDIUM,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SecretFinding:
    """A single detected secret."""
    secret_type:  SecretType
    severity:     Severity
    line_number:  int               # 1-indexed
    column_start: int               # 0-indexed character position in the line
    placeholder:  str               # what replaces the secret in sanitized code
    masked_value: str               # first 4 chars + "****" for audit trail
    description:  str               # human-readable explanation
    detector:     str               # "regex" | "detect-secrets"

    @classmethod
    def build(
        cls,
        secret_type: SecretType,
        raw_value: str,
        line_number: int,
        column_start: int,
        detector: str,
        description: str = "",
    ) -> "SecretFinding":
        severity    = _SEVERITY_MAP.get(secret_type, Severity.LOW)
        placeholder = _make_placeholder(secret_type, line_number)
        masked      = _mask(raw_value)
        return cls(
            secret_type=secret_type,
            severity=severity,
            line_number=line_number,
            column_start=column_start,
            placeholder=placeholder,
            masked_value=masked,
            description=description or secret_type.value.replace("_", " ").title(),
            detector=detector,
        )


@dataclass
class SanitizerInput:
    source_code: str
    filename:    str = "unnamed.txt"


@dataclass
class SanitizerResult:
    """Complete output of the sanitizer agent."""
    sanitized_code:   str
    findings:         list[SecretFinding]
    number_of_secrets: int
    highest_severity:  Optional[Severity]
    filename:          str
    warnings:          list[str] = field(default_factory=list)

    # Convenience helpers ─────────────────────────────────────────────────────

    @property
    def is_clean(self) -> bool:
        return self.number_of_secrets == 0

    def findings_by_severity(self) -> dict[Severity, list[SecretFinding]]:
        out: dict[Severity, list[SecretFinding]] = {s: [] for s in Severity}
        for f in self.findings:
            out[f.severity].append(f)
        return out

    def summary(self) -> str:
        if self.is_clean:
            return "No secrets detected."
        sev = self.highest_severity.value.upper() if self.highest_severity else "?"
        return (
            f"{self.number_of_secrets} secret(s) found — "
            f"highest severity: {sev}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regex-based detection patterns
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _RegexPattern:
    name:        SecretType
    pattern:     re.Pattern
    group:       int = 1    # capture group that contains the secret value
    description: str = ""


# Each pattern captures the SECRET VALUE in group `group`.
# We use named groups for clarity.
_REGEX_PATTERNS: list[_RegexPattern] = [

    # ── Private / PEM keys ───────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.PRIVATE_KEY,
        pattern=re.compile(
            r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[^-]+-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)",
            re.DOTALL,
        ),
        description="PEM private key block",
    ),

    # ── AWS ──────────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.AWS_ACCESS_KEY,
        pattern=re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        description="AWS Access Key ID",
    ),
    _RegexPattern(
        name=SecretType.AWS_SECRET_KEY,
        pattern=re.compile(
            r"""(?i)(?:aws[_\-\s]?secret[_\-\s]?(?:access[_\-\s]?)?key|aws_secret)\s*[=:]\s*['"]([A-Za-z0-9/+]{40})['"]"""
        ),
        description="AWS Secret Access Key",
    ),

    # ── Google / GCP ─────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.GOOGLE_API_KEY,
        # AIza followed by 35 or more characters (real keys are 39 chars total)
        pattern=re.compile(r"""(?<![A-Za-z0-9_])(AIza[0-9A-Za-z\-_]{35,})(?![A-Za-z0-9_])"""),
        description="Google / GCP API Key",
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.OPENAI_KEY,
        pattern=re.compile(r"""\b(sk-[A-Za-z0-9]{20,}(?:T3BlbkFJ[A-Za-z0-9]{20,})?)\b"""),
        description="OpenAI API Key",
    ),

    # ── GitHub ───────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.GITHUB_TOKEN,
        pattern=re.compile(r"""\b(gh[pousr]_[A-Za-z0-9_]{36,})\b"""),
        description="GitHub Personal Access Token",
    ),

    # ── Stripe ───────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.STRIPE_KEY,
        pattern=re.compile(r"""\b((?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,})\b"""),
        description="Stripe API Key",
    ),

    # ── Slack ────────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.SLACK_TOKEN,
        pattern=re.compile(r"""\b(xox[baprs]-[0-9A-Za-z\-]{10,})\b"""),
        description="Slack Token",
    ),

    # ── JWT ──────────────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.JWT_TOKEN,
        pattern=re.compile(
            r"""(?<![A-Za-z0-9_-])(eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})(?![A-Za-z0-9_-])"""
        ),
        description="JSON Web Token (JWT)",
    ),

    # ── Database URLs ────────────────────────────────────────────────────────
    _RegexPattern(
        name=SecretType.DATABASE_URL,
        pattern=re.compile(
            r"""((?:postgresql|postgres|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)\://[^:]+:[^@\s'"]+@[^\s'"]+)""",
            re.IGNORECASE,
        ),
        description="Database connection URL with embedded credentials",
    ),

    # ── Generic password assignment ───────────────────────────────────────────
    # Matches simple:      password = "..."
    # Matches dict access: config["passwd"] = "..."
    # Uses a lookahead approach: match keyword anywhere before the value.
    _RegexPattern(
        name=SecretType.PASSWORD,
        pattern=re.compile(
            r"""(?i)(?:password|passwd|pass|pwd|secret).{0,40}?=\s*['"]([^'"]{6,})['"]""" ,
            re.DOTALL,
        ),
        description="Password / secret assignment",
    ),

    # ── Generic token / key assignment ────────────────────────────────────────
    _RegexPattern(
        name=SecretType.TOKEN,
        pattern=re.compile(
            r"""(?i)(?:api[_\-]?key|access[_\-]?token|auth[_\-]?token|bearer[_\-]?token|secret[_\-]?key|private[_\-]?key)\s*[=:]\s*['"]([^'"]{8,})['"]"""
        ),
        description="Generic API key / token assignment",
    ),

    # ── Generic high-entropy string in assignment (low confidence) ────────────
    # MUST stay last: the deduplication logic keeps the highest-severity
    # finding per line, so specific patterns above override this catch-all.
    _RegexPattern(
        name=SecretType.GENERIC_SECRET,
        pattern=re.compile(
            r"""(?i)(?:key|token|secret|credential)\s*[=:]\s*['"]([A-Za-z0-9+/=_\-]{16,})['"]"""
        ),
        description="High-entropy string in a key/token/secret variable",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_placeholder(secret_type: SecretType, line: int) -> str:
    tag = secret_type.value.upper()
    return f"<REDACTED_{tag}_L{line}>"


def _mask(value: str) -> str:
    """Show first 4 characters, mask the rest — for audit trail."""
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * min(len(value) - 4, 12)


def _line_number_of_offset(source: str, offset: int) -> tuple[int, int]:
    """Return (1-indexed line number, 0-indexed column) for a char offset."""
    before = source[:offset]
    line   = before.count("\n") + 1
    col    = offset - before.rfind("\n") - 1
    return line, col


def _compute_severity(findings: list[SecretFinding]) -> Optional[Severity]:
    """Return the highest severity across all findings, or None if empty."""
    order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    for sev in order:
        if any(f.severity == sev for f in findings):
            return sev
    return None


# Severity ordering used to pick the more-specific finding when two patterns
# match the same line.  Lower index = higher priority (kept over others).
_PRIORITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
]


def _deduplicate(findings: list[SecretFinding]) -> list[SecretFinding]:
    """
    Remove duplicate findings on the same line.

    When two findings share the same line number, we keep the one with the
    higher severity (more specific pattern).  This prevents the low-priority
    GENERIC_SECRET catch-all from shadowing precise matches like GOOGLE_API_KEY.
    """
    # Map line_number → best (highest-severity) finding seen so far
    best: dict[int, SecretFinding] = {}
    for f in findings:
        ln = f.line_number
        if ln not in best:
            best[ln] = f
        else:
            existing_pri = _PRIORITY_ORDER.index(best[ln].severity)
            new_pri      = _PRIORITY_ORDER.index(f.severity)
            if new_pri < existing_pri:   # lower index = higher priority
                best[ln] = f
    return list(best.values())


# ─────────────────────────────────────────────────────────────────────────────
# Detection layers
# ─────────────────────────────────────────────────────────────────────────────

def _run_regex_detection(source: str) -> list[tuple[SecretFinding, int, int]]:
    """
    Run all regex patterns against `source`.

    Returns list of (finding, match_start, match_end) tuples so the
    caller can perform string substitution.
    """
    results: list[tuple[SecretFinding, int, int]] = []

    for rp in _REGEX_PATTERNS:
        for match in rp.pattern.finditer(source):
            try:
                secret_value = match.group(rp.group)
                value_start  = match.start(rp.group)
                value_end    = match.end(rp.group)
            except IndexError:
                secret_value = match.group(0)
                value_start  = match.start()
                value_end    = match.end()

            line, col = _line_number_of_offset(source, value_start)
            finding   = SecretFinding.build(
                secret_type  = rp.name,
                raw_value    = secret_value,
                line_number  = line,
                column_start = col,
                detector     = "regex",
                description  = rp.description,
            )
            results.append((finding, value_start, value_end))

    # Sort by start position so replacements don't overlap
    results.sort(key=lambda t: t[1])
    return results


def _run_detect_secrets(source: str, filename: str) -> list[SecretFinding]:
    """
    Run detect-secrets if installed; gracefully skip otherwise.
    Returns findings (no position data — we add line info from the library).
    """
    spec = importlib.util.find_spec("detect_secrets")
    if spec is None:
        return []

    try:
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import transient_settings

        config = {
            "plugins_used": [
                {"name": "AWSKeyDetector"},
                {"name": "ArtifactoryDetector"},
                {"name": "BasicAuthDetector"},
                {"name": "CloudantDetector"},
                {"name": "DiscordBotTokenDetector"},
                {"name": "GitHubTokenDetector"},
                {"name": "HexHighEntropyString", "limit": 3.0},
                {"name": "IbmCloudIamDetector"},
                {"name": "IbmCosHmacDetector"},
                {"name": "JwtTokenDetector"},
                {"name": "KeywordDetector"},
                {"name": "MailchimpDetector"},
                {"name": "NpmDetector"},
                {"name": "OpenAIDetector"},
                {"name": "PrivateKeyDetector"},
                {"name": "SendGridDetector"},
                {"name": "SlackDetector"},
                {"name": "SoftlayerDetector"},
                {"name": "SquareOAuthDetector"},
                {"name": "StripeDetector"},
                {"name": "TwilioKeyDetector"},
                {"name": "Base64HighEntropyString", "limit": 4.5},
            ],
        }

        findings: list[SecretFinding] = []
        lines = source.splitlines(keepends=True)

        with transient_settings(config):
            secrets = SecretsCollection()
            secrets.scan_diff(
                "".join(
                    f"+{l}" for l in lines
                ),
            )

            for _fname, secret_set in secrets:
                for secret in secret_set:
                    line_no = secret.line_number or 1
                    finding = SecretFinding.build(
                        secret_type  = SecretType.DETECT_SECRETS,
                        raw_value    = secret.secret_value or "****",
                        line_number  = line_no,
                        column_start = 0,
                        detector     = "detect-secrets",
                        description  = f"detect-secrets: {secret.type}",
                    )
                    findings.append(finding)

        return findings

    except Exception:
        # Never crash the caller due to detect-secrets internals
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Sanitization: apply replacements to source string
# ─────────────────────────────────────────────────────────────────────────────

def _apply_replacements(
    source: str,
    replacements: list[tuple[SecretFinding, int, int]],
) -> str:
    """
    Replace secret spans in `source` with their placeholders.
    Works right-to-left so earlier offsets remain valid after each replacement.
    """
    # Sort descending by start position to replace from end → start
    sorted_reps = sorted(replacements, key=lambda t: t[1], reverse=True)
    result = source
    for finding, start, end in sorted_reps:
        result = result[:start] + finding.placeholder + result[end:]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(inp: SanitizerInput) -> SanitizerResult:
    """
    Main entry point.  Run both detection layers, merge findings, apply
    placeholder substitution, and return a structured SanitizerResult.

    This function is pure (no side-effects) and ready to become a
    LangGraph node by wrapping it in a thin node function:

        def sanitizer_node(state: GraphState) -> GraphState:
            result = sanitize(SanitizerInput(source_code=state["raw_code"]))
            return {**state, "sanitizer_result": result}
    """
    source   = inp.source_code
    warnings: list[str] = []

    # ── Layer 1: regex ────────────────────────────────────────────────────────
    regex_hits = _run_regex_detection(source)
    regex_findings = [f for f, _, _ in regex_hits]

    # ── Layer 2: detect-secrets ───────────────────────────────────────────────
    ds_findings = _run_detect_secrets(source, inp.filename)
    if not importlib.util.find_spec("detect_secrets"):
        warnings.append(
            "detect-secrets is not installed. "
            "Only regex-based detection was performed. "
            "Install it with: pip install detect-secrets"
        )

    # ── Merge and deduplicate ─────────────────────────────────────────────────
    # Regex findings take priority (they have precise position data).
    # detect-secrets findings are added only if they report a line not
    # already covered by a regex finding.
    regex_lines: set[int] = {f.line_number for f in regex_findings}
    additional_ds = [f for f in ds_findings if f.line_number not in regex_lines]

    all_findings = _deduplicate(regex_findings + additional_ds)
    all_findings.sort(key=lambda f: f.line_number)

    # ── Apply replacements ────────────────────────────────────────────────────
    # Build a set of (line_number, secret_type) for the WINNING findings so we
    # only apply the replacement that corresponds to the deduped finding.
    # This prevents the GENERIC_SECRET catch-all from replacing a span that was
    # already won by a more-specific pattern (e.g. GOOGLE_API_KEY).
    winning_keys = {(f.line_number, f.secret_type) for f in all_findings}
    winning_hits = [
        (f, start, end)
        for f, start, end in regex_hits
        if (f.line_number, f.secret_type) in winning_keys
    ]
    sanitized = _apply_replacements(source, winning_hits)

    return SanitizerResult(
        sanitized_code    = sanitized,
        findings          = all_findings,
        number_of_secrets = len(all_findings),
        highest_severity  = _compute_severity(all_findings),
        filename          = inp.filename,
        warnings          = warnings,
    )
