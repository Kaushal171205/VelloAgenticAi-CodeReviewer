"""
tools/bandit_wrapper.py — Safe wrapper around the Bandit static analysis tool.

Runs Bandit as a subprocess, captures JSON output, and returns structured
findings.  Designed to never crash the caller — errors are returned as part
of the result rather than raised as exceptions.

Usage:
    from tools.bandit_wrapper import run_bandit, BanditInput

    result = run_bandit(BanditInput(source_code="import os; os.system('ls')"))
    for finding in result.findings:
        print(finding.title, finding.severity)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class BanditSeverity(str, Enum):
    """Severity levels matching Bandit's own classification."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class BanditConfidence(str, Enum):
    """Confidence levels matching Bandit's own classification."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BanditFinding:
    """A single finding from Bandit, normalised into a clean schema."""
    test_id:       str               # e.g. "B102"
    test_name:     str               # e.g. "exec_used"
    title:         str               # human-readable summary
    description:   str               # detailed Bandit issue text
    severity:      BanditSeverity
    confidence:    BanditConfidence
    file:          str               # filename
    line:          int               # 1-indexed line number
    end_line:      int               # ending line (may equal `line`)
    code_snippet:  str               # the offending source line(s)
    cwe_id:        Optional[str]     # CWE identifier if provided
    more_info:     str               # URL to Bandit docs for this test
    tool:          str = "bandit"


@dataclass
class BanditInput:
    """Input to the Bandit wrapper.

    Supply *either* ``source_code`` (arbitrary Python string) or
    ``file_path`` (path to an existing .py file).  If both are given,
    ``source_code`` takes precedence.
    """
    source_code: Optional[str] = None
    file_path:   Optional[str] = None
    filename:    str           = "input.py"     # display name when using source_code


@dataclass
class BanditResult:
    """Complete output of a Bandit run."""
    findings:     list[BanditFinding] = field(default_factory=list)
    errors:       list[str]           = field(default_factory=list)
    metrics:      dict                = field(default_factory=dict)
    return_code:  int                 = 0
    raw_json:     Optional[str]       = None

    # Convenience helpers ─────────────────────────────────────────────────────

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def highest_severity(self) -> Optional[BanditSeverity]:
        if not self.findings:
            return None
        rank = {BanditSeverity.HIGH: 3, BanditSeverity.MEDIUM: 2, BanditSeverity.LOW: 1}
        return max(self.findings, key=lambda f: rank[f.severity]).severity


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_severity(raw: str) -> BanditSeverity:
    try:
        return BanditSeverity(raw.lower())
    except ValueError:
        return BanditSeverity.LOW


def _parse_confidence(raw: str) -> BanditConfidence:
    try:
        return BanditConfidence(raw.lower())
    except ValueError:
        return BanditConfidence.LOW


def _parse_finding(raw: dict, display_filename: str) -> BanditFinding:
    """Convert a single Bandit JSON result dict into a BanditFinding."""
    cwe_data = raw.get("issue_cwe", {})
    cwe_id = None
    if isinstance(cwe_data, dict) and cwe_data.get("id"):
        cwe_id = f"CWE-{cwe_data['id']}"

    test_id   = raw.get("test_id", "B000")
    test_name = raw.get("test_name", "unknown")
    issue_text = raw.get("issue_text", "")

    return BanditFinding(
        test_id=test_id,
        test_name=test_name,
        title=f"[{test_id}] {issue_text}",
        description=issue_text,
        severity=_parse_severity(raw.get("issue_severity", "LOW")),
        confidence=_parse_confidence(raw.get("issue_confidence", "LOW")),
        file=display_filename,
        line=raw.get("line_number", 0),
        end_line=raw.get("end_col_offset", raw.get("line_number", 0)),
        code_snippet=raw.get("code", "").strip(),
        cwe_id=cwe_id,
        more_info=raw.get("more_info", ""),
        tool="bandit",
    )


def _find_bandit() -> str:
    """Resolve the Bandit executable path.

    Preference order:
    1. ``bandit`` on PATH
    2. ``python -m bandit`` via the current interpreter
    """
    # Check if bandit is on PATH
    try:
        result = subprocess.run(
            ["bandit", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return "bandit"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return ""  # will fall back to ``python -m bandit``


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_bandit(inp: BanditInput) -> BanditResult:
    """Run Bandit on the given input and return structured results.

    This function **never raises** — all errors are captured inside the
    returned ``BanditResult.errors`` list.
    """
    if inp.source_code is None and inp.file_path is None:
        return BanditResult(errors=["No input provided: set source_code or file_path."])

    # ── Determine target file ────────────────────────────────────────────
    tmp_file = None
    target_path: str

    try:
        if inp.source_code is not None:
            # Write source to a temp file for Bandit to analyse
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="bandit_input_",
                delete=False,
            )
            tmp_file.write(inp.source_code)
            tmp_file.flush()
            tmp_file.close()
            target_path = tmp_file.name
            display_filename = inp.filename
        else:
            target_path = str(inp.file_path)
            display_filename = os.path.basename(target_path)
            if not os.path.isfile(target_path):
                return BanditResult(errors=[f"File not found: {target_path}"])

        # ── Build the command ────────────────────────────────────────────
        bandit_exe = _find_bandit()
        if bandit_exe:
            cmd = [bandit_exe, "-f", "json", "-q", target_path]
        else:
            cmd = [sys.executable, "-m", "bandit", "-f", "json", "-q", target_path]

        # ── Execute ──────────────────────────────────────────────────────
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Bandit returns exit code 1 when it finds issues — that's normal.
        # Exit code 2 means Bandit itself errored out.
        raw_stdout = proc.stdout.strip()
        raw_stderr = proc.stderr.strip()

        if proc.returncode == 2 and not raw_stdout:
            return BanditResult(
                errors=[f"Bandit error (exit {proc.returncode}): {raw_stderr}"],
                return_code=proc.returncode,
            )

        if not raw_stdout:
            # No JSON output — Bandit found nothing or produced no output
            return BanditResult(return_code=proc.returncode, raw_json="")

        # ── Parse JSON output ────────────────────────────────────────────
        try:
            data = json.loads(raw_stdout)
        except json.JSONDecodeError as exc:
            return BanditResult(
                errors=[f"Failed to parse Bandit JSON output: {exc}"],
                return_code=proc.returncode,
                raw_json=raw_stdout,
            )

        # ── Convert results ──────────────────────────────────────────────
        findings: list[BanditFinding] = []
        for raw_finding in data.get("results", []):
            findings.append(_parse_finding(raw_finding, display_filename))

        # Sort by line number for consistent presentation
        findings.sort(key=lambda f: f.line)

        bandit_errors = data.get("errors", [])
        error_strings = []
        for err in bandit_errors:
            if isinstance(err, dict):
                error_strings.append(err.get("reason", str(err)))
            else:
                error_strings.append(str(err))

        return BanditResult(
            findings=findings,
            errors=error_strings,
            metrics=data.get("metrics", {}),
            return_code=proc.returncode,
            raw_json=raw_stdout,
        )

    except subprocess.TimeoutExpired:
        return BanditResult(errors=["Bandit timed out after 60 seconds."])
    except FileNotFoundError:
        return BanditResult(
            errors=[
                "Bandit is not installed. Install with: pip install bandit"
            ]
        )
    except Exception as exc:
        return BanditResult(errors=[f"Unexpected error running Bandit: {exc}"])
    finally:
        if tmp_file is not None:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
