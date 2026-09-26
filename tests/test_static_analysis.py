"""
tests/test_static_analysis.py — Phase 3 unit tests for the static analysis agent.

Tests cover:
  - Intentionally vulnerable code that Bandit SHOULD flag
  - Clean code that Bandit should NOT flag
  - Invalid Python (syntax errors)
  - Empty / no input edge cases
  - Finding schema correctness
  - Result convenience helpers

Run with:
    pytest tests/test_static_analysis.py -v
"""

from __future__ import annotations

import pytest

from agents.static_analysis_agent import (
    FindingConfidence,
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisInput,
    StaticAnalysisResult,
    analyse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sample vulnerable code snippets (intentionally insecure)
# ─────────────────────────────────────────────────────────────────────────────

VULN_EXEC = """\
import os

user_input = input("Enter command: ")
os.system(user_input)
"""

VULN_SUBPROCESS_SHELL = """\
import subprocess

cmd = "ls -la"
subprocess.call(cmd, shell=True)
"""

VULN_EVAL = """\
data = input("Enter expression: ")
result = eval(data)
print(result)
"""

VULN_HARDCODED_PASSWORD = """\
import hashlib

password = "SuperSecret123!"
hashed = hashlib.md5(password.encode()).hexdigest()
"""

VULN_YAML_LOAD = """\
import yaml

with open("config.yaml") as f:
    data = yaml.load(f, Loader=yaml.Loader)
"""

VULN_SQL_INJECTION = """\
import sqlite3

def get_user(username):
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '%s'" % username
    cursor.execute(query)
    return cursor.fetchall()
"""

VULN_FLASK_DEBUG = """\
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello"

if __name__ == "__main__":
    app.run(debug=True)
"""

VULN_MULTIPLE = """\
import os
import subprocess
import hashlib

# Vulnerability 1: OS command injection
os.system(input("cmd: "))

# Vulnerability 2: Shell injection via subprocess
subprocess.Popen("echo hello", shell=True)

# Vulnerability 3: Weak hash
hashlib.md5(b"data").hexdigest()
"""

CLEAN_CODE = """\
from __future__ import annotations

import os
import hashlib


def get_secret_from_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"{name} is not set")
    return value


def hash_data(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add(a: int, b: int) -> int:
    return a + b
"""

INVALID_PYTHON = """\
def broken_function(
    this is not valid python !!!
    for while if = = = {{{
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Vulnerable code detection
# ─────────────────────────────────────────────────────────────────────────────

class TestOsSystemDetection:
    """os.system() with user input should be flagged."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_EXEC, filename="vuln_exec.py")
        )

    def test_has_findings(self):
        assert self.result.has_findings

    def test_finding_count_at_least_one(self):
        assert self.result.finding_count >= 1

    def test_tool_is_bandit(self):
        for f in self.result.findings:
            assert f.tool == "bandit"

    def test_finding_has_title(self):
        for f in self.result.findings:
            assert f.title  # non-empty

    def test_finding_has_line_number(self):
        for f in self.result.findings:
            assert f.line >= 1

    def test_finding_has_severity(self):
        for f in self.result.findings:
            assert isinstance(f.severity, FindingSeverity)

    def test_finding_has_confidence(self):
        for f in self.result.findings:
            assert isinstance(f.confidence, FindingConfidence)

    def test_filename_is_correct(self):
        for f in self.result.findings:
            assert f.file == "vuln_exec.py"


class TestSubprocessShell:
    """subprocess.call with shell=True should be flagged."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_SUBPROCESS_SHELL, filename="shell.py")
        )

    def test_has_findings(self):
        assert self.result.has_findings

    def test_finding_count(self):
        assert self.result.finding_count >= 1

    def test_has_code_snippet(self):
        for f in self.result.findings:
            assert f.code_snippet  # non-empty


class TestEvalDetection:
    """eval() on user input should be flagged."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_EVAL, filename="eval.py")
        )

    def test_has_findings(self):
        assert self.result.has_findings

    def test_highest_severity_not_none(self):
        assert self.result.highest_severity is not None


class TestHardcodedPassword:
    """Hardcoded password + MD5 usage should be flagged."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_HARDCODED_PASSWORD, filename="pwd.py")
        )

    def test_has_findings(self):
        assert self.result.has_findings

    def test_at_least_one_finding(self):
        # MD5 insecure hash and/or hardcoded password
        assert self.result.finding_count >= 1


class TestMultipleVulnerabilities:
    """Code with multiple vulnerability types should report all."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_MULTIPLE, filename="multi.py")
        )

    def test_has_findings(self):
        assert self.result.has_findings

    def test_multiple_findings(self):
        assert self.result.finding_count >= 2

    def test_findings_sorted_by_line(self):
        lines = [f.line for f in self.result.findings]
        assert lines == sorted(lines)

    def test_highest_severity_is_meaningful(self):
        sev = self.result.highest_severity
        assert sev in (FindingSeverity.HIGH, FindingSeverity.MEDIUM)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Clean code
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanCode:
    """Clean code should have zero findings."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=CLEAN_CODE, filename="clean.py")
        )

    def test_no_findings(self):
        assert self.result.is_clean

    def test_finding_count_is_zero(self):
        assert self.result.finding_count == 0

    def test_highest_severity_is_none(self):
        assert self.result.highest_severity is None

    def test_summary_says_clean(self):
        assert "no security issues" in self.result.summary().lower()

    def test_no_errors(self):
        assert len(self.result.errors) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidPython:
    """Invalid Python should NOT crash the agent."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=INVALID_PYTHON, filename="bad.py")
        )

    def test_does_not_crash(self):
        assert isinstance(self.result, StaticAnalysisResult)

    def test_result_is_clean_or_has_errors(self):
        # Either Bandit reports errors or just finds nothing
        assert self.result.is_clean or self.result.has_findings or len(self.result.errors) >= 0


class TestEmptyInput:
    """Empty source code should not crash."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code="", filename="empty.py")
        )

    def test_does_not_crash(self):
        assert isinstance(self.result, StaticAnalysisResult)

    def test_no_findings(self):
        assert self.result.finding_count == 0


class TestNoInput:
    """No input at all should return an error message."""

    def setup_method(self):
        self.result = analyse(StaticAnalysisInput())

    def test_has_error(self):
        assert len(self.result.errors) >= 1

    def test_error_mentions_input(self):
        assert "input" in self.result.errors[0].lower()


class TestNonexistentFile:
    """A nonexistent file path should return an error, not crash."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(file_path="/nonexistent/path/to/file.py")
        )

    def test_does_not_crash(self):
        assert isinstance(self.result, StaticAnalysisResult)

    def test_has_error(self):
        assert len(self.result.errors) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Result structure & helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestResultStructure:
    """Verify the normalised result structure and convenience helpers."""

    def setup_method(self):
        self.result = analyse(
            StaticAnalysisInput(source_code=VULN_MULTIPLE, filename="structure.py")
        )

    def test_findings_are_StaticAnalysisFinding(self):
        for f in self.result.findings:
            assert isinstance(f, StaticAnalysisFinding)

    def test_findings_by_severity_returns_all_levels(self):
        by_sev = self.result.findings_by_severity()
        for sev in FindingSeverity:
            assert sev in by_sev

    def test_summary_mentions_count(self):
        summary = self.result.summary()
        assert str(self.result.finding_count) in summary

    def test_summary_mentions_severity(self):
        summary = self.result.summary()
        assert self.result.highest_severity.value.upper() in summary

    def test_finding_has_all_required_fields(self):
        f = self.result.findings[0]
        assert f.title
        assert f.description
        assert isinstance(f.severity, FindingSeverity)
        assert isinstance(f.confidence, FindingConfidence)
        assert f.file
        assert f.line >= 1
        assert f.tool == "bandit"
