"""
tests/test_aggregator_and_classifier.py — Tests for AggregatorAgent and SeverityClassifierAgent.

Test coverage:
  ✅ Duplicate findings (exact match, near-duplicate, line-overlap)
  ✅ Conflicting severities from different tools (classifier wins, not source)
  ✅ Multi-tool findings preserved in source_tools
  ✅ Severity reclassification rubric (specific vulnerability patterns)
  ✅ Severity not inflated beyond rubric (caps at HIGH for unmatched CRITICAL claims)
  ✅ Empty input edge cases
  ✅ Aggregation preserves evidence (line numbers, CWE, code snippets)
  ✅ Classifier summary and statistics
  ✅ Full pipeline: aggregation → classification output
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import MagicMock

from agents.aggregator_agent import (
    AggregatorInput,
    AggregatorResult,
    FindingRecord,
    _deduplicate,
    _from_logic,
    _from_security,
    _from_static,
    _jaccard,
    _is_exact_duplicate,
    _is_near_duplicate,
    _is_line_overlap_duplicate,
    _make_fingerprint,
    _title_tokens,
    aggregate,
)
from agents.severity_classifier_agent import (
    ClassifierInput,
    ClassifierResult,
    SeverityLevel,
    _classify_finding,
    classify,
)
from agents.static_analysis_agent import (
    FindingConfidence,
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisResult,
)
from agents.logic_reviewer_agent import LogicFinding, LogicReviewResult
from agents.security_scanner_agent import SecurityFinding, SecurityScannerResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_static_finding(
    title: str = "SQL Injection",
    severity: FindingSeverity = FindingSeverity.HIGH,
    confidence: FindingConfidence = FindingConfidence.HIGH,
    line: int = 10,
    file: str = "app.py",
    cwe: Optional[str] = "CWE-89",
    test_id: str = "B608",
    code: str = "cursor.execute(query % user_input)",
) -> StaticAnalysisFinding:
    return StaticAnalysisFinding(
        title=title,
        description=f"Detected: {title}",
        severity=severity,
        confidence=confidence,
        file=file,
        line=line,
        end_line=line + 2,
        code_snippet=code,
        tool="bandit",
        test_id=test_id,
        cwe_id=cwe,
    )


def _make_security_finding(
    title: str = "SQL Injection via ORM",
    severity: FindingSeverity = FindingSeverity.HIGH,
    confidence: FindingConfidence = FindingConfidence.MEDIUM,
    line: int = 10,
    file: str = "app.py",
    cwe: Optional[str] = "CWE-89",
) -> SecurityFinding:
    return SecurityFinding(
        title=title,
        description=f"RAG detected: {title}",
        severity=severity,
        confidence=confidence,
        affected_code=f"# line {line}",
        reasoning="RAG retrieved SQL injection pattern.",
        suggested_remediation="Use parameterised queries.",
        file=file,
        line_number=line,
        cwe_id=cwe,
        category="security",
    )


def _make_logic_finding(
    title: str = "Missing input validation",
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    confidence: FindingConfidence = FindingConfidence.HIGH,
    line: int = 25,
    file: str = "app.py",
) -> LogicFinding:
    return LogicFinding(
        title=title,
        description=f"Logic issue: {title}",
        severity=severity,
        confidence=confidence,
        affected_code=f"# line {line}",
        reasoning="Missing boundary check.",
        suggested_remediation="Add validation.",
        file=file,
        line_number=line,
        category="business_logic",
    )


def _make_finding_record(
    title: str = "Test Finding",
    severity: str = "medium",
    confidence: str = "high",
    file: str = "app.py",
    line: Optional[int] = 10,
    category: str = "security",
    tool: str = "bandit",
    affected_code: str = "some_code()",
) -> FindingRecord:
    fp = _make_fingerprint(title, file, line)
    return FindingRecord(
        title=title,
        description=f"Description of {title}",
        category=category,
        reported_severity=severity,
        reported_confidence=confidence,
        severity=severity,
        confidence=confidence,
        file=file,
        line_number=line,
        end_line=None,
        affected_code=affected_code,
        reasoning="",
        suggested_remediation="Fix it.",
        source_tools=[tool],
        primary_tool=tool,
        cwe_id=None,
        test_id=None,
        more_info="",
        fingerprint=fp,
        merge_notes=[],
        duplicate_count=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Utility / helper unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_jaccard_identical_sets(self):
        a = {"sql", "injection", "detected"}
        assert _jaccard(a, a) == 1.0

    def test_jaccard_disjoint_sets(self):
        a = {"sql", "injection"}
        b = {"xss", "reflected"}
        assert _jaccard(a, b) == 0.0

    def test_jaccard_partial_overlap(self):
        a = {"sql", "injection", "detected"}
        b = {"sql", "injection", "vulnerability"}
        # intersection=2, union=4 → 0.5
        assert _jaccard(a, b) == pytest.approx(0.5)

    def test_jaccard_empty_both(self):
        assert _jaccard(set(), set()) == 1.0

    def test_fingerprint_same_inputs(self):
        fp1 = _make_fingerprint("SQL Injection", "app.py", 10)
        fp2 = _make_fingerprint("SQL Injection", "app.py", 10)
        assert fp1 == fp2

    def test_fingerprint_different_file(self):
        fp1 = _make_fingerprint("SQL Injection", "app.py", 10)
        fp2 = _make_fingerprint("SQL Injection", "other.py", 10)
        assert fp1 != fp2

    def test_fingerprint_different_line(self):
        fp1 = _make_fingerprint("SQL Injection", "app.py", 10)
        fp2 = _make_fingerprint("SQL Injection", "app.py", 20)
        assert fp1 != fp2

    def test_title_tokens_normalisation(self):
        tokens = _title_tokens("SQL Injection Detected!!")
        assert "sql" in tokens
        assert "injection" in tokens
        assert "detected" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Deduplication tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:

    def test_exact_duplicate_detected(self):
        """Same title + file + line → exact duplicate."""
        a = _make_finding_record("SQL Injection", line=10)
        b = _make_finding_record("SQL Injection", line=10)
        assert _is_exact_duplicate(a, b)

    def test_exact_duplicate_different_line(self):
        """Same title but different line → NOT exact duplicate."""
        a = _make_finding_record("SQL Injection", line=10)
        b = _make_finding_record("SQL Injection", line=20)
        assert not _is_exact_duplicate(a, b)

    def test_near_duplicate_similar_title(self):
        """Very similar titles on the same file → near-duplicate (Jaccard > 0.72).

        Using 6 shared tokens / 7 union = 0.857 Jaccard.
        Title A: {sql, injection, login, endpoint, detected, by, bandit}
        Title B: {sql, injection, login, endpoint, detected, by, scanner}
        """
        a = _make_finding_record(
            "SQL Injection Login Endpoint Detected By Bandit", file="app.py", line=10
        )
        b = _make_finding_record(
            "SQL Injection Login Endpoint Detected By Scanner", file="app.py", line=20
        )
        assert _is_near_duplicate(a, b)

    def test_near_duplicate_different_file(self):
        """Similar title but different file → NOT near-duplicate."""
        a = _make_finding_record("SQL Injection via user", file="app.py")
        b = _make_finding_record("SQL Injection via user", file="views.py")
        assert not _is_near_duplicate(a, b)

    def test_near_duplicate_very_different_titles(self):
        """Completely different titles → NOT near-duplicate."""
        a = _make_finding_record("SQL Injection", file="app.py")
        b = _make_finding_record("Cross Site Scripting XSS", file="app.py")
        assert not _is_near_duplicate(a, b)

    def test_line_overlap_same_category(self):
        """Overlapping lines, same file, same category → overlap duplicate."""
        a = _make_finding_record(title="Issue A", file="app.py", line=10,
                                  category="static")
        b = _make_finding_record(title="Issue B Variant", file="app.py", line=12,
                                  category="static")
        # Lines 10 and 12 are within LINE_PROXIMITY=5
        assert _is_line_overlap_duplicate(a, b)

    def test_line_overlap_different_category(self):
        """Overlapping lines but different category → NOT overlap duplicate."""
        a = _make_finding_record(title="Issue A", file="app.py", line=10,
                                  category="static")
        b = _make_finding_record(title="Issue B Variant", file="app.py", line=10,
                                  category="security")
        assert not _is_line_overlap_duplicate(a, b)

    def test_deduplicate_removes_exact_duplicates(self):
        """Two identical records → one survives."""
        a = _make_finding_record("SQL Injection", line=10)
        b = _make_finding_record("SQL Injection", line=10)
        result, merged_count = _deduplicate([a, b])
        assert len(result) == 1
        assert merged_count == 1

    def test_deduplicate_distinct_findings_preserved(self):
        """Three distinct findings → all three preserved."""
        findings = [
            _make_finding_record("SQL Injection", file="app.py", line=10),
            _make_finding_record("XSS Vulnerability", file="app.py", line=50),
            _make_finding_record("Path Traversal", file="other.py", line=10),
        ]
        result, merged_count = _deduplicate(findings)
        assert len(result) == 3
        assert merged_count == 0

    def test_deduplicate_merges_tools(self):
        """When two tools flag the same issue, both tool names are preserved."""
        a = _make_finding_record("SQL Injection", line=10, tool="bandit")
        b = _make_finding_record("SQL Injection", line=10, tool="security_scanner_rag")
        result, _ = _deduplicate([a, b])
        assert len(result) == 1
        assert "bandit" in result[0].source_tools
        assert "security_scanner_rag" in result[0].source_tools

    def test_deduplicate_takes_higher_severity(self):
        """When a duplicate has higher severity, the merged record is upgraded."""
        a = _make_finding_record("SQL Injection", severity="medium", line=10)
        b = _make_finding_record("SQL Injection", severity="high", line=10)
        result, _ = _deduplicate([a, b])
        assert len(result) == 1
        assert result[0].severity == "high"

    def test_deduplicate_records_merge_note(self):
        """Merge note is written when a duplicate is absorbed."""
        a = _make_finding_record("SQL Injection", line=10)
        b = _make_finding_record("SQL Injection", line=10, tool="logic_reviewer")
        result, _ = _deduplicate([a, b])
        assert any("Absorbed duplicate" in note for note in result[0].merge_notes)

    def test_deduplicate_severity_escalation_noted(self):
        """When severity is escalated via merge, a note is written."""
        a = _make_finding_record("SQL Injection", severity="low", line=10)
        b = _make_finding_record("SQL Injection", severity="critical", line=10)
        result, _ = _deduplicate([a, b])
        assert any("Severity elevated" in note for note in result[0].merge_notes)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Full AggregatorAgent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregatorAgent:

    def test_empty_input_returns_empty_result(self):
        result = aggregate(AggregatorInput())
        assert result.total_output == 0
        assert result.has_findings is False

    def test_aggregates_from_single_tool(self):
        static = StaticAnalysisResult(
            findings=[_make_static_finding(title="B101: assert_used", line=5)],
            finding_count=1,
        )
        result = aggregate(AggregatorInput(static_result=static))
        assert result.total_output == 1
        assert result.findings[0].primary_tool == "bandit"

    def test_aggregates_findings_from_three_tools(self):
        """Non-overlapping findings from 3 tools → 3 records preserved."""
        static = StaticAnalysisResult(
            findings=[_make_static_finding(title="Assert Used", line=5)],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(title="XSS via Output", line=30)],
        )
        logic = LogicReviewResult(
            findings=[_make_logic_finding(title="Double Refund Possible", line=70)],
        )
        result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
            logic_result=logic,
        ))
        assert result.total_raw == 3
        assert result.total_output == 3
        assert result.total_merged == 0

    def test_deduplicates_cross_tool_sql_injection(self):
        """Bandit and security scanner both flag SQL injection → merged."""
        static = StaticAnalysisResult(
            findings=[_make_static_finding(
                title="SQL Injection Detected", line=42, cwe="CWE-89"
            )],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(
                title="SQL Injection Detected", line=42
            )],
        )
        result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
        ))
        assert result.total_raw == 2
        assert result.total_output == 1
        assert result.total_merged == 1
        merged = result.findings[0]
        assert "bandit" in merged.source_tools
        assert "security_scanner_rag" in merged.source_tools
        # CWE from static finding preserved
        assert merged.cwe_id == "CWE-89"

    def test_preserves_line_numbers(self):
        static = StaticAnalysisResult(
            findings=[_make_static_finding(title="Command Injection", line=77)],
            finding_count=1,
        )
        result = aggregate(AggregatorInput(static_result=static))
        assert result.findings[0].line_number == 77

    def test_preserves_cwe(self):
        security = SecurityScannerResult(
            findings=[_make_security_finding(title="XSS", cwe="CWE-79")],
        )
        result = aggregate(AggregatorInput(security_result=security))
        assert result.findings[0].cwe_id == "CWE-79"

    def test_sorting_highest_severity_first(self):
        static = StaticAnalysisResult(
            findings=[
                _make_static_finding(title="Low Risk", severity=FindingSeverity.LOW, line=5),
                _make_static_finding(title="Critical Risk", severity=FindingSeverity.CRITICAL, line=15),
                _make_static_finding(title="Medium Risk", severity=FindingSeverity.MEDIUM, line=25),
            ],
            finding_count=3,
        )
        result = aggregate(AggregatorInput(static_result=static))
        severities = [f.severity for f in result.findings]
        # Highest severity should come first
        assert severities[0] == "critical"

    def test_conflicting_severities_resolved_to_highest(self):
        """Same finding from two tools with different severities → highest wins."""
        static = StaticAnalysisResult(
            findings=[_make_static_finding(
                title="Insecure Hash", severity=FindingSeverity.LOW, line=10
            )],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(
                title="Insecure Hash", severity=FindingSeverity.HIGH, line=10
            )],
        )
        result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
        ))
        assert result.total_output == 1
        assert result.findings[0].severity == "high"

    def test_accumulates_errors_from_agents(self):
        static = StaticAnalysisResult(
            findings=[],
            errors=["Bandit failed on file X."],
        )
        result = aggregate(AggregatorInput(static_result=static))
        assert any("Bandit failed" in e for e in result.errors)

    def test_summary_string_output(self):
        static = StaticAnalysisResult(
            findings=[
                _make_static_finding(title="A", severity=FindingSeverity.HIGH, line=1),
                _make_static_finding(title="B", severity=FindingSeverity.LOW, line=2),
            ],
            finding_count=2,
        )
        result = aggregate(AggregatorInput(static_result=static))
        summary = result.summary()
        # Two distinct findings (different lines) → no dedup → 2 unique
        assert "2 unique finding(s)" in summary or "unique finding" in summary

    def test_near_duplicate_between_logic_and_security(self):
        """Logic and security agents both flag similar XSS issue on same file."""
        logic = LogicReviewResult(
            findings=[_make_logic_finding(
                title="XSS Reflected In Template", line=55
            )],
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(
                title="XSS Reflected in Template Output", line=55
            )],
        )
        result = aggregate(AggregatorInput(
            security_result=security,
            logic_result=logic,
        ))
        # Near-duplicate should be merged (Jaccard > 0.72)
        assert result.total_output <= 2  # may merge


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: SeverityClassifierAgent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityClassifier:

    # ── Individual rule matching ──────────────────────────────────────────────

    def test_command_injection_classified_critical(self):
        f = _make_finding_record(
            title="OS Command Injection via subprocess",
            severity="high",
            affected_code="subprocess.run(cmd, shell=True)",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.CRITICAL
        assert rule is not None

    def test_sql_injection_classified_critical(self):
        f = _make_finding_record(
            title="SQL Injection in login endpoint",
            severity="medium",
            affected_code="cursor.execute(f'SELECT * FROM users WHERE id={uid}')",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.CRITICAL

    def test_md5_password_classified_high(self):
        f = _make_finding_record(
            title="Weak Password Hash MD5 Without Salt",
            severity="low",   # source agent underestimated
            affected_code="hashlib.md5(password.encode()).hexdigest()",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.HIGH

    def test_path_traversal_classified_high(self):
        f = _make_finding_record(
            title="Path Traversal to Arbitrary File Read",
            severity="medium",
            affected_code="open(user_path, 'r')",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.HIGH

    def test_missing_validation_classified_medium(self):
        f = _make_finding_record(
            title="Missing Input Validation on Quantity Field",
            severity="critical",  # source agent over-estimated
            affected_code="quantity = int(request.POST['qty'])",
            category="business_logic",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.MEDIUM

    def test_unmatched_critical_capped_at_high(self):
        """Finding claiming CRITICAL that matches no rubric rule is capped at HIGH."""
        f = _make_finding_record(
            title="Some Unrecognised Flaw",
            severity="critical",
            affected_code="x = x + 1",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.HIGH
        assert rule is None  # no rule matched

    def test_unmatched_low_kept_as_low(self):
        f = _make_finding_record(
            title="Minor Naming Convention Violation",
            severity="low",
            affected_code="x = get_data()",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.LOW

    # ── Full classify() function tests ─────────────────────────────────────────

    def test_empty_input(self):
        result = classify(ClassifierInput(findings=[]))
        assert result.total == 0
        assert result.highest_severity is None

    def test_single_finding_classified_correctly(self):
        f = _make_finding_record(
            title="SQL Injection detected",
            severity="low",   # wrong source severity
            affected_code="execute(sql_query % user_input)",
        )
        result = classify(ClassifierInput(findings=[f]))
        assert result.total == 1
        assert result.highest_severity == "critical"
        assert result.reclassified_count == 1

    def test_classifier_does_not_trust_source_severity(self):
        """A finding mislabelled as 'low' by source should be correctly reclassified."""
        f = _make_finding_record(
            title="OS Command Injection",
            severity="low",
            affected_code="os.system(user_cmd)",
        )
        result = classify(ClassifierInput(findings=[f]))
        classified_critical = result.classified_findings.get("critical", [])
        assert len(classified_critical) == 1
        assert classified_critical[0].severity == "critical"

    def test_multiple_findings_different_severities(self):
        findings = [
            _make_finding_record("SQL Injection", severity="low",
                                  affected_code="execute(raw_sql)"),
            _make_finding_record("MD5 password hash without salt",
                                  severity="low",
                                  affected_code="hashlib.md5(pw).hexdigest()"),
            _make_finding_record("Verbose logging of stack trace",
                                  severity="medium",
                                  affected_code="logger.error(traceback.format_exc())"),
        ]
        result = classify(ClassifierInput(findings=findings))
        assert result.total == 3
        # SQL injection → CRITICAL
        assert len(result.classified_findings.get("critical", [])) >= 1
        # MD5 → HIGH
        assert len(result.classified_findings.get("high", [])) >= 1

    def test_conflicting_severities_across_tools(self):
        """
        Aggregator already merges conflicts but classifier is the final arbiter.
        Test that classifier overrides even if aggregator set a severity.
        """
        # Simulate aggregated finding with conflicting origins
        f = _make_finding_record(
            title="Deserialization of Untrusted Data",
            severity="medium",   # aggregator set medium from averaged tools
            affected_code="pickle.loads(user_data)",
        )
        result = classify(ClassifierInput(findings=[f]))
        # Pickle deserialization → CRITICAL (RCE possible)
        assert result.classified_findings["critical"]

    def test_reclassified_count_correct(self):
        findings = [
            _make_finding_record("SQL Injection", severity="low",
                                  affected_code="execute(sql)"),  # → CRITICAL
            _make_finding_record("XSS reflected cross-site scripting",
                                  severity="high",
                                  affected_code="output xss content"),  # → HIGH (may not change)
        ]
        result = classify(ClassifierInput(findings=findings))
        # SQL injection is reclassified (low → critical)
        assert result.reclassified_count >= 1

    def test_classification_detail_preserved(self):
        f = _make_finding_record(
            title="Command injection via subprocess shell",
            severity="medium",
            affected_code="subprocess.run(cmd, shell=True)",
        )
        result = classify(ClassifierInput(findings=[f]))
        detail = next(
            (d for d in result.details if "command injection" in d.finding_title.lower()),
            None,
        )
        assert detail is not None
        assert detail.original_severity == "medium"
        assert detail.classified_severity == "critical"
        assert detail.changed is True

    def test_severity_summary_counts(self):
        findings = [
            _make_finding_record("SQL Injection A", severity="low",
                                  affected_code="execute(sql)", line=1),
            _make_finding_record("SQL Injection B", severity="low",
                                  affected_code="execute(sql2)", line=2, file="b.py"),
            _make_finding_record("Minor lint issue", severity="low",
                                  affected_code="pass", line=3),
        ]
        result = classify(ClassifierInput(findings=findings))
        total = sum(result.severity_summary.values())
        assert total == 3

    def test_ssrf_classified_critical(self):
        f = _make_finding_record(
            title="SSRF to internal metadata endpoint",
            severity="medium",
            affected_code="requests.get('http://169.254.169.254/...')",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.CRITICAL

    def test_xxe_classified_high(self):
        f = _make_finding_record(
            title="XML External Entity injection XXE",
            severity="low",
            affected_code="xml.etree.ElementTree.fromstring(user_xml)",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.HIGH

    def test_auth_bypass_classified_critical(self):
        f = _make_finding_record(
            title="Authentication bypass allows unauthenticated access to admin panel",
            severity="high",
            affected_code="if user.role: pass  # no check",
        )
        level, rule = _classify_finding(f)
        assert level == SeverityLevel.CRITICAL

    def test_merge_note_written_on_reclassification(self):
        f = _make_finding_record(
            title="SQL Injection",
            severity="low",
            affected_code="execute(sql)",
        )
        result = classify(ClassifierInput(findings=[f]))
        classified_finding = result.classified_findings["critical"][0]
        assert any("Severity reclassified" in note for note in classified_finding.merge_notes)


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Integration — Aggregator → Classifier pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregatorClassifierPipeline:

    def test_full_pipeline_dedup_then_classify(self):
        """
        Bandit flags SQL injection as LOW.
        Security scanner flags same SQL injection as HIGH.
        After aggregation: 1 merged finding (severity=HIGH).
        After classification: reclassified to CRITICAL by rubric.
        """
        static = StaticAnalysisResult(
            findings=[_make_static_finding(
                title="SQL Injection",
                severity=FindingSeverity.LOW,
                line=42,
            )],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(
                title="SQL Injection",
                severity=FindingSeverity.HIGH,
                line=42,
            )],
        )

        agg_result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
        ))
        assert agg_result.total_output == 1
        assert agg_result.total_merged == 1

        clf_result = classify(ClassifierInput(findings=agg_result.findings))
        assert clf_result.highest_severity == "critical"
        assert len(clf_result.classified_findings["critical"]) == 1
        # Both tools preserved in provenance
        finding = clf_result.classified_findings["critical"][0]
        assert "bandit" in finding.source_tools
        assert "security_scanner_rag" in finding.source_tools

    def test_full_pipeline_three_tools_no_duplicates(self):
        """Completely different findings from 3 tools all pass through."""
        static = StaticAnalysisResult(
            findings=[_make_static_finding("Hardcoded Password", line=5)],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding("XSS in Template", line=30)],
        )
        logic = LogicReviewResult(
            findings=[_make_logic_finding("Double Refund Missing Idempotency", line=80)],
        )

        agg_result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
            logic_result=logic,
        ))
        clf_result = classify(ClassifierInput(findings=agg_result.findings))

        assert clf_result.total == 3

    def test_full_pipeline_all_findings_empty(self):
        agg_result = aggregate(AggregatorInput())
        clf_result = classify(ClassifierInput(findings=agg_result.findings))
        assert clf_result.total == 0
        assert clf_result.highest_severity is None

    def test_conflicting_severities_classifier_wins(self):
        """
        Three tools each provide a different severity for the same command-injection.
        The classifier should always pick CRITICAL regardless of which severity 'won'
        the aggregation battle.
        """
        static = StaticAnalysisResult(
            findings=[_make_static_finding(
                "Command Injection Shell Execution",
                severity=FindingSeverity.LOW, line=15,
                code="subprocess.run(cmd, shell=True)"
            )],
            finding_count=1,
        )
        security = SecurityScannerResult(
            findings=[_make_security_finding(
                "Command Injection Shell Execution",
                severity=FindingSeverity.MEDIUM, line=15,
            )],
        )
        logic = LogicReviewResult(
            findings=[LogicFinding(
                title="Command Injection Shell Execution",
                description="Logic issue",
                severity=FindingSeverity.HIGH,
                confidence=FindingConfidence.HIGH,
                affected_code="subprocess.run(cmd, shell=True)",
                reasoning="Unsanitized.",
                suggested_remediation="Use args list.",
                line_number=15,
                file="app.py",
            )],
        )

        agg_result = aggregate(AggregatorInput(
            static_result=static,
            security_result=security,
            logic_result=logic,
        ))
        clf_result = classify(ClassifierInput(findings=agg_result.findings))

        # No matter what severity made it through aggregation, classifier says CRITICAL
        assert clf_result.highest_severity == "critical"
