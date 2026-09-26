"""
tests/test_human_in_the_loop.py — Tests for Human-in-the-Loop Approval & SQLite Checkpoints.

Verifies:
  1. SQLite checkpointer creates required tables and records checkpoints.
  2. The review workflow pauses at `human_node` using LangGraph's genuine interrupt capability.
  3. Paused state snapshot provides complete reviewer payload: summary, severity counts,
     findings, affected code, AI explanations, and suggested fixes.
  4. Workflow is fully resumable from checkpoint with APPROVE decision and generates report.
  5. Workflow is resumable from checkpoint with REJECT decision and terminates.
  6. Checkpoints persist on disk in SQLite across connection restarts and can be resumed.
  7. Original code is preserved and repository is never mutated.
"""

from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.aggregator_agent import FindingRecord
from agents.critic_agent import CriticEvaluationSchema
from agents.fix_suggester_agent import FixSuggestionSchema
from agents.logic_reviewer_agent import LogicReviewResult
from agents.security_scanner_agent import (
    FindingConfidence,
    FindingSeverity,
    SecurityFinding,
    SecurityScannerResult,
)
from agents.static_analysis_agent import StaticAnalysisResult
from graph.state import AgentState
from graph.workflow import (
    build_review_graph,
    resume_review,
    run_review,
)
from llm.llm_client import BaseLLMProvider, LLMResponse, LLMStructuredResponse
from storage.checkpoint_db import (
    clear_saver_cache,
    get_sqlite_checkpointer,
    list_checkpoint_threads,
)


def _make_dummy_finding(
    title: str = "SQL Injection in User Login",
    description: str = "Unsanitized user input formatted into SQL query string.",
    severity: str = "critical",
    affected_code: str = 'db.execute(f"SELECT * FROM users WHERE name = \'{user}\'")',
    line_number: int = 42,
) -> FindingRecord:
    return FindingRecord(
        title=title,
        description=description,
        category="security",
        reported_severity=severity,
        reported_confidence="high",
        severity=severity,
        confidence="high",
        file="auth.py",
        line_number=line_number,
        end_line=None,
        affected_code=affected_code,
        reasoning="Raw string interpolation bypasses database escaping.",
        suggested_remediation="Use parameterized queries with prepared statements.",
        source_tools=["bandit", "security_scanner"],
        primary_tool="bandit",
        cwe_id="CWE-89",
        test_id="B608",
        more_info="https://cwe.mitre.org/data/definitions/89.html",
        fingerprint="fp-sql-inj-42",
        merge_notes=["Merged bandit and security scanner findings."],
        duplicate_count=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. SQLite Checkpointer Storage Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLiteCheckpointer:
    """Tests for storage/checkpoint_db.py."""

    def teardown_method(self):
        clear_saver_cache()

    def test_in_memory_checkpointer_initialization(self):
        saver = get_sqlite_checkpointer(":memory:")
        assert isinstance(saver, SqliteSaver)
        cursor = saver.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        assert cursor.fetchone() is not None

    def test_file_checkpointer_and_thread_listing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_checkpoints.db"
            saver = get_sqlite_checkpointer(db_file)
            assert isinstance(saver, SqliteSaver)

            # Initially no threads
            threads = list_checkpoint_threads(db_file)
            assert threads == []

            # Save a dummy checkpoint
            config = {"configurable": {"thread_id": "thread-101", "checkpoint_ns": ""}}
            saver.put(
                config,
                {"v": 1, "id": "cp-1", "ts": "2026-09-22T00:00:00Z", "channel_values": {}},
                {},
                {},
            )

            threads = list_checkpoint_threads(db_file)
            assert len(threads) == 1
            assert threads[0]["thread_id"] == "thread-101"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Workflow Pause & Resume Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanInTheLoopWorkflow:
    """Tests proving LangGraph pauses at human approval and resumes with decisions."""

    def teardown_method(self):
        clear_saver_cache()

    def _setup_mock_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "test"
        mock_provider.model_name = "test"

        mock_fix = FixSuggestionSchema(
            proposed_fix='db.execute("SELECT * FROM users WHERE name = ?", (user,))',
            explanation="Switched to parameterized query to eliminate SQL injection.",
            changed_summary="Used parameter binding.",
        )
        mock_critic = CriticEvaluationSchema(
            is_acceptable=True,
            feedback="Fix is safe, minimal, and resolves the SQL injection vulnerability.",
            meets_rubric=True,
            potential_regressions=[],
            specific_issues=[],
        )

        def mock_generate(prompt, schema, **kwargs):
            if schema == FixSuggestionSchema:
                data = mock_fix
            elif schema == CriticEvaluationSchema:
                data = mock_critic
            else:
                data = None
            return LLMStructuredResponse(
                parsed_data=data,
                raw_text="{}",
                is_valid=True,
                response_metadata=LLMResponse(
                    content="{}",
                    model="test",
                    provider="test",
                    status="success",
                ),
            )

        mock_provider.generate_structured.side_effect = mock_generate
        return mock_provider

    @contextmanager
    def _mock_pipeline(self):
        mock_llm = self._setup_mock_llm()
        dummy_sec_finding = SecurityFinding(
            title="SQL Injection in User Login",
            description="Unsanitized user input formatted into SQL query string.",
            severity=FindingSeverity.CRITICAL,
            confidence=FindingConfidence.HIGH,
            affected_code='db.execute(f"SELECT * FROM users WHERE name = \'{user}\'")',
            reasoning="Raw string interpolation bypasses database escaping.",
            suggested_remediation="Use parameterized queries with prepared statements.",
            file="auth.py",
            line_number=42,
            cwe_id="CWE-89",
        )
        with patch("graph.workflow.analyse") as mock_static, \
             patch("graph.workflow.scan_security") as mock_sec, \
             patch("graph.workflow.review_logic") as mock_log, \
             patch("agents.fix_suggester_agent.get_llm_provider", return_value=mock_llm), \
             patch("agents.critic_agent.get_llm_provider", return_value=mock_llm):
            mock_static.return_value = StaticAnalysisResult()
            mock_sec.return_value = SecurityScannerResult(findings=[dummy_sec_finding])
            mock_log.return_value = LogicReviewResult()
            yield mock_llm

    def test_workflow_pauses_at_human_approval_node(self):
        """Verify the graph halts at human_node and does not proceed without input."""
        saver = get_sqlite_checkpointer(":memory:")
        app = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver)

        thread_id = "thread-pause-test"
        finding = _make_dummy_finding()
        raw_code = 'def get_user(user): return db.execute(f"SELECT * FROM users WHERE name = \'{user}\'")'

        initial_state: AgentState = {
            "raw_code": raw_code,
            "filename": "auth.py",
            "aggregated_findings": [finding],
            "errors": [],
            "revision_count": 0,
            "needs_revision": False,
            "run_static": False,
            "run_security": False,
            "run_logic": False,
        }

        with self._mock_pipeline():
            config = {"configurable": {"thread_id": thread_id}}
            result = app.invoke(initial_state, config=config)

            # 1. Verify LangGraph emitted genuine __interrupt__
            assert "__interrupt__" in result
            interrupts = result["__interrupt__"]
            assert len(interrupts) > 0

            # 2. Verify state snapshot paused at human_node
            snapshot = app.get_state(config)
            assert "human_node" in snapshot.next
            assert snapshot.values.get("final_report") is None

            # 3. Verify reviewer payload contains all required information
            payload = interrupts[0].value
            assert payload["total_findings"] == 1
            assert payload["filename"] == "auth.py"
            assert len(payload["aggregated_findings"]) == 1
            assert len(payload["suggested_fixes"]) == 1

            # Check finding details in payload
            f_in_payload = payload["aggregated_findings"][0]
            title = f_in_payload.get("title") if isinstance(f_in_payload, dict) else f_in_payload.title
            assert "SQL Injection" in title

            # Check suggested fix in payload
            fix_in_payload = payload["suggested_fixes"][0]
            assert "parameterized query" in fix_in_payload["explanation"].lower()
            assert "?" in fix_in_payload["proposed_fix"]

    def test_workflow_resumes_and_generates_report_on_approve(self):
        """Verify workflow resumes from checkpoint upon APPROVE and renders report."""
        saver = get_sqlite_checkpointer(":memory:")
        app = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver)

        thread_id = "thread-approve-test"
        finding = _make_dummy_finding()

        initial_state: AgentState = {
            "raw_code": "def query(): pass",
            "filename": "auth.py",
            "aggregated_findings": [finding],
            "errors": [],
            "revision_count": 0,
            "needs_revision": False,
            "run_static": False,
            "run_security": False,
            "run_logic": False,
        }

        with self._mock_pipeline():
            config = {"configurable": {"thread_id": thread_id}}
            # Initial run -> pauses
            app.invoke(initial_state, config=config)

            # Confirm paused
            snap_paused = app.get_state(config)
            assert "human_node" in snap_paused.next

            # Resume with APPROVE
            resumed_state = resume_review(app, thread_id=thread_id, decision="approve")

            # Verify completion
            assert resumed_state["human_decision"] == "approve"
            assert "final_report" in resumed_state
            assert "## 📊 Summary" in resumed_state["final_report"]
            assert "Proposed Remediations" in resumed_state["final_report"]

            # Verify graph execution is complete (no remaining tasks)
            snap_final = app.get_state(config)
            assert snap_final.next == ()

    def test_workflow_resumes_and_terminates_on_reject(self):
        """Verify workflow resumes from checkpoint upon REJECT and terminates without report."""
        saver = get_sqlite_checkpointer(":memory:")
        app = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver)

        thread_id = "thread-reject-test"
        finding = _make_dummy_finding()

        initial_state: AgentState = {
            "raw_code": "def query(): pass",
            "filename": "auth.py",
            "aggregated_findings": [finding],
            "errors": [],
            "revision_count": 0,
            "needs_revision": False,
            "run_static": False,
            "run_security": False,
            "run_logic": False,
        }

        with self._mock_pipeline():
            config = {"configurable": {"thread_id": thread_id}}
            app.invoke(initial_state, config=config)

            # Resume with REJECT
            resumed_state = resume_review(app, thread_id=thread_id, decision="reject")

            # Verify rejection was recorded and graph halted at END without report_node
            assert resumed_state["human_decision"] == "reject"
            assert resumed_state.get("final_report") is None

            snap_final = app.get_state(config)
            assert snap_final.next == ()

    def test_disk_persistence_across_connection_reopen(self):
        """Verify checkpoints persist to SQLite disk file and survive process/connection restarts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "persistent_reviews.db"
            thread_id = "persist-thread-99"

            # Connection 1: Run graph and trigger pause
            saver1 = get_sqlite_checkpointer(db_path)
            app1 = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver1)

            finding = _make_dummy_finding()
            initial_state: AgentState = {
                "raw_code": "def login(): pass",
                "filename": "auth.py",
                "aggregated_findings": [finding],
                "errors": [],
                "revision_count": 0,
                "needs_revision": False,
                "run_static": False,
                "run_security": False,
                "run_logic": False,
            }

            with self._mock_pipeline():
                app1.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

            # Close connection 1 completely
            clear_saver_cache()

            # Connection 2: Connect fresh from disk
            saver2 = get_sqlite_checkpointer(db_path)
            app2 = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver2)

            # Verify paused state was recovered from disk
            config = {"configurable": {"thread_id": thread_id}}
            recovered_snapshot = app2.get_state(config)
            assert "human_node" in recovered_snapshot.next
            assert len(recovered_snapshot.values["aggregated_findings"]) == 1

            # Resume using the fresh connection!
            with self._mock_pipeline():
                resumed_state = resume_review(app2, thread_id=thread_id, decision="approve")
            assert resumed_state["human_decision"] == "approve"
            assert "final_report" in resumed_state
            assert "Code Review Report" in resumed_state["final_report"]

    def test_original_code_preserved_throughout_approval(self):
        """Verify original code is preserved byte-for-byte and repo is not modified."""
        raw_code = "class Payment:\n    def charge(self, amt):\n        return True\n"
        saver = get_sqlite_checkpointer(":memory:")
        app = build_review_graph(enable_critic=True, enable_human_review=True, checkpointer=saver)

        thread_id = "safety-test-thread"
        initial_state: AgentState = {
            "raw_code": raw_code,
            "sanitized_code": raw_code,
            "filename": "payment.py",
            "aggregated_findings": [_make_dummy_finding()],
            "errors": [],
            "revision_count": 0,
            "needs_revision": False,
            "run_static": False,
            "run_security": False,
            "run_logic": False,
        }

        with self._mock_pipeline():
            # 1. Run to pause
            app.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

            # 2. Resume with approve
            final = resume_review(app, thread_id=thread_id, decision="approve")

            # Invariant: raw_code is unchanged
            assert final["raw_code"] == raw_code
            assert final["sanitized_code"] == raw_code

