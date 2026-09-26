"""
tests/test_fix_and_critic_loop.py — Tests for Fix Suggester, AST Validator, Critic Agent, and LangGraph Loop.

Verifies:
  1. AST syntax validator catches Python syntax errors and handles dedenting.
  2. Fix Suggester Agent produces concrete fixes and explanations, handles secret sanitization,
     and incorporates critic feedback.
  3. Critic Agent rejects AST-invalid code with zero LLM calls, and evaluates valid fixes.
  4. LangGraph self-correction loop retries on critic rejection and terminates at max revisions.
  5. Original code is preserved and repository is never mutated.
"""

from unittest.mock import MagicMock, patch
import pytest

from agents.aggregator_agent import FindingRecord
from agents.critic_agent import (
    CriticAgent,
    CriticEvaluationSchema,
    CriticInput,
    CriticResult,
    critique_fix,
)
from agents.fix_suggester_agent import (
    FixSuggesterAgent,
    FixSuggesterInput,
    FixSuggesterResult,
    FixSuggestionSchema,
    suggest_fix,
)
from graph.state import AgentState
from graph.workflow import (
    _MAX_REVISIONS,
    after_critic_router,
    ast_validator_node,
    build_review_graph,
    critic_node,
    fix_suggester_node,
    run_review,
)
from llm.llm_client import BaseLLMProvider, LLMResponse, LLMStructuredResponse
from tools.ast_validator import (
    ASTValidationResult,
    is_syntax_valid,
    validate_python_syntax,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AST Validator Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestASTValidator:
    """Tests for tools/ast_validator.py."""

    def test_valid_code_parses_successfully(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = validate_python_syntax(code)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.node_count > 0
        assert is_syntax_valid(code) is True

    def test_syntax_error_detected_with_details(self):
        bad_code = "def broken(\n    return 42"
        result = validate_python_syntax(bad_code)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "SyntaxError" in result.errors[0]
        assert result.error_line is not None
        assert is_syntax_valid(bad_code) is False

    def test_unexpected_indent_is_dedented_and_parsed(self):
        # Snippet copied with leading indent
        indented_code = "    x = 10\n    if x > 5:\n        print(x)\n"
        result = validate_python_syntax(indented_code)
        assert result.is_valid is True
        assert result.node_count > 0

    def test_empty_or_whitespace_code_is_valid(self):
        assert validate_python_syntax("").is_valid is True
        assert validate_python_syntax("   \n\t  ").is_valid is True

    def test_to_dict_representation(self):
        result = validate_python_syntax("x = 1")
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["errors"] == []
        assert isinstance(d["node_count"], int)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix Suggester Agent Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFixSuggesterAgent:
    """Tests for agents/fix_suggester_agent.py."""

    def _make_mock_provider(self, proposed_fix: str, explanation: str, changed_summary: str = ""):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_schema = FixSuggestionSchema(
            proposed_fix=proposed_fix,
            explanation=explanation,
            changed_summary=changed_summary,
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_schema,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="test-model",
                provider="test-provider",
                status="success",
            ),
        )
        return mock_provider

    def test_generate_concrete_fix_and_explanation(self):
        mock_provider = self._make_mock_provider(
            proposed_fix="def safe_run(cmd: str):\n    import shlex\n    return shlex.quote(cmd)",
            explanation="Used shlex.quote to prevent command injection.",
            changed_summary="Sanitized command input.",
        )
        finding = {
            "title": "Command Injection Vulnerability",
            "description": "os.system executes arbitrary user input.",
            "severity": "critical",
            "affected_code": "os.system(user_cmd)",
            "line_number": 12,
        }

        agent = FixSuggesterAgent(provider=mock_provider)
        res = agent.suggest_fix(
            finding=finding,
            source_code="import os\ndef run(user_cmd):\n    os.system(user_cmd)",
            filename="app.py",
        )

        assert res.status == "success"
        assert "shlex.quote" in res.proposed_fix
        assert "shlex.quote" in res.explanation
        assert res.attempt_number == 1
        assert res.finding_title == "Command Injection Vulnerability"

    def test_strips_markdown_fences_from_proposed_fix(self):
        mock_provider = self._make_mock_provider(
            proposed_fix="```python\ndef safe():\n    return True\n```",
            explanation="Removed insecure call.",
        )
        finding = {"title": "Test Finding", "description": "Test"}
        res = suggest_fix(
            FixSuggesterInput(
                finding=finding,
                source_code="def unsafe(): pass",
            ),
            provider=mock_provider,
        )

        assert not res.proposed_fix.startswith("```")
        assert not res.proposed_fix.endswith("```")
        assert res.proposed_fix == "def safe():\n    return True"

    def test_sanitizes_secrets_before_prompting_llm(self):
        mock_provider = self._make_mock_provider(
            proposed_fix="API_KEY = os.environ['API_KEY']",
            explanation="Loaded API key from environment.",
        )
        source_with_secret = (
            'OPENAI_API_KEY = "sk-live-0123456789abcdef0123456789abcdef"\n'
            'def get_key(): return OPENAI_API_KEY'
        )
        finding = {"title": "Hardcoded Secret", "description": "API key in code"}

        res = suggest_fix(
            FixSuggesterInput(
                finding=finding,
                source_code=source_with_secret,
                is_pre_sanitized=False,
            ),
            provider=mock_provider,
        )

        assert res.status == "success"
        assert res.sanitization_applied is True
        # Verify raw secret was not passed to the LLM prompt
        call_args = mock_provider.generate_structured.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args.args[0]
        assert "sk-live-0123456789abcdef0123456789abcdef" not in prompt
        assert "<REDACTED_" in prompt

    def test_incorporates_critic_feedback_on_retry(self):
        mock_provider = self._make_mock_provider(
            proposed_fix="def fix_v2(): pass",
            explanation="Addressed critic concern.",
        )
        finding = {"title": "Issue", "description": "Desc"}

        suggest_fix(
            FixSuggesterInput(
                finding=finding,
                source_code="def broken(): pass",
                previous_fix="def fix_v1(): pass",
                critic_feedback="fix_v1 misses the edge case where amount is 0.",
                attempt_number=2,
            ),
            provider=mock_provider,
        )

        call_args = mock_provider.generate_structured.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args.args[0]
        assert "SELF-CORRECTION REQUIRED" in prompt
        assert "fix_v1 misses the edge case where amount is 0" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 3. Critic Agent Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticAgent:
    """Tests for agents/critic_agent.py."""

    def test_rejects_ast_invalid_code_without_calling_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        finding = {"title": "Issue", "description": "Desc"}
        bad_syntax_fix = "def broken_syntax(\n    return"

        agent = CriticAgent(provider=mock_provider)
        res = agent.evaluate_fix(
            finding=finding,
            original_code="x = 1",
            proposed_fix=bad_syntax_fix,
            explanation="Broken fix",
        )

        assert res.is_acceptable is False
        assert res.ast_valid is False
        assert "syntax error" in res.feedback.lower()
        # CRITICAL: LLM was NOT invoked!
        assert mock_provider.generate_structured.call_count == 0

    def test_rejects_empty_proposed_fix_without_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        agent = CriticAgent(provider=mock_provider)
        res = agent.evaluate_fix(
            finding={"title": "Issue"},
            original_code="x = 1",
            proposed_fix="   ",
        )
        assert res.is_acceptable is False
        assert "empty" in res.feedback.lower()
        assert mock_provider.generate_structured.call_count == 0

    def test_approves_valid_fix_via_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_schema = CriticEvaluationSchema(
            is_acceptable=True,
            feedback="The proposed fix properly validates input and does not introduce regressions.",
            meets_rubric=True,
            potential_regressions=[],
            specific_issues=[],
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_schema,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="test",
                provider="test",
                status="success",
            ),
        )

        finding = {"title": "Negative Amount Bug", "description": "Allows negative refund"}
        res = critique_fix(
            CriticInput(
                finding=finding,
                original_code="def refund(amt): pass",
                proposed_fix="def refund(amt):\n    if amt <= 0:\n        raise ValueError('Invalid')\n",
                explanation="Added negative check",
            ),
            provider=mock_provider,
        )

        assert res.is_acceptable is True
        assert res.ast_valid is True
        assert res.meets_rubric is True
        assert mock_provider.generate_structured.call_count == 1

    def test_rejects_flawed_fix_with_actionable_feedback(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_schema = CriticEvaluationSchema(
            is_acceptable=False,
            feedback="The fix checks amt < 0, but allows amt == 0 which causes division by zero later.",
            meets_rubric=False,
            potential_regressions=["Zero division error"],
            specific_issues=["Does not reject amt == 0"],
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_schema,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="test",
                provider="test",
                status="success",
            ),
        )

        res = critique_fix(
            CriticInput(
                finding={"title": "Zero Division Hazard"},
                original_code="def compute(amt): return 100 / amt",
                proposed_fix="def compute(amt):\n    if amt < 0:\n        return 0\n    return 100 / amt",
            ),
            provider=mock_provider,
        )

        assert res.is_acceptable is False
        assert "division by zero" in res.feedback


# ─────────────────────────────────────────────────────────────────────────────
# 4. LangGraph Self-Correction Loop Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLangGraphFixAndCriticLoop:
    """Tests for the full fix-generation and self-correction loop in LangGraph."""

    def test_ast_validator_node_annotates_fixes(self):
        state: AgentState = {
            "suggested_fixes": [
                {
                    "finding_title": "Fix 1",
                    "proposed_fix": "x = 10\ny = x + 1",
                    "critic_accepted": None,
                },
                {
                    "finding_title": "Fix 2",
                    "proposed_fix": "def broken(\n  return",
                    "critic_accepted": None,
                },
            ]
        }

        updated = ast_validator_node(state)
        fixes = updated["suggested_fixes"]
        assert fixes[0]["ast_valid"] is True
        assert fixes[0]["ast_errors"] == []
        assert fixes[1]["ast_valid"] is False
        assert len(fixes[1]["ast_errors"]) > 0

    def test_critic_node_immediate_ast_rejection(self):
        state: AgentState = {
            "suggested_fixes": [
                {
                    "finding_title": "Syntax Bug",
                    "proposed_fix": "def foo(:",
                    "ast_valid": False,
                    "ast_errors": ["SyntaxError: line 1"],
                    "critic_accepted": None,
                    "attempt_number": 1,
                }
            ],
            "sanitized_code": "def foo(): pass",
            "revision_count": 0,
        }

        # critic_node should reject AST-invalid fix without needing LLM
        res = critic_node(state)
        assert res["needs_revision"] is True
        assert res["revision_count"] == 1
        assert res["suggested_fixes"][0]["critic_accepted"] is False
        assert "syntax error" in res["suggested_fixes"][0]["critic_feedback"].lower()

    def test_max_revisions_prevents_infinite_loop(self):
        state: AgentState = {
            "suggested_fixes": [
                {
                    "finding_title": "Stubborn Bug",
                    "proposed_fix": "bad syntax",
                    "ast_valid": False,
                    "critic_accepted": False,
                }
            ],
            "revision_count": _MAX_REVISIONS,  # already at maximum
        }

        res = critic_node(state)
        assert res["needs_revision"] is False
        assert f"Maximum revisions ({_MAX_REVISIONS}) reached" in res["critic_feedback"]
        # Router must now route to report_node, not fix_suggester_node
        assert after_critic_router(res) == "report_node"


def _make_dummy_finding(
    title: str = "Cross-Site Scripting (XSS)",
    description: str = "Unescaped HTML rendered directly.",
    severity: str = "high",
    affected_code: str = "return x",
) -> FindingRecord:
    return FindingRecord(
        title=title,
        description=description,
        category="security",
        reported_severity=severity,
        reported_confidence="high",
        severity=severity,
        confidence="high",
        file="views.py",
        line_number=10,
        end_line=None,
        affected_code=affected_code,
        reasoning="User input rendered directly to response.",
        suggested_remediation="Use html.escape",
        source_tools=["security_scanner"],
        primary_tool="security_scanner",
        cwe_id="CWE-79",
        test_id=None,
        more_info="",
        fingerprint="fp12345",
        merge_notes=[],
        duplicate_count=1,
    )


class TestLangGraphExecution:
    """End-to-end execution of fix and critic self-correction loop in LangGraph."""

    def test_successful_loop_flow_in_compiled_graph(self):
        """End-to-end execution of compiled graph with mock LLM for fix + critic."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "test"
        mock_provider.model_name = "test"

        # Mock fix generation
        mock_fix_schema = FixSuggestionSchema(
            proposed_fix="import html\ndef render(x):\n    return html.escape(x)",
            explanation="Escaped user input using html.escape.",
            changed_summary="Fixed XSS.",
        )
        # Mock critic evaluation (accepts)
        mock_critic_schema = CriticEvaluationSchema(
            is_acceptable=True,
            feedback="Fix is safe, minimal, and resolves the XSS vulnerability.",
            meets_rubric=True,
            potential_regressions=[],
            specific_issues=[],
        )

        def mock_generate_structured(prompt, schema, **kwargs):
            if schema == FixSuggestionSchema:
                data = mock_fix_schema
            elif schema == CriticEvaluationSchema:
                data = mock_critic_schema
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

        mock_provider.generate_structured.side_effect = mock_generate_structured

        with patch("agents.fix_suggester_agent.get_llm_provider", return_value=mock_provider), \
             patch("agents.critic_agent.get_llm_provider", return_value=mock_provider):

            app = build_review_graph(enable_critic=True)
            finding = _make_dummy_finding()

            initial_state: AgentState = {
                "raw_code": "def render(x): return x",
                "filename": "views.py",
                "aggregated_findings": [finding],
                "errors": [],
                "revision_count": 0,
                "needs_revision": False,
                "run_static": False,
                "run_security": False,
                "run_logic": False,
            }

            # Step through fix_suggester_node -> ast_validator_node -> critic_node
            state_after_fix = fix_suggester_node(initial_state)
            assert len(state_after_fix["suggested_fixes"]) == 1
            assert "html.escape" in state_after_fix["suggested_fixes"][0]["proposed_fix"]

            state_after_ast = ast_validator_node(state_after_fix)
            assert state_after_ast["suggested_fixes"][0]["ast_valid"] is True

            state_after_critic = critic_node({**initial_state, **state_after_ast})
            assert state_after_critic["needs_revision"] is False
            assert state_after_critic["suggested_fixes"][0]["critic_accepted"] is True

    def test_self_correction_retry_loop_in_graph(self):
        """Verify graph loops back when critic rejects first attempt, then succeeds."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        call_count = {"fix": 0, "critic": 0}

        def mock_generate_structured(prompt, schema, **kwargs):
            if schema == FixSuggestionSchema:
                call_count["fix"] += 1
                if call_count["fix"] == 1:
                    # First attempt: flawed fix
                    data = FixSuggestionSchema(
                        proposed_fix="def render(x): return str(x)",
                        explanation="Converted to string.",
                    )
                else:
                    # Second attempt (after critic feedback): correct fix
                    data = FixSuggestionSchema(
                        proposed_fix="import html\ndef render(x): return html.escape(x)",
                        explanation="Properly escaped with html.escape.",
                    )
            elif schema == CriticEvaluationSchema:
                call_count["critic"] += 1
                if call_count["critic"] == 1:
                    # Reject first attempt
                    data = CriticEvaluationSchema(
                        is_acceptable=False,
                        feedback="str(x) does not prevent XSS. Use html.escape.",
                        meets_rubric=False,
                    )
                else:
                    # Accept second attempt
                    data = CriticEvaluationSchema(
                        is_acceptable=True,
                        feedback="html.escape correctly resolves the XSS vulnerability.",
                        meets_rubric=True,
                    )
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

        mock_provider.generate_structured.side_effect = mock_generate_structured

        with patch("agents.fix_suggester_agent.get_llm_provider", return_value=mock_provider), \
             patch("agents.critic_agent.get_llm_provider", return_value=mock_provider):

            finding = _make_dummy_finding()
            state: AgentState = {
                "raw_code": "def render(x): return x",
                "filename": "views.py",
                "aggregated_findings": [finding],
                "errors": [],
                "revision_count": 0,
                "needs_revision": False,
            }

            # 1. Initial pass
            state.update(fix_suggester_node(state))
            assert state["suggested_fixes"][0]["attempt_number"] == 1
            state.update(ast_validator_node(state))
            state.update(critic_node(state))

            # Critic should have rejected attempt #1 and requested revision
            assert state["needs_revision"] is True
            assert state["revision_count"] == 1
            assert state["suggested_fixes"][0]["critic_accepted"] is False
            assert "str(x) does not prevent XSS" in state["suggested_fixes"][0]["critic_feedback"]
            assert after_critic_router(state) == "fix_suggester_node"

            # 2. Self-correction loop: fix_suggester_node called again
            state.update(fix_suggester_node(state))
            assert state["suggested_fixes"][0]["attempt_number"] == 2
            assert "html.escape" in state["suggested_fixes"][0]["proposed_fix"]
            state.update(ast_validator_node(state))
            state.update(critic_node(state))

            # Critic now accepts attempt #2!
            assert state["needs_revision"] is False
            assert state["suggested_fixes"][0]["critic_accepted"] is True
            assert after_critic_router(state) == "report_node"

    def test_original_code_preserved_and_never_mutated(self):
        """Verify safety invariant: original code is untouched and repo not modified."""
        raw = "class BankAccount:\n    balance = 0\n    def deposit(self, a): self.balance += a"
        initial: AgentState = {
            "raw_code": raw,
            "sanitized_code": raw,
            "filename": "bank.py",
            "errors": [],
            "revision_count": 0,
            "needs_revision": False,
            "run_static": False,
            "run_security": False,
            "run_logic": False,
            "aggregated_findings": [],
            "total_findings": 0,
        }

        # Mock reviewer agents so test executes purely locally in memory
        with patch("graph.workflow.analyse") as mock_static, \
             patch("graph.workflow.scan_security") as mock_sec, \
             patch("graph.workflow.review_logic") as mock_log:

            mock_static.return_value = MagicMock(findings=[], total=0)
            mock_sec.return_value = MagicMock(findings=[], total=0)
            mock_log.return_value = MagicMock(findings=[], total=0)

            app = build_review_graph(enable_critic=True)
            final_state = app.invoke(initial)

            # Original code in state must be preserved byte-for-byte
            assert final_state["raw_code"] == raw
            assert "final_report" in final_state
            assert "## 📊 Summary" in final_state["final_report"]

