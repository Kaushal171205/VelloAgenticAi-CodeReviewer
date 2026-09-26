"""
tests/test_logic_reviewer.py — Unit and integration tests for Logic Reviewer Agent.

Tests the agent's ability to detect business-logic, correctness, and authorization
flaws (such as in payment/refund flows), verify secret sanitization prior to LLM calls,
and validate structured output normalization.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.logic_reviewer_agent import (
    FindingConfidence,
    FindingSeverity,
    LogicFinding,
    LogicIssueSchema,
    LogicReviewerAgent,
    LogicReviewerInput,
    LogicReviewResult,
    LogicReviewSchema,
    review_logic,
)
from llm.llm_client import BaseLLMProvider, LLMResponse, LLMStructuredResponse

# ─────────────────────────────────────────────────────────────────────────────
# Vulnerable Payment / Refund Sample Code
# ─────────────────────────────────────────────────────────────────────────────

VULNERABLE_PAYMENT_REFUND_CODE = """
class PaymentService:
    def __init__(self, db, payment_gateway):
        self.db = db
        self.gateway = payment_gateway

    def refund_order(self, user_id: str, order_id: str, refund_amount: float, target_account: str):
        # 1. Authorization flaw: user_id is never checked against order.owner_id (IDOR)
        order = self.db.find_order(order_id)

        # 2. Missing validation: Negative or zero refund amounts are not rejected
        # 3. Incorrect condition: Allows refunding more than order.amount
        if refund_amount > order.total_amount:
            # Bug: Just logs and continues instead of raising or returning error
            print(f"Notice: Refund {refund_amount} exceeds order total {order.total_amount}")

        # 4. Invalid state transition: No check if order is already REFUNDED (Double Refund Flaw)
        # 5. Missing idempotency: Can be called multiple times concurrently

        # 6. Unsafe assumption: External transfer invoked without verifying gateway status
        transfer_result = self.gateway.send_payout(
            amount=refund_amount,
            destination=target_account
        )

        # 7. Inconsistent state: Updates DB without verifying transfer_result.success
        order.status = "REFUNDED"
        order.total_refunded += refund_amount
        self.db.save(order)

        return {"success": True, "refunded": refund_amount}
"""

MOCK_STRIPE_KEY = "sk_" + "live_" + "99887766554433221100aabbccddeeff"

CODE_WITH_SECRET = f"""
# API key embedded in payment script
STRIPE_API_KEY = "{MOCK_STRIPE_KEY}"

def charge_customer(customer_id, amount):
    if amount < 0: # negative amount check
        return False
    return True
"""


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSecretSanitizationEnforcement:
    """Verify that the Logic Reviewer Agent NEVER sends raw secrets to the LLM."""

    def test_secrets_are_sanitized_before_llm(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "mock_provider"
        mock_provider.model_name = "mock_model"

        # Mock structured response
        mock_schema = LogicReviewSchema(
            summary="Cleaned code review",
            findings=[]
        )
        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_schema,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="mock",
                provider="mock",
                status="success"
            )
        )

        agent = LogicReviewerAgent(provider=mock_provider)
        result = agent.review_code(CODE_WITH_SECRET, filename="payment.py")

        assert result.status == "success"
        assert result.sanitization_applied is True
        assert result.secrets_detected >= 1
        assert MOCK_STRIPE_KEY not in result.sanitized_code
        assert "<REDACTED_" in result.sanitized_code

        # Verify prompt received by mock_provider does NOT contain the raw secret
        call_args = mock_provider.generate_structured.call_args
        prompt_passed_to_llm = call_args.kwargs.get("prompt", "") or call_args.args[0]
        assert MOCK_STRIPE_KEY not in prompt_passed_to_llm
        assert "<REDACTED_" in prompt_passed_to_llm


class TestEmptyInputHandling:
    """Test behavior on empty or whitespace code inputs."""

    def test_empty_code_returns_clean_result(self):
        result = review_logic(LogicReviewerInput(source_code=""))
        assert result.status == "success"
        assert len(result.findings) == 0
        assert "No source code provided" in result.summary


class TestVulnerablePaymentRefundMocked:
    """Test logic review findings parsing using the vulnerable payment example."""

    def test_parses_multiple_logic_findings(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "gemini"
        mock_provider.model_name = "gemini-3.5-flash"

        mock_findings = [
            LogicIssueSchema(
                title="Double Refund Vulnerability",
                description="The method does not check if order.status is already 'REFUNDED'.",
                severity="critical",
                confidence="high",
                affected_code='if refund_amount > order.total_amount:',
                reasoning="An attacker can initiate multiple concurrent refunds on the same order, draining merchant funds.",
                suggested_remediation='if order.status == "REFUNDED": raise ValueError("Order already refunded")',
                line_number=18,
                category="state_transition"
            ),
            LogicIssueSchema(
                title="Missing Negative Amount Validation",
                description="refund_amount is never checked to ensure it is positive.",
                severity="high",
                confidence="high",
                affected_code='def refund_order(self, user_id, order_id, refund_amount):',
                reasoning="Supplying a negative refund amount could cause an unexpected debit or inverted balance.",
                suggested_remediation='if refund_amount <= 0: raise ValueError("Amount must be positive")',
                line_number=9,
                category="missing_validation"
            ),
            LogicIssueSchema(
                title="Inverted / Ineffective Condition on Refund Limit",
                description="When refund_amount > order.total_amount, the code merely logs a warning and proceeds.",
                severity="critical",
                confidence="high",
                affected_code='if refund_amount > order.total_amount:\n    print(...)',
                reasoning="The check detects an over-refund but fails to abort the transaction.",
                suggested_remediation='if refund_amount > order.total_amount: raise ValueError("Refund exceeds total")',
                line_number=14,
                category="incorrect_condition"
            ),
            LogicIssueSchema(
                title="Missing Authorization Check (IDOR)",
                description="user_id is not validated as the owner of order_id.",
                severity="high",
                confidence="high",
                affected_code='order = self.db.find_order(order_id)',
                reasoning="Any user can request a refund for any other customer order.",
                suggested_remediation='if order.user_id != user_id: raise PermissionError("Unauthorized")',
                line_number=11,
                category="authorization_flaw"
            ),
        ]

        mock_schema = LogicReviewSchema(
            summary="Severe logic and financial flaws detected in payment refund handler.",
            findings=mock_findings
        )

        mock_provider.generate_structured.return_value = LLMStructuredResponse(
            parsed_data=mock_schema,
            raw_text="{}",
            is_valid=True,
            response_metadata=LLMResponse(
                content="{}",
                model="gemini-3.5-flash",
                provider="gemini",
                status="success"
            )
        )

        agent = LogicReviewerAgent(provider=mock_provider)
        result = agent.review_code(VULNERABLE_PAYMENT_REFUND_CODE, filename="payment_service.py")

        assert result.status == "success"
        assert len(result.findings) == 4
        assert result.highest_severity == FindingSeverity.CRITICAL

        # Verify specific findings
        titles = [f.title for f in result.findings]
        assert "Double Refund Vulnerability" in titles
        assert "Missing Negative Amount Validation" in titles
        assert "Inverted / Ineffective Condition on Refund Limit" in titles
        assert "Missing Authorization Check (IDOR)" in titles

        # Verify finding schema compliance
        first_finding = result.findings[0]
        assert isinstance(first_finding, LogicFinding)
        assert first_finding.tool == "logic_reviewer"
        assert first_finding.severity == FindingSeverity.CRITICAL
        assert first_finding.confidence == FindingConfidence.HIGH
        assert first_finding.file == "payment_service.py"
        assert first_finding.suggested_remediation != ""
        assert first_finding.reasoning != ""

        # Test dictionary conversion
        f_dict = first_finding.to_dict()
        assert f_dict["title"] == "Double Refund Vulnerability"
        assert f_dict["severity"] == "critical"
        assert f_dict["tool"] == "logic_reviewer"


class TestErrorHandling:
    """Verify that provider failures or invalid responses do not crash the application."""

    def test_llm_failure_captured_in_result(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "gemini"
        mock_provider.model_name = "gemini-3.5-flash"
        mock_provider.generate_structured.side_effect = RuntimeError("API quota exceeded")

        result = review_logic(
            LogicReviewerInput(source_code="def foo(): pass"),
            provider=mock_provider
        )

        assert result.status == "error"
        assert len(result.errors) > 0
        assert "API quota exceeded" in result.errors[0]
        assert len(result.findings) == 0
