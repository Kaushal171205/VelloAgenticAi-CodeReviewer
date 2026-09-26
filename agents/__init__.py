"""agents — Specialized review agents package."""

from agents.aggregator_agent import (
    AggregatorInput,
    AggregatorResult,
    FindingRecord,
    aggregate,
)
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
from agents.logic_reviewer_agent import (
    LogicFinding,
    LogicReviewerAgent,
    LogicReviewerInput,
    LogicReviewResult,
    review_logic,
)
from agents.sanitizer import (
    SanitizerInput,
    SanitizerResult,
    SecretFinding,
    SecretType,
    Severity,
    sanitize,
)
from agents.security_scanner_agent import (
    SecurityFinding,
    SecurityScannerAgent,
    SecurityScannerInput,
    SecurityScannerResult,
    scan_security,
)
from agents.severity_classifier_agent import (
    ClassificationDetail,
    ClassifierInput,
    ClassifierResult,
    SeverityLevel,
    classify,
)
from agents.static_analysis_agent import (
    FindingConfidence,
    FindingSeverity,
    StaticAnalysisFinding,
    StaticAnalysisInput,
    StaticAnalysisResult,
    analyse,
    analyze,
)

__all__ = [
    # Sanitizer (Phase 2)
    "SanitizerInput",
    "SanitizerResult",
    "SecretFinding",
    "SecretType",
    "Severity",
    "sanitize",
    # Static Analysis (Phase 3)
    "FindingConfidence",
    "FindingSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisInput",
    "StaticAnalysisResult",
    "analyse",
    "analyze",
    # Logic Reviewer (Phase 5)
    "LogicReviewerAgent",
    "LogicReviewerInput",
    "LogicReviewResult",
    "LogicFinding",
    "review_logic",
    # Security Scanner RAG (Phase 6)
    "SecurityScannerAgent",
    "SecurityScannerInput",
    "SecurityScannerResult",
    "SecurityFinding",
    "scan_security",
    # Aggregator (Phase 10)
    "AggregatorInput",
    "AggregatorResult",
    "FindingRecord",
    "aggregate",
    # Severity Classifier (Phase 10)
    "ClassificationDetail",
    "ClassifierInput",
    "ClassifierResult",
    "SeverityLevel",
    "classify",
    # Fix Suggester & Self-Correction (Phase 11)
    "FixSuggesterAgent",
    "FixSuggesterInput",
    "FixSuggesterResult",
    "FixSuggestionSchema",
    "suggest_fix",
    # Critic (Phase 11)
    "CriticAgent",
    "CriticEvaluationSchema",
    "CriticInput",
    "CriticResult",
    "critique_fix",
]

