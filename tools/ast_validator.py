"""
tools/ast_validator.py — Python AST and Syntax Validator.

Validates that proposed code snippets or full modules parse cleanly into
Python's Abstract Syntax Tree (AST) without syntax or indentation errors.
Operates completely locally (no LLM, no network) in under a millisecond.
"""

from __future__ import annotations

import ast
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ASTValidationResult:
    """Outcome of AST syntax validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    error_line: Optional[int] = None
    error_col: Optional[int] = None
    node_count: int = 0
    normalized_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "error_line": self.error_line,
            "error_col": self.error_col,
            "node_count": self.node_count,
        }


def validate_python_syntax(
    code: str,
    filename: str = "<unknown>",
) -> ASTValidationResult:
    """
    Validate Python syntax of `code` by attempting to compile it to an AST.

    If the code contains leading indentation (common when LLMs suggest a
    function body or block), it automatically tests ``textwrap.dedent(code)``.

    Parameters
    ----------
    code:
        Python source code or snippet to validate.
    filename:
        Display filename for error reporting.

    Returns
    -------
    ASTValidationResult indicating success or specific syntax/indentation errors.
    """
    if not code or not code.strip():
        return ASTValidationResult(
            is_valid=True,
            errors=[],
            error_line=None,
            error_col=None,
            node_count=0,
            normalized_code=code,
        )

    # First attempt: direct parse
    try:
        tree = ast.parse(code, filename=filename)
        node_count = sum(1 for _ in ast.walk(tree))
        return ASTValidationResult(
            is_valid=True,
            errors=[],
            error_line=None,
            error_col=None,
            node_count=node_count,
            normalized_code=code,
        )
    except (SyntaxError, IndentationError) as initial_err:
        # Check if dedenting resolves an unexpected indentation error
        dedented = textwrap.dedent(code)
        if dedented != code:
            try:
                tree = ast.parse(dedented, filename=filename)
                node_count = sum(1 for _ in ast.walk(tree))
                return ASTValidationResult(
                    is_valid=True,
                    errors=[],
                    error_line=None,
                    error_col=None,
                    node_count=node_count,
                    normalized_code=dedented,
                )
            except (SyntaxError, IndentationError):
                pass  # Use initial_err for error reporting

        line_info = f"line {initial_err.lineno}" if initial_err.lineno else "unknown line"
        col_info = f"col {initial_err.offset}" if initial_err.offset else ""
        loc = f" ({line_info}{', ' + col_info if col_info else ''})"
        msg = f"SyntaxError: {initial_err.msg}{loc}"

        return ASTValidationResult(
            is_valid=False,
            errors=[msg],
            error_line=initial_err.lineno,
            error_col=initial_err.offset,
            node_count=0,
            normalized_code=code,
        )
    except ValueError as val_err:
        return ASTValidationResult(
            is_valid=False,
            errors=[f"ValueError during AST parse: {val_err}"],
            error_line=None,
            error_col=None,
            node_count=0,
            normalized_code=code,
        )
    except Exception as exc:
        return ASTValidationResult(
            is_valid=False,
            errors=[f"Unexpected AST validation error: {exc}"],
            error_line=None,
            error_col=None,
            node_count=0,
            normalized_code=code,
        )


def is_syntax_valid(code: str) -> bool:
    """Convenience boolean check for valid Python syntax."""
    return validate_python_syntax(code).is_valid
