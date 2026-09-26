"""
tests/test_sanitizer.py — Phase 2 unit tests for the sanitizer agent.

Run with:
    pytest tests/test_sanitizer.py -v
"""

from __future__ import annotations

import textwrap

import pytest

from agents.sanitizer import (
    Severity,
    SecretType,
    SanitizerInput,
    SanitizerResult,
    sanitize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(code: str, filename: str = "test.py") -> SanitizerResult:
    return sanitize(SanitizerInput(source_code=textwrap.dedent(code), filename=filename))


def assert_secret_found(result: SanitizerResult, secret_type: SecretType) -> None:
    types = [f.secret_type for f in result.findings]
    assert secret_type in types, (
        f"Expected {secret_type.value!r} in findings, got: {types}"
    )


def assert_placeholder_present(result: SanitizerResult, secret_type: SecretType) -> None:
    for f in result.findings:
        if f.secret_type == secret_type:
            assert f.placeholder in result.sanitized_code, (
                f"Placeholder {f.placeholder!r} not found in sanitized code."
            )
            return
    pytest.fail(f"No finding of type {secret_type.value!r} found.")


def assert_raw_value_absent(result: SanitizerResult, raw_value: str) -> None:
    assert raw_value not in result.sanitized_code, (
        f"Raw secret value {raw_value!r} still present in sanitized code."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: clean code — no false positives
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanCode:
    def test_empty_code_is_clean(self):
        result = run("")
        assert result.is_clean
        assert result.number_of_secrets == 0
        assert result.sanitized_code == ""

    def test_plain_function_is_clean(self):
        code = """
            def add(a: int, b: int) -> int:
                return a + b

            result = add(1, 2)
            print(result)
        """
        result = run(code)
        assert result.is_clean, f"False positives: {result.findings}"

    def test_placeholder_variable_name_no_false_positive(self):
        code = """
            # This is a template placeholder — not a real key
            key = None
            token = ""
            password = ""
        """
        result = run(code)
        # None / empty string assignments should not trigger
        for f in result.findings:
            assert len("") >= 0  # value must be non-trivial to be flagged
        assert result.is_clean, f"Unexpected findings: {result.findings}"

    def test_env_var_lookup_is_clean(self):
        code = """
            import os
            api_key = os.environ.get("GEMINI_API_KEY")
            password = os.getenv("DB_PASSWORD")
        """
        result = run(code)
        assert result.is_clean, f"False positives: {result.findings}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Google API key
# ─────────────────────────────────────────────────────────────────────────────

class TestGoogleApiKey:
    FAKE_KEY = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"

    def test_detects_google_api_key(self):
        code = f'GOOGLE_KEY = "{self.FAKE_KEY}"'
        result = run(code)
        assert_secret_found(result, SecretType.GOOGLE_API_KEY)

    def test_raw_key_removed_from_code(self):
        code = f'GOOGLE_KEY = "{self.FAKE_KEY}"'
        result = run(code)
        assert_raw_value_absent(result, self.FAKE_KEY)

    def test_placeholder_is_in_sanitized_code(self):
        code = f'GOOGLE_KEY = "{self.FAKE_KEY}"'
        result = run(code)
        assert_placeholder_present(result, SecretType.GOOGLE_API_KEY)

    def test_severity_is_high(self):
        code = f'GOOGLE_KEY = "{self.FAKE_KEY}"'
        result = run(code)
        google_findings = [f for f in result.findings if f.secret_type == SecretType.GOOGLE_API_KEY]
        assert google_findings[0].severity == Severity.HIGH

    def test_line_number_is_correct(self):
        code = "\n".join([
            "import os",
            f'api_key = "{self.FAKE_KEY}"',
            "print('done')",
        ])
        result = run(code)
        google = [f for f in result.findings if f.secret_type == SecretType.GOOGLE_API_KEY]
        assert google[0].line_number == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test: Password
# ─────────────────────────────────────────────────────────────────────────────

class TestPassword:
    def test_detects_password_assignment(self):
        code = 'DB_PASSWORD = "super_secret_pass_123"'
        result = run(code)
        assert_secret_found(result, SecretType.PASSWORD)

    def test_detects_passwd_variant(self):
        code = 'config["passwd"] = "MyS3cr3tP4ss!"'
        result = run(code)
        assert_secret_found(result, SecretType.PASSWORD)

    def test_password_removed_from_output(self):
        raw = "hunter2_secret_pass"
        code = f'password = "{raw}"'
        result = run(code)
        assert_raw_value_absent(result, raw)

    def test_short_password_not_flagged(self):
        # Passwords under 6 chars are very likely false positives
        code = 'password = "abc"'
        result = run(code)
        password_findings = [f for f in result.findings if f.secret_type == SecretType.PASSWORD]
        assert len(password_findings) == 0

    def test_password_severity_is_medium(self):
        code = 'password = "correct_horse_battery_staple"'
        result = run(code)
        pw = [f for f in result.findings if f.secret_type == SecretType.PASSWORD]
        assert pw[0].severity == Severity.MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# Test: Database URL
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseUrl:
    POSTGRES_URL = "postgresql://admin:S3cr3tPa55@prod.db.example.com:5432/mydb"
    MYSQL_URL    = "mysql://root:rootpassword@localhost/app_db"
    MONGO_URL    = "mongodb://user:pass123@cluster.mongodb.net/db"

    def test_detects_postgres_url(self):
        code = f'DATABASE_URL = "{self.POSTGRES_URL}"'
        result = run(code)
        assert_secret_found(result, SecretType.DATABASE_URL)

    def test_detects_mysql_url(self):
        code = f'conn = "{self.MYSQL_URL}"'
        result = run(code)
        assert_secret_found(result, SecretType.DATABASE_URL)

    def test_detects_mongodb_url(self):
        code = f'MONGO_URI = "{self.MONGO_URL}"'
        result = run(code)
        assert_secret_found(result, SecretType.DATABASE_URL)

    def test_db_url_severity_is_high(self):
        code = f'DATABASE_URL = "{self.POSTGRES_URL}"'
        result = run(code)
        db = [f for f in result.findings if f.secret_type == SecretType.DATABASE_URL]
        assert db[0].severity == Severity.HIGH

    def test_raw_url_not_in_sanitized_code(self):
        code = f'DATABASE_URL = "{self.POSTGRES_URL}"'
        result = run(code)
        assert_raw_value_absent(result, self.POSTGRES_URL)

    def test_url_without_password_not_flagged(self):
        # No credentials embedded → not a secret
        code = 'host = "postgresql://localhost:5432/mydb"'
        result = run(code)
        db = [f for f in result.findings if f.secret_type == SecretType.DATABASE_URL]
        assert len(db) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: AWS keys
# ─────────────────────────────────────────────────────────────────────────────

class TestAwsKeys:
    FAKE_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"

    def test_detects_aws_access_key(self):
        code = f'AWS_KEY = "{self.FAKE_ACCESS_KEY}"'
        result = run(code)
        assert_secret_found(result, SecretType.AWS_ACCESS_KEY)

    def test_aws_key_severity_is_high(self):
        code = f'key = "{self.FAKE_ACCESS_KEY}"'
        result = run(code)
        aws = [f for f in result.findings if f.secret_type == SecretType.AWS_ACCESS_KEY]
        assert aws[0].severity == Severity.HIGH

    def test_raw_access_key_absent_from_sanitized(self):
        code = f'AWS_KEY = "{self.FAKE_ACCESS_KEY}"'
        result = run(code)
        assert_raw_value_absent(result, self.FAKE_ACCESS_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Multiple secrets in one file
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleSecrets:
    def test_detects_all_secrets(self):
        FAKE_GOOGLE  = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"
        FAKE_PW      = "super_secret_pass_2024"
        FAKE_DB      = "postgresql://admin:S3cr3tPa55@db.example.com/prod"
        FAKE_AWS     = "AKIAIOSFODNN7EXAMPLE"

        code = "\n".join([
            f'GOOGLE_KEY   = "{FAKE_GOOGLE}"',
            f'DB_PASSWORD  = "{FAKE_PW}"',
            f'DATABASE_URL = "{FAKE_DB}"',
            f'AWS_KEY      = "{FAKE_AWS}"',
        ])
        result = run(code)

        assert result.number_of_secrets >= 4
        types = {f.secret_type for f in result.findings}
        assert SecretType.GOOGLE_API_KEY in types
        assert SecretType.PASSWORD       in types
        assert SecretType.DATABASE_URL   in types
        assert SecretType.AWS_ACCESS_KEY in types

    def test_all_raw_values_absent(self):
        secrets = [
            "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678",
            "super_secret_pass_2024",
            "postgresql://admin:S3cr3tPa55@db.example.com/prod",
            "AKIAIOSFODNN7EXAMPLE",
        ]
        code = "\n".join([
            f'GOOGLE_KEY   = "{secrets[0]}"',
            f'DB_PASSWORD  = "{secrets[1]}"',
            f'DATABASE_URL = "{secrets[2]}"',
            f'AWS_KEY      = "{secrets[3]}"',
        ])
        result = run(code)
        for s in secrets:
            assert s not in result.sanitized_code, f"{s!r} still in sanitized code"

    def test_highest_severity_is_high(self):
        code = "\n".join([
            'GOOGLE_KEY   = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"',
            'DB_PASSWORD  = "super_secret_pass_2024"',
        ])
        result = run(code)
        assert result.highest_severity == Severity.HIGH

    def test_findings_have_unique_placeholders(self):
        code = "\n".join([
            'GOOGLE_KEY  = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"',
            'DB_PASSWORD = "another_password_here"',
        ])
        result = run(code)
        placeholders = [f.placeholder for f in result.findings]
        assert len(placeholders) == len(set(placeholders)), "Duplicate placeholders found"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Private key
# ─────────────────────────────────────────────────────────────────────────────

class TestPrivateKey:
    FAKE_PEM = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA2a2rwplBQLDqJKZK3SJHkMoMC0ALIQHM67MqDZKqCuhoY5Gs\n"
        "-----END RSA PRIVATE KEY-----"
    )

    def test_detects_rsa_private_key(self):
        code = f'key = """{self.FAKE_PEM}"""'
        result = run(code)
        assert_secret_found(result, SecretType.PRIVATE_KEY)

    def test_private_key_severity_is_critical(self):
        code = f'key = """{self.FAKE_PEM}"""'
        result = run(code)
        pk = [f for f in result.findings if f.secret_type == SecretType.PRIVATE_KEY]
        assert pk[0].severity == Severity.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# Test: Result structure
# ─────────────────────────────────────────────────────────────────────────────

class TestResultStructure:
    def test_masked_value_shows_only_first_four_chars(self):
        raw = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"
        code = f'key = "{raw}"'
        result = run(code)
        for f in result.findings:
            assert f.masked_value.startswith(raw[:4])
            assert raw[4:] not in f.masked_value

    def test_findings_sorted_by_line_number(self):
        code = "\n".join([
            'GOOGLE_KEY  = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"',
            'import os',
            'DB_PASSWORD = "super_secret_pass_2024"',
        ])
        result = run(code)
        lines = [f.line_number for f in result.findings]
        assert lines == sorted(lines)

    def test_summary_is_correct_for_clean_code(self):
        result = run("x = 1 + 2")
        assert "No secrets" in result.summary()

    def test_summary_mentions_count_and_severity(self):
        code = 'GOOGLE_KEY = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345678"'
        result = run(code)
        assert str(result.number_of_secrets) in result.summary()
        assert result.highest_severity.value.upper() in result.summary()
