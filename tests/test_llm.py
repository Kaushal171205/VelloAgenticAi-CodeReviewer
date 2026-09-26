"""
tests/test_llm.py — Unit tests for Phase 4 LLM Abstraction Layer.

Verifies provider factory, Gemini & Ollama implementations, health checks,
secret masking, response schema validation, and error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from config import LLMProvider, settings
from llm import (
    BaseLLMProvider,
    GeminiProvider,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMHealthStatus,
    LLMResponse,
    LLMStructuredResponse,
    LLMUsage,
    OllamaProvider,
    get_llm_provider,
)


class TestSecretMasking:
    """Test API key masking logic to ensure secrets are never leaked."""

    def test_mask_long_secret(self):
        secret = "AQ.A" + "b8RN6_MOCK_TESTING_SECRET_KEY_FOR_TESTS_ONLY_" + "vbw"
        masked = BaseLLMProvider.mask_secret(secret)
        assert masked.startswith("AQ.A...")
        assert masked.endswith("vbw")
        assert secret not in masked

    def test_mask_short_secret(self):
        secret = "1234567"
        assert BaseLLMProvider.mask_secret(secret) == "****"

    def test_mask_none_or_empty(self):
        assert BaseLLMProvider.mask_secret(None) == "<UNSET>"
        assert BaseLLMProvider.mask_secret("") == "<UNSET>"

    def test_repr_does_not_expose_raw_key(self):
        raw_key = "AQ.SECRET_KEY_123456789_TEST"
        provider = GeminiProvider(api_key=raw_key)
        representation = repr(provider)
        assert raw_key not in representation
        assert "AQ.S...TEST" in representation


class TestCleanJsonExtraction:
    """Test JSON cleaning from markdown code fences."""

    def test_extract_markdown_json_codeblock(self):
        text = "Here is the JSON:\n```json\n{\"key\": \"value\"}\n```\nHope that helps!"
        cleaned = BaseLLMProvider.clean_json_text(text)
        assert cleaned == '{"key": "value"}'

    def test_extract_plain_codeblock(self):
        text = "```\n{\"number\": 42}\n```"
        cleaned = BaseLLMProvider.clean_json_text(text)
        assert cleaned == '{"number": 42}'

    def test_plain_json(self):
        text = '{"status": "ok"}'
        assert BaseLLMProvider.clean_json_text(text) == '{"status": "ok"}'


class TestLLMFactory:
    """Test get_llm_provider factory function."""

    def test_get_gemini_provider(self):
        provider = get_llm_provider("gemini")
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_get_ollama_provider(self):
        provider = get_llm_provider("ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_name == "ollama"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError) as exc_info:
            get_llm_provider("invalid_backend")
        assert "Unsupported LLM provider" in str(exc_info.value)


class TestGeminiProvider:
    """Test Gemini LLM provider implementation."""

    def test_missing_api_key_health_check(self):
        provider = GeminiProvider(api_key=None)
        # Force unset
        provider._api_key = None
        health = provider.health_check()
        assert health.is_healthy is False
        assert "not set" in health.status_message.lower()

    def test_missing_api_key_generate_raises(self):
        provider = GeminiProvider(api_key=None)
        provider._api_key = None
        with pytest.raises(LLMAuthenticationError):
            provider.generate("Hello")

    @patch("llm.gemini_provider.genai")
    def test_mock_generate_success(self, mock_genai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        )
        mock_client.models.generate_content.return_value = mock_response

        provider = GeminiProvider(api_key="AQ.test_key_12345678")
        provider._client = mock_client

        res = provider.generate("Test prompt")
        assert res.status == "success"
        assert res.content == "Hello from Gemini!"
        assert res.usage.prompt_tokens == 10
        assert res.usage.completion_tokens == 5
        assert res.usage.total_tokens == 15
        assert res.provider == "gemini"

    def test_sanitize_error_msg_masks_api_key(self):
        raw_key = "AQ.SECRET_KEY_9999"
        provider = GeminiProvider(api_key=raw_key)
        raw_err = f"API request failed with key {raw_key} on endpoint."
        clean_err = provider._sanitize_error_msg(raw_err)
        assert raw_key not in clean_err
        assert provider.mask_secret(raw_key) in clean_err


class SampleSchema(BaseModel):
    is_vulnerable: bool = Field(description="Vulnerability status")
    issue_title: str = Field(description="Short title")


class TestStructuredOutput:
    """Test structured output parsing and validation."""

    def test_structured_output_success(self):
        provider = GeminiProvider(api_key="dummy_key")

        mock_llm_response = LLMResponse(
            content='{"is_vulnerable": true, "issue_title": "SQL Injection"}',
            model="gemini-3.5-flash",
            provider="gemini",
            status="success",
        )

        with patch.object(provider, "generate", return_value=mock_llm_response):
            res = provider.generate_structured("Check code", schema=SampleSchema)
            assert res.is_valid is True
            assert res.parsed_data.is_vulnerable is True
            assert res.parsed_data.issue_title == "SQL Injection"

    def test_structured_output_invalid_json(self):
        provider = GeminiProvider(api_key="dummy_key")

        mock_llm_response = LLMResponse(
            content="This is not JSON",
            model="gemini-3.5-flash",
            provider="gemini",
            status="success",
        )

        with patch.object(provider, "generate", return_value=mock_llm_response):
            res = provider.generate_structured("Check code", schema=SampleSchema)
            assert res.is_valid is False
            assert res.parsed_data is None
            assert "Failed to parse" in res.error_message


class TestOllamaProvider:
    """Test local Ollama provider implementation."""

    def test_ollama_init(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.2")
        assert provider.provider_name == "ollama"
        assert provider.model_name == "llama3.2"
        assert provider._base_url == "http://localhost:11434"

    @patch("requests.get")
    def test_health_check_offline(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        provider = OllamaProvider()
        health = provider.health_check()
        assert health.is_healthy is False
        assert "Cannot connect" in health.status_message

    @patch("requests.get")
    def test_health_check_online_model_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "codellama:latest"}]
        }
        mock_get.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        health = provider.health_check()
        assert health.is_healthy is True
        assert "online and model 'llama3.2' is available" in health.status_message

    @patch("requests.post")
    def test_ollama_generate_success(self, mock_post):
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "message": {"role": "assistant", "content": "Clean code!"},
            "prompt_eval_count": 12,
            "eval_count": 4,
        }
        mock_post.return_value = mock_res

        provider = OllamaProvider()
        res = provider.generate("Review this code")
        assert res.status == "success"
        assert res.content == "Clean code!"
        assert res.usage.prompt_tokens == 12
        assert res.usage.completion_tokens == 4
        assert res.usage.total_tokens == 16
