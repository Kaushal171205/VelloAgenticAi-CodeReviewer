"""
gemini_provider.py — Google Gemini LLM provider implementation.

Implements `BaseLLMProvider` using the official `google-genai` SDK.
Handles authentication, error handling, retry on transient errors, token tracking,
and structured output. Ensures API keys are never logged or exposed in raw output.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from config import settings
from llm.llm_client import (
    BaseLLMProvider,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMHealthStatus,
    LLMResponse,
    LLMResponseError,
    LLMStructuredResponse,
    LLMUsage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Lazy import check helper for google-genai SDK
_GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    APIError = Exception


class GeminiProvider(BaseLLMProvider):
    """
    LLM Provider implementation for Google Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Gemini LLM provider.

        Args:
            api_key: Gemini API key. Defaults to settings.gemini_api_key.
            model: Gemini model identifier. Defaults to settings.gemini_model.
        """
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model
        self._client = None

        if not _GENAI_AVAILABLE:
            logger.warning("google-genai package is not installed. Gemini provider will not function.")

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:
        masked_key = self.mask_secret(self._api_key)
        return f"<GeminiProvider(model='{self._model}', api_key='{masked_key}')>"

    def _get_client(self) -> Any:
        """Lazy-instantiate and return the google.genai Client."""
        if not _GENAI_AVAILABLE:
            raise LLMConnectionError(
                "google-genai library is not installed. Run `pip install google-genai`."
            )

        if not self._api_key:
            raise LLMAuthenticationError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please configure your Gemini API key in .env."
            )

        if self._client is None:
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                masked_err = self._sanitize_error_msg(str(e))
                logger.error(f"Failed to initialize Gemini client: {masked_err}")
                raise LLMAuthenticationError(f"Gemini client initialization failed: {masked_err}") from e

        return self._client

    def _sanitize_error_msg(self, msg: str) -> str:
        """Remove raw API key from error messages if present."""
        if self._api_key and self._api_key in msg:
            msg = msg.replace(self._api_key, self.mask_secret(self._api_key))
        return msg

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate completion using Google Gemini API with transient retry.
        """
        start_time = time.perf_counter()
        client = self._get_client()

        config_kwargs: Dict[str, Any] = {
            "temperature": max(0.0, min(1.0, temperature)),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs)

        last_error = ""
        # Up to 3 attempts for transient 503 / 429
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                text_content = response.text or ""

                usage = LLMUsage()
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    usage = LLMUsage(
                        prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                        completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(um, "total_token_count", 0) or 0,
                    )

                return LLMResponse(
                    content=text_content,
                    model=self._model,
                    provider=self.provider_name,
                    usage=usage,
                    latency_ms=round(elapsed_ms, 2),
                    status="success",
                )

            except Exception as e:
                clean_err = self._sanitize_error_msg(str(e))
                last_error = clean_err
                if "503" in clean_err or "429" in clean_err or "UNAVAILABLE" in clean_err:
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                logger.error(f"Gemini API error: {clean_err}")
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            content="",
            model=self._model,
            provider=self.provider_name,
            latency_ms=round(elapsed_ms, 2),
            status="error",
            error_message=last_error or "Generation failed",
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> LLMStructuredResponse[T]:
        """
        Generate structured output adhering to a Pydantic schema using Gemini.
        """
        schema_json_example = schema.model_json_schema()
        json_instruction = (
            f"\n\nReturn ONLY a single valid JSON object strictly adhering to this JSON schema:\n"
            f"{json.dumps(schema_json_example, indent=2)}\n"
            f"Do not include any text before or after the JSON."
        )

        full_prompt = prompt + json_instruction

        # Attempt native JSON mode if supported
        llm_resp = None
        try:
            client = self._get_client()
            config_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            config = types.GenerateContentConfig(**config_kwargs)

            start_time = time.perf_counter()
            for attempt in range(3):
                try:
                    res = client.models.generate_content(
                        model=self._model,
                        contents=full_prompt,
                        config=config,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    raw_text = res.text or ""

                    usage = LLMUsage()
                    if hasattr(res, "usage_metadata") and res.usage_metadata:
                        um = res.usage_metadata
                        usage = LLMUsage(
                            prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                            completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                            total_tokens=getattr(um, "total_token_count", 0) or 0,
                        )

                    llm_resp = LLMResponse(
                        content=raw_text,
                        model=self._model,
                        provider=self.provider_name,
                        usage=usage,
                        latency_ms=round(elapsed_ms, 2),
                        status="success",
                    )
                    break
                except Exception as e:
                    clean_err = self._sanitize_error_msg(str(e))
                    if ("503" in clean_err or "429" in clean_err) and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise

        except Exception as e:
            logger.debug(f"JSON mode call failed, falling back to standard generate: {e}")
            llm_resp = self.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        if not llm_resp or llm_resp.status != "success":
            return LLMStructuredResponse(
                parsed_data=None,
                raw_text=llm_resp.content if llm_resp else "",
                is_valid=False,
                error_message=llm_resp.error_message if llm_resp else "Generation failed",
                response_metadata=llm_resp,
            )

        raw_text = llm_resp.content

        # Parse and validate against Pydantic schema
        try:
            clean_json = self.clean_json_text(raw_text)
            parsed = schema.model_validate_json(clean_json)
            return LLMStructuredResponse(
                parsed_data=parsed,
                raw_text=raw_text,
                is_valid=True,
                response_metadata=llm_resp,
            )
        except Exception as e:
            err_msg = f"Failed to parse LLM output into schema {schema.__name__}: {e}"
            logger.warning(err_msg)
            return LLMStructuredResponse(
                parsed_data=None,
                raw_text=raw_text,
                is_valid=False,
                error_message=err_msg,
                response_metadata=llm_resp,
            )

    def health_check(self) -> LLMHealthStatus:
        """
        Verify Gemini API availability and credentials.
        """
        if not self._api_key:
            return LLMHealthStatus(
                is_healthy=False,
                provider=self.provider_name,
                model=self._model,
                status_message="GEMINI_API_KEY is not set in environment or config.",
            )

        if not _GENAI_AVAILABLE:
            return LLMHealthStatus(
                is_healthy=False,
                provider=self.provider_name,
                model=self._model,
                status_message="google-genai Python library is not installed.",
            )

        start_time = time.perf_counter()
        try:
            res = self.generate(prompt="Respond with 'OK' for health check.", max_tokens=5)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            if res.status == "success":
                return LLMHealthStatus(
                    is_healthy=True,
                    provider=self.provider_name,
                    model=self._model,
                    status_message="Gemini API is online and responding.",
                    latency_ms=latency_ms,
                )
            else:
                return LLMHealthStatus(
                    is_healthy=False,
                    provider=self.provider_name,
                    model=self._model,
                    status_message=f"Gemini API returned error: {res.error_message}",
                    latency_ms=latency_ms,
                )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            clean_err = self._sanitize_error_msg(str(e))
            return LLMHealthStatus(
                is_healthy=False,
                provider=self.provider_name,
                model=self._model,
                status_message=f"Gemini health check failed: {clean_err}",
                latency_ms=latency_ms,
            )
