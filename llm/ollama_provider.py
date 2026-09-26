"""
ollama_provider.py — Local Ollama LLM provider implementation.

Implements `BaseLLMProvider` using REST requests to a local Ollama server endpoint.
Handles connection checks, streaming disabled completions, token usage tracking,
and JSON mode structured output.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

import requests
from pydantic import BaseModel

from config import settings
from llm.llm_client import (
    BaseLLMProvider,
    LLMConnectionError,
    LLMHealthStatus,
    LLMResponse,
    LLMResponseError,
    LLMStructuredResponse,
    LLMUsage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(BaseLLMProvider):
    """
    LLM Provider implementation for local Ollama instances.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        """
        Initialize Ollama LLM provider.

        Args:
            base_url: Base URL of Ollama service (e.g. 'http://localhost:11434').
            model: Model tag (e.g. 'llama3.2').
            timeout_seconds: HTTP timeout for completion calls in seconds.
        """
        raw_url = base_url or settings.ollama_base_url
        self._base_url = raw_url.rstrip("/")
        self._model = model or settings.ollama_model
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"<OllamaProvider(model='{self._model}', base_url='{self._base_url}')>"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate completion via Ollama /api/chat endpoint.
        """
        start_time = time.perf_counter()
        endpoint = f"{self._base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {
            "temperature": max(0.0, min(1.0, temperature)),
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            res = requests.post(endpoint, json=payload, timeout=self._timeout)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if res.status_code != 200:
                err_msg = f"Ollama HTTP {res.status_code}: {res.text}"
                logger.error(err_msg)
                return LLMResponse(
                    content="",
                    model=self._model,
                    provider=self.provider_name,
                    latency_ms=round(elapsed_ms, 2),
                    status="error",
                    error_message=err_msg,
                )

            data = res.json()
            message_obj = data.get("message", {})
            content = message_obj.get("content", "")

            prompt_tokens = data.get("prompt_eval_count", 0) or 0
            completion_tokens = data.get("eval_count", 0) or 0
            total_tokens = prompt_tokens + completion_tokens

            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider=self.provider_name,
                usage=usage,
                latency_ms=round(elapsed_ms, 2),
                status="success",
                raw_response=data,
            )

        except requests.exceptions.ConnectionError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Failed to connect to Ollama server at {self._base_url}. Ensure Ollama is running."
            logger.error(err_msg)
            return LLMResponse(
                content="",
                model=self._model,
                provider=self.provider_name,
                latency_ms=round(elapsed_ms, 2),
                status="error",
                error_message=err_msg,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Ollama generate request failed: {e}"
            logger.error(err_msg)
            return LLMResponse(
                content="",
                model=self._model,
                provider=self.provider_name,
                latency_ms=round(elapsed_ms, 2),
                status="error",
                error_message=err_msg,
            )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> LLMStructuredResponse[T]:
        """
        Generate structured output adhering to a Pydantic schema using Ollama's JSON format mode.
        """
        schema_json_example = schema.model_json_schema()
        json_instruction = (
            f"\n\nReturn ONLY a single valid JSON object strictly matching this schema:\n"
            f"{json.dumps(schema_json_example, indent=2)}\n"
            f"Do not wrap in extra markdown text outside JSON."
        )

        full_prompt = prompt + json_instruction

        start_time = time.perf_counter()
        endpoint = f"{self._base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        }

        try:
            res = requests.post(endpoint, json=payload, timeout=self._timeout)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if res.status_code != 200:
                llm_resp = LLMResponse(
                    content="",
                    model=self._model,
                    provider=self.provider_name,
                    latency_ms=round(elapsed_ms, 2),
                    status="error",
                    error_message=f"Ollama HTTP {res.status_code}: {res.text}",
                )
                return LLMStructuredResponse(
                    parsed_data=None,
                    raw_text="",
                    is_valid=False,
                    error_message=llm_resp.error_message,
                    response_metadata=llm_resp,
                )

            data = res.json()
            raw_text = data.get("message", {}).get("content", "")

            prompt_tokens = data.get("prompt_eval_count", 0) or 0
            completion_tokens = data.get("eval_count", 0) or 0
            total_tokens = prompt_tokens + completion_tokens

            llm_resp = LLMResponse(
                content=raw_text,
                model=self._model,
                provider=self.provider_name,
                usage=LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=round(elapsed_ms, 2),
                status="success",
            )

            clean_json = self.clean_json_text(raw_text)
            parsed = schema.model_validate_json(clean_json)

            return LLMStructuredResponse(
                parsed_data=parsed,
                raw_text=raw_text,
                is_valid=True,
                response_metadata=llm_resp,
            )

        except Exception as e:
            err_msg = f"Structured generation/parsing failed for schema {schema.__name__}: {e}"
            logger.warning(err_msg)
            return LLMStructuredResponse(
                parsed_data=None,
                raw_text="",
                is_valid=False,
                error_message=err_msg,
            )

    def health_check(self) -> LLMHealthStatus:
        """
        Ping Ollama server and verify model availability.
        """
        start_time = time.perf_counter()
        tags_endpoint = f"{self._base_url}/api/tags"

        try:
            res = requests.get(tags_endpoint, timeout=5.0)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            if res.status_code != 200:
                return LLMHealthStatus(
                    is_healthy=False,
                    provider=self.provider_name,
                    model=self._model,
                    status_message=f"Ollama server responded with HTTP {res.status_code}",
                    latency_ms=latency_ms,
                )

            data = res.json()
            models_list = data.get("models", [])
            model_names = [m.get("name", "") for m in models_list]

            # Check if requested model or matching model prefix exists
            model_found = any(
                m == self._model or m.startswith(f"{self._model}:") or self._model.startswith(m)
                for m in model_names
            )

            if model_found:
                return LLMHealthStatus(
                    is_healthy=True,
                    provider=self.provider_name,
                    model=self._model,
                    status_message=f"Ollama is online and model '{self._model}' is available.",
                    latency_ms=latency_ms,
                )
            else:
                available_str = ", ".join(model_names) if model_names else "None"
                return LLMHealthStatus(
                    is_healthy=False,
                    provider=self.provider_name,
                    model=self._model,
                    status_message=(
                        f"Ollama is online, but model '{self._model}' is not pulled yet. "
                        f"Available models: [{available_str}]. Run `ollama pull {self._model}`."
                    ),
                    latency_ms=latency_ms,
                )

        except requests.exceptions.ConnectionError:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return LLMHealthStatus(
                is_healthy=False,
                provider=self.provider_name,
                model=self._model,
                status_message=f"Cannot connect to Ollama server at {self._base_url}. Ensure Ollama is running.",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return LLMHealthStatus(
                is_healthy=False,
                provider=self.provider_name,
                model=self._model,
                status_message=f"Ollama health check error: {e}",
                latency_ms=latency_ms,
            )
