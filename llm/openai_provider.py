"""
gestalt_extractor.llm.openai_provider
=====================================
Live OpenAI API extraction provider with dual transport (SDK or stdlib urllib)
and automatic fallback cascade to DeterministicMockProvider.
"""

import os
from pathlib import Path
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

from gestalt_extractor.parser import ParsedDocument
from gestalt_extractor.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    ENV_OPENAI_API_KEY,
    MAX_PARSED_CHARS,
)
from gestalt_extractor.llm.base import (
    LLMProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)

# Check for optional OpenAI SDK installation
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


SYSTEM_PROMPT = """You are an elite theoretical computer scientist and AI systems architect specialized in analyzing mathematical and algorithmic manuscripts.
Your mission is to semantically extract actionable computational bounds, mathematical invariants, asymptotic complexities, and optimization methodologies from manuscripts to serve as foundational data science techniques for building advanced LLM tuners.

You MUST respond strictly with a valid JSON object matching the following schema:
{
  "bounds": [
    {
      "name": "Concise name or title of the bound/invariant",
      "formula": "Exact mathematical formula or asymptotic expression (LaTeX or plain text)",
      "type": "algorithmic_bound | asymptotic_complexity | convergence_rate | parameter_bound | capacity_bound | spectral_bound | invariance_condition | general_bound",
      "context": "Operational context, assumptions, parameter range, or proof summary"
    }
  ],
  "summary": "Dense, publication-quality academic summary detailing theoretical foundations, key theorems, asymptotic bounds, and direct actionable implications for LLM tuner architectures.",
  "methodologies": [
    "Actionable technique 1",
    "Actionable technique 2"
  ]
}

DO NOT wrap the response in markdown code fences. Output raw JSON only."""


class OpenAIProvider(LLMProvider):
    """Live OpenAI API extraction provider with resilient stdlib fallback and cascade."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_LLM_MODEL,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        fallback_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get(ENV_OPENAI_API_KEY, "")).strip()
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._fallback_provider = fallback_provider
        self._sdk_client: Optional[Any] = None

        if HAS_OPENAI_SDK and self.api_key:
            try:
                self._sdk_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.warning("Failed to initialize OpenAI SDK client: %s. Using urllib transport.", e)
                self._sdk_client = None

    @property
    def fallback_provider(self) -> LLMProvider:
        """Lazily instantiates default mock provider if no fallback was explicitly passed."""
        if self._fallback_provider is None:
            from gestalt_extractor.llm.mock_provider import DeterministicMockProvider
            self._fallback_provider = DeterministicMockProvider()
        return self._fallback_provider

    def extract(self, doc: ParsedDocument, domain: str) -> Dict[str, Any]:
        """Extracts bounds and summary via OpenAI API with automatic fallback cascade."""
        if not self.api_key:
            logger.info("No OpenAI API key provided. Cascading to fallback mock provider.")
            return self.fallback_provider.extract(doc, domain)

        prompt = self._build_user_prompt(doc, domain)

        # Attempt extraction with retries
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw_json_str = self._call_api(prompt)
                payload = self._parse_json_response(raw_json_str)
                validated = self.validate_payload(payload)
                validated["metadata"] = {
                    "provider": "openai",
                    "model": self.model,
                    "attempt": attempt + 1,
                }
                return validated
            except urllib.error.HTTPError as e:
                last_error = e
                # 401: Unauthorized (bad key); 429: Rate limit / credit balance exhausted
                if e.code in (401, 429):
                    logger.warning(
                        "OpenAI API non-retriable HTTP %d error (%s). Triggering fallback mock provider.",
                        e.code, e.reason
                    )
                    break
                logger.warning("OpenAI API HTTP %d on attempt %d/%d: %s", e.code, attempt + 1, self.max_retries + 1, e)
            except Exception as e:
                last_error = e
                # Check for OpenAI SDK specific 401 / 429 exceptions
                err_str = str(e).lower()
                if "authentication" in err_str or "unauthorized" in err_str or "rate limit" in err_str or "quota" in err_str:
                    logger.warning("OpenAI SDK client error (%s). Triggering fallback mock provider.", e)
                    break
                logger.warning("OpenAI call failed on attempt %d/%d: %s", attempt + 1, self.max_retries + 1, e)

            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))

        # Cascade to fallback mock provider on failure
        doc_name = getattr(doc, "title", None) or (Path(doc.path).name if getattr(doc, "path", None) else "document")
        logger.warning(
            "OpenAI API extraction failed for '%s' (%s). Seamlessly cascading to DeterministicMockProvider.",
            doc_name, last_error
        )
        fallback_result = self.fallback_provider.extract(doc, domain)
        if "metadata" not in fallback_result or not isinstance(fallback_result["metadata"], dict):
            fallback_result["metadata"] = {}
        fallback_result["metadata"]["fallback_reason"] = str(last_error)
        return fallback_result

    def _build_user_prompt(self, doc: ParsedDocument, domain: str) -> str:
        """Constructs rich structural prompt from parsed document features."""
        lines = [
            f"Manuscript Title: {getattr(doc, 'title', 'Untitled')}",
            f"Classified Domain: {domain}",
            f"Document Type: {getattr(doc, 'doc_type', 'markdown')}",
            f"Word Count: {getattr(doc, 'word_count', 0)}",
        ]

        theorems = getattr(doc, "theorems", []) or []
        if theorems:
            lines.append("\nFormal Theorems, Lemmas, and Axioms:")
            for thm in theorems[:10]:
                lines.append(f"- {thm}")

        math_blocks = getattr(doc, "math_blocks", []) or []
        if math_blocks:
            lines.append("\nKey Mathematical Formulations:")
            for mb in math_blocks[:15]:
                lines.append(f"$$ {mb} $$")

        algorithms = getattr(doc, "algorithms", []) or []
        if algorithms:
            lines.append("\nAlgorithmic Procedures & Implementation Pseudocode:")
            for algo in algorithms[:3]:
                lines.append(f"```\n{algo[:1000]}\n```")

        # Include prose content bounded by MAX_PARSED_CHARS
        prose_budget = MAX_PARSED_CHARS - 5000
        clean_text = getattr(doc, "clean_text", "") or ""
        clean_excerpt = clean_text[:prose_budget]
        lines.append(f"\nManuscript Body Excerpt:\n{clean_excerpt}")

        return "\n".join(lines)

    def _call_api(self, prompt: str) -> str:
        """Dispatches request via OpenAI SDK or stdlib urllib."""
        if self._sdk_client is not None:
            return self._call_via_sdk(prompt)
        return self._call_via_urllib(prompt)

    def _call_via_sdk(self, prompt: str) -> str:
        """Invokes OpenAI SDK chat completions."""
        response = self._sdk_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return response.choices[0].message.content or "{}"

    def _call_via_urllib(self, prompt: str) -> str:
        """Invokes OpenAI REST endpoint via standard library urllib."""
        endpoint = f"{self.base_url}/chat/completions"
        payload_bytes = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "GestaltExtractor/0.1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "{}")
            return "{}"

    def _parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Parses and sanitizes LLM JSON output, stripping potential markdown fences."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


__all__ = [
    "OpenAIProvider",
    "SYSTEM_PROMPT",
]
