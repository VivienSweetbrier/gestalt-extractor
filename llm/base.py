"""
gestalt_extractor.llm.base
==========================
Abstract base class, exception hierarchy, and type contracts for LLM extraction providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from gestalt_extractor.parser import ParsedDocument

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class ProviderError(Exception):
    """Base exception for all LLM extraction provider failures."""
    pass


class RateLimitError(ProviderError):
    """Raised when an LLM provider encounters rate limiting (HTTP 429)."""
    pass


class AuthenticationError(ProviderError):
    """Raised when an LLM provider fails authentication or credential checks (HTTP 401)."""
    pass


class SchemaValidationError(ProviderError):
    """Raised when extraction payload fails validation against the canonical schema."""
    pass


# =============================================================================
# Abstract Base Class
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for manuscript semantic extraction providers."""

    @abstractmethod
    def extract(self, doc: ParsedDocument, domain: str) -> Dict[str, Any]:
        """Extracts computational bounds, semantic summary, and methodologies.

        Args:
            doc: ParsedDocument instance containing extracted manuscript features.
            domain: Mathematical domain category (e.g. 'ML_Optimization').

        Returns:
            Dictionary containing at minimum:
                - 'bounds': List[Dict[str, Any]] where each item has:
                    - 'name': str
                    - 'formula': str
                    - 'type': str
                    - 'context': str
                - 'summary': str (dense natural language semantic summary)
            Optional keys:
                - 'methodologies': List[str]
                - 'metadata': Dict[str, Any]
        """
        pass

    @staticmethod
    def validate_payload(payload: Any) -> Dict[str, Any]:
        """Validates and normalizes extraction payload into canonical schema.

        Ensures that 'bounds' is a list of valid dictionaries and 'summary'
        is a non-empty string.
        """
        if not isinstance(payload, dict):
            payload = {}

        # Normalize bounds
        raw_bounds = payload.get("bounds")
        valid_bounds: List[Dict[str, Any]] = []

        if isinstance(raw_bounds, list):
            for item in raw_bounds:
                if isinstance(item, dict):
                    valid_bounds.append({
                        "name": str(item.get("name") or "Mathematical Bound").strip(),
                        "formula": str(item.get("formula") or "").strip(),
                        "type": str(item.get("type") or "general_bound").strip(),
                        "context": str(item.get("context") or "").strip(),
                    })
                elif isinstance(item, str) and item.strip():
                    valid_bounds.append({
                        "name": "Mathematical Bound",
                        "formula": item.strip(),
                        "type": "general_bound",
                        "context": "",
                    })

        # Normalize summary
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary = "Semantic extraction completed without summary."

        # Normalize methodologies
        raw_methods = payload.get("methodologies")
        valid_methods: List[str] = []
        if isinstance(raw_methods, list):
            valid_methods = [str(m).strip() for m in raw_methods if str(m).strip()]

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "bounds": valid_bounds,
            "summary": summary,
            "methodologies": valid_methods,
            "metadata": metadata,
        }


__all__ = [
    "LLMProvider",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "SchemaValidationError",
]
