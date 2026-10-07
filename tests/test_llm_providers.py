"""
Unit tests for gestalt_extractor.llm (LLM Providers & Mock Engine)
==================================================================
Verifies Milestone 2 Acceptance Criteria:
- Abstract LLMProvider interface and custom exception hierarchy.
- DeterministicMockProvider high-fidelity AST and heuristic extraction:
  - Asymptotics (O, Omega, Theta), theorems, recurrences, and parameter bounds.
  - Domain-aware academic summary synthesis and actionable methodologies.
  - Bit-for-bit determinism and zero-network execution.
- OpenAIProvider live client:
  - Dual-transport handling and markdown fence stripping.
  - Automatic fallback cascade on empty API key, HTTP 401, HTTP 429, or errors.
- Dynamic provider factory: get_provider(config).
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.parser import DocumentParser, ParsedDocument
from gestalt_extractor.llm import (
    LLMProvider,
    OpenAIProvider,
    DeterministicMockProvider,
    MockProvider,
    get_provider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    SchemaValidationError,
)


class TestLLMProviderBase(unittest.TestCase):
    """Verifies LLMProvider abstract contract, exception classes, and payload sanitizer."""

    def test_abstract_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_custom_exception_hierarchy(self):
        self.assertTrue(issubclass(RateLimitError, ProviderError))
        self.assertTrue(issubclass(AuthenticationError, ProviderError))
        self.assertTrue(issubclass(SchemaValidationError, ProviderError))

    def test_validate_payload_defaults(self):
        # Empty payload input
        payload = LLMProvider.validate_payload({})
        self.assertEqual(payload["bounds"], [])
        self.assertEqual(payload["summary"], "Semantic extraction completed without summary.")
        self.assertEqual(payload["methodologies"], [])
        self.assertIsInstance(payload["metadata"], dict)

    def test_validate_payload_normalization(self):
        raw = {
            "bounds": [
                {"name": "Bound 1", "formula": "O(N)", "type": "asymptotic_complexity"},
                "\\lambda \\le 0.5",  # Raw string formula
            ],
            "summary": "   Dense academic summary.   ",
            "methodologies": ["Method A", "   Method B   "],
        }
        validated = LLMProvider.validate_payload(raw)
        self.assertEqual(len(validated["bounds"]), 2)
        self.assertEqual(validated["bounds"][0]["name"], "Bound 1")
        self.assertEqual(validated["bounds"][0]["formula"], "O(N)")
        self.assertEqual(validated["bounds"][1]["name"], "Mathematical Bound")
        self.assertEqual(validated["bounds"][1]["formula"], "\\lambda \\le 0.5")
        self.assertEqual(validated["summary"], "Dense academic summary.")
        self.assertEqual(validated["methodologies"], ["Method A", "Method B"])


class TestDeterministicMockProvider(unittest.TestCase):
    """Verifies high-fidelity heuristic extraction by DeterministicMockProvider."""

    def setUp(self):
        self.provider = DeterministicMockProvider()
        self.parser = DocumentParser()
        self.workspace_root = Path(__file__).resolve().parent.parent.parent

    def test_alias(self):
        self.assertIs(MockProvider, DeterministicMockProvider)

    def test_sample_bounds_tex_extraction(self):
        tex_path = self.workspace_root / "gestalt_extractor/testbed/fixtures/sample_bounds.tex"
        if not tex_path.exists():
            self.skipTest(f"Fixture {tex_path} not found")

        doc = self.parser.parse_file(tex_path)
        result = self.provider.extract(doc, "ML_Optimization")

        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertIn("methodologies", result)

        # Must extract multiple genuine bounds
        bounds = result["bounds"]
        self.assertGreaterEqual(len(bounds), 3)

        # Verify asymptotic bound extraction
        has_asymp = any(b["type"] == "asymptotic_complexity" for b in bounds)
        self.assertTrue(has_asymp, "Should extract at least one asymptotic bound")

        # Verify summary density
        self.assertGreater(len(result["summary"]), 150)
        self.assertIn("ML_Optimization", result["summary"])

    def test_sample_bounds_md_extraction(self):
        md_path = self.workspace_root / "gestalt_extractor/testbed/fixtures/sample_bounds.md"
        if not md_path.exists():
            self.skipTest(f"Fixture {md_path} not found")

        doc = self.parser.parse_file(md_path)
        result = self.provider.extract(doc, "ML_Optimization")

        bounds = result["bounds"]
        self.assertGreaterEqual(len(bounds), 2)
        self.assertGreater(len(result["summary"]), 100)

    def test_determinism_invariance(self):
        # Calling extract twice on the same document must produce identical output
        text = """# Optimization Invariants
We define grokfast momentum update:
$$m_t = \\alpha m_{t-1} + (1 - \\alpha) g_t$$
Complexity is bounded by $\\mathcal{O}(n \\log n)$.
Parameter range: $\\alpha \\in [0.95, 0.99]$.
"""
        doc = self.parser.parse_text(text)
        res1 = self.provider.extract(doc, "ML_Optimization")
        res2 = self.provider.extract(doc, "ML_Optimization")

        self.assertEqual(res1["bounds"], res2["bounds"])
        self.assertEqual(res1["summary"], res2["summary"])
        self.assertEqual(res1["methodologies"], res2["methodologies"])

    def test_empty_document_fallback(self):
        # Even on an empty document, never crash and produce valid structural bound
        doc = ParsedDocument(
            path=Path("empty.md"),
            title="Empty Manuscript",
            raw_text="",
            clean_text="",
        )
        res = self.provider.extract(doc, "General_Applied_Math")
        self.assertGreaterEqual(len(res["bounds"]), 1)
        self.assertTrue(len(res["summary"]) > 0)


class TestOpenAIProvider(unittest.TestCase):
    """Verifies OpenAIProvider live client behavior, fence sanitization, and fallback cascade."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_missing_api_key_cascades_to_mock(self):
        provider = OpenAIProvider(api_key="")
        doc = self.parser.parse_text("# Test Title\n$$O(n)$$")
        result = provider.extract(doc, "ML_Optimization")

        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertGreaterEqual(len(result["bounds"]), 1)

    def test_markdown_fence_stripping(self):
        provider = OpenAIProvider(api_key="sk-dummy-key")
        fenced_json = """```json
{
  "bounds": [
    {
      "name": "Linear Convergence",
      "formula": "\\|x_k - x^*\\| \\le c^k",
      "type": "convergence_rate",
      "context": "Strongly convex assumption"
    }
  ],
  "summary": "Verified linear convergence rate.",
  "methodologies": ["Gradient Descent"]
}
```"""
        parsed = provider._parse_json_response(fenced_json)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(len(parsed["bounds"]), 1)
        self.assertEqual(parsed["bounds"][0]["name"], "Linear Convergence")

    @patch("gestalt_extractor.llm.openai_provider.OpenAIProvider._call_api")
    def test_http_401_cascades_to_fallback_mock(self, mock_call):
        # Simulate HTTP 401 Unauthorized
        mock_call.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        provider = OpenAIProvider(api_key="sk-invalid-test-key", max_retries=0)
        doc = self.parser.parse_text("# Title\n$$O(n^2)$$")
        result = provider.extract(doc, "ML_Optimization")

        # Must NOT raise exception, must cascade to fallback mock
        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertIn("fallback_reason", result.get("metadata", {}))

    @patch("gestalt_extractor.llm.openai_provider.OpenAIProvider._call_api")
    def test_http_429_cascades_to_fallback_mock(self, mock_call):
        # Simulate HTTP 429 Rate Limit
        mock_call.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )

        provider = OpenAIProvider(api_key="sk-exhausted-quota-key", max_retries=0)
        doc = self.parser.parse_text("# Title\n$$O(n^3)$$")
        result = provider.extract(doc, "ML_Optimization")

        self.assertIn("bounds", result)
        self.assertIn("fallback_reason", result.get("metadata", {}))


class TestProviderFactory(unittest.TestCase):
    """Verifies get_provider factory logic."""

    def test_mock_flag_returns_mock_provider(self):
        config = ExtractorConfig(mock=True)
        provider = get_provider(config)
        self.assertIsInstance(provider, DeterministicMockProvider)

    def test_empty_api_key_returns_mock_provider(self):
        config = ExtractorConfig(mock=False, api_key="")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            provider = get_provider(config)
            self.assertIsInstance(provider, DeterministicMockProvider)

    def test_live_key_returns_openai_provider(self):
        config = ExtractorConfig(mock=False, api_key="sk-valid-key-format")
        provider = get_provider(config)
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.api_key, "sk-valid-key-format")
        self.assertIsInstance(provider.fallback_provider, DeterministicMockProvider)

    def test_env_var_gestalt_mock_enforces_mock(self):
        config = ExtractorConfig(mock=False, api_key="sk-valid-key-format")
        with patch.dict(os.environ, {"GESTALT_MOCK": "1"}):
            provider = get_provider(config)
            self.assertIsInstance(provider, DeterministicMockProvider)


if __name__ == "__main__":
    unittest.main()
