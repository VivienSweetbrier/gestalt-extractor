"""
gestalt_extractor.tests.test_adversarial_m2
===========================================
Hostile Adversarial Stress Test Suite for Milestone 2.
Authored by Empirical Challenger 2 to rigorously stress-test:
1. Network & HTTP Fault Injections:
   - HTTP 401 Unauthorized (bad API key, fast fail, zero leakage)
   - HTTP 429 Rate Limits / Quota Exhaustion (fast fail, zero lockup)
   - Network Dropouts (URLError, ConnectionResetError, RemoteDisconnected)
   - Request Timeouts (socket.timeout, TimeoutError)
   - HTTP 500 / 503 Internal Server Errors (retries then cascades)
2. Payload Mutilation & Normalization:
   - Non-JSON string responses (cascades cleanly)
   - Markdown-fenced JSON responses (```json ... ``` stripped and parsed)
   - JSON array instead of dictionary payloads
   - Empty JSON objects ({})
   - Type-polluted payloads (bounds=int, summary=None, methodologies=dict)
   - Corrupted bound items (None, missing keys, non-string values)
3. Deterministic Mock Provider Boundary Stress:
   - Empty math blocks and zero theorems (generates structural bound & summary)
   - Huge document stress (5,000,000 chars, 5,000 equations, no ReDoS)
   - Obscure mathematical notations (Dirac bra-ket, Category Hom, Einstein, Unicode)
   - Display math theorem body formula extraction edge cases
   - Path type duality (Path object vs str path)
   - Bit-for-bit determinism across repeated executions
4. Schema & Mathematical Quality Invariance:
   - Strict 4-key schema: {file, domain, bounds, summary}
   - Mathematical substance (authentic formulas, asymptotic complexities)
   - Domain-specific academic summaries and methodologies
"""

import os
import io
import sys
import json
import time
import socket
import unittest
import urllib.error
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gestalt_extractor.parser import ParsedDocument, DocumentParser
from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.models import (
    ExtractedBound,
    ExtractedRecord,
    ExtractionReport,
)
from gestalt_extractor.classifier import DomainClassifier
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


class TestNetworkFaultInjections(unittest.TestCase):
    """Stress tests simulating live network dropouts, HTTP error codes, and timeouts."""

    def setUp(self):
        self.doc = ParsedDocument(
            path=Path("sample_manuscript.tex"),
            title="Adversarial Testing Manuscript",
            raw_text="Sample text",
            clean_text="Sample clean text discussing gradient descent and convergence rates.",
            math_blocks=[r"\mathcal{O}(d^2)", r"\alpha \in [0.01, 0.1]"],
            theorems=["Theorem 1: Convergence bound is O(1/t)"],
            sections=[{"title": "Introduction", "level": 1, "type": "section"}],
            algorithms=[r"\alpha \in [0.01, 0.1]"],
            doc_type="latex",
            char_count=100,
            word_count=20,
        )

    @patch("urllib.request.urlopen")
    def test_http_401_bad_api_key_cascades_immediately(self, mock_urlopen):
        """HTTP 401 Unauthorized must immediately cascade to mock provider without retrying."""
        error_resp = io.BytesIO(b'{"error": {"message": "Incorrect API key provided"}}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=error_resp,
        )

        provider = OpenAIProvider(
            api_key="sk-invalid-test-key-12345",
            max_retries=3,
        )

        result = provider.extract(self.doc, "ML_Optimization")

        # Fast fail: should only have called urlopen ONCE (no useless retries on 401)
        self.assertEqual(mock_urlopen.call_count, 1)

        # Fallback provider must succeed and populate valid bounds
        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertGreaterEqual(len(result["bounds"]), 1)

        # Verify fallback reason recorded in metadata
        self.assertIn("metadata", result)
        self.assertIn("fallback_reason", result["metadata"])
        self.assertIn("401", result["metadata"]["fallback_reason"])

        # Security check: secret key must never be echoed in result or metadata
        self.assertNotIn("sk-invalid-test-key-12345", json.dumps(result))

    @patch("urllib.request.urlopen")
    def test_http_429_rate_limit_cascades_immediately(self, mock_urlopen):
        """HTTP 429 Rate Limit must immediately cascade to mock provider without loop stall."""
        error_resp = io.BytesIO(b'{"error": {"message": "Rate limit exceeded / Quota exhausted"}}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=error_resp,
        )

        provider = OpenAIProvider(
            api_key="sk-valid-key-but-rate-limited",
            max_retries=3,
        )

        result = provider.extract(self.doc, "ML_Optimization")

        # Should fast-fail on 429 without running through all retries
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertIn("429", result["metadata"]["fallback_reason"])

    @patch("urllib.request.urlopen")
    def test_network_dropout_connection_refused(self, mock_urlopen):
        """Simulates network dropouts (ConnectionRefused, URLError), retrying then cascading."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused by peer")

        provider = OpenAIProvider(
            api_key="sk-dummy-key",
            max_retries=1,  # 1 retry = 2 attempts total
        )

        # Patch time.sleep to run instantly without waiting
        with patch("time.sleep"):
            result = provider.extract(self.doc, "ML_Optimization")

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertIn("Connection refused", result["metadata"]["fallback_reason"])

    @patch("urllib.request.urlopen")
    def test_request_timeout_cascades(self, mock_urlopen):
        """Simulates socket read/connect timeout cascading to mock provider."""
        mock_urlopen.side_effect = socket.timeout("Timed out connecting to endpoint")

        provider = OpenAIProvider(
            api_key="sk-dummy-key",
            max_retries=1,
        )

        with patch("time.sleep"):
            result = provider.extract(self.doc, "ML_Optimization")

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertIn("bounds", result)
        self.assertIn("Timed out", result["metadata"]["fallback_reason"])

    @patch("urllib.request.urlopen")
    def test_http_500_internal_server_error_retries_and_cascades(self, mock_urlopen):
        """HTTP 500 retries up to max_retries before cleanly cascading to mock provider."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b'{"error": "Internal Error"}'),
        )

        provider = OpenAIProvider(
            api_key="sk-dummy-key",
            max_retries=2,
        )

        with patch("time.sleep"):
            result = provider.extract(self.doc, "ML_Optimization")

        self.assertEqual(mock_urlopen.call_count, 3)  # initial + 2 retries
        self.assertIn("bounds", result)
        self.assertIn("500", result["metadata"]["fallback_reason"])


class TestMalformedPayloadSanitization(unittest.TestCase):
    """Stress tests verifying normalization of mutilated, corrupted, and malformed JSON payloads."""

    def setUp(self):
        self.provider = OpenAIProvider(api_key="sk-test-key", max_retries=0)
        self.doc = ParsedDocument(
            path=Path("paper.tex"),
            title="Paper",
            raw_text="Body",
            clean_text="Clean body",
            doc_type="latex",
        )

    @patch("urllib.request.urlopen")
    def test_non_json_string_payload_cascades(self, mock_urlopen):
        """Non-JSON HTML or error text causes JSON parse error and cascades to mock provider."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body>502 Bad Gateway</body></html>"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.provider.extract(self.doc, "General_Applied_Math")

        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertIn("JSONDecodeError", result["metadata"]["fallback_reason"])

    @patch("urllib.request.urlopen")
    def test_markdown_fenced_json_is_stripped_and_parsed(self, mock_urlopen):
        """Markdown code fences (```json ... ```) are correctly stripped and parsed."""
        payload_data = {
            "bounds": [
                {
                    "name": "Spectral Radius Bound",
                    "formula": r"\rho(A) \le \|A\|",
                    "type": "spectral_bound",
                    "context": "Matrix norm inequality",
                }
            ],
            "summary": "Valid academic summary.",
            "methodologies": ["Matrix Norm Scaling"],
        }
        fenced_content = f"```json\n{json.dumps(payload_data)}\n```"
        api_response = {
            "choices": [{"message": {"content": fenced_content}}]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = self.provider.extract(self.doc, "General_Applied_Math")

        self.assertEqual(len(result["bounds"]), 1)
        self.assertEqual(result["bounds"][0]["name"], "Spectral Radius Bound")
        self.assertEqual(result["summary"], "Valid academic summary.")
        self.assertEqual(result["metadata"]["provider"], "openai")

    def test_validate_payload_empty_object(self):
        """Empty payload {} produces valid defaults matching schema."""
        validated = LLMProvider.validate_payload({})
        self.assertEqual(validated["bounds"], [])
        self.assertIsInstance(validated["summary"], str)
        self.assertGreater(len(validated["summary"]), 0)
        self.assertEqual(validated["methodologies"], [])
        self.assertEqual(validated["metadata"], {})

    def test_validate_payload_non_dict_input(self):
        """Passing a list or string to validate_payload gracefully produces defaults."""
        for invalid_input in [[1, 2, 3], "string_response", None]:
            validated = LLMProvider.validate_payload(invalid_input)
            self.assertEqual(validated["bounds"], [])
            self.assertIsInstance(validated["summary"], str)
            self.assertEqual(validated["methodologies"], [])

    def test_validate_payload_type_pollution(self):
        """Payload with polluted types (bounds=123, summary=None) is sanitized."""
        polluted = {
            "bounds": 12345,
            "summary": None,
            "methodologies": {"dict": "not_a_list"},
            "metadata": "not_a_dict",
        }
        validated = LLMProvider.validate_payload(polluted)
        self.assertEqual(validated["bounds"], [])
        self.assertEqual(validated["summary"], "Semantic extraction completed without summary.")
        self.assertEqual(validated["methodologies"], [])
        self.assertEqual(validated["metadata"], {})

    def test_validate_payload_corrupted_bound_elements(self):
        """Bounds list containing None, empty dict, or non-dict items is sanitized."""
        corrupted = {
            "bounds": [
                None,
                {},
                {"name": None, "formula": None, "type": None, "context": None},
                "r'\\lambda \\le 1'",  # raw formula string
                {"name": "Valid Bound", "formula": r"\mathcal{O}(1)"},
            ],
            "summary": "Valid summary",
        }
        validated = LLMProvider.validate_payload(corrupted)
        bounds = validated["bounds"]
        self.assertEqual(len(bounds), 3)  # {} and None without formula are filtered; dict with None gets default
        for b in bounds:
            self.assertIsInstance(b["name"], str)
            self.assertIsInstance(b["formula"], str)
            self.assertIsInstance(b["type"], str)
            self.assertIsInstance(b["context"], str)


class TestDeterministicMockProviderBoundaries(unittest.TestCase):
    """Stress tests evaluating boundary conditions and zero-crash invariance for DeterministicMockProvider."""

    def setUp(self):
        self.provider = DeterministicMockProvider()

    def test_empty_math_blocks_and_zero_theorems(self):
        """A document with 0 math blocks, 0 theorems, and 0 algorithms produces a structural bound."""
        doc = ParsedDocument(
            path=Path("pure_prose.md"),
            title="A Philosophical Treatise on Computational Limits",
            raw_text="Pure prose manuscript with zero mathematical formatting.",
            clean_text="Pure prose manuscript discussing computability and epistemic certainty.",
            math_blocks=[],
            theorems=[],
            sections=[{"title": "Epistemology", "level": 1, "type": "section"}],
            algorithms=[],
            doc_type="markdown",
            char_count=60,
            word_count=10,
        )

        result = self.provider.extract(doc, "General_Applied_Math")

        self.assertIn("bounds", result)
        self.assertGreaterEqual(len(result["bounds"]), 1)
        bound = result["bounds"][0]
        self.assertIn("Structural Formulation", bound["name"])
        self.assertIn("bounds", result)
        self.assertIn("summary", result)
        self.assertGreater(len(result["summary"]), 100)
        self.assertIn("methodologies", result)
        self.assertGreaterEqual(len(result["methodologies"]), 1)

    def test_huge_document_stress_scale(self):
        """Stress tests against a massive manuscript with 5,000,000 characters and 5,000 equations."""
        math_block_template = r"\mathcal{O}(n^{%d}) + \sum_{i=1}^n x_i \le C_%d"
        large_math_blocks = [math_block_template % (i % 10, i) for i in range(5000)]
        large_theorems = [f"Theorem {i}: Sequence converges in O(1/{i+1}) iterations" for i in range(200)]
        large_prose = ("This is an extensive analysis of regularized gradient trajectories in overparameterized spaces. " * 50000)

        doc = ParsedDocument(
            path=Path("massive_manuscript.tex"),
            title="Scale Stress Testing Manuscript",
            raw_text=large_prose,
            clean_text=large_prose,
            math_blocks=large_math_blocks,
            theorems=large_theorems,
            sections=[{"title": f"Section {i}", "level": 1, "type": "section"} for i in range(50)],
            algorithms=[r"\alpha \in [0.001, 0.01]", r"\beta \in [0.9, 0.99]"],
            doc_type="latex",
            char_count=len(large_prose),
            word_count=len(large_prose.split()),
        )

        start_time = time.time()
        result = self.provider.extract(doc, "ML_Optimization")
        duration = time.time() - start_time

        # Must complete within 5 seconds without memory exhaustion or ReDoS hang
        self.assertLess(duration, 5.0)
        self.assertGreater(len(result["bounds"]), 10)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 100)

    def test_obscure_mathematical_notations(self):
        """Verifies parsing of exotic notations (Dirac bra-ket, Category Hom, Einstein, Unicode)."""
        obscure_equations = [
            r"\langle \psi | \hat{H} | \psi \rangle \ge E_0",  # Dirac bra-ket
            r"\mathrm{Hom}_{\mathcal{C}}(A, B) \otimes \mathrm{Hom}_{\mathcal{C}}(B, C) \to \mathrm{Hom}(A, C)", # Category theory
            r"R^\mu{}_{\nu\rho\sigma} T^{\rho\sigma} \ge 0",   # Einstein notation
            r"\widetilde{\mathcal{O}}(N \operatorname{polylog}(N))", # Soft-O notation
            r"\forall x \in \mathbb{R},\ \exists y : x \prec y \land x \preceq y", # Unicode / logic symbols
            r"\kappa_{\mathrm{eff}} \le 10^4",                 # Effective condition number
        ]

        doc = ParsedDocument(
            path=Path("exotic_math.tex"),
            title="Exotic Mathematical Systems",
            raw_text="Exotic formulations.",
            clean_text="Exotic formulations.",
            math_blocks=obscure_equations,
            theorems=["Proposition 1: Spectral condition holds"],
            sections=[],
            algorithms=[],
            doc_type="latex",
        )

        result = self.provider.extract(doc, "General_Applied_Math")

        self.assertIn("bounds", result)
        self.assertGreaterEqual(len(result["bounds"]), 1)
        formulas = [b["formula"] for b in result["bounds"]]
        # Should extract soft-O and condition number bounds
        has_soft_o = any(r"\widetilde{\mathcal{O}}" in f or "polylog" in f for f in formulas)
        has_cond = any(r"\kappa" in f for f in formulas)
        self.assertTrue(has_soft_o or has_cond)

    def test_bit_for_bit_determinism(self):
        """Repeated extractions on the identical document must produce bit-for-bit identical results."""
        doc = ParsedDocument(
            path=Path("deterministic_test.tex"),
            title="Deterministic Invariance Test",
            raw_text="Discussion of gradient bounds.",
            clean_text="Discussion of gradient bounds.",
            math_blocks=[r"\mathcal{O}(d^2)", r"m_t = \alpha m_{t-1} + (1-\alpha) g_t"],
            theorems=["Theorem 1: Convergence bound is O(1/t)"],
            sections=[{"title": "Bounds", "level": 1, "type": "section"}],
            algorithms=[r"\alpha \in [0.9, 0.99]"],
            doc_type="latex",
        )

        first_run = json.dumps(self.provider.extract(doc, "ML_Optimization"), sort_keys=True)
        for _ in range(25):
            subsequent_run = json.dumps(self.provider.extract(doc, "ML_Optimization"), sort_keys=True)
            self.assertEqual(first_run, subsequent_run)


class TestSchemaAndMathematicalQuality(unittest.TestCase):
    """Stress tests asserting strict schema adherence and mathematical substance."""

    def setUp(self):
        self.provider = DeterministicMockProvider()
        self.parser = DocumentParser()
        self.tex_fixture = REPO_ROOT / "gestalt_extractor/testbed/fixtures/sample_bounds.tex"
        self.md_fixture = REPO_ROOT / "gestalt_extractor/testbed/fixtures/sample_bounds.md"

    def test_strict_four_key_schema_invariance(self):
        """ExtractedRecord serialization must strictly produce keys: file, domain, bounds, summary."""
        doc = ParsedDocument(
            path=Path("my_paper.md"),
            title="My Paper",
            raw_text="Paper content.",
            clean_text="Paper content.",
            math_blocks=[r"\mathcal{O}(N \log N)"],
            theorems=[],
            doc_type="markdown",
        )

        extracted = self.provider.extract(doc, "Graph_Combinatorics")
        record = ExtractedRecord(
            file=doc.path.name,
            domain="Graph_Combinatorics",
            bounds=extracted["bounds"],
            summary=extracted["summary"],
            methodologies=extracted.get("methodologies"),
            metadata=extracted.get("metadata"),
        )

        # Baseline dictionary without optional fields must have EXACTLY 4 keys
        strict_dict = record.to_dict(include_optional=False)
        self.assertEqual(set(strict_dict.keys()), {"file", "domain", "bounds", "summary"})

        # Serialized JSON string must parse into valid JSON containing all 4 keys
        json_str = record.to_json(include_optional=False)
        parsed_json = json.loads(json_str)
        self.assertEqual(set(parsed_json.keys()), {"file", "domain", "bounds", "summary"})
        self.assertEqual(parsed_json["file"], "my_paper.md")
        self.assertEqual(parsed_json["domain"], "Graph_Combinatorics")
        self.assertIsInstance(parsed_json["bounds"], list)
        self.assertIsInstance(parsed_json["summary"], str)

    def test_authentic_formulas_and_substance_from_tex_fixture(self):
        """Extracts authentic mathematical formulas and asymptotic bounds from sample_bounds.tex."""
        if not self.tex_fixture.exists():
            self.skipTest("sample_bounds.tex fixture not found")

        doc = self.parser.parse_file(self.tex_fixture)
        extracted = self.provider.extract(doc, "ML_Optimization")

        bounds = extracted["bounds"]
        formulas = [b["formula"] for b in bounds]

        # Must contain asymptotic bound O(d^2)
        has_asymp = any(r"\mathcal{O}(d^2)" in f or "O(d^2)" in f for f in formulas)
        self.assertTrue(has_asymp, f"Expected O(d^2) bound in formulas: {formulas}")

        # Must contain momentum recurrence
        has_recurrence = any(r"m_t" in f or r"\alpha" in f for f in formulas)
        self.assertTrue(has_recurrence, f"Expected momentum recurrence in formulas: {formulas}")

        # Summary must be dense, professional, and reference LLM tuning
        summary = extracted["summary"]
        self.assertIn("ML_Optimization", summary)
        self.assertIn("tuner", summary.lower())
        self.assertGreater(len(summary.split()), 40)

    def test_authentic_formulas_and_substance_from_md_fixture(self):
        """Extracts authentic bounds and parameter intervals from sample_bounds.md."""
        if not self.md_fixture.exists():
            self.skipTest("sample_bounds.md fixture not found")

        doc = self.parser.parse_file(self.md_fixture)
        extracted = self.provider.extract(doc, "ML_Optimization")

        bounds = extracted["bounds"]
        formulas = [b["formula"] for b in bounds]

        # Must contain parameter interval \alpha \in [0.9, 0.99]
        has_param = any(r"\alpha \in [0.9, 0.99]" in f or "[0.9, 0.99]" in f for f in formulas)
        self.assertTrue(has_param, f"Expected alpha in [0.9, 0.99] in formulas: {formulas}")

        # Must contain asymptotic complexity
        has_asymp = any(r"\mathcal{O}" in f or "O(" in f for f in formulas)
        self.assertTrue(has_asymp, f"Expected asymptotic bound in formulas: {formulas}")

    def test_domain_classifier_word_boundary_defense(self):
        """Verifies classifier word-boundary defenses ('knowledge' != 'edge')."""
        classifier = DomainClassifier()

        # Philosophy paper containing 'knowledge' 50 times should NOT classify as Graph_Combinatorics
        phil_doc = ParsedDocument(
            path=Path("epistemology.md"),
            title="Analysis of Epistemic Justification and Knowledge",
            raw_text="Knowledge and justified true belief. " * 50,
            clean_text="Knowledge and justified true belief. " * 50,
            doc_type="markdown",
        )

        domain, scores = classifier.classify_with_scores(phil_doc)
        self.assertEqual(scores.get("Graph_Combinatorics", 0.0), 0.0)
        self.assertEqual(domain, "General_Applied_Math")


class TestVulnerabilitiesReproduction(unittest.TestCase):
    """Specific reproduction cases for vulnerabilities identified during Milestone 2 audit."""

    def setUp(self):
        self.provider = DeterministicMockProvider()

    def test_repro_display_math_in_theorem_body_dropped(self):
        """Reproduction of Bug 1: $$ in theorem body causes math_matches[0] == '' and formula dropped.

        When a manuscript theorem defines an equation with display math:
        'Theorem 1: Let delta > 0. Then $$ \mathcal{O}(n \log n) \le C $$ holds.'
        re.findall(r'\$(.*?)\$', body) matches adjacent '$$' with empty content '',
        causing formula to be '' and the theorem to be silently dropped by _add_bound().
        """
        doc = ParsedDocument(
            path=Path("display_thm.md"),
            title="Display Theorem Doc",
            raw_text="...",
            clean_text="...",
            math_blocks=[],
            theorems=[r"Theorem 1: Let \delta > 0. Then $$ \mathcal{O}(n \log n) \le C $$ holds."],
            doc_type="markdown",
        )
        result = self.provider.extract(doc, "ML_Optimization")
        bounds = result["bounds"]
        formulas = [b["formula"] for b in bounds]
        # In the unpatched code, Theorem 1's formula is dropped!
        self.assertTrue(
            any(r"\mathcal{O}(n \log n)" in f for f in formulas),
            f"Vulnerability Repro: Theorem formula was dropped due to empty '$$' match! Extracted: {formulas}"
        )

    def test_repro_string_path_attribute_error(self):
        """Reproduction of Bug 2: Passing path as str causes AttributeError ('str' has no attribute 'stem').

        In DeterministicMockProvider._synthesize_semantic_summary():
        doc_path = getattr(doc, 'path', None)
        doc_stem = doc_path.stem if doc_path else 'manuscript'
        If doc.path is a str rather than a pathlib.Path, doc_path.stem raises AttributeError.
        Similarly, in OpenAIProvider: doc.path.name raises AttributeError.
        """
        doc = ParsedDocument(
            path="manuscripts/paper.tex",  # type: ignore[arg-type]
            title=None,
            raw_text="Sample text",
            clean_text="Sample text",
            math_blocks=[r"\mathcal{O}(1)"],
            theorems=[],
            doc_type="latex",
        )
        try:
            result = self.provider.extract(doc, "ML_Optimization")
            self.assertIn("summary", result)
        except AttributeError as e:
            self.fail(f"Vulnerability Repro: doc.path as str caused crash: {e}")


if __name__ == "__main__":
    unittest.main()

