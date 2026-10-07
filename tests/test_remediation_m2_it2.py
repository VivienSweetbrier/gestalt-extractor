"""
gestalt_extractor.tests.test_remediation_m2_it2
===============================================
Adversarial Verification & Remediation Test Suite for Milestone 2 (Iteration 2).
Authored by Empirical Challenger (challenger_m2_it2_1) to rigorously verify:

1. Schema Validation Invariant:
   - ExtractedRecord.from_dict() strictly rejects missing required keys ('file', 'domain', 'bounds', 'summary').
   - ExtractedRecord.from_dict() strictly rejects None for required keys.
   - Raises SchemaValidationError (and is a subclass of ValueError).
   - ExtractionReport.add_record() and ExtractionReport.from_dict() enforce schema validation.

2. Heterogeneous Bounds Serialization:
   - ExtractedRecord.to_dict() safely serializes bounds containing dicts, strings, objects, numbers, and None.
   - Eliminates 'AttributeError: dict object has no attribute to_dict'.
   - Generates strictly conforming 4-key JSON schema.

3. Thread-Safe File Persistence:
   - ExtractionReport.save() under concurrent multi-threaded execution (ThreadPoolExecutor).
   - Temporary file isolation via unique UUID tokens preventing Windows PID collisions / WinError 32.

4. Null Sections & Feature Robustness:
   - ParsedDocument(sections=None) does not crash classifier or providers.
   - Malformed/non-dict sections list items do not crash classifier.
   - Fully uninitialized ParsedDocument does not crash.

5. Mathematical Domain Disambiguation:
   - Fluid divergence (\\nabla \\cdot u = 0), curl (\\nabla \\times u = w), and Laplacian (\\nabla^2 u = 0)
     strictly award 0.0 points to ML_Optimization.
   - Authentic ML gradient formulations (\\nabla \\mathcal{L}, \\nabla_\\theta f) award > 0.0 points to ML_Optimization.

6. Provider Ergonomics & Formula Preservation:
   - Path type duality (str vs Path) in both MockProvider and OpenAIProvider fallback.
   - Theorem body display math ($$ ... $$) extracted without dropping.
"""

import os
import sys
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gestalt_extractor.parser import ParsedDocument
from gestalt_extractor.classifier import DomainClassifier, DEFAULT_FALLBACK_DOMAIN
from gestalt_extractor.models import (
    PYDANTIC_AVAILABLE,
    ExtractedBound,
    ExtractedRecord,
    ExtractionReport,
    SchemaValidationError,
    DataclassExtractedBound,
    DataclassExtractedRecord,
    DataclassExtractionReport,
)
from gestalt_extractor.llm import (
    DeterministicMockProvider,
    OpenAIProvider,
)

if PYDANTIC_AVAILABLE:
    from gestalt_extractor.models import (
        PydanticExtractedBound,
        PydanticExtractedRecord,
        PydanticExtractionReport,
    )


class TestRemediationSchemaValidation(unittest.TestCase):
    """Rigorous verification of Schema Validation in ExtractedRecord and ExtractionReport."""

    def test_from_dict_rejects_empty_dict(self):
        """Calling ExtractedRecord.from_dict({}) must raise SchemaValidationError."""
        with self.assertRaises(SchemaValidationError) as ctx:
            ExtractedRecord.from_dict({})
        self.assertIn("file", str(ctx.exception))
        self.assertTrue(issubclass(SchemaValidationError, ValueError))

    def test_from_dict_rejects_missing_individual_required_keys(self):
        """Verify that each required key ('file', 'domain', 'bounds', 'summary') is strictly required."""
        valid_payload = {
            "file": "test_paper.tex",
            "domain": "ML_Optimization",
            "bounds": [{"name": "Bound 1", "formula": r"\mathcal{O}(1)"}],
            "summary": "Academic summary of the paper.",
        }

        # Check each required key missing one-by-one
        for required_key in ("file", "domain", "bounds", "summary"):
            incomplete = dict(valid_payload)
            del incomplete[required_key]
            with self.assertRaises(SchemaValidationError, msg=f"Should raise on missing '{required_key}'") as ctx:
                ExtractedRecord.from_dict(incomplete)
            self.assertIn(required_key, str(ctx.exception))

    def test_from_dict_rejects_none_values_for_required_keys(self):
        """Verify that setting required keys to None raises SchemaValidationError."""
        base_payload = {
            "file": "test_paper.tex",
            "domain": "ML_Optimization",
            "bounds": [],
            "summary": "Valid summary.",
        }

        for required_key in ("file", "domain", "bounds", "summary"):
            corrupted = dict(base_payload)
            corrupted[required_key] = None
            with self.assertRaises(SchemaValidationError, msg=f"Should raise when '{required_key}' is None") as ctx:
                ExtractedRecord.from_dict(corrupted)
            self.assertIn(required_key, str(ctx.exception))

    def test_from_dict_rejects_non_dict_inputs(self):
        """Passing non-dict types (list, str, int) raises TypeError."""
        for invalid_input in (["file", "domain"], "raw_string", 12345):
            with self.assertRaises(TypeError):
                ExtractedRecord.from_dict(invalid_input)

    def test_dataclass_and_pydantic_schema_validation_parity(self):
        """Verify both stdlib Dataclass and Pydantic implementations enforce identical schema validation."""
        with self.assertRaises(SchemaValidationError):
            DataclassExtractedRecord.from_dict({})

        if PYDANTIC_AVAILABLE:
            with self.assertRaises(SchemaValidationError):
                PydanticExtractedRecord.from_dict({})

    def test_report_add_record_enforces_schema_validation(self):
        """ExtractionReport.add_record() with an empty dict must raise SchemaValidationError."""
        report = ExtractionReport()
        with self.assertRaises(SchemaValidationError):
            report.add_record({})

    def test_report_from_dict_enforces_schema_validation(self):
        """ExtractionReport.from_dict() with invalid records must raise SchemaValidationError."""
        with self.assertRaises(SchemaValidationError):
            ExtractionReport.from_dict([{}])


class TestRemediationHeterogeneousBounds(unittest.TestCase):
    """Verifies that to_dict() and to_json() safely serialize heterogeneous bounds collections."""

    def test_heterogeneous_bounds_list_serialization(self):
        """Record with raw dicts, raw strings, objects, and numbers serializes safely."""
        record = ExtractedRecord(
            file="hetero_paper.tex",
            domain="Graph_Combinatorics",
            bounds=[],
            summary="Heterogeneous bounds test.",
        )

        # Mutate bounds directly (common in pipeline transformations)
        record.bounds.append({"name": "Raw Dict Bound", "formula": r"\chi(G) \ge \omega(G)", "type": "chromatic"})
        record.bounds.append(r"\deg(v) \le \Delta")  # raw string
        record.bounds.append(ExtractedBound(name="Object Bound", formula=r"|E| \le \binom{V}{2}"))
        record.bounds.append(42)  # scalar number
        record.bounds.append(None)  # None value

        # Must NOT raise AttributeError: 'dict' object has no attribute 'to_dict'
        d = record.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d.keys()), {"file", "domain", "bounds", "summary"})
        self.assertEqual(len(d["bounds"]), 5)

        # Verify each bound normalized properly
        self.assertEqual(d["bounds"][0]["name"], "Raw Dict Bound")
        self.assertEqual(d["bounds"][1]["formula"], r"\deg(v) \le \Delta")
        self.assertEqual(d["bounds"][2]["name"], "Object Bound")

        # Verify JSON roundtrip
        json_str = record.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed["bounds"]), 5)

    def test_report_serialization_with_heterogeneous_bounds(self):
        """ExtractionReport serializes records with heterogeneous bounds without error."""
        record = ExtractedRecord(
            file="report_doc.md",
            domain="General_Applied_Math",
            bounds=[{"name": "Dict In Bound", "formula": "x <= y"}],
            summary="Report summary.",
        )
        report = ExtractionReport(records=[record])
        array_dict = report.to_dict(as_envelope=False)
        self.assertIsInstance(array_dict, list)
        self.assertEqual(len(array_dict), 1)
        self.assertEqual(array_dict[0]["bounds"][0]["name"], "Dict In Bound")


class TestRemediationThreadSafety(unittest.TestCase):
    """Verifies thread-safe atomic file writing in ExtractionReport.save()."""

    def test_concurrent_saves_to_same_path(self):
        """Concurrent threads saving reports to the exact same file path must not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "concurrent_output.json"

            def worker_save(thread_idx: int) -> bool:
                record = ExtractedRecord(
                    file=f"thread_doc_{thread_idx}.md",
                    domain="ML_Optimization",
                    bounds=[{"name": f"Bound {thread_idx}", "formula": f"O({thread_idx})"}],
                    summary=f"Summary from worker {thread_idx}",
                )
                report = ExtractionReport(records=[record])
                report.save(target_file)
                return True

            # Execute 20 concurrent saves across 8 threads
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker_save, i) for i in range(20)]
                results = [f.result() for f in futures]

            self.assertEqual(len(results), 20)
            self.assertTrue(all(results))
            self.assertTrue(target_file.exists())

            # Verify target file is valid JSON
            with open(target_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsInstance(loaded, list)
            self.assertGreaterEqual(len(loaded), 1)
            self.assertIn("file", loaded[0])


class TestRemediationNullSections(unittest.TestCase):
    """Verifies defensive handling of ParsedDocument(sections=None) and uninitialized fields."""

    def setUp(self):
        self.classifier = DomainClassifier()

    def test_sections_none_does_not_crash(self):
        """ParsedDocument(sections=None) must not raise TypeError."""
        doc = ParsedDocument(
            path=Path("null_sec.tex"),
            title="Convex Optimization in Reproducing Kernel Spaces",
            raw_text="Discussion of gradient descent.",
            clean_text="Discussion of gradient descent.",
            sections=None,  # Explicitly None
            math_blocks=[r"\nabla \mathcal{L}(\theta) = 0"],
            theorems=["Theorem 1: Convergence rate"],
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "ML_Optimization")
        self.assertGreater(scores["ML_Optimization"], 0.0)

    def test_sections_with_malformed_items(self):
        """Sections list containing non-dict items or dicts with missing/None title."""
        doc = ParsedDocument(
            path=Path("malformed_sec.tex"),
            title="Analysis of Random Graphs",
            raw_text="Vertices and edges.",
            clean_text="Vertices and edges.",
            sections=[None, 123, "not_a_dict", {"level": 1}, {"title": ""}],
            math_blocks=[r"G = (V, E)"],
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, "Graph_Combinatorics")

    def test_all_document_attributes_none(self):
        """ParsedDocument with all nullable attributes set to None."""
        doc = ParsedDocument(
            path=Path("all_null.tex"),
            title=None,
            raw_text=None,
            clean_text=None,
            sections=None,
            math_blocks=None,
            theorems=None,
            algorithms=None,
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(domain, DEFAULT_FALLBACK_DOMAIN)
        self.assertTrue(all(s == 0.0 for s in scores.values()))


class TestRemediationFluidVsMLDisambiguation(unittest.TestCase):
    """Verifies isolation between Fluid Dynamics differential operators and ML loss gradients."""

    def setUp(self):
        self.classifier = DomainClassifier()

    def test_pure_fluid_divergence_awards_zero_ml_points(self):
        """\\nabla \\cdot u = 0 must award 0.0 points to ML_Optimization and > 0 to Fluid_Dynamics."""
        doc = ParsedDocument(
            path=Path("divergence.tex"),
            title="Incompressible Continuity",
            raw_text=r"$$\nabla \cdot \mathbf{u} = 0$$",
            clean_text="Incompressible fluid continuity equation.",
            math_blocks=[r"\nabla \cdot \mathbf{u} = 0"],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["ML_Optimization"], 0.0)
        self.assertGreater(scores["Fluid_Dynamics"], 0.0)
        self.assertEqual(domain, "Fluid_Dynamics")

    def test_pure_fluid_curl_awards_zero_ml_points(self):
        """\\nabla \\times u = \\omega must award 0.0 points to ML_Optimization."""
        doc = ParsedDocument(
            path=Path("curl.tex"),
            title="Vorticity Invariance",
            raw_text=r"$$\nabla \times \mathbf{u} = \boldsymbol{\omega}$$",
            clean_text="Vorticity dynamics.",
            math_blocks=[r"\nabla \times \mathbf{u} = \boldsymbol{\omega}"],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["ML_Optimization"], 0.0)
        self.assertGreater(scores["Fluid_Dynamics"], 0.0)

    def test_vector_laplacian_awards_zero_ml_points(self):
        """\\nabla^2 u and \\nabla^{2} u must award 0.0 points to ML_Optimization."""
        doc = ParsedDocument(
            path=Path("laplacian.tex"),
            title="Viscous Dissipation",
            raw_text=r"$$\nabla^2 \mathbf{u} = 0, \quad \nabla^{2} \phi = 0$$",
            clean_text="Viscous dissipation.",
            math_blocks=[r"\nabla^2 \mathbf{u} = 0", r"\nabla^{2} \phi = 0"],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["ML_Optimization"], 0.0)
        self.assertGreater(scores["Fluid_Dynamics"], 0.0)

    def test_whitespace_variation_in_divergence(self):
        """Spacing variations in \\nabla \\cdot (no space, multi-space) must not leak to ML."""
        doc = ParsedDocument(
            path=Path("whitespace.tex"),
            title="Fluid Flow",
            raw_text="",
            clean_text="Fluid flow.",
            math_blocks=[
                r"\nabla\cdot\mathbf{u} = 0",
                r"\nabla   \cdot   \mathbf{u} = 0",
                r"\nabla   \times   \mathbf{u} = 0",
            ],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertEqual(scores["ML_Optimization"], 0.0)

    def test_genuine_ml_loss_gradients_awarded_points(self):
        """\\nabla \\mathcal{L}(\\theta) must award points to ML_Optimization."""
        doc = ParsedDocument(
            path=Path("ml_grad.tex"),
            title="Loss Minimization",
            raw_text=r"$$\nabla \mathcal{L}(\theta) = 0$$",
            clean_text="Parameter update step.",
            math_blocks=[r"\nabla \mathcal{L}(\theta) = 0", r"\nabla_\theta f(x)"],
            doc_type="latex",
        )
        domain, scores = self.classifier.classify_with_scores(doc)
        self.assertGreater(scores["ML_Optimization"], 0.0)
        self.assertEqual(domain, "ML_Optimization")


class TestRemediationProviderResilience(unittest.TestCase):
    """Verifies provider resilience fixes: string path duality and display math preservation."""

    def test_mock_provider_string_path_duality(self):
        """DeterministicMockProvider handles doc.path as str without AttributeError."""
        provider = DeterministicMockProvider()
        doc = ParsedDocument(
            path="manuscripts/paper.tex",  # type: ignore[arg-type]
            title=None,
            raw_text="Paper",
            clean_text="Paper text.",
            math_blocks=[r"\mathcal{O}(1)"],
        )
        result = provider.extract(doc, "ML_Optimization")
        self.assertIn("summary", result)
        self.assertIn("Paper", result["summary"])

    def test_openai_provider_fallback_string_path_duality(self):
        """OpenAIProvider fallback handles doc.path as str without AttributeError."""
        provider = OpenAIProvider(api_key="sk-test", max_retries=0)
        doc = ParsedDocument(
            path="manuscripts/paper.tex",  # type: ignore[arg-type]
            title=None,
            raw_text="Paper",
            clean_text="Paper text.",
            math_blocks=[r"\mathcal{O}(1)"],
        )
        result = provider.extract(doc, "ML_Optimization")
        self.assertIn("summary", result)
        self.assertIn("Paper", result["summary"])

    def test_mock_provider_display_math_preserved_in_theorem(self):
        """Theorem with $$ display math extracts formula without dropping."""
        provider = DeterministicMockProvider()
        doc = ParsedDocument(
            path=Path("thm.md"),
            title="Display Thm Doc",
            raw_text="",
            clean_text="",
            theorems=[r"Theorem 1: Under condition C, $$ \mathcal{O}(n \log n) \le K $$ holds."],
        )
        result = provider.extract(doc, "Graph_Combinatorics")
        formulas = [b["formula"] for b in result["bounds"]]
        self.assertTrue(
            any(r"\mathcal{O}(n \log n)" in f for f in formulas),
            f"Expected display math formula to be extracted, got: {formulas}",
        )


if __name__ == "__main__":
    unittest.main()
