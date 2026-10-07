"""
gestalt_extractor.tests.test_extractor
======================================
Unit tests for GestaltExtractor:
- extract_document, extract_file, and extract_text workflows.
- Verification of mandatory keys: file, domain, bounds, summary.
- Multi-format ingestion (.md and .tex).
- Resilient error handling (missing files, invalid inputs, empty docs).
"""

import tempfile
import unittest
from pathlib import Path

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.extractor import (
    GestaltExtractor,
    extract_document,
    extract_file,
    extract_text,
)
from gestalt_extractor.models import ExtractedRecord, ExtractedBound
from gestalt_extractor.parser import DocumentParser, ParsedDocument
from gestalt_extractor.classifier import DomainClassifier
from gestalt_extractor.llm import DeterministicMockProvider


class TestGestaltExtractor(unittest.TestCase):
    """Unit tests verifying GestaltExtractor behavior across manuscripts."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent.parent / "testbed" / "fixtures"
        self.config = ExtractorConfig(mock=True)
        self.extractor = GestaltExtractor(config=self.config)

    def test_extract_text_markdown(self):
        md_text = (
            "# Grokfast Optimization Bounds\n\n"
            "We analyze gradient descent dynamics under delayed generalization.\n\n"
            "$$\n\\| \\nabla \\mathcal{L}(\\theta) \\| \\le \\epsilon\n$$\n\n"
            "**Theorem 1**: Under exponential moving average momentum, learning converges.\n"
        )
        record = self.extractor.extract_text(md_text, filename="grokfast.md")

        self.assertIsInstance(record, ExtractedRecord)
        self.assertEqual(record.file, "grokfast.md")
        self.assertEqual(record.domain, "ML_Optimization")
        self.assertIsInstance(record.bounds, list)
        self.assertTrue(len(record.bounds) > 0)
        self.assertTrue(len(record.summary) > 0)

        # Invariant 4-key dictionary test
        d = record.to_dict()
        for k in ("file", "domain", "bounds", "summary"):
            self.assertIn(k, d)

    def test_extract_text_latex(self):
        tex_text = (
            r"\documentclass{article}" "\n"
            r"\title{Elliptic Curve Cryptography Bounds}" "\n"
            r"\begin{document}" "\n"
            r"We study discrete logarithm bounds over finite field $\mathbb{F}_p$." "\n"
            r"\begin{equation}" "\n"
            r"|E(\mathbb{F}_p)| \le p + 1 + 2\sqrt{p}" "\n"
            r"\end{equation}" "\n"
            r"\end{document}"
        )
        record = self.extractor.extract_text(tex_text, filename="crypto.tex")

        self.assertIsInstance(record, ExtractedRecord)
        self.assertEqual(record.file, "crypto.tex")
        self.assertEqual(record.domain, "Crypto_Number_Theory")
        self.assertTrue(len(record.bounds) > 0)
        self.assertTrue(len(record.summary) > 0)

    def test_extract_file_markdown_fixture(self):
        sample_md = self.fixtures_dir / "sample_bounds.md"
        self.assertTrue(sample_md.exists(), f"Missing fixture: {sample_md}")

        record = self.extractor.extract_file(sample_md)
        self.assertIsInstance(record, ExtractedRecord)
        self.assertEqual(record.file, "sample_bounds.md")
        self.assertEqual(record.domain, "ML_Optimization")
        self.assertTrue(len(record.bounds) >= 2)
        self.assertIn("Grokfast", record.summary)

    def test_extract_file_latex_fixture(self):
        sample_tex = self.fixtures_dir / "sample_bounds.tex"
        self.assertTrue(sample_tex.exists(), f"Missing fixture: {sample_tex}")

        record = self.extractor.extract_file(sample_tex)
        self.assertIsInstance(record, ExtractedRecord)
        self.assertEqual(record.file, "sample_bounds.tex")
        self.assertEqual(record.domain, "ML_Optimization")
        self.assertTrue(len(record.bounds) >= 2)
        self.assertTrue(len(record.summary) > 0)

    def test_extract_document_direct(self):
        doc = ParsedDocument(
            path=Path("manual_doc.md"),
            title="Graph Invariant Bounds",
            raw_text="Directed acyclic graph with topological sort.",
            clean_text="Directed acyclic graph with topological sort.",
            math_blocks=[r"\omega(G) \le \chi(G)"],
            theorems=["Theorem: Clique number is bounded by chromatic number."],
        )
        record = self.extractor.extract_document(doc)
        self.assertIsInstance(record, ExtractedRecord)
        self.assertEqual(record.file, "manual_doc.md")
        self.assertEqual(record.domain, "Graph_Combinatorics")
        self.assertTrue(len(record.bounds) > 0)

    def test_extract_document_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.extractor.extract_document(None)

    def test_extract_file_missing_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_file("non_existent_file_path_12345.md")

    def test_extract_file_directory_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                self.extractor.extract_file(tmpdir)

    def test_extract_file_empty_returns_graceful_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.md"
            empty_file.write_text("", encoding="utf-8")
            record = self.extractor.extract_file(empty_file)

            self.assertIsInstance(record, ExtractedRecord)
            self.assertEqual(record.file, "empty.md")
            self.assertEqual(record.bounds, [])
            self.assertTrue(len(record.summary) > 0)

    def test_convenience_functions(self):
        md_text = "# Test\nGradient descent optimizer with loss $\\mathcal{L}$."
        rec = extract_text(md_text, filename="helper_test.md", config=self.config)
        self.assertEqual(rec.file, "helper_test.md")
        self.assertIn("file", rec.to_dict())

        sample_md = self.fixtures_dir / "sample_bounds.md"
        rec_file = extract_file(sample_md, config=self.config)
        self.assertEqual(rec_file.file, "sample_bounds.md")


if __name__ == "__main__":
    unittest.main()
