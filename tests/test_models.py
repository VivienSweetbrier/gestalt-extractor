"""
Unit tests for gestalt_extractor.models
=======================================
Verifies Milestone 2 Acceptance Criteria:
- Standardized data models: ExtractedBound, ExtractedRecord, ExtractionReport.
- Mandatory 4-key schema invariant: file, domain, bounds, summary.
- Resilient bounds normalization across types (strings, dicts, objects).
- Zero-crash Pydantic and stdlib Dataclass duality.
- ExtractionReport list operations and atomic serialization.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from gestalt_extractor.models import (
    PYDANTIC_AVAILABLE,
    ExtractedBound,
    ExtractedRecord,
    ExtractionReport,
    ExtractionResult,
    Bound,
    Report,
    DataclassExtractedBound,
    DataclassExtractedRecord,
    DataclassExtractionReport,
)

if PYDANTIC_AVAILABLE:
    from gestalt_extractor.models import (
        PydanticExtractedBound,
        PydanticExtractedRecord,
        PydanticExtractionReport,
    )


class TestExtractedBound(unittest.TestCase):
    """Verifies ExtractedBound field constraints and resilient parsing."""

    def test_default_instantiation(self):
        bound = ExtractedBound()
        self.assertEqual(bound.name, "Extracted Bound")
        self.assertEqual(bound.formula, "")
        self.assertEqual(bound.type, "algorithmic_bound")
        self.assertEqual(bound.context, "")

    def test_explicit_instantiation(self):
        bound = ExtractedBound(
            name="SVD Rank Bound",
            formula=r"\operatorname{rank}(A) \le r",
            type="spectral_bound",
            context="Under low-rank subspace projection",
        )
        self.assertEqual(bound.name, "SVD Rank Bound")
        self.assertEqual(bound.formula, r"\operatorname{rank}(A) \le r")
        self.assertEqual(bound.type, "spectral_bound")
        self.assertEqual(bound.context, "Under low-rank subspace projection")

    def test_to_dict_keys(self):
        bound = ExtractedBound(name="Test Bound", formula="O(n log n)")
        d = bound.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(set(d.keys()), {"name", "formula", "type", "context"})
        self.assertEqual(d["name"], "Test Bound")
        self.assertEqual(d["formula"], "O(n log n)")

    def test_from_dict_with_dict(self):
        data = {
            "name": "Asymptotic Bound",
            "formula": r"\mathcal{O}(N)",
            "type": "asymptotic_complexity",
            "context": "Linear time complexity",
            "extra_key": "ignored",
        }
        bound = ExtractedBound.from_dict(data)
        self.assertEqual(bound.name, "Asymptotic Bound")
        self.assertEqual(bound.formula, r"\mathcal{O}(N)")
        self.assertEqual(bound.type, "asymptotic_complexity")
        self.assertEqual(bound.context, "Linear time complexity")

    def test_from_dict_with_raw_string(self):
        # Edge case: raw formula string passed instead of dict
        bound = ExtractedBound.from_dict(r"\lambda \le 0.5")
        self.assertEqual(bound.name, "Mathematical Bound")
        self.assertEqual(bound.formula, r"\lambda \le 0.5")
        self.assertEqual(bound.type, "algorithmic_bound")

    def test_to_json_roundtrip(self):
        bound = ExtractedBound(name="EMA Filter", formula="g'_t = g_t + lambda * m_t")
        json_str = bound.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["name"], "EMA Filter")
        self.assertEqual(data["formula"], "g'_t = g_t + lambda * m_t")


class TestExtractedRecord(unittest.TestCase):
    """Verifies ExtractedRecord schema enforcement and normalization."""

    def test_mandatory_4_keys_in_to_dict(self):
        record = ExtractedRecord(
            file="memo.md",
            domain="ML_Optimization",
            summary="Test summary of paper.",
        )
        d = record.to_dict(include_optional=False)
        self.assertEqual(set(d.keys()), {"file", "domain", "bounds", "summary"})
        self.assertEqual(d["file"], "memo.md")
        self.assertEqual(d["domain"], "ML_Optimization")
        self.assertEqual(d["summary"], "Test summary of paper.")
        self.assertEqual(d["bounds"], [])

    def test_bounds_normalization_from_dicts(self):
        record = ExtractedRecord(
            file="paper.tex",
            domain="Graph_Combinatorics",
            bounds=[
                {"name": "Clique Bound", "formula": r"\omega(G) \le \chi(G)"},
                {"name": "Degree Bound", "formula": r"\deg(v) \le \Delta"},
            ],
            summary="Graph invariants.",
        )
        self.assertEqual(len(record.bounds), 2)
        self.assertIsInstance(record.bounds[0], ExtractedBound)
        self.assertEqual(record.bounds[0].name, "Clique Bound")
        self.assertEqual(record.bounds[1].name, "Degree Bound")

    def test_bounds_normalization_from_raw_strings(self):
        record = ExtractedRecord(
            file="formula.md",
            domain="General_Applied_Math",
            bounds=["O(n^2)", r"\int f(x) dx < \infty"],
            summary="Formulas only.",
        )
        self.assertEqual(len(record.bounds), 2)
        self.assertEqual(record.bounds[0].formula, "O(n^2)")
        self.assertEqual(record.bounds[1].formula, r"\int f(x) dx < \infty")

    def test_optional_fields_preservation(self):
        record = ExtractedRecord(
            file="test.md",
            domain="ML_Optimization",
            bounds=[],
            summary="A test.",
            path="/abs/path/test.md",
            methodologies=["EMA Low-Pass Filtering"],
            metadata={"source": "unit_test"},
        )
        d = record.to_dict(include_optional=True)
        self.assertIn("file", d)
        self.assertIn("domain", d)
        self.assertIn("bounds", d)
        self.assertIn("summary", d)
        self.assertEqual(d["path"], "/abs/path/test.md")
        self.assertEqual(d["methodologies"], ["EMA Low-Pass Filtering"])
        self.assertEqual(d["metadata"], {"source": "unit_test"})

    def test_from_dict_roundtrip(self):
        data = {
            "file": "roundtrip.tex",
            "domain": "Crypto_Number_Theory",
            "bounds": [
                {"name": "Prime Bound", "formula": r"\pi(x) \approx x/\ln(x)", "type": "asymptotic_complexity", "context": ""}
            ],
            "summary": "Prime number distribution.",
            "extra_unexpected_field": 12345,
        }
        record = ExtractedRecord.from_dict(data)
        self.assertEqual(record.file, "roundtrip.tex")
        self.assertEqual(record.domain, "Crypto_Number_Theory")
        self.assertEqual(len(record.bounds), 1)
        self.assertEqual(record.bounds[0].name, "Prime Bound")


class TestExtractionReport(unittest.TestCase):
    """Verifies ExtractionReport collection semantics and persistence."""

    def setUp(self):
        self.records = [
            ExtractedRecord(
                file=f"doc_{i}.md",
                domain="ML_Optimization",
                bounds=[{"name": f"Bound {i}", "formula": f"O(n^{i})"}],
                summary=f"Summary {i}",
            )
            for i in range(1, 4)
        ]
        self.report = ExtractionReport(records=self.records, metadata={"run_id": "test_run"})

    def test_collection_interface(self):
        self.assertEqual(len(self.report), 3)
        self.assertTrue(bool(self.report))
        self.assertEqual(self.report[0].file, "doc_1.md")
        self.assertEqual(self.report[2].file, "doc_3.md")

        files = [r.file for r in self.report]
        self.assertEqual(files, ["doc_1.md", "doc_2.md", "doc_3.md"])

    def test_add_record(self):
        new_rec = ExtractedRecord(file="doc_4.md", domain="Fluid_Dynamics", summary="Fluid memo.")
        self.report.add_record(new_rec)
        self.assertEqual(len(self.report), 4)
        self.assertEqual(self.report[3].file, "doc_4.md")

    def test_to_dict_as_json_array_default(self):
        # Default as_envelope=False returns List[Dict]
        d = self.report.to_dict(as_envelope=False)
        self.assertIsInstance(d, list)
        self.assertEqual(len(d), 3)
        for item in d:
            self.assertIn("file", item)
            self.assertIn("domain", item)
            self.assertIn("bounds", item)
            self.assertIn("summary", item)

    def test_to_dict_as_envelope(self):
        # as_envelope=True returns {"metadata": ..., "records": [...]}
        d = self.report.to_dict(as_envelope=True)
        self.assertIsInstance(d, dict)
        self.assertIn("metadata", d)
        self.assertIn("records", d)
        self.assertEqual(len(d["records"]), 3)

    def test_atomic_save_and_from_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "reports/output.json"
            result_path = self.report.save(save_path)
            self.assertTrue(result_path.exists())

            # Load and verify content
            loaded = ExtractionReport.from_file(result_path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0].file, "doc_1.md")
            self.assertEqual(loaded[0].bounds[0].name, "Bound 1")


class TestDualityConsistency(unittest.TestCase):
    """Verifies that Dataclass implementation behaves identically to Pydantic if available."""

    def test_dataclass_explicit_behavior(self):
        dc_bound = DataclassExtractedBound(name="DC Bound", formula="O(1)")
        dc_rec = DataclassExtractedRecord(
            file="dc.md",
            domain="General_Applied_Math",
            bounds=[dc_bound],
            summary="Dataclass summary",
        )
        d = dc_rec.to_dict()
        self.assertEqual(set(d.keys()), {"file", "domain", "bounds", "summary"})
        self.assertEqual(d["bounds"][0]["name"], "DC Bound")

    @unittest.skipUnless(PYDANTIC_AVAILABLE, "Pydantic not installed in this environment")
    def test_pydantic_explicit_behavior(self):
        py_bound = PydanticExtractedBound(name="PY Bound", formula="O(1)")
        py_rec = PydanticExtractedRecord(
            file="py.md",
            domain="General_Applied_Math",
            bounds=[py_bound],
            summary="Pydantic summary",
        )
        d = py_rec.to_dict()
        self.assertEqual(set(d.keys()), {"file", "domain", "bounds", "summary"})
        self.assertEqual(d["bounds"][0]["name"], "PY Bound")

    def test_aliases(self):
        self.assertIs(ExtractionResult, ExtractedRecord)
        self.assertIs(Bound, ExtractedBound)
        self.assertIs(Report, ExtractionReport)


if __name__ == "__main__":
    unittest.main()
