"""
gestalt_extractor.tests.test_e2e
================================
Milestone 4 End-to-End (E2E) Verification Test Suite.

Verifies:
1. Full end-to-end execution of testbed runner against fixtures.
2. CLI execution against real DreamLLM manuscripts (research/2026-09-18/.../memo.md).
3. CLI execution against testbed fixtures (sample_bounds.tex, sample_bounds.md).
4. Strict schema compliance for all generated reports: {"file", "domain", "bounds", "summary"}.
5. Authentic mathematical bounds extraction, domain classification (ML_Optimization), and summaries.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.pipeline import ExtractionPipeline
from gestalt_extractor.extractor import GestaltExtractor
from gestalt_extractor.models import ExtractedRecord, ExtractionReport
from gestalt_extractor.cli import main as cli_main
from gestalt_extractor.testbed.runner import run_testbed, resolve_default_target_dir


class TestE2EWorkflow(unittest.TestCase):
    """End-to-end integration test suite verifying full pipeline workflows."""

    def setUp(self):
        self.fixtures_dir = REPO_ROOT / "gestalt_extractor" / "testbed" / "fixtures"
        self.research_dir = (
            REPO_ROOT
            / "research"
            / "2026-09-18"
            / "grokking-swarm-autoresearch"
            / "grokking-notes"
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_e2e_fixtures_extraction_via_pipeline(self):
        """Verify pipeline processing both .tex and .md fixtures with authentic extraction."""
        self.assertTrue(self.fixtures_dir.exists(), f"Fixtures dir missing: {self.fixtures_dir}")
        tex_fixture = self.fixtures_dir / "sample_bounds.tex"
        md_fixture = self.fixtures_dir / "sample_bounds.md"
        self.assertTrue(tex_fixture.exists(), "sample_bounds.tex missing")
        self.assertTrue(md_fixture.exists(), "sample_bounds.md missing")

        out_json = self.out_dir / "pipeline_fixtures_report.json"
        config = ExtractorConfig(output_file=str(out_json), mock=True, max_workers=2)
        pipeline = ExtractionPipeline(config=config, max_workers=2)
        report = pipeline.run(target_dir=self.fixtures_dir, output_file=out_json)

        self.assertEqual(len(report), 2, "Expected exactly 2 fixture files processed")
        self.assertTrue(out_json.exists(), "Output JSON report was not created")

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

        required_keys = {"file", "domain", "bounds", "summary"}
        for record in data:
            self.assertTrue(required_keys.issubset(record.keys()))
            self.assertEqual(record["domain"], "ML_Optimization")
            self.assertIsInstance(record["bounds"], list)
            self.assertGreater(len(record["bounds"]), 0, "Expected non-zero extracted bounds")
            for b in record["bounds"]:
                self.assertIn("name", b)
                self.assertIn("formula", b)
                self.assertIn("type", b)
                self.assertIn("context", b)
            self.assertIsInstance(record["summary"], str)
            self.assertGreater(len(record["summary"]), 50)

    def test_e2e_real_research_notes_extraction_via_cli(self):
        """Verify CLI execution against real DreamLLM research notes memo.md."""
        self.assertTrue(self.research_dir.exists(), f"Research dir missing: {self.research_dir}")
        memo_path = self.research_dir / "memo.md"
        self.assertTrue(memo_path.exists(), "memo.md missing in research notes")

        out_json = self.out_dir / "research_report.json"
        exit_code = cli_main([
            "--target", str(self.research_dir),
            "--output", str(out_json),
            "--mock",
            "--quiet",
        ])

        self.assertEqual(exit_code, 0, "CLI execution failed with non-zero exit code")
        self.assertTrue(out_json.exists(), "Research report JSON was not created")

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1, "Expected exactly memo.md discovered and processed")

        record = data[0]
        self.assertEqual(record["file"], "memo.md")
        self.assertEqual(record["domain"], "ML_Optimization")
        self.assertIsInstance(record["bounds"], list)
        self.assertGreater(len(record["bounds"]), 0, "Expected authentic bounds in memo.md")
        self.assertIsInstance(record["summary"], str)
        self.assertIn("ML_Optimization", record["summary"])

    def test_e2e_fixtures_extraction_via_cli(self):
        """Verify CLI execution against fixtures directory."""
        out_json = self.out_dir / "fixture_report.json"
        exit_code = cli_main([
            "--target", str(self.fixtures_dir),
            "--output", str(out_json),
            "--mock",
            "--quiet",
        ])

        self.assertEqual(exit_code, 0, "CLI execution failed on fixtures")
        self.assertTrue(out_json.exists(), "Fixture report JSON was not created")

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        filenames = {r["file"] for r in data}
        self.assertEqual(filenames, {"sample_bounds.md", "sample_bounds.tex"})

    def test_e2e_testbed_runner_execution(self):
        """Verify testbed runner execution against local workspace."""
        out_json = self.out_dir / "testbed_runner_report.json"
        result_path = run_testbed(
            target_dir=self.fixtures_dir,
            output_file=out_json,
            mock=True,
            max_workers=2,
        )

        self.assertEqual(result_path, out_json)
        self.assertTrue(out_json.exists())

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        for r in data:
            self.assertEqual(set(r.keys()), {"file", "domain", "bounds", "summary", "path", "methodologies", "metadata"})


if __name__ == "__main__":
    unittest.main()
