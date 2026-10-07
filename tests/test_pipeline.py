"""
gestalt_extractor.tests.test_pipeline
=====================================
Unit tests for ExtractionPipeline:
- Multi-pattern exclusion filtering (.agents, node_modules, build, dist, .git).
- Concurrency execution via ThreadPoolExecutor.
- Thread-safe error containment (resilience against per-file failures).
- Aggregated ExtractionReport and atomic JSON serialization.
- Top-level run_extraction convenience interface.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.pipeline import ExtractionPipeline, run_extraction
from gestalt_extractor.models import ExtractionReport, ExtractedRecord


class TestExtractionPipeline(unittest.TestCase):
    """Unit tests for ExtractionPipeline batch processing and concurrency."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent.parent / "testbed" / "fixtures"
        self.config = ExtractorConfig(mock=True, max_workers=2)
        self.pipeline = ExtractionPipeline(config=self.config)

    def test_run_on_fixtures_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "output.json"
            report = self.pipeline.run(target_dir=self.fixtures_dir, output_file=output_json)

            self.assertIsInstance(report, ExtractionReport)
            self.assertTrue(len(report) >= 2)
            self.assertTrue(output_json.exists())

            # Verify JSON payload on disk
            with open(output_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertEqual(len(data), len(report))
            for item in data:
                self.assertIn("file", item)
                self.assertIn("domain", item)
                self.assertIn("bounds", item)
                self.assertIn("summary", item)

    def test_concurrency_workers_execution(self):
        # Create a set of temporary test manuscripts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = []
            for i in range(5):
                f = tmp_path / f"paper_{i}.md"
                f.write_text(
                    f"# Paper {i}\nOptimization with learning rate $\\eta$ and bounds $O(1/{i+1})$.",
                    encoding="utf-8",
                )
                files.append(f)

            # Test sequential (max_workers=1)
            pipeline_seq = ExtractionPipeline(config=ExtractorConfig(mock=True, max_workers=1))
            report_seq = pipeline_seq.run_batch(files)
            self.assertEqual(len(report_seq), 5)

            # Test concurrent (max_workers=4)
            pipeline_conc = ExtractionPipeline(config=ExtractorConfig(mock=True, max_workers=4))
            report_conc = pipeline_conc.run_batch(files)
            self.assertEqual(len(report_conc), 5)

            # Ensure both produce deterministic sorted files
            files_seq = [r.file for r in report_seq]
            files_conc = [r.file for r in report_conc]
            self.assertEqual(files_seq, files_conc)

    def test_error_isolation_per_file(self):
        """Verify that an exception on one file does not abort the entire batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            good_file = tmp_path / "good.md"
            good_file.write_text("# Good Paper\nFormula $\\lambda < 1$.", encoding="utf-8")

            bad_file = tmp_path / "bad.md"
            bad_file.write_text("# Bad Paper\nCorrupt manuscript.", encoding="utf-8")

            pipeline = ExtractionPipeline(config=ExtractorConfig(mock=True))

            # Mock extract_file to raise on bad_file
            orig_extract_file = pipeline.extractor.extract_file

            def side_effect(path):
                p = Path(path)
                if p.name == "bad.md":
                    raise RuntimeError("Simulated extraction crash on bad file.")
                return orig_extract_file(p)

            with patch.object(pipeline.extractor, "extract_file", side_effect=side_effect):
                report = pipeline.run_batch([good_file, bad_file])

            # Batch must succeed for good_file
            self.assertEqual(len(report), 1)
            self.assertEqual(report[0].file, "good.md")

            # Errors must be safely logged in metadata
            meta = report.metadata or {}
            self.assertEqual(meta.get("failed"), 1)
            self.assertEqual(len(meta.get("errors", [])), 1)
            self.assertIn("bad.md", meta["errors"][0]["file"])

    def test_multi_pattern_exclusion_filters(self):
        """Verify that DirectoryScanner inside pipeline filters noise directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Valid manuscript
            (root / "valid.md").write_text("# Valid\nContent with math $x = 1$.", encoding="utf-8")

            # Excluded directories
            for excl_dir in ("node_modules", ".agents", ".git", "build", "dist"):
                d = root / excl_dir
                d.mkdir(parents=True, exist_ok=True)
                (d / f"ignored_{excl_dir}.md").write_text("# Noise\nShould not be extracted.", encoding="utf-8")

            report = self.pipeline.run(target_dir=root)

            # Only valid.md should be extracted
            self.assertEqual(len(report), 1)
            self.assertEqual(report[0].file, "valid.md")

    def test_run_on_empty_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty_dir"
            empty_dir.mkdir()
            output_file = empty_dir / "empty_report.json"

            report = self.pipeline.run(target_dir=empty_dir, output_file=output_file)
            self.assertEqual(len(report), 0)
            self.assertTrue(output_file.exists())

    def test_run_extraction_functional_helper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_file = Path(tmpdir) / "doc.md"
            doc_file.write_text("# Title\nFormula $y = f(x)$.", encoding="utf-8")
            out_file = Path(tmpdir) / "report.json"

            report = run_extraction(target_dir=tmpdir, output_file=out_file, mock=True, max_workers=1)
            self.assertIsInstance(report, ExtractionReport)
            self.assertEqual(len(report), 1)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
