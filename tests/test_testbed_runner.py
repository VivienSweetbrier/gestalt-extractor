"""
gestalt_extractor.tests.test_testbed_runner
===========================================
Unit tests for the local DreamLLM testbed runner:
- Target directory resolution.
- run_testbed() execution on fixture directories.
- Strict validation of mandatory schema keys: file, domain, bounds, summary.
- Error handling on non-existent or empty target directories.
"""

import json
import tempfile
import unittest
from pathlib import Path

from gestalt_extractor.testbed.runner import (
    resolve_default_target_dir,
    run_testbed,
    main as testbed_main,
)


class TestTestbedRunner(unittest.TestCase):
    """Unit tests for the DreamLLM testbed execution harness."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent.parent / "testbed" / "fixtures"

    def test_resolve_default_target_dir(self):
        resolved = resolve_default_target_dir()
        self.assertIsInstance(resolved, Path)
        self.assertTrue(resolved.exists())
        self.assertTrue(resolved.is_dir())

    def test_run_testbed_with_explicit_fixtures_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "testbed_output.json"
            result_path = run_testbed(
                target_dir=self.fixtures_dir,
                output_file=out_file,
                mock=True,
                max_workers=2,
            )

            self.assertIsInstance(result_path, Path)
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.resolve(), out_file.resolve())

            # Verify file content
            with open(result_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            self.assertIsInstance(records, list)
            self.assertTrue(len(records) >= 2)

            for rec in records:
                self.assertIn("file", rec)
                self.assertIn("domain", rec)
                self.assertIn("bounds", rec)
                self.assertIn("summary", rec)
                self.assertIsInstance(rec["bounds"], list)
                self.assertTrue(len(rec["summary"]) > 0)

    def test_run_testbed_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "default_testbed.json"
            result_path = run_testbed(output_file=out_file, mock=True)

            self.assertTrue(result_path.exists())
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertTrue(len(data) > 0)

    def test_run_testbed_missing_directory_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            run_testbed(target_dir="non_existent_dir_random_99999")

    def test_run_testbed_empty_directory_raises_assertion_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            out_file = Path(tmpdir) / "empty_out.json"

            with self.assertRaises(AssertionError):
                run_testbed(target_dir=empty_dir, output_file=out_file, mock=True)

    def test_testbed_main_entrypoint(self):
        exit_code = testbed_main()
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
