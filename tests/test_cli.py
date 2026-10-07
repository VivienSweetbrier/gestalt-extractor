"""
gestalt_extractor.tests.test_cli
================================
Unit tests for Gestalt Extractor CLI:
- Argument parsing and defaults.
- Successful main() execution with --mock, --target, --output flags.
- Output report file generation and JSON schema verification.
- Quiet and verbose flag operation.
- Non-zero exit code on non-existent target.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

from gestalt_extractor.cli import build_parser, main
from gestalt_extractor.config import DEFAULT_OUTPUT_FILE, DEFAULT_LLM_MODEL


class TestCLI(unittest.TestCase):
    """Unit tests for the Gestalt Extractor command-line interface."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).resolve().parent.parent / "testbed" / "fixtures"

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.target, ".")
        self.assertEqual(args.output, DEFAULT_OUTPUT_FILE)
        self.assertFalse(args.mock)
        self.assertEqual(args.model, DEFAULT_LLM_MODEL)
        self.assertFalse(args.quiet)
        self.assertFalse(args.verbose)

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--target", "custom_dir",
            "--output", "custom_out.json",
            "--mock",
            "--model", "gpt-4-turbo",
            "--max-workers", "8",
            "--quiet",
            "--verbose",
        ])
        self.assertEqual(args.target, "custom_dir")
        self.assertEqual(args.output, "custom_out.json")
        self.assertTrue(args.mock)
        self.assertEqual(args.model, "gpt-4-turbo")
        self.assertEqual(args.max_workers, 8)
        self.assertTrue(args.quiet)
        self.assertTrue(args.verbose)

    def test_main_success_with_fixtures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "cli_output.json"
            argv = [
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
                "--max-workers", "2",
            ]

            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            # Validate saved JSON payload
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertTrue(len(data) >= 2)
            for item in data:
                self.assertIn("file", item)
                self.assertIn("domain", item)
                self.assertIn("bounds", item)
                self.assertIn("summary", item)

    def test_main_quiet_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "quiet_output.json"
            argv = [
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ]

            # Suppress stdout and verify quiet execution
            with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
                exit_code = main(argv)
                stdout_content = fake_stdout.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout_content.strip(), "")
            self.assertTrue(out_file.exists())

    def test_main_nonexistent_target_returns_error(self):
        argv = [
            "--target", "non_existent_directory_xyz_98765",
            "--mock",
            "--quiet",
        ]
        exit_code = main(argv)
        self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
