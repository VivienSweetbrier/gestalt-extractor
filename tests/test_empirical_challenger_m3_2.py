"""
gestalt_extractor.tests.test_empirical_challenger_m3_2
======================================================
Comprehensive Empirical Challenger verification suite for Milestone 3:
1. CLI execution (`python -m gestalt_extractor.cli`):
   - Ingestion of fixtures and individual manuscripts.
   - Command line flag verification (--target, --output, --mock, --max-workers, --quiet, --verbose).
   - Exit code verification (0 on success, 1 on non-existent targets).
2. Output file serialization & schema invariants:
   - Valid JSON syntax and structure.
   - Mandatory keys: `file`, `domain`, `bounds`, `summary`.
   - Structural typing and non-empty assertions.
3. Testbed runner execution (`python -m gestalt_extractor.testbed.runner`):
   - Local directory resolution.
   - Execution against bundled fixtures and real DreamLLM `research/` manuscripts.
   - Schema enforcement and empty directory assertions.
4. Acceptance criteria satisfaction:
   - Requirement R1: LLM/Mock-augmented extraction on .tex and .md.
   - Requirement R2: Execution strictly against local workspace without external network/cloning.
   - Requirement R3: Structured JSON reporting containing required schema fields.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

from gestalt_extractor.cli import build_parser, main as cli_main
from gestalt_extractor.testbed.runner import (
    resolve_default_target_dir,
    run_testbed,
    main as testbed_main,
)
from gestalt_extractor.config import DEFAULT_OUTPUT_FILE


class TestCLIExecutionAndSchema(unittest.TestCase):
    """Empirical verification of the Gestalt Extractor CLI interface."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.fixtures_dir = self.repo_root / "gestalt_extractor" / "testbed" / "fixtures"
        self.assertTrue(self.fixtures_dir.exists(), f"Fixtures directory missing: {self.fixtures_dir}")

    def test_cli_fixtures_clean_execution_exit_0(self):
        """CLI runs cleanly against testbed fixtures and exits with code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_report.json"
            argv = [
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0, "CLI did not return exit code 0 on fixtures target")
            self.assertTrue(out_file.exists(), "Output file test_report.json was not created")
            self.assertGreater(out_file.stat().st_size, 0, "Output file test_report.json is 0 bytes")

            # Verify JSON syntax and schema
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIsInstance(data, list, "Report output is not a JSON list")
            self.assertGreaterEqual(len(data), 2, "Expected at least 2 manuscript records (TeX and MD)")

            required_keys = {"file", "domain", "bounds", "summary"}
            for idx, rec in enumerate(data):
                self.assertIsInstance(rec, dict, f"Record #{idx} is not a dictionary")
                missing = required_keys - set(rec.keys())
                self.assertEqual(missing, set(), f"Record #{idx} missing mandatory keys: {missing}")

                self.assertIsInstance(rec["file"], str)
                self.assertTrue(len(rec["file"]) > 0)

                self.assertIsInstance(rec["domain"], str)
                self.assertTrue(len(rec["domain"]) > 0)

                self.assertIsInstance(rec["bounds"], list)
                self.assertIsInstance(rec["summary"], str)
                self.assertTrue(len(rec["summary"]) > 0)

    def test_cli_single_markdown_file(self):
        """CLI successfully processes a single .md file target directly."""
        md_file = self.fixtures_dir / "sample_bounds.md"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "single_md_report.json"
            argv = [
                "--target", str(md_file),
                "--output", str(out_file),
                "--mock",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data), 1)
            rec = data[0]
            self.assertEqual(rec["file"], "sample_bounds.md")
            self.assertEqual(rec["domain"], "ML_Optimization")
            self.assertGreater(len(rec["bounds"]), 0, "Expected extracted bounds from sample_bounds.md")

    def test_cli_single_latex_file(self):
        """CLI successfully processes a single .tex file target directly."""
        tex_file = self.fixtures_dir / "sample_bounds.tex"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "single_tex_report.json"
            argv = [
                "--target", str(tex_file),
                "--output", str(out_file),
                "--mock",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data), 1)
            rec = data[0]
            self.assertEqual(rec["file"], "sample_bounds.tex")
            self.assertEqual(rec["domain"], "ML_Optimization")
            self.assertGreater(len(rec["bounds"]), 0, "Expected extracted bounds from sample_bounds.tex")

    def test_cli_nested_output_directory_creation(self):
        """CLI automatically creates missing nested destination directories for --output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_out = Path(tmpdir) / "nested" / "sub" / "deep" / "report.json"
            self.assertFalse(nested_out.parent.exists())

            argv = [
                "--target", str(self.fixtures_dir),
                "--output", str(nested_out),
                "--mock",
                "--quiet",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(nested_out.exists(), "Nested output file was not created")

            with open(nested_out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list)

    def test_cli_nonexistent_target_exits_1(self):
        """CLI returns exit code 1 when target path does not exist."""
        argv = [
            "--target", "non_existent_path_xy_99999",
            "--mock",
            "--quiet",
        ]
        exit_code = cli_main(argv)
        self.assertEqual(exit_code, 1, "CLI should return exit code 1 for non-existent target")

    def test_cli_empty_directory_handling(self):
        """CLI on empty directory exits cleanly with code 0 and serializes empty record array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty_dir"
            empty_dir.mkdir()
            out_file = Path(tmpdir) / "empty_report.json"

            argv = [
                "--target", str(empty_dir),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data, [], "Empty target directory should serialize empty record array")

    def test_cli_quiet_mode_suppresses_stdout(self):
        """CLI with --quiet suppresses all standard output banners and summaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "quiet.json"
            argv = [
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ]

            with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
                exit_code = cli_main(argv)
                captured = fake_stdout.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured.strip(), "", "Quiet mode should produce zero stdout output")

    def test_cli_concurrency_workers_scaling(self):
        """CLI produces deterministic identical results under single-thread and multi-thread workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_seq = Path(tmpdir) / "seq.json"
            out_par = Path(tmpdir) / "par.json"

            exit_seq = cli_main([
                "--target", str(self.fixtures_dir),
                "--output", str(out_seq),
                "--mock",
                "--max-workers", "1",
                "--quiet",
            ])
            exit_par = cli_main([
                "--target", str(self.fixtures_dir),
                "--output", str(out_par),
                "--mock",
                "--max-workers", "4",
                "--quiet",
            ])

            self.assertEqual(exit_seq, 0)
            self.assertEqual(exit_par, 0)

            with open(out_seq, "r", encoding="utf-8") as f1, open(out_par, "r", encoding="utf-8") as f2:
                data_seq = json.load(f1)
                data_par = json.load(f2)

            self.assertEqual(len(data_seq), len(data_par))
            files_seq = [r["file"] for r in data_seq]
            files_par = [r["file"] for r in data_par]
            self.assertEqual(files_seq, files_par, "Multi-threading should preserve deterministic sorted filenames")


class TestTestbedRunnerExecution(unittest.TestCase):
    """Empirical verification of the local DreamLLM testbed runner."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.fixtures_dir = self.repo_root / "gestalt_extractor" / "testbed" / "fixtures"

    def test_runner_resolves_default_target_directory(self):
        """resolve_default_target_dir() returns a valid existing directory containing manuscripts."""
        target = resolve_default_target_dir()
        self.assertIsInstance(target, Path)
        self.assertTrue(target.exists())
        self.assertTrue(target.is_dir())

        has_manuscripts = any(
            p.suffix.lower() in {".tex", ".md"}
            for p in target.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        )
        self.assertTrue(has_manuscripts, "Resolved target directory has no .tex or .md manuscripts")

    def test_runner_with_explicit_fixtures_directory(self):
        """run_testbed() executes cleanly on fixtures directory and returns path to verified report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "testbed_out.json"
            res = run_testbed(
                target_dir=self.fixtures_dir,
                output_file=out_path,
                mock=True,
                max_workers=2,
            )

            self.assertEqual(res.resolve(), out_path.resolve())
            self.assertTrue(out_path.exists())

            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertGreaterEqual(len(data), 2)
            required_keys = {"file", "domain", "bounds", "summary"}
            for rec in data:
                self.assertEqual(required_keys - set(rec.keys()), set())

    def test_runner_main_entrypoint_exits_0(self):
        """testbed.runner.main() completes and returns exit code 0."""
        # Run testbed main entrypoint (writes to default gestalt_extraction_report.json)
        exit_code = testbed_main()
        self.assertEqual(exit_code, 0, "Testbed main() entrypoint did not return exit code 0")

        default_report = Path("gestalt_extraction_report.json")
        self.assertTrue(default_report.exists(), "Default report gestalt_extraction_report.json missing")
        self.assertGreater(default_report.stat().st_size, 0)

        with open(default_report, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_runner_nonexistent_directory_raises_filenotfound(self):
        """run_testbed() raises FileNotFoundError on missing target directory."""
        with self.assertRaises(FileNotFoundError):
            run_testbed(target_dir="totally_non_existent_dir_001122")

    def test_runner_empty_directory_raises_assertion_error(self):
        """run_testbed() raises AssertionError if target contains 0 manuscripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty = Path(tmpdir) / "empty"
            empty.mkdir()
            out = Path(tmpdir) / "empty.json"
            with self.assertRaises(AssertionError):
                run_testbed(target_dir=empty, output_file=out, mock=True)


class TestRealDreamLLMWorkspaceManuscripts(unittest.TestCase):
    """Empirical verification targeting real DreamLLM research manuscripts."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.research_dir = self.repo_root / "research"
        self.grokking_notes_dir = (
            self.research_dir
            / "2026-09-18"
            / "grokking-swarm-autoresearch"
            / "grokking-notes"
        )

    def test_testbed_runner_on_real_research_notes_directory(self):
        """Testbed runner successfully ingests and extracts from real research notes without crashing."""
        if not self.grokking_notes_dir.exists():
            self.skipTest(f"Research directory not found: {self.grokking_notes_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "research_report.json"
            result_path = run_testbed(
                target_dir=self.grokking_notes_dir,
                output_file=out_file,
                mock=True,
                max_workers=2,
            )

            self.assertTrue(result_path.exists())
            with open(result_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            self.assertIsInstance(records, list)
            self.assertGreater(len(records), 0, "No records extracted from research notes directory")

            # Check that memo.md was processed
            memo_records = [r for r in records if "memo.md" in r["file"].lower()]
            self.assertTrue(len(memo_records) > 0, "Expected memo.md to be extracted from research notes")

            memo_rec = memo_records[0]
            self.assertIn("file", memo_rec)
            self.assertIn("domain", memo_rec)
            self.assertIn("bounds", memo_rec)
            self.assertIn("summary", memo_rec)

    def test_cli_on_real_research_memo_file(self):
        """CLI directly processes research/memo.md and extracts mathematical bounds."""
        memo_path = self.grokking_notes_dir / "memo.md"
        if not memo_path.exists():
            self.skipTest(f"memo.md not found at {memo_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "memo_report.json"
            argv = [
                "--target", str(memo_path),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ]

            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data), 1)
            rec = data[0]
            self.assertEqual(rec["file"], "memo.md")
            self.assertIsInstance(rec["domain"], str)
            self.assertIsInstance(rec["bounds"], list)
            self.assertIsInstance(rec["summary"], str)


class TestAcceptanceCriteriaR1R2R3(unittest.TestCase):
    """Direct verification of requirements R1, R2, R3 from ORIGINAL_REQUEST.md."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.fixtures_dir = self.repo_root / "gestalt_extractor" / "testbed" / "fixtures"

    def test_r1_llm_augmented_extraction_logic(self):
        """R1: Scans .tex and .md files, parses contents, and extracts computational bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "r1_report.json"
            exit_code = cli_main([
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ])
            self.assertEqual(exit_code, 0)

            with open(out_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            extensions = {Path(r["file"]).suffix.lower() for r in records}
            self.assertIn(".tex", extensions, "R1 violation: .tex files not processed")
            self.assertIn(".md", extensions, "R1 violation: .md files not processed")

            total_bounds = sum(len(r["bounds"]) for r in records)
            self.assertGreater(total_bounds, 0, "R1 violation: no computational bounds extracted")

    def test_r2_workspace_testbed_integration(self):
        """R2: Configured to run against local DreamLLM workspace without external cloning."""
        # Verify run_testbed defaults to local workspace without network or cloning
        resolved = resolve_default_target_dir()
        self.assertTrue(str(resolved).startswith(str(self.repo_root.resolve())))

        report_path = run_testbed(mock=True)
        self.assertTrue(report_path.exists())

    def test_r3_structured_output_reporting(self):
        """R3: Output serialized to valid JSON containing file, domain, bounds, summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "r3_report.json"
            exit_code = cli_main([
                "--target", str(self.fixtures_dir),
                "--output", str(out_file),
                "--mock",
                "--quiet",
            ])
            self.assertEqual(exit_code, 0)

            with open(out_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            self.assertIsInstance(records, list)
            for rec in records:
                self.assertSetEqual(
                    {"file", "domain", "bounds", "summary"},
                    set(rec.keys()) & {"file", "domain", "bounds", "summary"},
                    "R3 violation: mandatory keys missing from output",
                )


if __name__ == "__main__":
    unittest.main()
