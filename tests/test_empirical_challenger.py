"""
Adversarial Empirical Stress-Test Suite for Gestalt Extractor (Milestone 1).

Engineered by Empirical Challenger 1 to hostilely stress-test:
- gestalt_extractor.scanner (DirectoryScanner, scan_directory)
- gestalt_extractor.config (ScannerConfig, ExtractorConfig)
- gestalt_extractor.parser (DocumentParser, ParsedDocument)

Validates extreme edge cases:
- Exact folder boundary matching (node_modules_backup vs node_modules)
- Case-folding variations on Windows (NoDe_MoDuLeS, .AgEnTs)
- Subdirectory and relative path exclusion mechanics
- Deeply nested directory traversal (25+ levels)
- Scale stress testing (1,000+ files)
- ScannerConfig property enforcement (max_file_size_bytes)
- Mathematical parser grammar isolation (inter-equation prose bleed)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Set

# Ensure gestalt_extractor is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gestalt_extractor.config import (
    ScannerConfig,
    ExtractorConfig,
    DEFAULT_EXCLUDES,
    DEFAULT_EXCLUDE_DIR_NAMES,
    SUPPORTED_EXTENSIONS,
)
from gestalt_extractor.scanner import (
    DirectoryScanner,
    Scanner,
    scan_directory,
    TargetNotFoundError,
    InvalidTargetError,
)
from gestalt_extractor.parser import (
    DocumentParser,
    Parser,
    ParsedDocument,
)


class TestEmpiricalScannerBoundaries(unittest.TestCase):
    """Stress tests for exact directory boundary matching and case-folding."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scanner = Scanner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_folder_boundary_matching(self):
        """
        Verify exact directory boundary matching:
        - 'node_modules' MUST be excluded
        - 'node_modules_backup', 'node_modules_old', 'my_node_modules' MUST NOT be excluded
        - 'build' MUST be excluded
        - 'build_v2', 'rebuild' MUST NOT be excluded
        - 'env' MUST be excluded
        - 'env_prod', 'environment' MUST NOT be excluded
        """
        # Excluded directories
        excluded_files = [
            self.root / "node_modules" / "pkg" / "doc.md",
            self.root / "build" / "out.tex",
            self.root / "env" / "readme.md",
            self.root / ".agents" / "teamwork" / "memo.md",
            self.root / ".git" / "config.md",
            self.root / "__pycache__" / "cached.md",
        ]

        # Valid boundary-adjacent directories that MUST be included
        included_files = [
            self.root / "node_modules_backup" / "paper.md",
            self.root / "node_modules_old" / "pkg" / "paper.md",
            self.root / "my_node_modules" / "paper.md",
            self.root / "build_v2" / "notes.tex",
            self.root / "rebuild" / "notes.tex",
            self.root / "env_prod" / "setup.md",
            self.root / "environment" / "setup.md",
            self.root / "agents_research" / "memo.md",
        ]

        for p in excluded_files + included_files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"Content of {p.name}", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        # Assert all boundary-adjacent files are discovered
        for f in included_files:
            self.assertIn(
                f.resolve(),
                result_paths,
                f"Boundary failure: {f} should NOT be excluded!",
            )

        # Assert all strictly excluded files are omitted
        for f in excluded_files:
            self.assertNotIn(
                f.resolve(),
                result_paths,
                f"Exclusion failure: {f} MUST be excluded!",
            )

        self.assertEqual(len(results), len(included_files))

    def test_windows_case_folding_exclusions(self):
        """
        Verify case-insensitive exclusions on Windows/NTFS:
        - 'NoDe_MoDuLeS', 'NODE_MODULES', 'NoDe_MoDuLeS'
        - '.AgEnTs', '.GIT', 'BUILD', 'Dist', '__PyCaChE__'
        """
        case_variations = [
            self.root / "NODE_MODULES" / "test.md",
            self.root / "NoDe_MoDuLeS" / "test.md",
            self.root / ".AgEnTs" / "test.md",
            self.root / ".GIT" / "test.md",
            self.root / "BUILD" / "test.tex",
            self.root / "Dist" / "test.md",
            self.root / "__PyCaChE__" / "test.md",
        ]

        valid_file = self.root / "VALID_PAPER.MD"
        valid_file.write_text("Valid content", encoding="utf-8")

        for p in case_variations:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Should be excluded", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertEqual(len(results), 1)
        self.assertIn(valid_file.resolve(), result_paths)
        for noise in case_variations:
            self.assertNotIn(noise.resolve(), result_paths)


class TestEmpiricalSubdirectoryExclusions(unittest.TestCase):
    """Hostile challenge on relative and subdirectory exclusion patterns."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_relative_path_exclusion_behavior(self):
        """
        CHALLENGE TEST:
        Passing relative paths like 'docs/private' or 'research/drafts'
        in custom_excludes should exclude all files inside those directories.
        """
        f_keep = self.root / "docs" / "public" / "paper.md"
        f_drop1 = self.root / "docs" / "private" / "secret.md"
        f_drop2 = self.root / "research" / "drafts" / "wip.tex"

        for f in [f_keep, f_drop1, f_drop2]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        # Caller specifies subdirectory relative paths
        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["docs/private", "research/drafts"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        # If scanner fails on relative directory paths without wildcards,
        # f_drop1 and f_drop2 will erroneously appear in results.
        self.assertIn(f_keep.resolve(), result_paths)
        self.assertNotIn(
            f_drop1.resolve(),
            result_paths,
            "BUG: Relative path 'docs/private' failed to exclude 'docs/private/secret.md'!",
        )
        self.assertNotIn(
            f_drop2.resolve(),
            result_paths,
            "BUG: Relative path 'research/drafts' failed to exclude 'research/drafts/wip.tex'!",
        )


class TestEmpiricalConfigEnforcement(unittest.TestCase):
    """Stress tests verifying config attributes are actually enforced."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scanner_max_file_size_enforcement(self):
        """
        CHALLENGE TEST:
        ScannerConfig defines max_file_size_bytes.
        If a user configures max_file_size_bytes=1024, files larger than 1024 bytes
        should be excluded by the scanner.
        """
        small_file = self.root / "small.md"
        large_file = self.root / "large.md"

        small_file.write_text("Small content", encoding="utf-8")  # ~13 bytes
        large_file.write_text("A" * 10_000, encoding="utf-8")      # 10,000 bytes

        cfg = ScannerConfig(max_file_size_bytes=500)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(small_file.resolve(), result_paths)
        self.assertNotIn(
            large_file.resolve(),
            result_paths,
            "BUG: Scanner ignored max_file_size_bytes configuration in ScannerConfig!",
        )


class TestEmpiricalParserVulnerabilities(unittest.TestCase):
    """Adversarial stress-tests against mathematical extraction grammar."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_markdown_display_math_bleed_bug(self):
        """
        CHALLENGE TEST:
        When a Markdown manuscript has two display math blocks:
        $$
        E = mc^2
        $$
        This prose is between equations and is NOT a mathematical formula.
        $$
        F = ma
        $$
        
        The parser must NOT extract the prose between $$ blocks as an inline math block!
        """
        doc_text = (
            "# Physics Notes\n\n"
            "$$\nE = mc^2\n$$\n\n"
            "This prose is between equations and is NOT a mathematical formula.\n\n"
            "$$\nF = ma\n$$\n"
        )
        parsed = self.parser.parse_text(doc_text, Path("physics.md"))

        # Check math_blocks
        leaked_prose = [
            m for m in parsed.math_blocks
            if "This prose is between equations" in m
        ]
        self.assertEqual(
            leaked_prose,
            [],
            f"CRITICAL PARSER BUG: Inter-equation prose was captured as math: {leaked_prose}",
        )

    def test_latex_display_math_duplicate_extraction(self):
        """
        CHALLENGE TEST:
        When a LaTeX manuscript has $$ x = 1 $$, it should not be extracted twice
        (once as display math and once as inline math).
        """
        doc_text = (
            "\\begin{document}\n"
            "$$\nx = 1\n$$\n"
            "\\end{document}\n"
        )
        parsed = self.parser.parse_text(doc_text, Path("eq.tex"))
        occurrences = [m for m in parsed.math_blocks if "x = 1" in m]
        self.assertLessEqual(
            len(occurrences),
            1,
            f"PARSER BUG: Display equation was extracted {len(occurrences)} times (duplicate extraction)!",
        )


class TestEmpiricalScaleAndNesting(unittest.TestCase):
    """Stress tests for deeply nested directories, empty trees, and scale."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scanner = Scanner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deeply_nested_directory_traversal(self):
        """Verify scanner navigates a 25-level nested path without crashing."""
        nested_dir = self.root
        for i in range(25):
            nested_dir = nested_dir / f"level_{i:02d}"
        nested_dir.mkdir(parents=True, exist_ok=True)

        target_file = nested_dir / "deep_manuscript.md"
        target_file.write_text("# Deep Research", encoding="utf-8")

        results = self.scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), target_file.resolve())

    def test_scale_thousand_files_discovery(self):
        """Verify performance and deterministic sorting on 1,000 files."""
        # Create 10 folders, each with 50 markdown files and 50 noise files
        expected_files = []
        for d in range(10):
            sub = self.root / f"dir_{d:02d}"
            sub.mkdir(exist_ok=True)
            for f in range(50):
                md_file = sub / f"doc_{f:03d}.md"
                md_file.write_text(f"Content {f}", encoding="utf-8")
                expected_files.append(md_file.resolve())
                # Add noise
                (sub / f"noise_{f:03d}.txt").write_text("noise", encoding="utf-8")
                (sub / f"script_{f:03d}.py").write_text("print(1)", encoding="utf-8")

        results = self.scanner.scan(self.root)
        self.assertEqual(len(results), 500)  # 500 .md files out of 1,500 total files
        # Check sorting
        result_str_list = [str(p).lower() for p in results]
        self.assertEqual(result_str_list, sorted(result_str_list))


if __name__ == "__main__":
    unittest.main()
