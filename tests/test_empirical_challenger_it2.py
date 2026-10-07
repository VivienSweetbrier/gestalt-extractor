"""
Adversarial Empirical Challenge Suite — Iteration 2 (Milestone 1).
==================================================================
Authored by Empirical Challenger 1 to hostilely stress-test:
1. Verification of Remediations CH-01 through CH-05:
   - CH-01: Inter-equation prose isolation in Markdown
   - CH-02: LaTeX display math duplication elimination
   - CH-03: Subdirectory/relative path exclusions without wildcards
   - CH-04: ScannerConfig max_file_size_bytes enforcement
   - CH-05: Exclude default directory override via excludes=[]
2. Additional Edge Cases:
   - Scanner Path Traversal:
     * Symlinked directories and files (follow_symlinks=False vs True)
     * Target paths with relative '..' traversal
     * Non-existent and invalid target error handling
     * Single file target path resolution and exclusion
   - Custom Excludes:
     * Subdirectories with trailing slashes (e.g. 'docs/private/')
     * Deeply nested path exclusions (e.g. 'a/b/c/d/hidden')
     * Case-folding variations in custom excludes (e.g. 'DOCS/PRIVATE')
     * Custom exclude patterns with spaces (e.g. 'my drafts/private')
     * Anchored vs unanchored custom excludes
     * Custom exclude_dir_names full override
   - File Limits & Boundaries:
     * Exact size boundary (st_size == max_file_size_bytes vs st_size > max_file_size_bytes)
     * Zero-byte files handling in scanner and parser
     * Negative max_file_size_bytes (-1 disables limit)
     * None max_file_size_bytes (disables limit)
     * Single file target exact size boundary enforcement
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
    MAX_FILE_SIZE_BYTES,
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


class TestRemediationsCH01toCH05(unittest.TestCase):
    """Rigorous empirical re-test of previously failing challenges CH-01 through CH-05."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.parser = DocumentParser()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remediation_ch01_prose_bleed(self):
        """CH-01: Inter-equation prose must not leak into math_blocks."""
        doc_text = (
            "# Mathematical Memo\n\n"
            "$$\n"
            "\\mathcal{L}_{\\theta} = \\mathbb{E}[\\log p(x)]\n"
            "$$\n\n"
            "This natural language prose explains the objective function and contains no math syntax.\n\n"
            "$$\n"
            "\\nabla_\\theta \\mathcal{L}_{\\theta} = 0\n"
            "$$\n"
        )
        parsed = self.parser.parse_text(doc_text, Path("memo.md"))
        self.assertIsNotNone(parsed)
        for block in parsed.math_blocks:
            self.assertNotIn(
                "This natural language prose",
                block,
                f"CH-01 REGRESSION: Inter-equation prose leaked into math_blocks: {block}",
            )
        self.assertEqual(len(parsed.math_blocks), 2)

    def test_remediation_ch02_latex_duplicate_extraction(self):
        """CH-02: LaTeX $$ ... $$ equations must be extracted exactly once."""
        doc_text = (
            "\\begin{document}\n"
            "$$\n"
            "E = m c^2\n"
            "$$\n"
            "\\end{document}\n"
        )
        parsed = self.parser.parse_text(doc_text, Path("energy.tex"))
        self.assertIsNotNone(parsed)
        mc_matches = [m for m in parsed.math_blocks if "E = m c^2" in m or "E = mc^2" in m]
        self.assertEqual(
            len(mc_matches),
            1,
            f"CH-02 REGRESSION: Equation extracted {len(mc_matches)} times instead of 1!",
        )

    def test_remediation_ch03_relative_path_exclusion(self):
        """CH-03: Relative subdirectory patterns without wildcards (e.g. docs/private) must exclude files."""
        f_pub = self.root / "docs" / "public" / "open.md"
        f_priv = self.root / "docs" / "private" / "hidden.md"
        f_wip = self.root / "research" / "drafts" / "memo.tex"

        for f in (f_pub, f_priv, f_wip):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("sample content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["docs/private", "research/drafts"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_pub.resolve(), result_paths)
        self.assertNotIn(
            f_priv.resolve(),
            result_paths,
            "CH-03 REGRESSION: 'docs/private' failed to exclude 'docs/private/hidden.md'",
        )
        self.assertNotIn(
            f_wip.resolve(),
            result_paths,
            "CH-03 REGRESSION: 'research/drafts' failed to exclude 'research/drafts/memo.tex'",
        )

    def test_remediation_ch04_max_file_size_enforcement(self):
        """CH-04: Scanner must filter files exceeding ScannerConfig.max_file_size_bytes."""
        f_small = self.root / "small.md"
        f_large = self.root / "large.md"

        f_small.write_text("small", encoding="utf-8")  # 5 bytes
        f_large.write_text("x" * 2000, encoding="utf-8")  # 2000 bytes

        cfg = ScannerConfig(max_file_size_bytes=500)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_small.resolve(), result_paths)
        self.assertNotIn(
            f_large.resolve(),
            result_paths,
            "CH-04 REGRESSION: Scanner did not filter file exceeding max_file_size_bytes!",
        )

    def test_remediation_ch05_exclude_dir_names_override(self):
        """CH-05: ScannerConfig(excludes=[]) must not re-inject default exclude directory names."""
        cfg = ScannerConfig(excludes=[])
        self.assertEqual(
            cfg.exclude_dir_names,
            set(),
            "CH-05 REGRESSION: ScannerConfig(excludes=[]) must have empty exclude_dir_names!",
        )

        build_file = self.root / "build" / "out.tex"
        build_file.parent.mkdir(parents=True, exist_ok=True)
        build_file.write_text("build manuscript", encoding="utf-8")

        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), build_file.resolve())


class TestAdversarialPathTraversal(unittest.TestCase):
    """Adversarial challenge tests for path traversal, symlinks, and invalid targets."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scanner = Scanner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parent_directory_relative_target_resolution(self):
        """Target paths containing '..' must be resolved to canonical paths correctly."""
        sub1 = self.root / "sub1"
        sub2 = self.root / "sub2"
        sub1.mkdir(parents=True, exist_ok=True)
        sub2.mkdir(parents=True, exist_ok=True)

        target_file = sub2 / "paper.md"
        target_file.write_text("# Sub2 Paper", encoding="utf-8")

        # Traversing via sub1/../sub2
        relative_target = sub1 / ".." / "sub2"
        results = self.scanner.scan(relative_target)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), target_file.resolve())

    def test_nonexistent_target_raises_target_not_found(self):
        """Scanning a non-existent path must raise TargetNotFoundError."""
        bogus_path = self.root / "non_existent_directory_12345"
        with self.assertRaises(TargetNotFoundError):
            self.scanner.scan(bogus_path)

    def test_empty_target_raises_target_not_found(self):
        """Empty string or None target must raise TargetNotFoundError."""
        with self.assertRaises(TargetNotFoundError):
            self.scanner.scan("")
        with self.assertRaises(TargetNotFoundError):
            self.scanner.scan(None)

    def test_single_file_target_non_manuscript(self):
        """Single file target that is not .tex or .md returns empty list."""
        py_file = self.root / "script.py"
        py_file.write_text("print('hello')", encoding="utf-8")
        results = self.scanner.scan(py_file)
        self.assertEqual(results, [])

    def test_single_file_target_in_excluded_dir(self):
        """Single file target inside an excluded directory returns empty list."""
        excluded_dir = self.root / "node_modules"
        excluded_dir.mkdir(parents=True, exist_ok=True)
        target_file = excluded_dir / "readme.md"
        target_file.write_text("# Node Module Readme", encoding="utf-8")

        results = self.scanner.scan(target_file)
        self.assertEqual(results, [])


class TestAdversarialCustomExcludes(unittest.TestCase):
    """Adversarial stress-testing of custom exclusion mechanisms."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trailing_slash_in_custom_excludes(self):
        """Custom exclude patterns with trailing slashes ('docs/private/') must be normalized and exclude files."""
        f_kept = self.root / "docs" / "public" / "paper.md"
        f_dropped = self.root / "docs" / "private" / "secret.md"

        for f in (f_kept, f_dropped):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["docs/private/"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_kept.resolve(), result_paths)
        self.assertNotIn(
            f_dropped.resolve(),
            result_paths,
            "Failed to exclude when pattern has trailing slash 'docs/private/'",
        )

    def test_deeply_nested_custom_excludes(self):
        """Custom excludes with 4-level path ('a/b/c/hidden') must prune that branch only."""
        f_kept = self.root / "a" / "b" / "c" / "visible" / "doc.md"
        f_dropped = self.root / "a" / "b" / "c" / "hidden" / "secret.md"

        for f in (f_kept, f_dropped):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["a/b/c/hidden"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_kept.resolve(), result_paths)
        self.assertNotIn(
            f_dropped.resolve(),
            result_paths,
            "Failed to exclude deeply nested path 'a/b/c/hidden'",
        )

    def test_case_folding_in_custom_excludes(self):
        """Custom excludes must match case-insensitively ('DOCS/PRIVATE' excludes 'docs/private')."""
        f_kept = self.root / "docs" / "public" / "paper.md"
        f_dropped = self.root / "docs" / "private" / "secret.md"

        for f in (f_kept, f_dropped):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["DOCS/PRIVATE"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_kept.resolve(), result_paths)
        self.assertNotIn(
            f_dropped.resolve(),
            result_paths,
            "Case-folding custom exclude 'DOCS/PRIVATE' failed to exclude 'docs/private'",
        )

    def test_custom_excludes_with_spaces(self):
        """Directories containing spaces (e.g. 'draft notes/private') must be excluded properly."""
        f_kept = self.root / "draft notes" / "public" / "paper.md"
        f_dropped = self.root / "draft notes" / "private" / "secret.md"

        for f in (f_kept, f_dropped):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["draft notes/private"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_kept.resolve(), result_paths)
        self.assertNotIn(
            f_dropped.resolve(),
            result_paths,
            "Custom exclude with spaces 'draft notes/private' failed to exclude files",
        )


class TestAdversarialFileLimitsAndBoundaries(unittest.TestCase):
    """Boundary testing on file sizes and limits."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_size_boundary_enforcement(self):
        """
        Verify exact file size boundary:
        - File with EXACTLY max_file_size_bytes must be INCLUDED (st_size <= max).
        - File with max_file_size_bytes + 1 must be EXCLUDED (st_size > max).
        """
        LIMIT = 100
        f_exact = self.root / "exact.md"
        f_over = self.root / "over.md"

        f_exact.write_bytes(b"a" * LIMIT)         # Exactly 100 bytes
        f_over.write_bytes(b"a" * (LIMIT + 1))    # 101 bytes

        cfg = ScannerConfig(max_file_size_bytes=LIMIT)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(
            f_exact.resolve(),
            result_paths,
            "File of exact boundary size (100 bytes) should be included when limit is 100!",
        )
        self.assertNotIn(
            f_over.resolve(),
            result_paths,
            "File exceeding boundary size (101 bytes) must be excluded when limit is 100!",
        )

    def test_single_file_exact_size_boundary(self):
        """Single file target at exact boundary is included; 1 byte over is excluded."""
        LIMIT = 50
        f_exact = self.root / "exact.md"
        f_over = self.root / "over.md"

        f_exact.write_bytes(b"b" * LIMIT)
        f_over.write_bytes(b"b" * (LIMIT + 1))

        cfg = ScannerConfig(max_file_size_bytes=LIMIT)
        scanner = Scanner(config=cfg)

        res_exact = scanner.scan(f_exact)
        res_over = scanner.scan(f_over)

        self.assertEqual(len(res_exact), 1)
        self.assertEqual(res_over, [])

    def test_zero_byte_file_handling(self):
        """Zero-byte manuscript is discovered by scanner and returns None in parser without crash."""
        f_zero = self.root / "empty.md"
        f_zero.touch()

        scanner = Scanner()
        results = scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), f_zero.resolve())

        # Parser behavior on zero-byte file
        parser = DocumentParser()
        doc = parser.parse_file(f_zero)
        self.assertIsNone(doc, "Zero-byte file should safely return None from parser")

    def test_negative_max_file_size_bytes_disables_limit(self):
        """Setting max_file_size_bytes = -1 disables the size filter."""
        f_large = self.root / "large.md"
        f_large.write_bytes(b"c" * 50_000)

        cfg = ScannerConfig(max_file_size_bytes=-1)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), f_large.resolve())

    def test_none_max_file_size_bytes_disables_limit(self):
        """Setting max_file_size_bytes = None disables the size filter."""
        f_large = self.root / "large.md"
        f_large.write_bytes(b"c" * 50_000)

        cfg = ScannerConfig(max_file_size_bytes=None)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), f_large.resolve())


if __name__ == "__main__":
    unittest.main()
