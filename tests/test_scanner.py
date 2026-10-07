"""
Unit tests for gestalt_extractor.scanner (Directory Scanner & Path Exclusion Filter).

Verifies Milestone 1 Acceptance Criteria:
- Discovers valid .tex and .md manuscripts recursively.
- Enforces case-insensitive extension matching (.tex, .TEX, .md, .MD).
- Strictly excludes blacklisted directories (node_modules, .agents, .git, build, dist).
- Handles edge cases: single files, empty dirs, non-existent paths, special characters.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

# Ensure package importability from tests directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from gestalt_extractor.scanner import Scanner, DirectoryScanner, scan_directory
    from gestalt_extractor.config import DEFAULT_EXCLUDES, ScannerConfig
except ImportError:
    try:
        from scanner import Scanner, DirectoryScanner, scan_directory
        from config import DEFAULT_EXCLUDES, ScannerConfig
    except ImportError:
        Scanner = None
        DirectoryScanner = None
        scan_directory = None
        ScannerConfig = None
        DEFAULT_EXCLUDES = [
            "node_modules",
            ".agents",
            ".git",
            "build",
            "dist",
            "__pycache__",
            ".venv",
            "env",
        ]


class TestScannerDiscovery(unittest.TestCase):
    """Test standard discovery of valid manuscript files across directories."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        if Scanner is not None:
            self.scanner = Scanner()
        else:
            self.scanner = None

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_recursive_discovery(self):
        """Verify discovery of .tex and .md files in nested directories."""
        f1 = self.root / "doc1.md"
        f2 = self.root / "sub1" / "doc2.tex"
        f3 = self.root / "sub1" / "sub2" / "doc3.md"
        f_ignored1 = self.root / "script.py"
        f_ignored2 = self.root / "notes.txt"

        f2.parent.mkdir(parents=True, exist_ok=True)
        f3.parent.mkdir(parents=True, exist_ok=True)

        for p in [f1, f2, f3, f_ignored1, f_ignored2]:
            p.write_text("sample content", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertEqual(len(results), 3)
        self.assertIn(f1.resolve(), result_paths)
        self.assertIn(f2.resolve(), result_paths)
        self.assertIn(f3.resolve(), result_paths)
        self.assertNotIn(f_ignored1.resolve(), result_paths)
        self.assertNotIn(f_ignored2.resolve(), result_paths)

    def test_case_insensitive_extension_matching(self):
        """Verify extensions matching .TEX, .TeX, .MD, .Md, .tex, .md."""
        files = [
            self.root / "paper1.tex",
            self.root / "paper2.TEX",
            self.root / "paper3.TeX",
            self.root / "memo1.md",
            self.root / "memo2.MD",
            self.root / "memo3.Md",
        ]
        for f in files:
            f.write_text("math text", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertEqual(len(results), len(files))
        for f in files:
            self.assertIn(f.resolve(), result_paths)


class TestScannerExclusions(unittest.TestCase):
    """Test blacklist exclusion filtering to eliminate noise and false positives."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scanner = Scanner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_blacklist_directories_excluded(self):
        """Verify that files inside node_modules, .agents, .git, etc. are 100% ignored."""
        valid_file1 = self.root / "valid_research.md"
        valid_file2 = self.root / "papers" / "proof.tex"
        valid_file2.parent.mkdir(parents=True, exist_ok=True)
        valid_file1.write_text("Valid", encoding="utf-8")
        valid_file2.write_text("Valid", encoding="utf-8")

        noise_files = [
            self.root / "node_modules" / "@package" / "README.md",
            self.root / ".agents" / "teamwork" / "handoff.md",
            self.root / ".git" / "description.md",
            self.root / "build" / "output.tex",
            self.root / "dist" / "bundle.md",
            self.root / "__pycache__" / "cached.md",
            self.root / ".venv" / "lib" / "info.md",
            self.root / "env" / "readme.md",
        ]

        for p in noise_files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Noise content that must be filtered out", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertEqual(len(results), 2)
        self.assertIn(valid_file1.resolve(), result_paths)
        self.assertIn(valid_file2.resolve(), result_paths)
        for noise in noise_files:
            self.assertNotIn(noise.resolve(), result_paths)

    def test_custom_exclusion_patterns(self):
        """Verify that custom exclusion rules correctly filter user-specified patterns."""
        f_keep = self.root / "keep.md"
        f_drop1 = self.root / "drafts" / "wip.md"
        f_drop2 = self.root / "test.backup.tex"

        f_drop1.parent.mkdir(parents=True, exist_ok=True)
        f_keep.write_text("keep", encoding="utf-8")
        f_drop1.write_text("drop", encoding="utf-8")
        f_drop2.write_text("drop", encoding="utf-8")

        custom_excludes = list(DEFAULT_EXCLUDES) + ["**/drafts/**", "*.backup.tex"]
        custom_scanner = Scanner(excludes=custom_excludes)

        results = custom_scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertEqual(len(results), 1)
        self.assertIn(f_keep.resolve(), result_paths)
        self.assertNotIn(f_drop1.resolve(), result_paths)
        self.assertNotIn(f_drop2.resolve(), result_paths)


class TestScannerEdgeCases(unittest.TestCase):
    """Test defensive edge cases: single files, missing dirs, empty dirs, symbols."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scanner = Scanner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_file_target(self):
        """Verify scanner behavior when target is a single file rather than a directory."""
        single_md = self.root / "single.md"
        single_md.write_text("single file", encoding="utf-8")

        results = self.scanner.scan(single_md)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), single_md.resolve())

    def test_single_file_non_matching_extension(self):
        """Verify scanning a single non-manuscript file returns empty list or handles cleanly."""
        single_txt = self.root / "single.txt"
        single_txt.write_text("not math", encoding="utf-8")

        results = self.scanner.scan(single_txt)
        self.assertEqual(len(results), 0)

    def test_empty_directory(self):
        """Verify scanning an empty directory returns an empty list without error."""
        results = self.scanner.scan(self.root)
        self.assertEqual(results, [])

    def test_directory_with_no_matching_files(self):
        """Verify directory with only .py, .json, .csv returns empty list."""
        (self.root / "test.py").write_text("print(1)", encoding="utf-8")
        (self.root / "data.json").write_text("{}", encoding="utf-8")

        results = self.scanner.scan(self.root)
        self.assertEqual(results, [])

    def test_nonexistent_directory_raises_error(self):
        """Verify FileNotFoundError is raised if root target does not exist."""
        bad_path = self.root / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            self.scanner.scan(bad_path)

    def test_deterministic_sorting(self):
        """Verify that scan results are deterministically sorted by path."""
        file_names = ["z_doc.md", "a_doc.tex", "m_doc.md", "b_doc.tex"]
        for name in file_names:
            (self.root / name).write_text("content", encoding="utf-8")

        results = self.scanner.scan(self.root)
        result_names = [p.name for p in results]

        self.assertEqual(result_names, sorted(file_names))

    def test_filenames_with_special_characters(self):
        """Verify scanner handles spaces, brackets, hyphens, and unicode characters."""
        special_names = [
            "paper (v2) [final].md",
            "grokking-bounds_2026.tex",
            "study with spaces in title.md",
        ]
        for name in special_names:
            (self.root / name).write_text("content", encoding="utf-8")

        results = self.scanner.scan(self.root)
        self.assertEqual(len(results), len(special_names))


class TestScannerRemediation(unittest.TestCase):
    """Regression tests covering Milestone 1 Iteration 2 scanner remediations."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_relative_path_exclusion_without_wildcards(self):
        """Verify relative directory patterns without wildcards (e.g. docs/private) exclude nested files."""
        f_keep = self.root / "docs" / "public" / "paper.md"
        f_drop1 = self.root / "docs" / "private" / "secret.md"
        f_drop2 = self.root / "research" / "drafts" / "wip.tex"

        for f in [f_keep, f_drop1, f_drop2]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("content", encoding="utf-8")

        scanner = Scanner(excludes=list(DEFAULT_EXCLUDES) + ["docs/private", "research/drafts"])
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_keep.resolve(), result_paths)
        self.assertNotIn(f_drop1.resolve(), result_paths)
        self.assertNotIn(f_drop2.resolve(), result_paths)

    def test_max_file_size_enforcement_directory(self):
        """Verify files exceeding max_file_size_bytes are excluded during directory scan."""
        small_file = self.root / "small.md"
        large_file = self.root / "large.md"

        small_file.write_text("Small content", encoding="utf-8")
        large_file.write_text("A" * 10_000, encoding="utf-8")

        cfg = ScannerConfig(max_file_size_bytes=500)
        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(small_file.resolve(), result_paths)
        self.assertNotIn(large_file.resolve(), result_paths)

    def test_max_file_size_enforcement_single_file(self):
        """Verify single-file target exceeding max_file_size_bytes returns empty list."""
        large_file = self.root / "large.md"
        large_file.write_text("A" * 10_000, encoding="utf-8")

        cfg = ScannerConfig(max_file_size_bytes=500)
        scanner = Scanner(config=cfg)
        results = scanner.scan(large_file)
        self.assertEqual(results, [])

    def test_empty_excludes_allows_build_scan(self):
        """Verify passing excludes=[] allows scanning build/ or dist/ directories."""
        build_file = self.root / "build" / "manuscript.tex"
        build_file.parent.mkdir(parents=True, exist_ok=True)
        build_file.write_text("build manuscript", encoding="utf-8")

        cfg = ScannerConfig(excludes=[])
        self.assertEqual(cfg.exclude_dir_names, set())

        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].resolve(), build_file.resolve())

    def test_custom_exclude_dir_names_override(self):
        """Verify explicit exclude_dir_names overrides default directory blacklist."""
        doc1 = self.root / "build" / "doc.md"
        doc2 = self.root / "custom_drop" / "doc.md"
        doc1.parent.mkdir(parents=True, exist_ok=True)
        doc2.parent.mkdir(parents=True, exist_ok=True)
        doc1.write_text("doc1", encoding="utf-8")
        doc2.write_text("doc2", encoding="utf-8")

        cfg = ScannerConfig(exclude_dir_names=["custom_drop"])
        self.assertEqual(cfg.exclude_dir_names, {"custom_drop"})

        scanner = Scanner(config=cfg)
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(doc1.resolve(), result_paths)
        self.assertNotIn(doc2.resolve(), result_paths)

    def test_boundary_adjacent_directories(self):
        """Verify node_modules_backup is kept while node_modules is dropped."""
        f_kept = self.root / "node_modules_backup" / "paper.md"
        f_dropped = self.root / "node_modules" / "paper.md"
        f_kept.parent.mkdir(parents=True, exist_ok=True)
        f_dropped.parent.mkdir(parents=True, exist_ok=True)
        f_kept.write_text("kept", encoding="utf-8")
        f_dropped.write_text("dropped", encoding="utf-8")

        scanner = Scanner()
        results = scanner.scan(self.root)
        result_paths = {p.resolve() for p in results}

        self.assertIn(f_kept.resolve(), result_paths)
        self.assertNotIn(f_dropped.resolve(), result_paths)


if __name__ == "__main__":
    unittest.main()
