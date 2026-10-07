"""
Adversarial Stress-Testing Suite for gestalt_extractor.parser.
==============================================================
Empirically stress-tests DocumentParser against hostile, corrupted,
and boundary-case inputs to verify that it NEVER crashes or hangs.

Attack Vectors Covered:
1. Zero-byte files (0 bytes).
2. Oversize manuscripts (>10MB cap, tested with 15MB-20MB simulation).
3. Raw null bytes (\x00) and corrupted binary streams.
4. Non-ASCII Unicode (RTL Arabic, CJK, Emoji, Greek, mathematical scripts).
5. Unclosed math environments ($$ without $$, \\begin{equation} without \\end).
6. Unclosed LaTeX groups (\\title{... without }).
7. Escaped percent symbols (\\%, \\\\%, \\\\\\%).
8. Complex nested theorems and environments.
9. ReDoS resistance (long pathological strings with unclosed delimiters).
10. Prose isolation between display math delimiters.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gestalt_extractor.parser import DocumentParser, ParsedDocument
from gestalt_extractor.config import MAX_FILE_SIZE_BYTES


class TestAdversarialInputs(unittest.TestCase):
    """Hostile edge case test suite for DocumentParser."""

    def setUp(self):
        self.parser = DocumentParser()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Vector 1: Zero-byte and Empty Files
    # -------------------------------------------------------------------------

    def test_zero_byte_tex_file(self):
        """Verify 0-byte .tex file returns None safely without exception."""
        p = self.root / "zero.tex"
        p.touch()
        result = self.parser.parse_file(p)
        self.assertIsNone(result)

    def test_zero_byte_md_file(self):
        """Verify 0-byte .md file returns None safely without exception."""
        p = self.root / "zero.md"
        p.touch()
        result = self.parser.parse_file(p)
        self.assertIsNone(result)

    def test_whitespace_only_files(self):
        """Verify files with spaces, tabs, and newlines return None safely."""
        p = self.root / "spaces.md"
        p.write_text("   \n\t\r\n   \n", encoding="utf-8")
        result = self.parser.parse_file(p)
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # Vector 2: Oversize Files (>10MB, 20MB)
    # -------------------------------------------------------------------------

    def test_oversize_file_rejection(self):
        """Verify files exceeding MAX_FILE_SIZE_BYTES (>10MB) are rejected with None without crashing."""
        oversize_path = self.root / "huge_manuscript.tex"
        target_size = MAX_FILE_SIZE_BYTES + 1024 * 1024  # 11 MB
        with open(oversize_path, "wb") as f:
            f.seek(target_size - 1)
            f.write(b"\0")
        self.assertGreater(oversize_path.stat().st_size, MAX_FILE_SIZE_BYTES)

        start = time.time()
        result = self.parser.parse_file(oversize_path)
        elapsed = time.time() - start

        self.assertIsNone(result)
        self.assertLess(elapsed, 0.5, "Oversize file check must be instantaneous O(1)")

    # -------------------------------------------------------------------------
    # Vector 3: Raw Null Bytes & Binary Garbage
    # -------------------------------------------------------------------------

    def test_raw_null_bytes_and_binary_garbage_tex(self):
        """Verify parsing raw null bytes and non-decodable bytes in .tex does not crash."""
        p = self.root / "garbage.tex"
        # Mix null bytes, invalid UTF-8 sequences (0xFF, 0xFE, 0xC0), and valid text
        p.write_bytes(b"\x00\x00\x00\\title{Binary Garbage}\x00\xff\xfe\x00\\begin{document}Hello\x00\end{document}")
        
        result = self.parser.parse_file(p)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ParsedDocument)
        self.assertIn("Binary Garbage", result.title)

    def test_raw_null_bytes_in_markdown(self):
        """Verify markdown with embedded null bytes does not crash or raise."""
        p = self.root / "garbage.md"
        p.write_bytes(b"# Binary MD Title\x00\x00\nContent with nulls \x00 here.\n")
        
        result = self.parser.parse_file(p)
        self.assertIsNotNone(result)
        self.assertIn("Binary MD Title", result.title)

    def test_pure_binary_random_stream(self):
        """Verify pure random non-text byte streams do not crash."""
        p = self.root / "random.md"
        # High-entropy bytes that are not valid UTF-8
        p.write_bytes(bytes(range(256)) * 10)
        
        result = self.parser.parse_file(p)
        # Should either return ParsedDocument or None without throwing uncaught exceptions
        self.assertTrue(result is None or isinstance(result, ParsedDocument))

    # -------------------------------------------------------------------------
    # Vector 4: Non-ASCII & Multi-Language Unicode
    # -------------------------------------------------------------------------

    def test_multilingual_unicode_support(self):
        """Verify full Unicode preservation: Chinese, Arabic RTL, Math Greek, and Emojis."""
        content = """# 机器学习与谱收敛: Spectral Convergence 🚀
---
title: 深度学习中的对偶理论
author: 阿尔伯特 · 爱因斯坦 & الخوارزمي
---

## 理论陈述 (Theoretical Formulation)

考虑黎曼流形 $\mathcal{M}$ 上的测地线流:
$$
\int_{\Omega} \nabla f(x) \cdot \mathbf{n} \, dS = \sum_{i=1}^k \lambda_i \langle v_i, w_i \rangle
$$

> **Theorem 1 (谱界限 / Spectral Bound)**: 设哈密顿量 $\mathcal{H}$ 满足全纯约束，则收敛率为 $\mathcal{O}(n^2 \log n)$。
"""
        doc = self.parser.parse_text(content, path=Path("unicode_test.md"))
        self.assertIsNotNone(doc)
        self.assertIn("深度学习中的对偶理论", doc.title)
        self.assertIn("Spectral Bound", " ".join(doc.theorems))
        self.assertGreaterEqual(len(doc.math_blocks), 1)

    # -------------------------------------------------------------------------
    # Vector 5: Unclosed Math Environments
    # -------------------------------------------------------------------------

    def test_unclosed_double_dollars_tex(self):
        """Verify single unmatched $$ in LaTeX does not crash or hang."""
        content = r"""
        \begin{document}
        This is an unclosed display math block:
        $$ \int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
        No closing double dollars anywhere in this document!
        \end{document}
        """
        start = time.time()
        doc = self.parser.parse_text(content, format_hint="latex")
        elapsed = time.time() - start

        self.assertIsNotNone(doc)
        self.assertLess(elapsed, 1.0)

    def test_unclosed_double_dollars_md(self):
        """Verify single unmatched $$ in Markdown does not crash or hang."""
        content = """# Incomplete Math
Here is an open formula:
$$
E = mc^2 + \mathcal{O}(p^4)
And we never closed it.
"""
        start = time.time()
        doc = self.parser.parse_text(content, format_hint="markdown")
        elapsed = time.time() - start

        self.assertIsNotNone(doc)
        self.assertLess(elapsed, 1.0)

    def test_unclosed_latex_begin_environments(self):
        """Verify unclosed \\begin{equation}, \\begin{align} do not hang."""
        content = r"""
        \begin{document}
        \begin{equation}
        x_{t+1} = x_t - \eta \nabla f(x_t)
        % Missing \end{equation}
        \begin{align}
        y = mx + b
        """
        start = time.time()
        doc = self.parser.parse_text(content, format_hint="latex")
        elapsed = time.time() - start

        self.assertIsNotNone(doc)
        self.assertLess(elapsed, 1.0)

    # -------------------------------------------------------------------------
    # Vector 6: Unclosed LaTeX Groups & Formatting
    # -------------------------------------------------------------------------

    def test_unclosed_title_group(self):
        """Verify unclosed \\title{... falls back safely to filename stem."""
        content = r"""
        \documentclass{article}
        \title{Unclosed Title Group without closing bracket
        \begin{document}
        Prose goes here.
        \end{document}
        """
        doc = self.parser.parse_text(content, path=Path("unclosed_doc.tex"))
        self.assertIsNotNone(doc)
        # Should gracefully fall back to title-cased filename
        self.assertEqual(doc.title, "Unclosed Doc")

    def test_unclosed_sections_and_formatting(self):
        """Verify unmatched braces in \\section{ and \\textbf{ do not raise errors."""
        content = r"""
        \begin{document}
        \section{Normal Section}
        This is \textbf{unclosed bold formatting
        \subsection{Unclosed subsection
        More prose text.
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        self.assertIsNotNone(doc)
        self.assertIn("Normal Section", [s["title"] for s in doc.sections])

    # -------------------------------------------------------------------------
    # Vector 7: Escaped Percent Characters & Comment Parity
    # -------------------------------------------------------------------------

    def test_escaped_percent_parity_adversarial(self):
        """Verify backslash parity calculation: \\% is literal, \\\\% is comment, \\\\\\% is literal."""
        content = r"""
        \begin{document}
        Case 1: 99\% valid data. % comment 1
        Case 2: Double slash \\% This is a comment because \\ escapes the slash!
        Case 3: Triple slash \\\% This has escaped percent and should survive!
        Case 4: Empty percent %
        Case 5: Trailing backslash and percent \%
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        self.assertIsNotNone(doc)
        # Case 1: 99% must be in clean_text, comment 1 must not
        self.assertIn("99%", doc.clean_text)
        self.assertNotIn("comment 1", doc.clean_text)
        # Case 2: "This is a comment because" must be stripped
        self.assertNotIn("This is a comment because", doc.clean_text)
        # Case 3: "This has escaped percent" must be preserved
        self.assertIn("This has escaped percent", doc.clean_text)

    # -------------------------------------------------------------------------
    # Vector 8: Nested Theorems & Complex Environments
    # -------------------------------------------------------------------------

    def test_nested_theorems_handling(self):
        """Verify parser processes nested theorem and lemma environments without crashing."""
        content = r"""
        \begin{document}
        \begin{theorem}[Main Stability]
        Let $\mathbf{A}$ be symmetric positive definite.
        \begin{lemma}[Eigenvalue Sub-Bound]
        All eigenvalues satisfy $\lambda_i > 0$.
        \end{lemma}
        Then the condition number is finite.
        \end{theorem}
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        self.assertIsNotNone(doc)
        # Verify parser did not crash and captured at least the outer theorem
        self.assertGreaterEqual(len(doc.theorems), 1)
        self.assertIn("Main Stability", doc.theorems[0])

    # -------------------------------------------------------------------------
    # Vector 9: ReDoS Pathological Stress Tests
    # -------------------------------------------------------------------------

    def test_redos_deeply_nested_braces(self):
        """Stress-test regex against 2,000 open braces to verify no exponential backtracking."""
        pathological = "\\title{" + "{" * 2000 + "word" + "}" * 1000
        start = time.time()
        doc = self.parser.parse_text(pathological, path=Path("redos.tex"))
        elapsed = time.time() - start

        self.assertIsNotNone(doc)
        self.assertLess(elapsed, 1.0, f"ReDoS suspected: took {elapsed:.2f}s on nested braces")

    def test_redos_long_unmatched_math(self):
        """Stress-test regex against 50,000 characters of unmatched math syntax."""
        pathological = "$" + ("x + y * z " * 5000)
        start = time.time()
        doc = self.parser.parse_text(pathological, format_hint="markdown")
        elapsed = time.time() - start

        self.assertIsNotNone(doc)
        self.assertLess(elapsed, 1.0, f"ReDoS suspected: took {elapsed:.2f}s on long unmatched math")

    # -------------------------------------------------------------------------
    # Vector 10: Prose Delimiter Isolation Between Math Blocks
    # -------------------------------------------------------------------------

    def test_display_math_prose_isolation_tex(self):
        """Test whether prose between $$ ... $$ is accidentally extracted into math_blocks."""
        content = r"""
        \begin{document}
        First equation:
        $$ E = mc^2 $$
        Here is standard narrative prose that discusses the implications of relativistic energy.
        Second equation:
        $$ F = ma $$
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        self.assertIsNotNone(doc)
        
        # Check math blocks
        for block in doc.math_blocks:
            # Prose should NEVER be classified as a mathematical formula
            self.assertNotIn("Here is standard narrative prose", block, 
                             "CRITICAL FLAW: Narrative prose between $$ blocks was extracted as math!")

    def test_display_math_prose_isolation_md(self):
        """Test whether prose between $$ ... $$ in Markdown is extracted into math_blocks."""
        content = """# Relativity Notes
First equation:
$$
E = mc^2
$$
Here is standard narrative prose explaining the dynamics of velocity and energy.
Second equation:
$$
F = ma
$$
"""
        doc = self.parser.parse_text(content, format_hint="markdown")
        self.assertIsNotNone(doc)
        for block in doc.math_blocks:
            self.assertNotIn("standard narrative prose", block,
                             "CRITICAL FLAW: Markdown prose between $$ blocks was extracted as math!")

    def test_code_fence_comment_header_isolation(self):
        """Test that Python comments like '# My Comment' in code fences are not treated as H1 headers."""
        content = """```python
# Setup Configuration Script
x = 1
## Sub configuration
y = 2
```

# Real Document Title

Prose body here.
"""
        doc = self.parser.parse_text(content, format_hint="markdown")
        self.assertIsNotNone(doc)
        # The document title should NOT be the code fence comment
        self.assertEqual(doc.title, "Real Document Title",
                         f"CRITICAL FLAW: Code comment was mistaken for document title: {doc.title}")
        # The section list should NOT contain code comments as headers
        section_titles = [s["title"] for s in doc.sections]
        self.assertNotIn("Setup Configuration Script", section_titles,
                         "CRITICAL FLAW: Code fence comment was extracted as document section!")


if __name__ == "__main__":
    unittest.main()

