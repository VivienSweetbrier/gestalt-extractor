"""
Unit tests for gestalt_extractor.parser (Multi-Format Document Parser).

Verifies Milestone 1 Acceptance Criteria:
- ParsedDocument data model compliance.
- LaTeX parsing: preamble/comment stripping, theorem/math extraction.
- Markdown parsing: title/header extraction, code fence handling, math blocks.
- Resilience: zero-byte files, non-UTF8 decoding fallback, malformed syntax.
- Real fixture verification on sample_bounds.tex and sample_bounds.md.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure package importability from tests directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from gestalt_extractor.parser import DocumentParser, Parser, ParsedDocument
except ImportError:
    try:
        from parser import DocumentParser, Parser, ParsedDocument
    except ImportError:
        DocumentParser = None
        Parser = None
        ParsedDocument = None


class TestParsedDocumentContract(unittest.TestCase):
    """Verify ParsedDocument data model structure and attributes."""

    def test_parsed_document_attributes(self):
        """Verify ParsedDocument instantiates with required interface attributes."""
        doc = ParsedDocument(
            path=Path("sample.tex"),
            title="Sample Title",
            raw_text="raw content",
            clean_text="clean content",
            math_blocks=["\\mathcal{O}(n)"],
            theorems=["Theorem 1: Bound"],
        )
        self.assertEqual(doc.path, Path("sample.tex"))
        self.assertEqual(doc.title, "Sample Title")
        self.assertEqual(doc.raw_text, "raw content")
        self.assertEqual(doc.clean_text, "clean content")
        self.assertEqual(doc.math_blocks, ["\\mathcal{O}(n)"])
        self.assertEqual(doc.theorems, ["Theorem 1: Bound"])


class TestLatexParser(unittest.TestCase):
    """Verify LaTeX (.tex) parsing, preamble/comment stripping, and math extraction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.parser = DocumentParser()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_latex_title_extraction(self):
        """Verify extraction of title from \\title{...}."""
        content = r"""
        \documentclass{article}
        \title{Spectral Convergence of Gradient Descents}
        \begin{document}
        Hello World
        \end{document}
        """
        tex_file = self.root / "test_title.tex"
        tex_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(tex_file)
        self.assertEqual(doc.title, "Spectral Convergence of Gradient Descents")

    def test_latex_preamble_and_macro_stripping(self):
        """Verify preamble commands and macro definitions are stripped from clean_text."""
        content = r"""
        \documentclass[11pt]{article}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \newcommand{\R}{\mathbb{R}}
        \begin{document}
        This is the actual prose of the manuscript.
        \end{document}
        """
        tex_file = self.root / "test_preamble.tex"
        tex_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(tex_file)
        self.assertNotIn(r"\documentclass", doc.clean_text)
        self.assertNotIn(r"\usepackage", doc.clean_text)
        self.assertNotIn(r"\newcommand", doc.clean_text)
        self.assertIn("This is the actual prose of the manuscript.", doc.clean_text)

    def test_latex_comment_stripping(self):
        """Verify % comments are stripped while escaped \% is preserved."""
        content = r"""
        \begin{document}
        Accuracy reached 98.5\% on validation data. % This is a private author comment
        % Entire line comment that must disappear
        Final conclusion confirmed.
        \end{document}
        """
        tex_file = self.root / "test_comment.tex"
        tex_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(tex_file)
        self.assertIn("98.5%", doc.clean_text)
        self.assertNotIn("This is a private author comment", doc.clean_text)
        self.assertNotIn("Entire line comment that must disappear", doc.clean_text)
        self.assertIn("Final conclusion confirmed.", doc.clean_text)

    def test_latex_math_environment_extraction(self):
        """Verify extraction of equation, align, gather, and display math blocks."""
        content = r"""
        \begin{document}
        First we consider:
        \begin{equation}
        \mathbf{G} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T
        \end{equation}
        And the multiline recurrence:
        \begin{align}
        \mathbf{m}_t &= \alpha \mathbf{m}_{t-1} + (1-\alpha) \mathbf{g}_t \\
        \mathbf{g}'_t &= \mathbf{g}_t + \lambda \mathbf{m}_t
        \end{align}
        And display math:
        $$ \mathcal{T}(n) = \mathcal{O}(n^2 \log n) $$
        \end{document}
        """
        tex_file = self.root / "test_math.tex"
        tex_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(tex_file)
        self.assertGreaterEqual(len(doc.math_blocks), 3)

        math_joined = " ".join(doc.math_blocks)
        self.assertIn(r"\mathbf{G} = \mathbf{U}_r", math_joined)
        self.assertIn(r"\mathbf{m}_t", math_joined)
        self.assertIn(r"\mathcal{O}(n^2 \log n)", math_joined)

    def test_latex_theorem_environments(self):
        """Verify extraction of theorem, lemma, proposition, and corollary blocks."""
        content = r"""
        \begin{document}
        \begin{theorem}
        Let $G$ be full rank. Then convergence occurs in $\mathcal{O}(n^2 \log n)$ steps.
        \end{theorem}
        \begin{lemma}
        Gradient variance is strictly bounded by $\sigma^2 (1 + \lambda)$.
        \end{lemma}
        \begin{proposition}
        The condition number satisfies $\kappa \le 1.0$.
        \end{proposition}
        \end{document}
        """
        tex_file = self.root / "test_theorems.tex"
        tex_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(tex_file)
        self.assertGreaterEqual(len(doc.theorems), 3)

        theorems_joined = " ".join(doc.theorems)
        self.assertIn("convergence occurs in", theorems_joined)
        self.assertIn("Gradient variance is strictly bounded", theorems_joined)
        self.assertIn("condition number satisfies", theorems_joined)


class TestMarkdownParser(unittest.TestCase):
    """Verify Markdown (.md) parsing, header extraction, code fence handling, and math preservation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.parser = DocumentParser()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_markdown_title_extraction(self):
        """Verify title extraction from top H1 header."""
        content = "# Grokfast Optimizer Dynamics\n\nThis manuscript investigates grokking."
        md_file = self.root / "test_title.md"
        md_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(md_file)
        self.assertEqual(doc.title, "Grokfast Optimizer Dynamics")

    def test_markdown_frontmatter_title(self):
        """Verify title extraction from YAML frontmatter if present."""
        content = "---\ntitle: Spectral Equalization for Deep Networks\ndate: 2026-10-07\n---\n\n# Fallback Header\n\nContent here."
        md_file = self.root / "test_fm.md"
        md_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(md_file)
        self.assertEqual(doc.title, "Spectral Equalization for Deep Networks")

    def test_markdown_display_and_inline_math(self):
        """Verify extraction of $$...$$ display math and $...$ inline math."""
        content = """
        # Model Capacity

        We evaluate capacity ceiling $C \\le 3.6 P$ bits.

        The loss gradient is defined as:
        $$
        g'_t = g_t + \\lambda m_t
        $$

        Where $\\lambda = 2.0$ is the amplification constant.
        """
        md_file = self.root / "test_math.md"
        md_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(md_file)
        self.assertGreaterEqual(len(doc.math_blocks), 1)
        math_joined = " ".join(doc.math_blocks)
        self.assertIn("g'_t = g_t + \\lambda m_t", math_joined)

    def test_markdown_code_fence_stripping(self):
        """Verify code blocks are removed from prose to avoid corrupting language analysis."""
        content = """
        # Algorithm Specification

        Here is the implementation:

        ```python
        def optimizer_step(g, m):
            return g + 2.0 * m
        ```

        This confirms that the optimizer performs low-pass filtering.
        """
        md_file = self.root / "test_fence.md"
        md_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(md_file)
        self.assertIn("This confirms that the optimizer performs low-pass filtering.", doc.clean_text)
        self.assertNotIn("def optimizer_step(g, m):", doc.clean_text)


class TestParserRobustnessAndEdgeCases(unittest.TestCase):
    """Test error handling, malformed syntax, 0-byte files, and non-UTF8 encoding."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.parser = DocumentParser()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zero_byte_empty_file(self):
        """Verify parsing a 0-byte file returns None or safe empty document without crashing."""
        empty_file = self.root / "empty.md"
        empty_file.touch()

        doc = self.parser.parse_file(empty_file)
        if doc is not None:
            self.assertEqual(doc.clean_text.strip(), "")
            self.assertEqual(len(doc.math_blocks), 0)

    def test_whitespace_only_file(self):
        """Verify parsing file with only spaces and newlines handles safely."""
        ws_file = self.root / "whitespace.tex"
        ws_file.write_text("   \n\t  \n  ", encoding="utf-8")

        doc = self.parser.parse_file(ws_file)
        if doc is not None:
            self.assertEqual(doc.clean_text.strip(), "")

    def test_corrupted_non_utf8_encoding(self):
        """Verify parser gracefully handles non-UTF8 or Latin-1 byte streams without crashing."""
        bad_file = self.root / "corrupted.md"
        with open(bad_file, "wb") as f:
            f.write(b"# Corrupted Title \x93Smart Quotes\x94 and \xff invalid byte\n")

        doc = self.parser.parse_file(bad_file)
        self.assertIsNotNone(doc)
        self.assertIn("Corrupted Title", doc.clean_text)

    def test_malformed_unclosed_latex_environment(self):
        """Verify parser does not crash or hang on unclosed LaTeX environments."""
        content = r"""
        \documentclass{article}
        \begin{document}
        \begin{equation}
        \mathbf{G} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
        % Missing \end{equation} at EOF
        """
        malformed_file = self.root / "malformed.tex"
        malformed_file.write_text(content, encoding="utf-8")

        doc = self.parser.parse_file(malformed_file)
        self.assertIsNotNone(doc)

    def test_nonexistent_file_raises_not_found(self):
        """Verify FileNotFoundError is raised when file does not exist."""
        missing_file = self.root / "nonexistent.tex"
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_file(missing_file)


class TestParserFixtureIntegration(unittest.TestCase):
    """Verify parser against the real testbed fixtures."""

    def setUp(self):
        self.parser = DocumentParser()
        project_root = Path(__file__).resolve().parent.parent
        self.fixtures_dir = project_root / "testbed" / "fixtures"

    def test_parse_sample_bounds_tex_fixture(self):
        """Verify parsing the bundled sample_bounds.tex fixture."""
        tex_path = self.fixtures_dir / "sample_bounds.tex"
        if not tex_path.exists():
            self.skipTest(f"Fixture {tex_path} not yet created.")

        doc = self.parser.parse_file(tex_path)
        self.assertIsNotNone(doc)
        self.assertIn("Spectral Bounds", doc.title)
        self.assertGreaterEqual(len(doc.theorems), 2)
        self.assertGreaterEqual(len(doc.math_blocks), 3)

        all_text = doc.raw_text
        self.assertTrue(r"\mathcal{O}(n^2 \log n)" in all_text or "n^2 \log n" in all_text)

    def test_parse_sample_bounds_md_fixture(self):
        """Verify parsing the bundled sample_bounds.md fixture."""
        md_path = self.fixtures_dir / "sample_bounds.md"
        if not md_path.exists():
            self.skipTest(f"Fixture {md_path} not yet created.")

        doc = self.parser.parse_file(md_path)
        self.assertIsNotNone(doc)
        self.assertIn("Dual-Regime Dynamics", doc.title)
        self.assertGreaterEqual(len(doc.math_blocks), 2)
        self.assertIn("Grokfast", doc.raw_text)
        self.assertIn("3.6", doc.raw_text)


class TestParserRemediation(unittest.TestCase):
    """Regression tests covering Milestone 1 Iteration 2 parser remediations."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_latex_inter_equation_prose_isolation(self):
        """Verify LaTeX display math does not capture narrative prose into math_blocks."""
        content = r"""
        \begin{document}
        $$ E = mc^2 $$
        Here is standard narrative prose that must never be a formula.
        $$ F = ma $$
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        self.assertFalse(any("narrative prose" in b for b in doc.math_blocks))
        e_matches = [b for b in doc.math_blocks if "E = mc^2" in b]
        f_matches = [b for b in doc.math_blocks if "F = ma" in b]
        self.assertEqual(len(e_matches), 1)
        self.assertEqual(len(f_matches), 1)

    def test_latex_ams_inline_parentheses(self):
        """Verify LaTeX \( ... \) inline formulas are extracted into math_blocks."""
        content = r"""
        \begin{document}
        Let \( x \in \mathcal{H} \) be a bounded vector and \( y \ge 0 \).
        \end{document}
        """
        doc = self.parser.parse_text(content, format_hint="latex")
        math_str = " ".join(doc.math_blocks)
        self.assertIn(r"x \in \mathcal{H}", math_str)
        self.assertIn(r"y \ge 0", math_str)

    def test_markdown_inter_equation_prose_isolation(self):
        """Verify Markdown $$...$$ blocks do not bleed inter-equation prose into math_blocks."""
        content = """# Physics
$$
E = mc^2
$$
This prose is between equations and is NOT a mathematical formula.
$$
F = ma
$$
"""
        doc = self.parser.parse_text(content, format_hint="markdown")
        self.assertFalse(any("between equations" in b for b in doc.math_blocks))
        e_matches = [b for b in doc.math_blocks if "E = mc^2" in b]
        f_matches = [b for b in doc.math_blocks if "F = ma" in b]
        self.assertEqual(len(e_matches), 1)
        self.assertEqual(len(f_matches), 1)

    def test_markdown_interleaving_variable_preservation(self):
        """Verify inline variables between display equations are preserved without prose bleed."""
        content = """# Theory
$$
A = B
$$
where $x > 0$ holds strictly
$$
C = D
$$
"""
        doc = self.parser.parse_text(content, format_hint="markdown")
        self.assertIn("A = B", doc.math_blocks)
        self.assertIn("C = D", doc.math_blocks)
        self.assertIn("x > 0", doc.math_blocks)
        self.assertFalse(any("where" in b for b in doc.math_blocks))
        self.assertFalse(any("holds" in b for b in doc.math_blocks))

    def test_code_fence_title_and_section_isolation(self):
        """Verify code comments in code blocks do not pollute titles or section headers."""
        md = """```bash
# Setup Environment
export PATH="$HOME/bin:$PATH"
## Sub setup step
npm install
```

# Primary Manuscript Title

## 1. Introduction
This is genuine prose.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        self.assertEqual(doc.title, "Primary Manuscript Title")
        section_titles = [s["title"] for s in doc.sections]
        self.assertNotIn("Setup Environment", section_titles)
        self.assertNotIn("Sub setup step", section_titles)
        self.assertIn("1. Introduction", section_titles)

    def test_code_fence_math_variable_isolation(self):
        """Verify shell variables ($PATH, $DIR) in code fences are not extracted as math."""
        md = """# Shell Analysis
```sh
echo "$DIR:$PATH"
$VAR = $OTHER
```
We evaluate $x \\in \\mathcal{X}$.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        for m in doc.math_blocks:
            self.assertNotIn("DIR", m)
            self.assertNotIn("VAR", m)
            self.assertNotIn("PATH", m)
        self.assertTrue(any("x \\in \\mathcal{X}" in m for m in doc.math_blocks))

    def test_code_fence_crlf_windows_isolation(self):
        """Verify code fence masking succeeds with Windows CRLF line endings."""
        md = "```python\r\n# CRLF Comment\r\nx = 1\r\n```\r\n\r\n# Title CRLF\r\nProse text.\r\n"
        doc = self.parser.parse_text(md, path=Path("crlf.md"))
        self.assertEqual(doc.title, "Title CRLF")
        self.assertNotIn("CRLF Comment", [s["title"] for s in doc.sections])

    def test_theorem_number_preservation_hierarchical(self):
        """Verify hierarchical theorem numbers (1.1) and titles are preserved."""
        md = """# Theory
**Theorem 1.1 (Complexity Bound):** For all $n \\ge 1$, runtime is bounded.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        self.assertEqual(len(doc.theorems), 1)
        self.assertTrue(
            doc.theorems[0].startswith("Theorem 1.1 (Complexity Bound):"),
            f"Theorem number truncated: {doc.theorems[0]}"
        )

    def test_theorem_number_preservation_simple(self):
        """Verify simple theorem numbers (**Theorem 2:**) are preserved."""
        md = """# Theory
**Theorem 2:** Convergence holds monotonically.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        self.assertEqual(len(doc.theorems), 1)
        self.assertTrue(
            doc.theorems[0].startswith("Theorem 2:"),
            f"Theorem number truncated: {doc.theorems[0]}"
        )

    def test_axiom_preservation(self):
        """Verify Axiom numbers (**Axiom 1 (The Incarnation):**) are preserved."""
        md = """# Ontology
**Axiom 1 (The Incarnation):** Continuity requires biological embodiment.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        self.assertEqual(len(doc.theorems), 1)
        self.assertTrue(
            doc.theorems[0].startswith("Axiom 1 (The Incarnation):"),
            f"Axiom number truncated: {doc.theorems[0]}"
        )

    def test_blockquote_theorem_preservation(self):
        """Verify blockquote theorem formatting is handled cleanly."""
        md = """# Ontology
> **Lemma 3 (Duality):** Dual formulation preserves convex hull.
"""
        doc = self.parser.parse_text(md, path=Path("test.md"))
        self.assertEqual(len(doc.theorems), 1)
        self.assertTrue(
            doc.theorems[0].startswith("Lemma 3 (Duality):"),
            f"Lemma blockquote truncated: {doc.theorems[0]}"
        )


if __name__ == "__main__":
    unittest.main()
