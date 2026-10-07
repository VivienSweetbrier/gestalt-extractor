"""
gestalt_extractor.parser
========================
Multi-format parser for LaTeX (.tex) and Markdown (.md) manuscripts.
Extracts clean prose, mathematical formulations, theorem environments,
and structured metadata with robust multi-tier encoding resilience.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from gestalt_extractor.config import MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Standardized representation of an ingested manuscript.
    
    Attributes:
        path: Filesystem path to the manuscript.
        title: Extracted or inferred manuscript title.
        raw_text: Verbatim source text as read from disk.
        clean_text: Sanitized prose text with stripped macros and formatting.
        math_blocks: Mathematical formulas extracted from display and inline environments.
        theorems: Formal theorem, lemma, proposition, and axiom environments.
        sections: Hierarchical list of section dictionaries with title, level, and content.
        algorithms: Algorithmic pseudocode or code implementations.
        doc_type: Manuscript category ('latex' or 'markdown').
        metadata: Document metadata dictionary.
        char_count: Character length of clean_text.
        word_count: Word count of clean_text.
    """
    path: Path
    title: str
    raw_text: str
    clean_text: str
    math_blocks: List[str] = field(default_factory=list)
    theorems: List[str] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    doc_type: str = "markdown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed document into a serializable dictionary."""
        return {
            "path": str(self.path),
            "file": self.path.name,
            "title": self.title,
            "doc_type": self.doc_type,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "clean_text": self.clean_text,
            "math_blocks_count": len(self.math_blocks),
            "theorems_count": len(self.theorems),
            "algorithms_count": len(self.algorithms),
            "sections": self.sections,
            "metadata": self.metadata,
        }


class DocumentParser:
    """Multi-format manuscript parser supporting LaTeX and Markdown."""

    # LaTeX Regex Patterns
    RE_TEX_TITLE = re.compile(r'\\title(?:\[.*?\])?\{((?:[^{}]|\{[^{}]*\})+)\}', re.DOTALL)
    RE_TEX_AUTHOR = re.compile(r'\\author(?:\[.*?\])?\{((?:[^{}]|\{[^{}]*\})+)\}', re.DOTALL)
    RE_TEX_MATH_ENV = re.compile(
        r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath\*?)\}(.*?)\\end\{\1\}',
        re.DOTALL
    )
    RE_TEX_DISPLAY_BRACKETS = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
    RE_TEX_DISPLAY_DOLLARS = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
    RE_TEX_INLINE_MATH = re.compile(r'(?<![\$\\])\$(?![\$\\])(.*?)(?<![\$\\])\$(?![\$\\])', re.DOTALL)
    RE_TEX_INLINE_PARENS = re.compile(r'\\\((.*?)\\\)', re.DOTALL)
    RE_TEX_THEOREM_ENV = re.compile(
        r'\\begin\{(theorem\*?|lemma\*?|proposition\*?|corollary\*?|definition\*?|conjecture\*?|axiom\*?|claim\*?|remark\*?|algorithm\*?)\}'
        r'(?:\[(.*?)\])?'
        r'(.*?)'
        r'\\end\{\1\}',
        re.DOTALL | re.IGNORECASE
    )
    RE_TEX_SECTIONS = re.compile(
        r'\\(section|subsection|subsubsection|paragraph)\*?\{((?:[^{}]|\{[^{}]*\})+)\}',
        re.DOTALL
    )
    RE_TEX_CITES = re.compile(r'\\cite[pt]?\*?(?:\[.*?\])*\{.*?\}')
    RE_TEX_REFS = re.compile(r'\\(?:eq)?ref\*?\{.*?\}')
    RE_TEX_LABELS = re.compile(r'\\label\{.*?\}')
    RE_TEX_FORMATTING = re.compile(r'\\(?:textbf|textit|emph|texttt|textsc|underline|textrm)\{((?:[^{}]|\{[^{}]*\})+)\}')
    RE_TEX_MACROS_STRIP = re.compile(
        r'\\(?:documentclass|usepackage|newcommand|def|bibliographystyle|bibliography)\b.*?$',
        re.MULTILINE
    )

    # Markdown Regex Patterns
    RE_MD_FRONTMATTER = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    RE_MD_H1 = re.compile(r'^\#\s+(.+)$', re.MULTILINE)
    RE_MD_SETEXT_H1 = re.compile(r'^([^\n]+)\n={3,}$', re.MULTILINE)
    RE_MD_H2 = re.compile(r'^\#\#\s+(.+)$', re.MULTILINE)
    RE_MD_ALL_HEADERS = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    RE_MD_CODE_FENCE = re.compile(
        r'^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[a-zA-Z0-9_\-\+]*)[^\n\r]*\r?\n'
        r'(?P<body>.*?)'
        r'^[ \t]*(?P=fence)[ \t]*(?:\r?\n|\Z)',
        re.MULTILINE | re.DOTALL
    )
    RE_MD_DISPLAY_MATH = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
    RE_MD_INLINE_MATH = re.compile(r'(?<![\$\\])\$(?![\$\\])(.*?)(?<![\$\\])\$(?![\$\\])', re.DOTALL)
    RE_MD_THEOREM_BOLD = re.compile(
        r'^\s*(?:>\s*)?(?:\*\*|__)'
        r'(?P<type>Theorem|Lemma|Proposition|Corollary|Definition|Axiom|Claim|Conjecture|Remark|Algorithm)'
        r'\b[ \t]*(?P<label>[^\*_]*)'
        r'(?:\*\*|__)'
        r'[ \t]*:?[ \t]*'
        r'(?P<body>[^\n\r]*(?:\r?\n(?![ \t]*(?:#|(?:\*\*|__)|\Z))[^\n\r]+)*)',
        re.MULTILINE | re.IGNORECASE
    )
    RE_MD_HTML_TAGS = re.compile(r'<[^>]+>')
    RE_MD_LINKS = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
    RE_MD_IMAGES = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')

    # Algorithmic signature keywords
    ALGORITHM_KEYWORDS = re.compile(
        r'\b(Algorithm\s+\d+|Require:|Ensure:|def\s+|class\s+|while\s+|for\s+|EMA|Grokfast|EGD|SVD|loss|optimizer|learning_rate|weight_decay)\b',
        re.IGNORECASE
    )

    def parse_file(
        self_or_cls,
        file_path: Optional[Union[Path, str]] = None,
    ) -> Optional[ParsedDocument]:
        """Reads and parses a manuscript file into a ParsedDocument object.
        
        Supports dual invocation: instance method or classmethod.
        """
        if isinstance(self_or_cls, DocumentParser):
            instance = self_or_cls
            target = file_path
        else:
            instance = DocumentParser()
            target = self_or_cls if file_path is None else file_path

        if target is None:
            raise FileNotFoundError("Manuscript file path cannot be None.")

        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Manuscript file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Manuscript path is not a regular file: {path}")

        # Check file size cap
        try:
            size = path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                logger.warning("Skipping oversize file (>10MB): %s (%d bytes)", path, size)
                return None
            if size == 0:
                logger.debug("Skipping 0-byte empty file: %s", path)
                return None
        except OSError as e:
            logger.warning("Error checking file size for %s: %s", path, e)
            return None

        # Robust multi-tier file reading
        raw_text = instance._read_file_safely(path)
        if not raw_text or not raw_text.strip():
            logger.debug("Skipping empty or whitespace-only file: %s", path)
            return None

        return instance.parse_text(raw_text, path=path)

    def parse_text(
        self,
        text: str,
        path: Optional[Path] = None,
        format_hint: Optional[str] = None,
    ) -> ParsedDocument:
        """Parses in-memory manuscript text.
        
        Args:
            text: Raw manuscript text string.
            path: Optional associated filesystem Path.
            format_hint: Optional format override ('latex' or 'markdown').
            
        Returns:
            ParsedDocument object.
        """
        doc_path = path or Path("document.md")
        ext = doc_path.suffix.lower()

        is_latex = (format_hint == "latex") or (ext in {".tex", ".latex", ".ltx"})
        if is_latex:
            return self._parse_latex(text, doc_path)
        return self._parse_markdown(text, doc_path)

    # -------------------------------------------------------------------------
    # Multi-Tier Encoding Safe Reader
    # -------------------------------------------------------------------------

    def _read_file_safely(self, path: Path) -> str:
        """Reads file using multi-tier encoding fallback to ensure zero decoding crashes."""
        # Tier 1: UTF-8 with BOM stripping
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            pass

        # Tier 2: UTF-8 with character replacement
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            pass

        # Tier 3: Latin-1 failsafe (maps all 256 byte values directly)
        try:
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error("Fatal I/O error reading %s: %s", path, e)
            return ""

    # -------------------------------------------------------------------------
    # LaTeX Engine
    # -------------------------------------------------------------------------

    def _strip_tex_comments(self, text: str) -> str:
        """Strips LaTeX comments (%) while preserving literal escaped \\%."""
        lines = []
        for line in text.splitlines():
            idx = 0
            comment_start = None
            while idx < len(line):
                if line[idx] == '%':
                    bs_count = 0
                    b_idx = idx - 1
                    while b_idx >= 0 and line[b_idx] == '\\':
                        bs_count += 1
                        b_idx -= 1
                    if bs_count % 2 == 0:
                        comment_start = idx
                        break
                idx += 1
            lines.append(line[:comment_start] if comment_start is not None else line)
        return "\n".join(lines)

    def _parse_latex(self, raw_text: str, path: Path) -> ParsedDocument:
        """Parses LaTeX manuscripts."""
        # 1. Strip comments
        text_no_comments = self._strip_tex_comments(raw_text)

        # 2. Extract title
        title = self._extract_tex_title(text_no_comments, path)

        # 3. Extract preamble vs body
        body_text = text_no_comments
        doc_begin = text_no_comments.find(r"\begin{document}")
        doc_end = text_no_comments.rfind(r"\end{document}")
        if doc_begin != -1:
            if doc_end != -1 and doc_end > doc_begin:
                body_text = text_no_comments[doc_begin + len(r"\begin{document}"):doc_end]
            else:
                body_text = text_no_comments[doc_begin + len(r"\begin{document}"):]

        # 4. Extract math environments with two-phase masking
        math_blocks = []
        body_masked = body_text

        # Display environments (equation, align, gather, etc.)
        for env_match in self.RE_TEX_MATH_ENV.finditer(body_text):
            inner_math = env_match.group(2).strip()
            math_blocks.append(inner_math if inner_math else env_match.group(0).strip())
        body_masked = self.RE_TEX_MATH_ENV.sub(
            lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
            body_masked
        )

        # Display brackets \[ ... \]
        for br_match in self.RE_TEX_DISPLAY_BRACKETS.finditer(body_masked):
            m = br_match.group(1).strip()
            if m:
                math_blocks.append(m)
        body_masked = self.RE_TEX_DISPLAY_BRACKETS.sub(
            lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
            body_masked
        )

        # Display dollars $$ ... $$
        for dd_match in self.RE_TEX_DISPLAY_DOLLARS.finditer(body_masked):
            m = dd_match.group(1).strip()
            if m:
                math_blocks.append(m)
        body_masked = self.RE_TEX_DISPLAY_DOLLARS.sub(
            lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
            body_masked
        )

        # AMS-LaTeX Inline parentheses \( ... \)
        for ip_match in self.RE_TEX_INLINE_PARENS.finditer(body_masked):
            m = ip_match.group(1).strip()
            if m:
                math_blocks.append(m)
        body_masked = self.RE_TEX_INLINE_PARENS.sub(
            lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
            body_masked
        )

        # Inline math $ ... $
        for in_match in self.RE_TEX_INLINE_MATH.finditer(body_masked):
            m = in_match.group(1).strip()
            if m and not m.startswith("$"):
                math_blocks.append(m)

        # 5. Extract theorem environments
        theorems = []
        for thm_match in self.RE_TEX_THEOREM_ENV.finditer(body_text):
            env_name = thm_match.group(1).capitalize()
            opt_title = thm_match.group(2)
            content = thm_match.group(3).strip()
            if opt_title:
                theorems.append(f"{env_name} [{opt_title.strip()}]: {content}")
            else:
                theorems.append(f"{env_name}: {content}")

        # 6. Extract section tree
        sections = []
        level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
        for sec_match in self.RE_TEX_SECTIONS.finditer(body_text):
            sec_type = sec_match.group(1)
            sec_title = self._clean_tex_snippet(sec_match.group(2))
            sections.append({
                "level": level_map.get(sec_type, 1),
                "title": sec_title,
                "type": sec_type
            })

        # 7. Extract algorithm blocks
        algorithms = []
        re_algo = re.compile(r'\\begin\{algorithmic\}(.*?)\\end\{algorithmic\}', re.DOTALL)
        for alg_match in re_algo.finditer(body_text):
            algorithms.append(alg_match.group(1).strip())

        # 8. Clean prose text
        clean_text = self._clean_tex_prose(body_text)

        return ParsedDocument(
            path=path,
            title=title,
            raw_text=raw_text,
            clean_text=clean_text,
            math_blocks=math_blocks,
            theorems=theorems,
            sections=sections,
            algorithms=algorithms,
            doc_type="latex",
            metadata={"format": "latex"},
            char_count=len(clean_text),
            word_count=len(clean_text.split()),
        )

    def _extract_tex_title(self, text: str, path: Path) -> str:
        """Extracts title from LaTeX \\title{} or falls back to file stem."""
        match = self.RE_TEX_TITLE.search(text)
        if match:
            raw_title = match.group(1)
            clean_title = self._clean_tex_snippet(raw_title)
            if clean_title:
                return clean_title
        return path.stem.replace("_", " ").title()

    def _clean_tex_snippet(self, text: str) -> str:
        """Sanitizes short LaTeX snippets (titles, headers)."""
        text = self.RE_TEX_FORMATTING.sub(r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        text = text.replace('{', '').replace('}', '').replace('\\', '').strip()
        return " ".join(text.split())

    def _clean_tex_prose(self, text: str) -> str:
        """Transforms LaTeX body into clean, readable natural language prose."""
        t = text
        # Remove preambles/macros
        t = self.RE_TEX_MACROS_STRIP.sub('', t)
        # Remove bibliographies
        t = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', t, flags=re.DOTALL)
        # Remove document tags
        t = t.replace(r"\begin{document}", "").replace(r"\end{document}", "")
        # Strip cites, refs, labels
        t = self.RE_TEX_CITES.sub('[cite]', t)
        t = self.RE_TEX_REFS.sub('(ref)', t)
        t = self.RE_TEX_LABELS.sub('', t)
        # Unwrap text formatting
        for _ in range(3):  # Repeat for nested macros
            t = self.RE_TEX_FORMATTING.sub(r'\1', t)
        # Section titles to markdown-like headers
        t = re.sub(r'\\section\*?\{([^}]+)\}', r'\n\n# \1\n', t)
        t = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n\n## \1\n', t)
        t = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n\n### \1\n', t)
        # List items
        t = re.sub(r'\\item\b', '\n- ', t)
        t = re.sub(r'\\(?:begin|end)\{(?:itemize|enumerate|description)\}', '', t)
        # Unescape special characters
        t = t.replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_')
        t = t.replace(r'\#', '#').replace(r'\{', '{').replace(r'\}', '}')
        # Clean extra macros
        t = re.sub(r'\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?', '', t)
        # Clean multiple whitespaces
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()

    # -------------------------------------------------------------------------
    # Markdown Engine
    # -------------------------------------------------------------------------

    def _parse_markdown(self, raw_text: str, path: Path) -> ParsedDocument:
        """Parses Markdown manuscripts with robust code fence and math isolation."""
        metadata = {}
        content = raw_text

        # 1. Extract YAML Frontmatter
        fm_match = self.RE_MD_FRONTMATTER.match(raw_text)
        if fm_match:
            fm_text = fm_match.group(1)
            content = raw_text[fm_match.end():]
            metadata = self._parse_yaml_frontmatter(fm_text)

        # 2. Extract Algorithms from raw code fences prior to masking
        algorithms = []
        for fence_match in self.RE_MD_CODE_FENCE.finditer(content):
            lang = (fence_match.group("lang") or "").lower()
            code_body = fence_match.group("body").strip()
            # Inspect for algorithmic markers
            if (
                lang in {"python", "pseudo", "pseudocode", "algo", "algorithm"}
                or self.ALGORITHM_KEYWORDS.search(code_body)
            ):
                algorithms.append(code_body)

        # 3. Mask code fences to prevent code comments and shell variables from polluting NLP/headers
        content_no_code = self._mask_code_fences(content)

        # 4. Extract Document Title from masked content
        title = self._extract_md_title(content_no_code, metadata, path)

        # 5. Extract Math Blocks (Display & Inline) with two-phase masking
        math_blocks = []
        # Display math $$ ... $$
        for dd_match in self.RE_MD_DISPLAY_MATH.finditer(content_no_code):
            m = dd_match.group(1).strip()
            if m:
                math_blocks.append(m)

        # Mask display math before inline math extraction to prevent delimiter desync
        content_no_display_math = self.RE_MD_DISPLAY_MATH.sub(
            lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)),
            content_no_code
        )

        # Inline math $ ... $
        for in_match in self.RE_MD_INLINE_MATH.finditer(content_no_display_math):
            m = in_match.group(1).strip()
            if m and not m.startswith("$"):
                math_blocks.append(m)

        # 6. Extract Theorems, Axioms, and Propositions with full number preservation
        theorems = []
        for thm_match in self.RE_MD_THEOREM_BOLD.finditer(content_no_code):
            thm_type = thm_match.group("type").capitalize()
            raw_label = (thm_match.group("label") or "").strip()
            clean_label = raw_label.rstrip(':').strip()
            raw_body = thm_match.group("body").strip()

            if raw_body.startswith('>'):
                body_lines = [re.sub(r'^\s*>\s*', '', line) for line in raw_body.splitlines()]
                clean_body = " ".join(line.strip() for line in body_lines if line.strip())
            else:
                clean_body = " ".join(raw_body.split())

            if clean_body:
                if clean_label:
                    theorems.append(f"{thm_type} {clean_label}: {clean_body}")
                else:
                    theorems.append(f"{thm_type}: {clean_body}")

        # Check for LaTeX Axiom/Theorem tags inside Markdown display math
        for m in math_blocks:
            if "Axiom" in m or "Theorem" in m:
                theorems.append(m)

        # 7. Extract Section Hierarchy from masked content
        sections = []
        for h_match in self.RE_MD_ALL_HEADERS.finditer(content_no_code):
            level = len(h_match.group(1))
            sec_title = h_match.group(2).strip()
            sections.append({
                "level": level,
                "title": sec_title
            })

        # 8. Clean prose text from masked content
        clean_text = self._clean_md_prose(content_no_code)

        return ParsedDocument(
            path=path,
            title=title,
            raw_text=raw_text,
            clean_text=clean_text,
            math_blocks=math_blocks,
            theorems=theorems,
            sections=sections,
            algorithms=algorithms,
            doc_type="markdown",
            metadata=metadata,
            char_count=len(clean_text),
            word_count=len(clean_text.split()),
        )

    def _mask_code_fences(self, text: str) -> str:
        """Masks code fences by replacing their text with equivalent newlines.
        
        Neutralizes shell comments (# ...), environment variables ($VAR),
        and formatting constructs inside code fences while preserving exact
        line numbering and document structure for downstream extractors.
        """
        def _replacer(m: re.Match) -> str:
            matched = m.group(0)
            newlines = matched.count('\n')
            nl = '\r\n' if '\r\n' in matched else '\n'
            return nl * newlines

        return self.RE_MD_CODE_FENCE.sub(_replacer, text)

    def _parse_yaml_frontmatter(self, yaml_text: str) -> Dict[str, Any]:
        """Simple YAML frontmatter key-value parser without external dependencies."""
        data = {}
        for line in yaml_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                clean_key = key.strip()
                clean_val = val.strip().strip("'\"")
                data[clean_key] = clean_val
        return data

    def _extract_md_title(self, text: str, metadata: Dict[str, Any], path: Path) -> str:
        """Resolves Markdown document title via priority hierarchy."""
        if "title" in metadata and metadata["title"]:
            return metadata["title"]

        h1_match = self.RE_MD_H1.search(text)
        if h1_match:
            return h1_match.group(1).strip()

        setext_match = self.RE_MD_SETEXT_H1.search(text)
        if setext_match:
            return setext_match.group(1).strip()

        h2_match = self.RE_MD_H2.search(text)
        if h2_match:
            return h2_match.group(1).strip()

        return path.stem.replace("_", " ").replace("-", " ").title()

    def _clean_md_prose(self, text: str) -> str:
        """Sanitizes Markdown formatting into readable plain prose."""
        t = text
        # Strip code fences from prose so code doesn't pollute NLP analysis
        t = self.RE_MD_CODE_FENCE.sub('', t)
        # Strip HTML tags
        t = self.RE_MD_HTML_TAGS.sub('', t)
        # Strip images
        t = self.RE_MD_IMAGES.sub('', t)
        # Unwrap links [text](url) -> text
        t = self.RE_MD_LINKS.sub(r'\1', t)
        # Unwrap bold / italic styling
        t = re.sub(r'(\*\*|__)(.*?)\1', r'\2', t)
        t = re.sub(r'(\*|_)(.*?)\1', r'\2', t)
        # Clean blockquotes
        t = re.sub(r'^\s*>\s*', '', t, flags=re.MULTILINE)
        # Clean multiple newlines
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip()


# Universal alias for interface contract compliance
Parser = DocumentParser
