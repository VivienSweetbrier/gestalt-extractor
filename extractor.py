"""
gestalt_extractor.extractor
===========================
High-level extraction orchestrator tying DocumentParser, DomainClassifier,
and LLMProvider together into a unified manuscript extraction engine.

Guarantees output records strictly adhere to the mandatory 4-key schema:
`{"file": str, "domain": str, "bounds": List[ExtractedBound], "summary": str}`
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Sequence

from gestalt_extractor.config import ExtractorConfig, get_default_config
from gestalt_extractor.parser import DocumentParser, ParsedDocument
from gestalt_extractor.classifier import DomainClassifier
from gestalt_extractor.llm import LLMProvider, get_provider
from gestalt_extractor.models import (
    ExtractedBound,
    ExtractedRecord,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


class GestaltExtractor:
    """High-level manuscript extractor coordinating parsing, classification, and LLM bound extraction.

    Coordinates:
    - DocumentParser (sanitization, math blocks, theorem extraction)
    - DomainClassifier (heuristic and formula taxonomy assignment)
    - LLMProvider (OpenAI API with automatic mock fallback or forced mock)
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        parser: Optional[DocumentParser] = None,
        classifier: Optional[DomainClassifier] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.config = config or get_default_config()
        self.parser = parser or DocumentParser()
        self.classifier = classifier or DomainClassifier(config=self.config)
        self.provider = provider or get_provider(self.config)

    def extract_document(self, doc: ParsedDocument) -> ExtractedRecord:
        """Extracts computational bounds, semantic summary, and domain from a ParsedDocument.

        Args:
            doc: Ingested and parsed manuscript representation.

        Returns:
            Validated ExtractedRecord containing mandatory keys:
            `file`, `domain`, `bounds`, and `summary`.

        Raises:
            ValueError: If doc is None.
            SchemaValidationError: If constructed record violates schema.
        """
        if doc is None:
            raise ValueError("ParsedDocument cannot be None.")

        doc_name = doc.path.name if (doc.path and hasattr(doc.path, "name")) else "document"
        logger.debug("Extracting bounds and domain for manuscript: %s", doc_name)

        # 1. Classify mathematical domain
        domain = self.classifier.classify(doc)

        # 2. Invoke LLM provider (or deterministic mock provider)
        raw_payload = self.provider.extract(doc, domain)

        # 3. Normalize bounds into ExtractedBound instances
        norm_bounds = []
        raw_bounds = raw_payload.get("bounds", []) if isinstance(raw_payload, dict) else []
        for b in raw_bounds:
            if isinstance(b, ExtractedBound):
                norm_bounds.append(b)
            elif isinstance(b, dict):
                norm_bounds.append(ExtractedBound.from_dict(b))
            elif isinstance(b, str) and b.strip():
                norm_bounds.append(
                    ExtractedBound(
                        name="Mathematical Bound",
                        formula=b.strip(),
                        type="algorithmic_bound",
                        context="",
                    )
                )

        # 4. Normalize summary
        summary = ""
        if isinstance(raw_payload, dict):
            summary = str(raw_payload.get("summary") or "").strip()
        if not summary:
            summary = f"Mathematical bounds extracted from {doc_name} for domain {domain}."

        # 5. Extract optional metadata
        methodologies = []
        metadata = {}
        if isinstance(raw_payload, dict):
            raw_methods = raw_payload.get("methodologies")
            if isinstance(raw_methods, list):
                methodologies = [str(m).strip() for m in raw_methods if str(m).strip()]
            raw_meta = raw_payload.get("metadata")
            if isinstance(raw_meta, dict):
                metadata = dict(raw_meta)

        if doc.metadata and isinstance(doc.metadata, dict):
            metadata.setdefault("doc_metadata", doc.metadata)
        metadata.setdefault("title", doc.title)
        metadata.setdefault("char_count", doc.char_count)
        metadata.setdefault("word_count", doc.word_count)

        path_str = str(doc.path.resolve()) if hasattr(doc.path, "resolve") else str(doc.path)

        # 6. Assemble and validate ExtractedRecord
        record = ExtractedRecord(
            file=doc_name,
            domain=domain,
            bounds=norm_bounds,
            summary=summary,
            path=path_str,
            methodologies=methodologies,
            metadata=metadata,
        )

        return record

    def extract_file(self, file_path: Union[str, Path]) -> ExtractedRecord:
        """Parses a manuscript from disk and extracts bounds and semantic metadata.

        Args:
            file_path: Filesystem path to a .tex or .md manuscript.

        Returns:
            Validated ExtractedRecord instance.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If file is unreadable or unsupported.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Manuscript file does not exist: {p}")
        if not p.is_file():
            raise ValueError(f"Manuscript path is not a regular file: {p}")

        doc = self.parser.parse_file(p)
        if doc is None:
            # Handle empty or oversized files gracefully with an empty ExtractedRecord
            logger.warning("File %s yielded empty ParsedDocument. Generating empty record.", p)
            return ExtractedRecord(
                file=p.name,
                domain="General_Applied_Math",
                bounds=[],
                summary="Manuscript is empty or could not be parsed.",
                path=str(p.resolve()),
                metadata={"status": "empty_or_unparseable"},
            )

        return self.extract_document(doc)

    def extract_text(
        self,
        text: str,
        filename: str = "snippet.md",
    ) -> ExtractedRecord:
        """Parses in-memory text and extracts computational bounds.

        Args:
            text: Manuscript text content.
            filename: Synthetic filename for format hinting and record attribution.

        Returns:
            Validated ExtractedRecord instance.
        """
        path = Path(filename)
        doc = self.parser.parse_text(text, path=path)
        return self.extract_document(doc)


# Top-level functional helpers
def extract_document(
    doc: ParsedDocument,
    config: Optional[ExtractorConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> ExtractedRecord:
    """Convenience helper to extract from an existing ParsedDocument."""
    extractor = GestaltExtractor(config=config, provider=provider)
    return extractor.extract_document(doc)


def extract_file(
    file_path: Union[str, Path],
    config: Optional[ExtractorConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> ExtractedRecord:
    """Convenience helper to extract from a manuscript file path."""
    extractor = GestaltExtractor(config=config, provider=provider)
    return extractor.extract_file(file_path)


def extract_text(
    text: str,
    filename: str = "snippet.md",
    config: Optional[ExtractorConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> ExtractedRecord:
    """Convenience helper to extract from manuscript text."""
    extractor = GestaltExtractor(config=config, provider=provider)
    return extractor.extract_text(text, filename=filename)


__all__ = [
    "GestaltExtractor",
    "extract_document",
    "extract_file",
    "extract_text",
]
