"""
Gestalt Extractor
=================
Automated manuscript ingestion, classification, and mathematical bound extraction.
"""

from gestalt_extractor.config import (
    DEFAULT_EXCLUDES,
    SUPPORTED_EXTENSIONS,
    DEFAULT_DOMAINS,
    ScannerConfig,
    ExtractorConfig,
    get_default_config,
)
from gestalt_extractor.scanner import Scanner, DirectoryScanner, scan_directory
from gestalt_extractor.parser import DocumentParser, Parser, ParsedDocument
from gestalt_extractor.classifier import (
    DomainClassifier,
    Classifier,
    classify_document,
    classify_document_with_scores,
)
from gestalt_extractor.models import (
    ExtractedBound,
    ExtractedRecord,
    ExtractionReport,
    ExtractionResult,
    Bound,
    Report,
    PYDANTIC_AVAILABLE,
)
from gestalt_extractor.llm import (
    LLMProvider,
    OpenAIProvider,
    DeterministicMockProvider,
    MockProvider,
    get_provider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    SchemaValidationError,
)
from gestalt_extractor.extractor import (
    GestaltExtractor,
    extract_document,
    extract_file,
    extract_text,
)
from gestalt_extractor.pipeline import (
    ExtractionPipeline,
    run_extraction,
)

__all__ = [
    # Config & Constants
    "DEFAULT_EXCLUDES",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_DOMAINS",
    "ScannerConfig",
    "ExtractorConfig",
    "get_default_config",
    # Scanner
    "Scanner",
    "DirectoryScanner",
    "scan_directory",
    # Parser
    "DocumentParser",
    "Parser",
    "ParsedDocument",
    # Classifier
    "DomainClassifier",
    "Classifier",
    "classify_document",
    "classify_document_with_scores",
    # Models
    "ExtractedBound",
    "ExtractedRecord",
    "ExtractionReport",
    "ExtractionResult",
    "Bound",
    "Report",
    "PYDANTIC_AVAILABLE",
    # LLM
    "LLMProvider",
    "OpenAIProvider",
    "DeterministicMockProvider",
    "MockProvider",
    "get_provider",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "SchemaValidationError",
    # Extractor
    "GestaltExtractor",
    "extract_document",
    "extract_file",
    "extract_text",
    # Pipeline
    "ExtractionPipeline",
    "run_extraction",
]
