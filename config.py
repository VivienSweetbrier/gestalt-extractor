"""
gestalt_extractor.config
========================
Configuration constants, domain taxonomies, and dataclass models for the
Gestalt Extractor pipeline.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict, Optional, Sequence, Union

# =============================================================================
# Filesystem Ingestion Defaults
# =============================================================================

DEFAULT_EXCLUDE_DIR_NAMES: List[str] = [
    "node_modules",
    "bower_components",
    ".agents",
    ".git",
    ".svn",
    ".hg",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "venv",
    ".venv",
    "env",
    ".env",
    ".idea",
    ".vscode",
    ".vs",
    ".cache",
    ".gemini",
]

DEFAULT_EXCLUDE_GLOB_PATTERNS: List[str] = [
    "**/node_modules/**",
    "**/.agents/**",
    "**/.git/**",
    "**/build/**",
    "**/dist/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/env/**",
    "*.egg-info",
    "*.egg-info/*",
    "*.dist-info",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",
]

# Combined default exclusion list (using List[str] to support concatenation with list overrides)
DEFAULT_EXCLUDES: List[str] = DEFAULT_EXCLUDE_DIR_NAMES + [
    "*.egg-info",
    "*.egg-info/*",
    "*.dist-info",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",
]

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".tex", ".md")

# =============================================================================
# Mathematical Domain Taxonomy
# =============================================================================

DEFAULT_DOMAINS: Tuple[str, ...] = (
    "ML_Optimization",
    "Graph_Combinatorics",
    "Crypto_Number_Theory",
    "Fluid_Dynamics",
    "General_Applied_Math",
)

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "ML_Optimization": [
        r"(?i)convex",
        r"(?i)gradient",
        r"(?i)loss function",
        r"(?i)learning rate",
        r"(?i)weight decay",
        r"(?i)adamw",
        r"(?i)grokfast",
        r"(?i)momentum",
        r"(?i)backpropagation",
        r"(?i)stochastic gradient",
        r"(?i)eigenvalue",
        r"(?i)singular value",
        r"(?i)transformer",
        r"(?i)attention",
        r"(?i)bits/parameter",
    ],
    "Graph_Combinatorics": [
        r"(?i)vertex",
        r"(?i)vertices",
        r"(?i)edge",
        r"(?i)clique",
        r"(?i)network",
        r"(?i)adjacency",
        r"(?i)directed acyclic graph",
        r"(?i)dag",
        r"(?i)topological",
        r"(?i)graph isomorphism",
        r"(?i)bipartite",
        r"(?i)structural causal model",
    ],
    "Crypto_Number_Theory": [
        r"(?i)finite field",
        r"(?i)prime",
        r"(?i)l-function",
        r"(?i)zeta",
        r"(?i)elliptic",
        r"(?i)residue",
        r"(?i)discrete log",
        r"(?i)modulus",
        r"(?i)isogeny",
        r"(?i)zero-knowledge",
    ],
    "Fluid_Dynamics": [
        r"(?i)navier-stokes",
        r"(?i)fluid",
        r"(?i)viscosity",
        r"(?i)turbulence",
        r"(?i)velocity field",
        r"(?i)reynolds number",
        r"(?i)incompressible",
        r"(?i)boundary layer",
        r"(?i)vorticity",
    ],
}

# =============================================================================
# Safety Caps & Limits
# =============================================================================

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB safety cap
MAX_PARSED_CHARS: int = 150_000               # Token bounding ceiling
DEFAULT_MAX_WORKERS: int = min(16, max(1, (os.cpu_count() or 1) * 2))
DEFAULT_OUTPUT_FILE: str = "gestalt_extraction_report.json"

# =============================================================================
# LLM Settings & Environment Variables
# =============================================================================

DEFAULT_LLM_MODEL: str = "gpt-4o"
DEFAULT_REQUEST_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 2
ENV_OPENAI_API_KEY: str = "OPENAI_API_KEY"
ENV_GESTALT_MOCK: str = "GESTALT_MOCK"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ScannerConfig:
    """Configuration governing directory discovery and path exclusion rules."""
    supported_extensions: Sequence[str] = SUPPORTED_EXTENSIONS
    excludes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    follow_symlinks: bool = False
    ignore_hidden: bool = True
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    custom_exclude_dir_names: Optional[Sequence[str]] = None

    def __init__(
        self,
        supported_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
        excludes: Optional[Sequence[str]] = None,
        follow_symlinks: bool = False,
        ignore_hidden: bool = True,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
        exclude_dir_names: Optional[Sequence[str]] = None,
        custom_exclude_dir_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.supported_extensions = supported_extensions
        self.excludes = list(DEFAULT_EXCLUDES) if excludes is None else list(excludes)
        self.follow_symlinks = follow_symlinks
        self.ignore_hidden = ignore_hidden
        self.max_file_size_bytes = max_file_size_bytes
        self.custom_exclude_dir_names = (
            exclude_dir_names if exclude_dir_names is not None else custom_exclude_dir_names
        )

    @property
    def normalized_extensions(self) -> Set[str]:
        """Returns lowercase extensions with leading dots."""
        return {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.supported_extensions
        }

    @property
    def exclude_dir_names(self) -> Set[str]:
        """Set of directory names for O(1) in-place subtree pruning.
        
        If custom_exclude_dir_names is explicitly passed, respects that set.
        Otherwise extracts directory names directly from self.excludes without forced defaults.
        """
        if self.custom_exclude_dir_names is not None:
            return {
                d.lower().strip().strip("/").replace("\\", "/")
                for d in self.custom_exclude_dir_names
                if d and d.strip()
            }

        names = set()
        for name in self.excludes:
            clean = name.strip().strip("/")
            if clean.startswith("**/") and clean.endswith("/**"):
                clean = clean[3:-3]
            elif clean.startswith("**/"):
                clean = clean[3:]
            elif clean.endswith("/**"):
                clean = clean[:-3]
            if not ("*" in clean or "?" in clean or "/" in clean or "\\" in clean):
                names.add(clean.lower())
        return names

    @property
    def exclude_glob_patterns(self) -> List[str]:
        """Glob patterns for fnmatch filtering."""
        return [
            pat for pat in self.excludes
            if ("*" in pat or "?" in pat or "/" in pat or "\\" in pat)
        ]


@dataclass
class ExtractorConfig:
    """Master configuration for the Gestalt Extractor pipeline."""
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    output_file: str = DEFAULT_OUTPUT_FILE
    mock: bool = False
    model: str = DEFAULT_LLM_MODEL
    max_workers: int = DEFAULT_MAX_WORKERS
    timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    verbose: bool = False

    def is_mock_enabled(self) -> bool:
        """Determines whether mock mode should be enforced."""
        if self.mock:
            return True
        if os.environ.get(ENV_GESTALT_MOCK, "").strip().lower() in ("1", "true", "yes"):
            return True
        key = self.api_key or os.environ.get(ENV_OPENAI_API_KEY, "")
        return not bool(key and key.strip())


def get_default_config() -> ExtractorConfig:
    """Returns a freshly instantiated default ExtractorConfig."""
    return ExtractorConfig()
