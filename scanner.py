"""
gestalt_extractor.scanner
=========================
Filesystem manuscript discovery engine with in-place subtree pruning,
strict blacklist filtering, and defensive edge case containment.
"""

import os
import sys
import fnmatch
import logging
from pathlib import Path
from typing import List, Set, Sequence, Optional, Union, Any

from gestalt_extractor.config import (
    ScannerConfig,
    DEFAULT_EXCLUDES,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class ScannerError(Exception):
    """Base exception for all scanner failures."""
    pass


class TargetNotFoundError(ScannerError, FileNotFoundError):
    """Raised when the specified target directory or file does not exist."""
    pass


class InvalidTargetError(ScannerError, ValueError):
    """Raised when target exists but is not a directory or supported manuscript."""
    pass


class ScanPermissionError(ScannerError, PermissionError):
    """Raised when directory scanning is completely blocked by OS permissions."""
    pass


# =============================================================================
# Scanner Engine
# =============================================================================

class DirectoryScanner:
    """
    High-performance directory crawler for LaTeX and Markdown manuscripts.
    
    Features:
    - O(1) in-place subtree pruning for blacklisted directories (node_modules, .agents, .git, etc.)
    - Case-insensitive extension matching (.tex, .md)
    - Defensive handling of symlinks, permission errors, and missing directories
    - Deterministically sorted output paths
    """

    def __init__(
        self,
        config: Optional[ScannerConfig] = None,
        excludes: Optional[Sequence[str]] = None,
        supported_extensions: Optional[Sequence[str]] = None,
        follow_symlinks: bool = False,
        ignore_hidden: bool = True,
        max_file_size_bytes: Optional[int] = None,
        exclude_dir_names: Optional[Sequence[str]] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            eff_excludes = list(excludes) if excludes is not None else list(DEFAULT_EXCLUDES)
            eff_exts = tuple(supported_extensions) if supported_extensions is not None else SUPPORTED_EXTENSIONS
            eff_max_size = max_file_size_bytes if max_file_size_bytes is not None else MAX_FILE_SIZE_BYTES
            self.config = ScannerConfig(
                supported_extensions=eff_exts,
                excludes=eff_excludes,
                follow_symlinks=follow_symlinks,
                ignore_hidden=ignore_hidden,
                max_file_size_bytes=eff_max_size,
                exclude_dir_names=exclude_dir_names,
            )

        self._normalized_exts: Set[str] = self.config.normalized_extensions
        self._exclude_dirs: Set[str] = self.config.exclude_dir_names
        self._exclude_globs: List[str] = self.config.exclude_glob_patterns

    def is_supported_extension(self, path: Union[str, Path]) -> bool:
        """
        Checks if file path matches supported extensions case-insensitively.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        return suffix in self._normalized_exts

    def is_excluded(
        self,
        path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        extra_excludes: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Determines whether a given path must be excluded.
        
        Evaluates:
        1. Subtree directory names against blacklisted components.
        2. Relative path string and file name against fnmatch glob patterns and path prefixes.
        3. Hidden file prefixes if ignore_hidden is active.
        """
        p = Path(path)
        base = Path(base_dir) if base_dir is not None else None

        # Hidden file rule
        if self.config.ignore_hidden and p.name.startswith("."):
            return True

        # Compute relative parts if within base_dir
        if base is not None:
            try:
                rel = p.relative_to(base)
                parts = rel.parts
            except ValueError:
                parts = p.parts
                rel = p
        else:
            parts = p.parts
            rel = p

        # Check path components against exact blacklist dir names
        for part in parts[:-1]:
            part_lower = part.lower()
            if part_lower in self._exclude_dirs:
                return True

        rel_posix = rel.as_posix().lower()
        file_name = p.name.lower()

        # Combine configured patterns and extra excludes
        all_patterns = list(self.config.excludes)
        if extra_excludes:
            all_patterns.extend(extra_excludes)

        for pat in all_patterns:
            pat_str = str(pat).strip()
            if not pat_str:
                continue

            pat_lower = pat_str.lower()
            has_wildcard = "*" in pat_lower or "?" in pat_lower
            is_anchored = pat_lower.startswith("/")
            clean_pat = pat_lower.replace("\\", "/").strip("/")

            # 1. Exact directory name match (e.g. "node_modules")
            if not has_wildcard and "/" not in clean_pat:
                if any(part.lower() == clean_pat for part in parts[:-1]):
                    return True
                if file_name == clean_pat:
                    return True

            # 2. Path prefix / Subdirectory match (e.g. "docs/private")
            if not has_wildcard:
                if rel_posix == clean_pat or rel_posix.startswith(f"{clean_pat}/"):
                    return True
                if not is_anchored and f"/{clean_pat}/" in f"/{rel_posix}":
                    return True

            # 3. Wildcard / Glob pattern match
            if has_wildcard:
                if (
                    fnmatch.fnmatch(file_name, pat_lower)
                    or fnmatch.fnmatch(rel_posix, pat_lower)
                    or fnmatch.fnmatch(f"/{rel_posix}", pat_lower)
                    or fnmatch.fnmatch(f"{rel_posix}/", pat_lower)
                    or fnmatch.fnmatch(f"/{rel_posix}/", pat_lower)
                ):
                    return True

                # Segment wildcard check for **/name/**
                if clean_pat.startswith("**/") and clean_pat.endswith("/**"):
                    seg = clean_pat[3:-3]
                    if any(part.lower() == seg for part in parts[:-1]):
                        return True
                elif clean_pat.startswith("**/"):
                    seg = clean_pat[3:]
                    if any(part.lower() == seg for part in parts[:-1]):
                        return True
                elif clean_pat.endswith("/**"):
                    seg = clean_pat[:-3]
                    if any(part.lower() == seg for part in parts[:-1]):
                        return True

        return False

    def _should_prune_dir(
        self,
        dir_name: str,
        dir_path: Path,
        base_dir: Path,
        extra_excludes: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Evaluates whether an encountered directory should be pruned in-place
        during os.walk.
        """
        dir_lower = dir_name.lower()

        # Hidden directory rule
        if self.config.ignore_hidden and dir_name.startswith("."):
            return True

        # Exact component blacklist match
        if dir_lower in self._exclude_dirs:
            return True

        # Symlink directory check when follow_symlinks is False
        if not self.config.follow_symlinks and dir_path.is_symlink():
            return True

        try:
            rel_dir_posix = dir_path.relative_to(base_dir).as_posix().lower()
        except ValueError:
            rel_dir_posix = dir_lower

        all_patterns = list(self.config.excludes)
        if extra_excludes:
            all_patterns.extend(extra_excludes)

        for pat in all_patterns:
            pat_str = str(pat).strip()
            if not pat_str:
                continue

            pat_lower = pat_str.lower()
            has_wildcard = "*" in pat_lower or "?" in pat_lower
            is_anchored = pat_lower.startswith("/")
            clean_pat = pat_lower.replace("\\", "/").strip("/")

            # 1. Exact directory name match
            if not has_wildcard and "/" not in clean_pat:
                if dir_lower == clean_pat:
                    return True

            # 2. Subdirectory / Path prefix match (e.g. "docs/private")
            if not has_wildcard:
                if rel_dir_posix == clean_pat or rel_dir_posix.startswith(f"{clean_pat}/"):
                    return True
                if not is_anchored and f"/{clean_pat}/" in f"/{rel_dir_posix}/":
                    return True

            # 3. Glob matching
            if has_wildcard:
                if (
                    fnmatch.fnmatch(dir_lower, pat_lower)
                    or fnmatch.fnmatch(rel_dir_posix, pat_lower)
                    or fnmatch.fnmatch(f"/{rel_dir_posix}", pat_lower)
                    or fnmatch.fnmatch(f"/{rel_dir_posix}/", pat_lower)
                    or fnmatch.fnmatch(f"{rel_dir_posix}/", pat_lower)
                ):
                    return True

                if clean_pat.startswith("**/") and clean_pat.endswith("/**"):
                    seg = clean_pat[3:-3]
                    if dir_lower == seg or f"/{seg}/" in f"/{rel_dir_posix}/":
                        return True
                elif clean_pat.startswith("**/"):
                    seg = clean_pat[3:]
                    if dir_lower == seg or f"/{seg}/" in f"/{rel_dir_posix}/":
                        return True
                elif clean_pat.endswith("/**"):
                    seg = clean_pat[:-3]
                    if dir_lower == seg or f"/{seg}/" in f"/{rel_dir_posix}/":
                        return True

        return False

    def _scan_single_file(
        self,
        file_path: Path,
        extra_excludes: Optional[Sequence[str]] = None,
    ) -> List[Path]:
        """Validates single file input mode."""
        if not file_path.exists():
            raise TargetNotFoundError(f"Target file does not exist: {file_path}")
        if not file_path.is_file():
            raise InvalidTargetError(f"Target path is not a regular file: {file_path}")
        if not self.is_supported_extension(file_path):
            logger.debug("Single file target %s does not match supported extensions.", file_path)
            return []
        if self.is_excluded(file_path, extra_excludes=extra_excludes):
            logger.warning("Target file %s matches exclusion rules. Returning empty queue.", file_path)
            return []
        try:
            if (
                self.config.max_file_size_bytes is not None
                and self.config.max_file_size_bytes >= 0
                and file_path.stat().st_size > self.config.max_file_size_bytes
            ):
                logger.warning(
                    "Target file %s exceeds maximum allowed size (%d > %d bytes). Returning empty queue.",
                    file_path,
                    file_path.stat().st_size,
                    self.config.max_file_size_bytes,
                )
                return []
        except OSError as err:
            logger.warning("Could not stat single file target %s: %s", file_path, err)
            return []

        return [file_path.resolve()]

    def scan(
        self,
        target: Optional[Union[str, Path]] = None,
        custom_excludes: Optional[Sequence[str]] = None,
        excludes: Optional[Sequence[str]] = None,
    ) -> List[Path]:
        """
        Recursively scans target for .tex and .md manuscripts.
        
        Parameters:
            target: Directory path or single manuscript file path.
            custom_excludes: Optional extra exclusion patterns for this run.
            excludes: Alias for custom_excludes.
            
        Returns:
            List[Path]: Deterministically sorted list of resolved Path instances.
            
        Raises:
            TargetNotFoundError: If target path does not exist.
            InvalidTargetError: If target path is invalid.
        """
        eff_target = target
        eff_excludes = custom_excludes if custom_excludes is not None else excludes

        if not eff_target:
            raise TargetNotFoundError("Target path cannot be empty or None.")

        target_path = Path(eff_target)
        if not target_path.exists():
            raise TargetNotFoundError(f"Target directory or file does not exist: {target_path}")

        if target_path.is_file():
            return self._scan_single_file(target_path, extra_excludes=eff_excludes)

        if not target_path.is_dir():
            raise InvalidTargetError(f"Target path is neither a file nor a directory: {target_path}")

        results: List[Path] = []
        visited_nodes: Set[tuple] = set()

        def _on_walk_error(err: OSError) -> None:
            logger.warning("Permission denied or access error during scan: %s", err)

        resolved_target = target_path.resolve()

        for root_str, dirs, files in os.walk(
            resolved_target,
            followlinks=self.config.follow_symlinks,
            onerror=_on_walk_error,
        ):
            root_path = Path(root_str)

            # Detect symlink cycles if follow_symlinks is enabled
            if self.config.follow_symlinks:
                try:
                    stat_info = root_path.stat()
                    node_id = (stat_info.st_dev, stat_info.st_ino)
                    if node_id in visited_nodes:
                        logger.warning("Symlink cycle detected at %s. Pruning subtree.", root_path)
                        dirs[:] = []
                        continue
                    visited_nodes.add(node_id)
                except OSError:
                    dirs[:] = []
                    continue

            # In-place directory pruning
            dirs[:] = [
                d for d in dirs
                if not self._should_prune_dir(d, root_path / d, resolved_target, extra_excludes=eff_excludes)
            ]

            # Process files
            for file_name in files:
                file_path = root_path / file_name

                if not self.is_supported_extension(file_path):
                    continue

                if self.is_excluded(file_path, base_dir=resolved_target, extra_excludes=eff_excludes):
                    continue

                if not self.config.follow_symlinks and file_path.is_symlink():
                    continue

                try:
                    if not file_path.is_file():
                        continue
                    if (
                        self.config.max_file_size_bytes is not None
                        and self.config.max_file_size_bytes >= 0
                        and file_path.stat().st_size > self.config.max_file_size_bytes
                    ):
                        logger.debug(
                            "File %s exceeds maximum allowed size (%d > %d bytes). Skipping.",
                            file_path,
                            file_path.stat().st_size,
                            self.config.max_file_size_bytes,
                        )
                        continue
                except OSError as err:
                    logger.warning("Failed to access or stat file %s: %s", file_path, err)
                    continue

                results.append(file_path.resolve())

        # Return deterministically sorted list
        results.sort(key=lambda p: str(p).lower())
        return results

    @classmethod
    def scan_path(
        cls,
        target: Union[str, Path],
        excludes: Optional[Sequence[str]] = None,
        custom_excludes: Optional[Sequence[str]] = None,
        config: Optional[ScannerConfig] = None,
        **kwargs,
    ) -> List[Path]:
        """Convenience classmethod satisfying Scanner.scan(target_dir, excludes) contract."""
        eff_excludes = custom_excludes if custom_excludes is not None else excludes
        scanner = cls(config=config, excludes=eff_excludes, **kwargs)
        return scanner.scan(target, custom_excludes=eff_excludes)


# Class and functional aliases for universal contract compliance
Scanner = DirectoryScanner


def scan_directory(
    target_dir: Union[str, Path],
    custom_excludes: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[Path]:
    """Top-level convenience function scanning target directory."""
    eff_excludes = custom_excludes if custom_excludes is not None else excludes
    return DirectoryScanner.scan_path(target_dir, excludes=eff_excludes, **kwargs)
