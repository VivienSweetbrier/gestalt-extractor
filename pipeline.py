"""
gestalt_extractor.pipeline
==========================
Batch extraction pipeline orchestrating filesystem discovery, concurrent
multithreaded manuscript processing, thread-safe error isolation,
and atomic structured JSON report serialization.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union, Sequence, List, Dict, Any

from gestalt_extractor.config import ExtractorConfig, get_default_config
from gestalt_extractor.scanner import DirectoryScanner
from gestalt_extractor.extractor import GestaltExtractor
from gestalt_extractor.models import (
    ExtractedRecord,
    ExtractionReport,
)

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Batch manuscript processing pipeline with thread-pooled execution and atomic output serialization.

    Features:
    - Scans target paths using DirectoryScanner with exclusion filtering (.agents, node_modules, etc.).
    - Concurrently processes manuscripts via ThreadPoolExecutor.
    - Thread-safe error containment: failures on individual files are logged and recorded without halting batch.
    - Serializes aggregated results atomically to disk via ExtractionReport.save().
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        extractor: Optional[GestaltExtractor] = None,
        scanner: Optional[DirectoryScanner] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.config = config or get_default_config()
        self.extractor = extractor or GestaltExtractor(config=self.config)
        self.scanner = scanner or DirectoryScanner(config=self.config.scanner)
        self.max_workers = max_workers or self.config.max_workers

    def run(
        self,
        target_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        excludes: Optional[Sequence[str]] = None,
    ) -> ExtractionReport:
        """Discovers manuscripts in target path and runs concurrent batch extraction.

        Args:
            target_dir: Target directory path or single file path.
            output_file: Optional path to save the resulting JSON report.
            excludes: Optional custom exclusion patterns for this execution.

        Returns:
            Aggregated ExtractionReport instance.
        """
        target = Path(target_dir)
        logger.info("Starting Gestalt extraction pipeline on target: %s", target)

        # 1. Discover manuscripts
        file_paths = self.scanner.scan(target, custom_excludes=excludes)
        logger.info("Discovered %d candidate manuscript(s) for extraction.", len(file_paths))

        if not file_paths:
            logger.warning("No valid manuscripts found in target: %s", target)
            report = ExtractionReport(
                records=[],
                metadata={
                    "target": str(target),
                    "total_found": 0,
                    "successful": 0,
                    "failed": 0,
                    "errors": [],
                },
            )
            eff_output = output_file or self.config.output_file
            if eff_output:
                report.save(eff_output, indent=2, as_envelope=False)
            return report

        # 2. Execute batch extraction
        return self.run_batch(file_paths, output_file=output_file)

    def run_batch(
        self,
        file_paths: Sequence[Union[str, Path]],
        output_file: Optional[Union[str, Path]] = None,
    ) -> ExtractionReport:
        """Concurrently extracts bounds from a designated sequence of file paths.

        Thread-safe: errors occurring during individual file extraction are caught,
        logged, and attached to report metadata without causing pipeline termination.

        Args:
            file_paths: Sequence of manuscript file paths to process.
            output_file: Optional path to save the resulting JSON report.

        Returns:
            Aggregated ExtractionReport instance containing all extracted records.
        """
        paths = [Path(p) for p in file_paths]
        total_files = len(paths)
        workers = max(1, min(self.max_workers, total_files))

        logger.info("Executing batch extraction for %d files across %d worker threads.", total_files, workers)

        records: List[ExtractedRecord] = []
        errors: List[Dict[str, Any]] = []

        if workers == 1 or total_files <= 1:
            # Sequential execution for single-file or single-worker scenarios
            for p in paths:
                try:
                    record = self.extractor.extract_file(p)
                    if record is not None:
                        records.append(record)
                except Exception as exc:
                    logger.error("Failed to process manuscript '%s': %s", p, exc, exc_info=True)
                    errors.append({
                        "file": str(p),
                        "error": str(exc),
                        "type": type(exc).__name__,
                    })
        else:
            # Concurrent execution via ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_path = {
                    executor.submit(self.extractor.extract_file, p): p
                    for p in paths
                }

                for future in as_completed(future_to_path):
                    p = future_to_path[future]
                    try:
                        record = future.result()
                        if record is not None:
                            records.append(record)
                    except Exception as exc:
                        logger.error("Worker error processing manuscript '%s': %s", p, exc, exc_info=True)
                        errors.append({
                            "file": str(p),
                            "error": str(exc),
                            "type": type(exc).__name__,
                        })

        # Deterministically sort records by file name
        records.sort(key=lambda r: str(r.file).lower())

        # Build aggregated report
        report = ExtractionReport(
            records=records,
            metadata={
                "total_files": total_files,
                "successful": len(records),
                "failed": len(errors),
                "errors": errors,
            },
        )

        logger.info(
            "Batch extraction completed: %d succeeded, %d failed.",
            len(records),
            len(errors),
        )

        # 3. Atomic persistence to disk if requested
        eff_output = output_file or self.config.output_file
        if eff_output:
            out_path = Path(eff_output)
            logger.info("Saving extraction report atomically to: %s", out_path)
            report.save(out_path, indent=2, as_envelope=False)

        return report


def run_extraction(
    target_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[ExtractorConfig] = None,
    mock: bool = False,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> ExtractionReport:
    """Convenience functional interface executing the extraction pipeline on a target.

    Args:
        target_dir: Directory or file to extract.
        output_file: Path to output JSON file.
        config: Optional pre-configured ExtractorConfig.
        mock: Force deterministic mock mode if True.
        max_workers: Thread worker pool limit.
        **kwargs: Additional configuration overrides.

    Returns:
        ExtractionReport instance.
    """
    eff_config = config or get_default_config()
    if mock:
        eff_config.mock = True
    if max_workers is not None:
        eff_config.max_workers = max_workers
    for k, v in kwargs.items():
        if hasattr(eff_config, k):
            setattr(eff_config, k, v)

    pipeline = ExtractionPipeline(config=eff_config, max_workers=eff_config.max_workers)
    return pipeline.run(target_dir=target_dir, output_file=output_file)


__all__ = [
    "ExtractionPipeline",
    "run_extraction",
]
