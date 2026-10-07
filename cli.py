"""
gestalt_extractor.cli
=====================
Command-line interface for the Gestalt Extractor pipeline.
Ingests target directory or file, classifies domains, extracts computational
bounds via LLM or deterministic mock provider, and writes structured JSON reports.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from gestalt_extractor.config import (
    ExtractorConfig,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_WORKERS,
)
from gestalt_extractor.pipeline import ExtractionPipeline

logger = logging.getLogger("gestalt_extractor")


def build_parser() -> argparse.ArgumentParser:
    """Builds the standard library argparse ArgumentParser for Gestalt Extractor."""
    parser = argparse.ArgumentParser(
        prog="gestalt-extractor",
        description="Gestalt Extractor: Automated manuscript parsing, domain classification, and mathematical bound extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=".",
        help="Target manuscript file or directory containing .tex/.md manuscripts to scan.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination path for the serialized JSON extraction report.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force offline deterministic heuristic extraction mode without invoking external APIs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="OpenAI LLM model identifier to use when mock mode is not active.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent worker threads for batch processing.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress informational stdout output and banner messages.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging output.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint function for Gestalt Extractor CLI.

    Args:
        argv: Optional command-line argument sequence (defaults to sys.argv[1:]).

    Returns:
        int: 0 on success, non-zero exit code on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    target_path = Path(args.target)
    if not target_path.exists():
        if not args.quiet:
            sys.stderr.write(f"Error: Target path does not exist: {target_path}\n")
        return 1

    # Build config from CLI arguments
    config = ExtractorConfig(
        output_file=args.output,
        mock=args.mock,
        model=args.model,
        max_workers=max(1, args.max_workers),
        verbose=args.verbose,
    )

    if not args.quiet:
        print("==================================================")
        print("          GESTALT EXTRACTOR PIPELINE              ")
        print("==================================================")
        print(f" Target : {target_path.resolve()}")
        print(f" Output : {Path(args.output).resolve()}")
        print(f" Mode   : {'Deterministic Mock' if config.is_mock_enabled() else f'OpenAI ({args.model})'}")
        print(f" Workers: {config.max_workers}")
        print("--------------------------------------------------")

    try:
        pipeline = ExtractionPipeline(config=config, max_workers=config.max_workers)
        report = pipeline.run(target_dir=target_path, output_file=args.output)

        num_records = len(report)
        meta = report.metadata or {}
        num_errors = meta.get("failed", 0)

        if not args.quiet:
            print("--------------------------------------------------")
            print(f" Extraction Complete: {num_records} manuscript(s) processed.")
            if num_errors > 0:
                print(f" Warnings: {num_errors} file(s) encountered processing errors.")
            print(f" Report successfully saved to: {Path(args.output).resolve()}")
            print("==================================================")

        return 0

    except Exception as exc:
        logger.exception("Fatal error during extraction pipeline execution: %s", exc)
        if not args.quiet:
            sys.stderr.write(f"Fatal Extraction Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
