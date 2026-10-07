"""
gestalt_extractor.testbed.runner
================================
Local testbed execution harness targeting DreamLLM workspace manuscripts
(e.g., testbed fixtures or research directories) without external network
access or repository cloning.

Validates that:
1. Target directory is discovered and processed without crashes.
2. The pipeline executes using offline deterministic mock extraction.
3. The final report is saved to disk as a valid JSON file.
4. Every extracted record strictly contains mandatory keys:
   `file`, `domain`, `bounds`, and `summary`.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from gestalt_extractor.config import ExtractorConfig
from gestalt_extractor.pipeline import ExtractionPipeline

logger = logging.getLogger("gestalt_extractor.testbed")


def resolve_default_target_dir() -> Path:
    """Discovers a valid local testbed manuscript directory in DreamLLM."""
    module_dir = Path(__file__).resolve().parent
    fixtures_dir = module_dir / "fixtures"

    candidates = [
        fixtures_dir,
        Path("gestalt_extractor/testbed/fixtures"),
        Path("research"),
        Path("fixtures"),
        Path("."),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if directory contains at least one .tex or .md manuscript
            has_manuscript = any(
                p.suffix.lower() in {".tex", ".md"}
                for p in candidate.rglob("*")
                if p.is_file() and not any(part.startswith(".") for part in p.parts)
            )
            if has_manuscript:
                logger.debug("Resolved testbed target directory: %s", candidate.resolve())
                return candidate.resolve()

    # Fallback to fixtures_dir
    return fixtures_dir.resolve()


def run_testbed(
    target_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    mock: bool = True,
    max_workers: int = 4,
) -> Path:
    """Executes the extraction pipeline against a local workspace testbed directory.

    Args:
        target_dir: Directory containing .tex/.md manuscripts. Defaults to local testbed fixtures.
        output_file: Destination path for JSON report. Defaults to 'gestalt_extraction_report.json'.
        mock: Enforce deterministic mock extraction mode (True by default).
        max_workers: Concurrent worker threads.

    Returns:
        Path: Resolved filesystem path to the verified output JSON file.

    Raises:
        FileNotFoundError: If target directory does not exist.
        AssertionError: If output file is missing or records violate the schema.
    """
    # 1. Resolve target directory
    if target_dir is None:
        target_path = resolve_default_target_dir()
    else:
        target_path = Path(target_dir).resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Testbed target directory does not exist: {target_path}")

    # 2. Resolve output path
    if output_file is None:
        out_path = Path("gestalt_extraction_report.json").resolve()
    else:
        out_path = Path(output_file).resolve()

    logger.info("Running Gestalt Extractor testbed targeting: %s", target_path)
    logger.info("Destination report path: %s", out_path)

    # 3. Configure and execute extraction pipeline
    config = ExtractorConfig(
        output_file=str(out_path),
        mock=mock,
        max_workers=max_workers,
    )
    pipeline = ExtractionPipeline(config=config, max_workers=max_workers)
    report = pipeline.run(target_dir=target_path, output_file=out_path)

    # 4. Strict Verification of Output File and Invariant Schema
    if not out_path.exists():
        raise AssertionError(f"Testbed verification failed: output file was not created at {out_path}")

    if out_path.stat().st_size == 0:
        raise AssertionError(f"Testbed verification failed: output file at {out_path} is 0 bytes")

    with open(out_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as err:
            raise AssertionError(f"Testbed verification failed: output file is not valid JSON: {err}")

    # Normalize records list from JSON payload
    records: List[Dict[str, Any]] = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, dict) and "file" in data:
        records = [data]
    else:
        raise AssertionError(f"Testbed verification failed: unexpected JSON structure ({type(data).__name__})")

    if not records:
        raise AssertionError(f"Testbed verification failed: report contains 0 extracted records from {target_path}")

    required_keys = {"file", "domain", "bounds", "summary"}
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise AssertionError(f"Record #{idx} is not a dictionary: {rec}")

        missing = required_keys - set(rec.keys())
        if missing:
            raise AssertionError(
                f"Record #{idx} ({rec.get('file')}) is missing mandatory schema keys: {missing}"
            )

        # Validate types of mandatory keys
        if not isinstance(rec["file"], str) or not rec["file"]:
            raise AssertionError(f"Record #{idx} has invalid 'file' field: {rec['file']}")

        if not isinstance(rec["domain"], str) or not rec["domain"]:
            raise AssertionError(f"Record #{idx} has invalid 'domain' field: {rec['domain']}")

        if not isinstance(rec["bounds"], list):
            raise AssertionError(f"Record #{idx} 'bounds' field must be a list: {type(rec['bounds'])}")

        if not isinstance(rec["summary"], str) or not rec["summary"]:
            raise AssertionError(f"Record #{idx} has empty or non-string 'summary': {rec['summary']}")

    logger.info(
        "Testbed verification succeeded: %d record(s) validated against mandatory schema in %s",
        len(records),
        out_path,
    )
    return out_path


def main() -> int:
    """CLI entrypoint for standalone testbed execution."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        out_file = run_testbed()
        print(f"\nSUCCESS: Testbed run completed and verified at: {out_file}\n")
        return 0
    except Exception as exc:
        logger.exception("Testbed run failed: %s", exc)
        print(f"\nFAILURE: Testbed run failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
