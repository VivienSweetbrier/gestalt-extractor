"""
gestalt_extractor.testbed
=========================
Local workspace testbed and fixture manuscripts.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

__all__ = ["FIXTURES_DIR"]
