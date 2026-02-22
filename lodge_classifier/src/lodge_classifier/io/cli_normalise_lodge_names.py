from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from lodge_classifier.io.normalise_lodge_names import normalise_lodge_names_csv


def build_parser() -> argparse.ArgumentParser:
    """
    Build an argument parser for the lodge name normalisation CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="lodge-normalise-names",
        description="Normalise lodge names so each name event becomes its own row.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the cleaned input CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the normalised CSV.",
    )
    parser.add_argument(
        "--encoding",
        required=False,
        type=str,
        default=None,
        help="Optional file encoding to use when reading the CSV.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point for the lodge name normalisation CLI.

    Args:
        argv: Optional list of CLI arguments (primarily for testing).

    Returns:
        Process exit code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.info("Running lodge name normalisation")
    normalise_lodge_names_csv(
        input_csv=args.input,
        output_csv=args.output,
        encoding=args.encoding,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
