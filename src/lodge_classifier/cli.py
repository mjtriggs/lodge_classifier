from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from lodge_classifier.config import build_default_config
from lodge_classifier.pipeline import run_pipeline


def main() -> None:
    """CLI entrypoint for running the lodge classification pipeline."""
    parser = argparse.ArgumentParser(description="Run lodge name classification.")
    parser.add_argument("--repo-root", type=str, default=".", help="Path to repository root.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV containing lodge_name_raw column.",
    )
    parser.add_argument(
        "--manual",
        type=str,
        default="",
        help="Optional path to manual curation CSV.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg = build_default_config(repo_root)

    input_path = Path(args.input).resolve()
    manual_path = Path(args.manual).resolve() if args.manual else None

    logger.info("Running from repo_root={root}", root=str(repo_root))
    run_pipeline(cfg=cfg, lodge_names_path=input_path, manual_curation_path=manual_path)


if __name__ == "__main__":
    main()
