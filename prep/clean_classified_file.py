from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def create_review_file(
    input_csv_path: str | Path,
    *,
    output_filename: str = "test_review.csv",
) -> Path:
    """
    Create a structured review file from a lodge classification output CSV.

    The function:
        - Loads the input CSV.
        - Selects key classification columns.
        - Adds manual review columns:
            LANGUAGE_CORRECT
            PRIMARY_ONTOLOGY_CORRECT
            SECONDARY_ONTOLOGY_CORRECT
            THEME_PRIMARY_CORRECT
            THEME_SECONDARY_CORRECT
          These are prefilled with "N/A" where the corresponding prediction is null,
          otherwise left blank for reviewer entry as "Y" or "N".
        - Adds REVIEW_COMPLETE (blank) for reviewer sign-off.
        - Enforces a strict, audit-friendly column order.
        - Writes the output to `output_filename` in the same directory as the input file.

    Reviewer guidance:
        - Use "Y" if the model output is correct, "N" if incorrect, "N/A" if not applicable.
        - Optionally set REVIEW_COMPLETE to "Y" when the row has been fully reviewed.

    Args:
        input_csv_path: Path to the classification output CSV.
        output_filename: Name for the output file (written into the input file's folder).

    Returns:
        Path to the created review CSV file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Loading input file: %s", input_path)
    df = pd.read_csv(input_path)

    required_columns = [
        "id",
        "lodge_name_raw",
        "lodge_name_clean",
        "language_primary",
        "ontology_primary",
        "ontology_secondary",
        "theme_primary",
        "theme_secondary",
    ]
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found columns: {sorted(df.columns)}"
        )

    logger.info("Selecting review columns")
    review_df = df.loc[:, required_columns].copy()

    logger.info("Adding review columns (Y/N/N/A) and REVIEW_COMPLETE")

    def _blank_or_na(series: pd.Series) -> pd.Series:
        """
        Return a review column initialised to blank, but set to "N/A" where series is null.
        """
        out = pd.Series([""] * len(series), index=series.index, dtype="string")
        out = out.mask(series.isna(), "N/A")
        return out

    review_df["LANGUAGE_CORRECT"] = _blank_or_na(review_df["language_primary"])
    review_df["PRIMARY_ONTOLOGY_CORRECT"] = _blank_or_na(review_df["ontology_primary"])
    review_df["SECONDARY_ONTOLOGY_CORRECT"] = _blank_or_na(review_df["ontology_secondary"])
    review_df["THEME_PRIMARY_CORRECT"] = _blank_or_na(review_df["theme_primary"])
    review_df["THEME_SECONDARY_CORRECT"] = _blank_or_na(review_df["theme_secondary"])

    # Reviewer sign-off / completeness flag (left blank by default)
    review_df["REVIEW_COMPLETE"] = pd.Series([""] * len(review_df), dtype="string")

    # Enforce strict column order
    ordered_cols = [
        "id",
        "lodge_name_raw",
        "lodge_name_clean",
        "language_primary",
        "ontology_primary",
        "ontology_secondary",
        "theme_primary",
        "theme_secondary",
        "LANGUAGE_CORRECT",
        "PRIMARY_ONTOLOGY_CORRECT",
        "SECONDARY_ONTOLOGY_CORRECT",
        "THEME_PRIMARY_CORRECT",
        "THEME_SECONDARY_CORRECT",
        "REVIEW_COMPLETE",
    ]
    review_df = review_df.loc[:, ordered_cols]

    output_path = input_path.parent / output_filename
    logger.info("Writing review file to: %s", output_path)
    review_df.to_csv(output_path, index=False)

    logger.info("Review file created successfully")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    INPUT_CSV = "data/test_files/output/test_2_2026_02_23/classified_b973d36a1e.csv"

    created_path = create_review_file(INPUT_CSV, output_filename="test_review.csv")
    logger.info("Done. Review file available at: %s", created_path)
