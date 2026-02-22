from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def create_random_samples(
    input_csv_path: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 20,
    sample_size: int = 100,
    seed: int = 42,
) -> list[Path]:
    """
    Read a lodge CSV, retain new_id and lodge_name, rename them to
    id and lodge_name_raw, and write multiple repeatable random samples.

    Sampling is deterministic: each output file uses a random_state
    derived from `seed`.

    Args:
        input_csv_path: Path to the source CSV file.
        output_dir: Directory to write sample CSV files into (created if missing).
        n_samples: Number of sample files to create.
        sample_size: Number of rows per sample file.
        seed: Base random seed for repeatability.

    Returns:
        A list of Paths to the created sample files.

    Raises:
        FileNotFoundError: If input_csv_path does not exist.
        ValueError: If required columns are missing or sample_size is invalid.
    """
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = {"new_id", "lodge_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )

    df_small = (
        df.loc[:, ["new_id", "lodge_name"]]
        .rename(columns={"new_id": "new_id", "lodge_name": "lodge_name_raw"})
        .copy()
    )

    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    if len(df_small) < sample_size:
        raise ValueError(
            f"Not enough rows to sample without replacement: "
            f"rows={len(df_small)} < sample_size={sample_size}."
        )

    written_files: list[Path] = []
    logger.info(
        "Creating %s samples of size %s from %s rows (seed=%s). Output: %s",
        n_samples,
        sample_size,
        len(df_small),
        seed,
        out_dir,
    )

    for i in range(1, n_samples + 1):
        sample_df = df_small.sample(
            n=sample_size,
            replace=False,
            random_state=seed + i,
        )

        out_path = out_dir / f"lodge_name_sample_{i:02d}_n{sample_size}_seed{seed}.csv"
        sample_df.to_csv(out_path, index=False)

        written_files.append(out_path)
        logger.info("Wrote sample %s/%s: %s", i, n_samples, out_path)

    return written_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Update these two paths:
    INPUT_CSV = "data/processed/test_input_lodges_normalised.csv"
    OUTPUT_DIR = "data/test_files/input"

    created = create_random_samples(
        input_csv_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        n_samples=20,
        sample_size=100,
        seed=12345,
    )

    logger.info("Done. Created %s files.", len(created))
