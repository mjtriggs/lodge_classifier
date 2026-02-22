from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def stable_new_id(
    *,
    original_id: Any,
    name_slot: str,
    lodge_name: Optional[str],
    name_date: Optional[str],
) -> str:
    """
    Create a deterministic surrogate ID for a lodge name event.

    Args:
        original_id: Original lodge identifier from the source dataset.
        name_slot: Slot identifier (e.g. "original", "other_1", ..., "other_6").
        lodge_name: The lodge name for this event.
        name_date: The date associated with the name event.

    Returns:
        A 32-character hex string (MD5 hash).
    """
    base = f"{original_id}||{name_slot}||{lodge_name or ''}||{name_date or ''}"
    # MD5 is used for determinism/compactness, not for security.
    return hashlib.md5(base.encode("utf-8")).hexdigest()  # noqa: S324


def normalise_lodge_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise lodge names so each lodge name event becomes its own row.

    Rules:
        - Keep all LodgeNumber* and ReNumber* columns.
        - Keep Lapsed and Erased flags.
        - Create one row for the original LodgeName/DateNamed.
        - Create additional rows for OtherName1..6 where present.
        - Add name_type/name_slot fields to preserve provenance.
        - Create a deterministic new_id per name event.

    Args:
        df: Input DataFrame with the original wide-format lodge data.

    Returns:
        A normalised DataFrame with one row per lodge name event.
    """
    required_cols = {"ID", "LodgeName", "DateNamed", "Lapsed", "Erased"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    number_cols: List[str] = [
        c for c in df.columns if c.startswith("LodgeNumber") or c.startswith("ReNumber")
    ]
    keep_cols = number_cols + ["Lapsed", "Erased"]

    events: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        original_id = r["ID"]

        original_name_raw = r.get("LodgeName")
        original_name = "" if pd.isna(original_name_raw) else str(original_name_raw).strip()

        events.append(
            {
                "original_id": original_id,
                "lodge_name": original_name,
                "name_date": None if pd.isna(r.get("DateNamed")) else r.get("DateNamed"),
                "name_type": "original",
                "name_slot": "original",
                **{c: r.get(c) for c in keep_cols},
            }
        )

        for i in range(1, 7):
            other_name_col = f"OtherName{i}"
            other_date_col = f"OtherDate{i}"

            if other_name_col not in df.columns:
                continue

            other_name_raw = r.get(other_name_col)
            if pd.isna(other_name_raw) or str(other_name_raw).strip() == "":
                continue

            other_name = str(other_name_raw).strip()
            other_date = r.get(other_date_col) if other_date_col in df.columns else None
            other_date = None if pd.isna(other_date) else other_date

            events.append(
                {
                    "original_id": original_id,
                    "lodge_name": other_name,
                    "name_date": other_date,
                    "name_type": "alternate",
                    "name_slot": f"other_{i}",
                    **{c: r.get(c) for c in keep_cols},
                }
            )

    out = pd.DataFrame(events)

    out.insert(
        0,
        "new_id",
        [
            stable_new_id(
                original_id=row["original_id"],
                name_slot=row["name_slot"],
                lodge_name=row.get("lodge_name"),
                name_date=None if pd.isna(row.get("name_date")) else str(row.get("name_date")),
            )
            for _, row in out.iterrows()
        ],
    )

    logger.info("Normalised {} input rows into {} name-event rows", len(df), len(out))
    return out


def normalise_lodge_names_csv(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    encoding: Optional[str] = None,
) -> None:
    """
    Read a cleaned lodge CSV, normalise lodge names, and write the result to CSV.

    Args:
        input_csv: Path to input CSV.
        output_csv: Path to output CSV.
        encoding: Optional file encoding to pass to pandas.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    logger.info("Reading input CSV: {}", input_csv)
    df = pd.read_csv(input_csv, encoding=encoding)

    logger.info("Normalising lodge names")
    out = normalise_lodge_names(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing output CSV: {}", output_csv)
    out.to_csv(output_csv, index=False)

    logger.info("Done")
