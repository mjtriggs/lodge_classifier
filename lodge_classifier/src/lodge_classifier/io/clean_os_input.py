from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from loguru import logger


def _read_header_names(header_file: str) -> List[str]:
    """
    Read OS Open Names header file and return the column names in order.

    Parameters
    ----------
    header_file : str
        Path to the header file containing column names.

    Returns
    -------
    List[str]
        Ordered list of column names.
    """
    p = Path(header_file)
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"Header file is empty: {header_file}")

    if "\t" in text and "," not in text:
        cols = text.split("\t")
    elif "," in text and "\t" not in text:
        cols = text.split(",")
    else:
        cols = text.split("\t") if "\t" in text else text.split(",")

    cols = [c.strip() for c in cols if c.strip()]
    if not cols:
        raise ValueError(f"Could not parse column names from header file: {header_file}")

    return cols


def _normalise_name_series(s: pd.Series) -> pd.Series:
    """
    Normalise names for matching.

    Parameters
    ----------
    s : pd.Series
        Series containing candidate names.

    Returns
    -------
    pd.Series
        Normalised names: stripped, lowercased, curly apostrophes replaced.
    """
    return s.astype("string").str.strip().str.lower().str.replace("’", "'", regex=False)


def _normalise_type_series(s: pd.Series) -> pd.Series:
    """
    Normalise TYPE-like series.

    Parameters
    ----------
    s : pd.Series
        Series containing candidate types.

    Returns
    -------
    pd.Series
        Normalised types: stripped and lowercased.
    """
    return s.astype("string").str.strip().str.lower()


def _add_names(target: Set[str], series: pd.Series) -> None:
    """
    Add normalised, non-empty names to a set.

    Parameters
    ----------
    target : Set[str]
        Set to add into.
    series : pd.Series
        Candidate names.
    """
    if series is None:
        return

    s = _normalise_name_series(series).dropna()
    s = s[s.str.len() > 0]
    target.update(s.tolist())


def build_os_open_names_lists_chunked(
    input_folder: str,
    header_file: str,
    output_folder: str,
    chunksize: int = 250_000,
    include_district_borough_in_reg: bool = False,
    file_glob: str = "*.csv",
    debug_first_chunk_type_counts: bool = True,
) -> Dict[str, Path]:
    """
    Build three OS Open Names derived name lists aligned to taxonomy buckets using
    chunked processing and a separate header definition file.

    Outputs:
    - loc_reg.csv: region/county/territory style admin areas
    - loc_cty.csv: populated places (cities, towns, villages etc.)
    - loc_lan.csv: natural landmarks and physical features

    Parameters
    ----------
    input_folder : str
        Folder containing OS Open Names data CSV files (often no header row).
    header_file : str
        Path to the file containing the header names in order.
    output_folder : str
        Folder to write output CSVs into.
    chunksize : int
        Number of rows per chunk.
    include_district_borough_in_reg : bool
        If True, include DISTRICT_BOROUGH in loc_reg.csv.
    file_glob : str
        Glob pattern for data files.
    debug_first_chunk_type_counts : bool
        If True, logs the top TYPE values from the first chunk for quick validation.

    Returns
    -------
    Dict[str, Path]
        Mapping of bucket name to output CSV path.
    """
    input_path = Path(input_folder)
    csv_files = sorted(input_path.glob(file_glob))
    if not csv_files:
        raise FileNotFoundError(f"No files matched {file_glob} in: {input_folder}")

    cols = _read_header_names(header_file)
    logger.info(f"Parsed {len(cols)} header columns from {header_file}")

    required = [
        "NAME1",
        "TYPE",
        "LOCAL_TYPE",
        "COUNTY_UNITARY",
        "REGION",
        "COUNTRY",
    ]
    if include_district_borough_in_reg:
        required.append("DISTRICT_BOROUGH")

    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns in header definition: {missing}")

    loc_reg: Set[str] = set()
    loc_cty: Set[str] = set()
    loc_lan: Set[str] = set()

    # Local types that are strong signals for LOC_LAN, used as a robust fallback
    landmark_local_types = {
        "river",
        "stream",
        "brook",
        "burn",
        "beck",
        "water",
        "lake",
        "loch",
        "reservoir",
        "mere",
        "pond",
        "hill",
        "mountain",
        "fell",
        "tor",
        "down",
        "moor",
        "valley",
        "glen",
        "wood",
        "woodland",
        "forest",
        "park",
        "island",
        "cliff",
        "head",
        "bay",
        "cove",
        "beach",
        "coast",
        "ridge",
    }

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(csv_files)} data files. Reading chunks of {chunksize:,} rows.")

    for file_idx, f in enumerate(csv_files):
        logger.info(f"Processing file: {f.name}")

        chunk_iter = pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=required,
            chunksize=chunksize,
            low_memory=False,
        )

        for i, chunk in enumerate(chunk_iter, start=1):
            chunk["TYPE"] = _normalise_type_series(chunk["TYPE"])
            chunk["LOCAL_TYPE"] = _normalise_type_series(chunk["LOCAL_TYPE"])
            chunk["NAME1"] = _normalise_name_series(chunk["NAME1"])

            if debug_first_chunk_type_counts and file_idx == 0 and i == 1:
                logger.info("Top TYPE values in first chunk (lowercased):")
                logger.info(chunk["TYPE"].value_counts(dropna=False).head(25))

            # LOC_CTY: populated places
            _add_names(loc_cty, chunk.loc[chunk["TYPE"] == "populatedplace", "NAME1"])

            # LOC_REG: admin boundaries plus admin columns
            _add_names(loc_reg, chunk.loc[chunk["TYPE"] == "administrativeboundary", "NAME1"])
            for col in ["COUNTY_UNITARY", "REGION", "COUNTRY"]:
                _add_names(loc_reg, chunk[col])

            if include_district_borough_in_reg and "DISTRICT_BOROUGH" in chunk.columns:
                _add_names(loc_reg, chunk["DISTRICT_BOROUGH"])

            # LOC_LAN: physicalfeature OR LOCAL_TYPE in landmark list
            is_physical_feature = chunk["TYPE"] == "physicalfeature"
            is_landmark_local = chunk["LOCAL_TYPE"].isin(landmark_local_types)
            _add_names(loc_lan, chunk.loc[is_physical_feature | is_landmark_local, "NAME1"])

            if i % 10 == 0:
                logger.info(
                    f"{f.name}: chunks {i:,} | "
                    f"LOC_REG={len(loc_reg):,} LOC_CTY={len(loc_cty):,} LOC_LAN={len(loc_lan):,}"
                )

    def _write(names: Set[str], path: Path) -> None:
        pd.DataFrame({"name": sorted(names)}).to_csv(path, index=False)

    loc_reg_path = out_path / "loc_reg.csv"
    loc_cty_path = out_path / "loc_cty.csv"
    loc_lan_path = out_path / "loc_lan.csv"

    _write(loc_reg, loc_reg_path)
    _write(loc_cty, loc_cty_path)
    _write(loc_lan, loc_lan_path)

    logger.success(
        "Done. " f"LOC_REG={len(loc_reg):,} LOC_CTY={len(loc_cty):,} LOC_LAN={len(loc_lan):,}"
    )

    return {"LOC_REG": loc_reg_path, "LOC_CTY": loc_cty_path, "LOC_LAN": loc_lan_path}


if __name__ == "__main__":

    INPUT_PATH = "data/input/os_open_names"
    OUTPUT_FOLDER = "data/processed"
    HEADER_PATH = "data/input/OS_Open_names_Header.csv"

    # build_os_open_names_location_lists(
    #     input_folder=INPUT_PATH, output_folder=OUTPUT_FOLDER, include_district_borough_in_reg=False
    # )

    paths = build_os_open_names_lists_chunked(
        input_folder=INPUT_PATH,
        header_file=HEADER_PATH,
        output_folder=OUTPUT_FOLDER,
        chunksize=300_000,
        include_district_borough_in_reg=False,
    )

    print(paths)
